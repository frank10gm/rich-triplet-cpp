#pragma once

// =============================================================================
// Llama 3.2 -- inference only
// =============================================================================
//
// The backbone Orpheus is fine-tuned from. Structurally it is the plainest
// architecture in this project: one attention block, one FFN, two norms, full
// causal attention everywhere.
//
// ## What it is not
//
// Read alongside `transformer4.hpp` (Gemma 3), because the differences are all
// places a shared implementation would go quietly wrong:
//
// | Feature                 | Gemma 3                        | Llama 3.2         |
// |-------------------------|--------------------------------|-------------------|
// | Norms per block         | 4                              | **2**             |
// | Per-head Q/K RMSNorm    | yes                            | **no**            |
// | Attention scale         | 1/sqrt(query_pre_attn_scalar)  | 1/sqrt(head_dim)  |
// | Embedding scaling       | x sqrt(hidden_size)            | **none**          |
// | GGUF norm gamma         | stored as `1 + gamma`          | **stored raw**    |
// | FFN gate                | gelu_pytorch_tanh              | **SiLU**          |
// | Attention span          | 5 local : 1 global, two thetas | all global, one   |
// | lm_head                 | weight-tied                    | **separate**      |
// | RoPE frequency scaling  | one scalar                     | **per dimension** |
//
// Two of those bite hardest. Subtracting 1 from a Llama norm gamma the way the
// Gemma loader must produces immediate garbage -- loud, easy. The RoPE scaling
// is the quiet one; see `Config5::inv_freq`.
//
// ## Llama 3 RoPE scaling
//
// Llama 3.2 does not scale RoPE by a single factor. It divides each frequency
// band by a different amount: high-frequency dimensions are left alone so
// local structure is preserved, low-frequency ones are divided by 32 so
// positions stretch, with a smooth ramp between. GGUF ships the resulting
// per-dimension divisors in `rope_freqs.weight`, so there is no need to
// reimplement the piecewise formula -- but they are **divisors**, running from
// 1.0 up to 32.0, not multipliers running down to 1/32.
//
// Applying them the wrong way round scales the low-frequency dimensions by 32
// instead of by 1/32 -- a factor of 1024 on exactly the dimensions that carry
// long-range position. Because the unscaled high-frequency dimensions still
// dominate local structure, the output starts out plausible and degrades as
// the sequence grows, which is close to the worst failure mode to debug.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "rt/gguf.hpp"
#include "rt/mat.hpp"
#include "rt/nn2.hpp"
#include "rt/result.hpp"
#include "rt/sampling.hpp"
#include "rt/tokenizer.hpp"
#include "rt/tensor_node.hpp"

namespace rt {

// =============================================================================
// RoPE conventions
// =============================================================================

/// Which dimensions RoPE rotates against each other.
///
/// This is not a free choice -- it has to match how the weights were stored,
/// and the two conventions in circulation disagree:
///
///   HalfSplit    dimension `i` rotates with `i + head_dim/2`
///   Interleaved  dimension `2i` rotates with `2i + 1`
///
/// HuggingFace's `LlamaAttention` uses `rotate_half`, which is HalfSplit.
/// llama.cpp's llama path uses Interleaved -- and reconciles the two by
/// **permuting the Q and K weight rows during conversion**, so that
/// interleaved rotation of the permuted weights reproduces half-split rotation
/// of the originals. `convert_hf_to_gguf.py` does this for the llama
/// architecture but not for Gemma, which is why `transformer4` rotates
/// half-split and this rotates interleaved off the very same file format.
///
/// Getting it wrong is close to undetectable at first. Both conventions rotate
/// by the same angles and differ only in which pairs those angles apply to, so
/// at low positions -- where every angle is small and every rotation near
/// identity -- the output looks fine. It falls apart as positions grow, which
/// for a speech model means the first few tokens are right and the rest is
/// noise.
enum class RopePairing {
    HalfSplit,
    Interleaved,
};

// =============================================================================
// Config5
// =============================================================================

struct Config5 {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t num_hidden_layers = 0;
    std::size_t num_attention_heads = 0;
    std::size_t num_key_value_heads = 0;
    std::size_t intermediate_size = 0;
    /// Explicit, though for Llama 3.2 it equals hidden_size / n_heads.
    std::size_t head_dim = 0;
    float rope_theta = 500000.0f;
    float rms_norm_eps = 1e-5f;
    std::size_t max_position_embeddings = 131072;
    std::size_t eos_token_id = 128009;

    /// Per-dimension RoPE divisors, `head_dim / 2` of them, as shipped in
    /// GGUF's `rope_freqs.weight`. Empty means unscaled RoPE.
    std::vector<float> rope_freq_divisors;

    /// Interleaved for GGUF, which stores permuted Q/K weights. See
    /// `RopePairing`.
    RopePairing rope_pairing = RopePairing::Interleaved;

    /// canopylabs/orpheus-3b-0.1-ft, whose backbone is Llama 3.2 3B Instruct
    /// with the vocabulary extended by 28 672 audio tokens.
    ///
    /// Values are those in the published GGUF metadata: 28 layers, hidden
    /// 3072, 24 query heads over 8 KV heads, head_dim 128, FFN 8192,
    /// rope_theta 500000, RMS epsilon 1e-5, vocabulary 156 940.
    [[nodiscard]] static Config5 orpheus_3b();

    /// Inverse RoPE frequencies, one per rotated pair.
    ///
    /// `inv_freq[i] = theta^(-2i/head_dim) / rope_freq_divisors[i]`, so a
    /// divisor of 32 stretches that band by 32x. With no divisors this is
    /// plain RoPE.
    [[nodiscard]] std::vector<float> inv_freq() const;

    /// Query heads sharing each KV head.
    [[nodiscard]] std::size_t gqa_group() const {
        return num_key_value_heads == 0 ? 1 : num_attention_heads / num_key_value_heads;
    }
};

// =============================================================================
// KV cache
// =============================================================================

/// One layer's cached keys and values, preallocated to the session length.
struct LlamaLayerKvCache {
    /// [max_seq_len, n_kv_heads * head_dim]
    Mat k;
    Mat v;
    std::size_t seq_len = 0;

    LlamaLayerKvCache(std::size_t n_kv_heads, std::size_t head_dim, std::size_t max_seq_len)
        : k(Mat::zeros(max_seq_len, n_kv_heads * head_dim)),
          v(Mat::zeros(max_seq_len, n_kv_heads * head_dim)) {}

    /// Append rows, both [n_new, n_kv_heads * head_dim].
    void append(const Mat& new_k, const Mat& new_v);
};

struct LlamaKvCache {
    std::vector<LlamaLayerKvCache> layers;

    LlamaKvCache(const Config5& config, std::size_t max_tokens);

    void clear();
    /// Release the memory, for after a GPU upload.
    void free();
};

// =============================================================================
// Layers
// =============================================================================

/// Grouped-query attention with full causal span and no per-head norms.
class LlamaAttention {
   public:
    Linear2 q_proj;
    Linear2 k_proj;
    Linear2 v_proj;
    Linear2 o_proj;
    std::size_t n_q_heads = 0;
    std::size_t n_kv_heads = 0;
    std::size_t head_dim = 0;
    /// 1/sqrt(head_dim).
    float attn_scale = 0.0f;
    /// Shared with the model; `head_dim / 2` entries.
    const std::vector<float>* inv_freq = nullptr;
    RopePairing rope_pairing = RopePairing::Interleaved;

    explicit LlamaAttention(const Config5& cfg);

    [[nodiscard]] static LlamaAttention new_for_inference(const Config5& cfg);

    /// Project to Q/K/V, rotate Q and K at their absolute positions, append to
    /// the cache, then attend over the whole cache.
    [[nodiscard]] Mat forward_cached(const Mat& x, LlamaLayerKvCache& cache) const;
};

/// `down_proj(silu(gate_proj(x)) * up_proj(x))`.
class LlamaMlp {
   public:
    Linear2 gate_proj;
    Linear2 up_proj;
    Linear2 down_proj;

    explicit LlamaMlp(const Config5& cfg);

    [[nodiscard]] static LlamaMlp new_for_inference(const Config5& cfg);

    [[nodiscard]] Mat forward(const Mat& x) const;
};

/// Pre-norm block:
///
///   x  = x + attn(input_layernorm(x))
///   x  = x + mlp(post_attention_layernorm(x))
///
/// Two norms, not Gemma's four. There is no norm on either sub-block's output.
class LlamaBlock {
   public:
    RmsNorm2 input_layernorm;
    LlamaAttention self_attn;
    RmsNorm2 post_attention_layernorm;
    LlamaMlp mlp;

    explicit LlamaBlock(const Config5& cfg);

    [[nodiscard]] static LlamaBlock new_for_inference(const Config5& cfg);

    [[nodiscard]] Mat forward_cached(const Mat& x, LlamaLayerKvCache& cache) const;
};

// =============================================================================
// Model
// =============================================================================

class LlamaModel {
   public:
    /// BF16 embedding table, [vocab_size, hidden_size]. Looked up row by row,
    /// so it never needs an f32 copy.
    std::optional<MatBf16> embed_bf16;
    Config5 config;
    std::vector<LlamaBlock> layers;
    RmsNorm2 norm;
    /// Separate from the embedding: Llama 3.2 3B ties them, but Orpheus does
    /// not, and the extended vocabulary makes this the largest single weight.
    Linear2 lm_head;
    /// Precomputed once and shared by every layer.
    std::vector<float> inv_freq_cache;

    explicit LlamaModel(Config5 cfg);

    [[nodiscard]] static LlamaModel new_for_inference(Config5 cfg);

    /// Load from a GGUF file with `general.architecture == "llama"`.
    ///
    /// Reads `token_embd`, `output`, `output_norm`, `rope_freqs` and the
    /// `blk.{i}.*` tensors. Norm gammas are taken **verbatim** -- the `1 +
    /// gamma` convention is Gemma's alone.
    [[nodiscard]] Result<void> load_weights_from_gguf(const std::string& path);

    /// Quantize `lm_head` from BF16 to Q4_K, freeing the BF16 copy.
    ///
    /// Worth doing for a large vocabulary. Orpheus ships `output.weight` as
    /// Q6_K, which this loader widens to BF16 at 2 bytes an element -- 964 MB
    /// for 156 940 x 3072. Q4_K brings that to 271 MB, and since the lm_head
    /// GEMV is the single largest memory read per token, it speeds decode up
    /// as well.
    void quantize_lm_head();

    /// Wire `head_dim / 2` inverse frequencies into every layer.
    ///
    /// Called after loading, and again after any move, because the layers hold
    /// a pointer into `inv_freq_cache`.
    void rebind_inv_freq();

    /// Prefill the cache with `token_ids`, returning logits for the last token.
    [[nodiscard]] Mat forward_cached(const std::vector<std::size_t>& token_ids,
                                     LlamaKvCache& cache) const;

    /// Look up embedding rows. Unlike Gemma there is no sqrt(hidden) scaling.
    [[nodiscard]] Mat embed_rows(const std::vector<std::size_t>& token_ids) const;

    /// Autoregressive generation over a KV cache.
    ///
    /// `on_token` receives each sampled id and returns false to stop, which is
    /// how a caller detects an end-of-audio token it recognises but the config
    /// does not. Returns the number of tokens generated.
    std::size_t generate(const std::vector<std::size_t>& prompt, std::size_t max_new,
                         const SamplingParams& params, bool debug,
                         const std::function<bool(std::size_t)>& on_token) const;

    /// Total resident weight bytes, for reporting.
    [[nodiscard]] std::size_t weight_bytes() const;
};

// =============================================================================
// Helpers (exposed for testing)
// =============================================================================

/// Build a byte-level BPE tokenizer from a GGUF file's embedded vocabulary.
///
/// Reads `tokenizer.ggml.tokens`, `tokenizer.ggml.merges` and
/// `tokenizer.ggml.pre`. A `pre` of `llama-bpe` selects Llama 3's digit
/// grouping, which differs from GPT-2's in a way that changes the ids for
/// every number in the input.
///
/// This exists so a GGUF file is self-sufficient. Orpheus is gated on
/// HuggingFace, so requiring a separate `tokenizer.json` would mean requiring
/// an account for weights that are otherwise freely mirrored.
[[nodiscard]] Result<HfBpeTokenizer> load_gguf_tokenizer(const GgufFile& gguf);

/// Apply RoPE to every head of a [T, n_heads * head_dim] matrix.
///
/// Row `r` sits at absolute position `offset + r`.
[[nodiscard]] Mat llama_rope(const Mat& x, std::size_t n_heads, std::size_t head_dim,
                             std::size_t offset, const std::vector<float>& inv_freq,
                             RopePairing pairing);

/// Sample from a logits row without sorting the whole vocabulary.
///
/// Same pipeline as `sample_token`: repetition penalty, temperature, softmax,
/// top-k, top-p, then draw. The difference is that top-k uses
/// `std::nth_element` and only sorts the survivors, which for a 156 940-entry
/// vocabulary is the difference between ~10 ms and well under 1 ms per token.
/// At a realtime budget of 11.9 ms per token, a full sort would consume most of
/// it on its own.
///
/// `seen` holds previously generated ids for the repetition penalty.
[[nodiscard]] std::size_t sample_token_large_vocab(const Mat& logits, std::size_t row,
                                                   const SamplingParams& params,
                                                   const std::vector<std::size_t>& seen,
                                                   LcgRng& rng);

}  // namespace rt

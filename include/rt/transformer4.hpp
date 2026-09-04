#pragma once

// =============================================================================
// Gemma 3 -- inference-first architecture
// =============================================================================
//
// Implements the Gemma 3 family (1b, 4b, 12b, 27b) for text-only inference.
// Weights load from HuggingFace .safetensors shards or GGUF files.
//
// ## How it differs from GPT-OSS (transformer3)
//
// | Feature                | GPT-OSS               | Gemma 3                       |
// |------------------------|-----------------------|-------------------------------|
// | head_dim               | hidden / n_heads      | explicit (256)                |
// | Q/K per-head RMSNorm   | no                    | yes                           |
// | Attention scale        | 1/sqrt(d_head)        | 1/sqrt(query_pre_attn_scalar) |
// | Local/global attention | odd layers only       | 5 local : 1 global            |
// | FFN type               | MoE                   | dense SwiGLU (GELU-gated)     |
// | Vocab size             | 201088                | 262144 (1b) / 262208 (4b)     |
//
// The `Trainable` implementation carries backward rules, so the same
// architecture can be trained from scratch; the loaded-weight paths are
// inference-only.

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rt/init_rng.hpp"
#include "rt/nn2.hpp"
#include "rt/sampling.hpp"
#include "rt/tensor_node.hpp"

namespace rt {

/// Tag selecting the zero-placeholder constructors used by the inference paths.
struct InferenceInit {};

// =============================================================================
// Config4 -- Gemma 3 hyperparameters
// =============================================================================

/// `head_dim` is stored explicitly because Gemma 3 fixes it at 256 regardless
/// of `hidden_size / num_attention_heads`.
struct Config4 {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t num_hidden_layers = 0;
    /// Number of Q heads per layer.
    std::size_t num_attention_heads = 0;
    /// Number of K/V heads per layer -- fewer than Q heads (GQA).
    std::size_t num_key_value_heads = 0;
    std::size_t intermediate_size = 0;
    /// Explicit head dimension; always 256 in Gemma 3.
    std::size_t head_dim = 0;
    /// Sliding window for local attention layers; unset means full attention.
    std::optional<std::size_t> sliding_window;
    /// RoPE base theta: 10k for local layers, 1M for global.
    float rope_theta_local = 10000.0f;
    float rope_theta_global = 1000000.0f;
    /// RoPE linear frequency scale. 1.0 = none; Gemma 3 4B global layers use
    /// 1/8 because rope_scaling.factor is 8.
    float rope_freq_scale_local = 1.0f;
    float rope_freq_scale_global = 1.0f;
    float rms_norm_eps = 1e-6f;
    /// Attention score scale is 1/sqrt(query_pre_attn_scalar); Gemma 3 uses
    /// 256.0, so the scale is 1/16.
    float query_pre_attn_scalar = 256.0f;
    /// EOS token id, used to stop generation.
    std::size_t eos_token_id = 1;
    /// Maximum sequence length, for KV cache allocation.
    std::size_t max_position_embeddings = 32768;

    /// google/gemma-3-1b-it
    [[nodiscard]] static Config4 gemma3_1b();

    /// google/gemma-3-4b-it (also unsloth/gemma-3-4b-pt).
    ///
    /// `rope_scaling: {factor: 8.0, rope_type: "linear"}` applies to global
    /// layers only. Linear scaling divides inv_freq by the factor -- equivalent
    /// to scaling the position by 1/factor -- rather than multiplying theta by
    /// it. Local layers use rope_local_base_freq = 10000 with no scaling.
    [[nodiscard]] static Config4 gemma3_4b();

    /// True when layer `layer_idx` uses global (full) attention.
    ///
    /// Gemma 3 repeats 5 local layers then 1 global: layers 0-4 local, 5
    /// global, 6-10 local, 11 global, and so on.
    [[nodiscard]] bool is_global_layer(std::size_t layer_idx) const {
        return layer_idx % 6 == 5;
    }
};

// =============================================================================
// KV cache
// =============================================================================

/// One layer's cached keys and values.
struct Gemma3LayerKvCache {
    /// [max_seq_len, n_kv_heads * head_dim]
    Mat k;
    Mat v;
    /// Number of valid (filled) rows.
    std::size_t seq_len = 0;

    Gemma3LayerKvCache(std::size_t n_kv_heads, std::size_t head_dim, std::size_t max_seq_len)
        : k(Mat::zeros(max_seq_len, n_kv_heads * head_dim)),
          v(Mat::zeros(max_seq_len, n_kv_heads * head_dim)) {}

    /// Append `new_k` / `new_v` rows, both [n_new, n_kv_heads * head_dim].
    void append(const Mat& new_k, const Mat& new_v);

    /// The filled portion of K / V: [seq_len, cols].
    [[nodiscard]] Mat k_filled() const;
    [[nodiscard]] Mat v_filled() const;

    /// The last `window` rows of K / V, or all of them when shorter.
    [[nodiscard]] Mat k_last(std::size_t window) const;
    [[nodiscard]] Mat v_last(std::size_t window) const;
};

/// The KV cache across all layers.
struct Gemma3KvCache {
    std::vector<Gemma3LayerKvCache> layers;

    Gemma3KvCache(const Config4& config, std::size_t max_tokens);

    /// Reset every layer, starting a new sequence.
    void clear();

    /// Release all cache memory, replacing each layer's matrices with 0x0
    /// placeholders. Call after syncing to GPU buffers.
    void free();
};

// =============================================================================
// Layers
// =============================================================================

/// One Gemma 3 attention layer.
///
/// Against GPT-OSS attention: per-head RMSNorm on Q and K (after projection,
/// before RoPE), an explicit head_dim, a scale of 1/sqrt(query_pre_attn_scalar),
/// a sliding-window mask on local layers, and a different RoPE theta for local
/// (10k) versus global (1M) layers.
class Gemma3Attention : public Module2 {
   public:
    Linear2 q_proj;  // [hidden, n_q_heads * head_dim]
    Linear2 k_proj;  // [hidden, n_kv_heads * head_dim]
    Linear2 v_proj;  // [hidden, n_kv_heads * head_dim]
    Linear2 o_proj;  // [n_q_heads * head_dim, hidden]
    /// Per-head RMSNorm on Q and K (Gemma 3 specific), both [1, head_dim].
    RmsNorm2 q_norm;
    RmsNorm2 k_norm;
    std::size_t n_q_heads = 0;
    std::size_t n_kv_heads = 0;
    std::size_t head_dim = 0;
    /// QK dot-product scale = 1/sqrt(query_pre_attn_scalar).
    float attn_scale = 0.0f;
    /// Unset means global attention (full causal mask); set means local
    /// attention limited to the last `window` tokens.
    std::optional<std::size_t> sliding_window;
    /// RoPE theta for this layer: 10k local, 1M global.
    float rope_theta = 10000.0f;
    /// RoPE linear frequency scale: 1.0 local, 1/8 global on Gemma 3 4B.
    float rope_freq_scale = 1.0f;

    Gemma3Attention(const Config4& cfg, std::size_t layer_idx, InitRng& rng);

    /// Zero-placeholder weights, for inference. Avoids the ~42 MB random-init
    /// allocation per layer; weights must be loaded before use.
    [[nodiscard]] static Gemma3Attention new_for_inference(const Config4& cfg,
                                                           std::size_t layer_idx);

    /// x [T, hidden] -> [T, hidden].
    [[nodiscard]] TensorNode forward(const TensorNode& x) const;

    /// Cached forward, with `x` typically a single decode token.
    ///
    /// Projects to Q/K/V, per-head-normalizes Q and K, applies RoPE at absolute
    /// positions starting at the current cache length, appends K and V to the
    /// cache, then attends over the last `window` rows (local) or the whole
    /// cache (global). No causal mask is needed -- the cache only holds past
    /// tokens.
    [[nodiscard]] TensorNode forward_cached(const TensorNode& x,
                                            Gemma3LayerKvCache& cache) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;

    Gemma3Attention(const Config4& cfg, std::size_t layer_idx, InferenceInit);
};

/// Dense SwiGLU FFN: `down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))`.
///
/// Gemma 3 gates with `gelu_pytorch_tanh`, not SiLU.
class Gemma3Mlp : public Module2 {
   public:
    Linear2 gate_proj;  // [hidden, intermediate]
    Linear2 up_proj;    // [hidden, intermediate]
    Linear2 down_proj;  // [intermediate, hidden]

    Gemma3Mlp(const Config4& cfg, InitRng& rng);

    /// Zero-placeholder weights, for inference.
    [[nodiscard]] static Gemma3Mlp new_for_inference(const Config4& cfg);

    [[nodiscard]] TensorNode forward(const TensorNode& x) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;

    Gemma3Mlp(const Config4& cfg, InferenceInit);
};

/// One transformer block. Gemma 3 wraps both sub-blocks in norms:
///
///   1. normed  = input_layernorm(x)
///   2. attn    = self_attn(normed)
///   3. attn    = post_attention_layernorm(attn)
///   4. x2      = x + attn
///   5. normed2 = pre_feedforward_layernorm(x2)
///   6. mlp     = mlp(normed2)
///   7. mlp     = post_feedforward_layernorm(mlp)
///   8. out     = x2 + mlp
class Gemma3Block : public Module2 {
   public:
    RmsNorm2 input_layernorm;
    Gemma3Attention self_attn;
    RmsNorm2 post_attention_layernorm;
    RmsNorm2 pre_feedforward_layernorm;
    RmsNorm2 post_feedforward_layernorm;
    Gemma3Mlp mlp;

    Gemma3Block(const Config4& cfg, std::size_t layer_idx, InitRng& rng);

    /// Zero-placeholder weights, for inference.
    [[nodiscard]] static Gemma3Block new_for_inference(const Config4& cfg,
                                                       std::size_t layer_idx);

    [[nodiscard]] TensorNode forward(const TensorNode& x) const;

    /// The same structure, using the KV cache for attention. The MLP always
    /// processes all n_new tokens -- there is no FFN cache.
    [[nodiscard]] TensorNode forward_cached(const TensorNode& x,
                                            Gemma3LayerKvCache& cache) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;

    Gemma3Block(const Config4& cfg, std::size_t layer_idx, InferenceInit);
};

// =============================================================================
// Helpers (exposed for testing)
// =============================================================================

/// Apply a per-head RMSNorm across a concatenated [T, n_heads * head_dim]
/// tensor, normalizing each head's slice independently.
[[nodiscard]] TensorNode apply_per_head_norm(const TensorNode& x, const RmsNorm2& norm,
                                             std::size_t t, std::size_t n_heads,
                                             std::size_t head_dim);

/// Apply RoPE to every head of a [T, n_heads * head_dim] tensor.
///
/// Gemma 3 uses the NeoX half-split pairing: element `i` rotates with
/// `i + head_dim/2`, not with `i + 1`.
[[nodiscard]] TensorNode apply_rope_to_all_heads(const TensorNode& x, std::size_t n_heads,
                                                 std::size_t t, std::size_t head_dim, float theta,
                                                 float freq_scale);

/// RoPE for the KV-cached decode path, where row `r` sits at absolute
/// position `offset + r`.
[[nodiscard]] TensorNode apply_rope_at_offset(const TensorNode& x, std::size_t n_heads,
                                              std::size_t n_new, std::size_t head_dim, float theta,
                                              std::size_t offset, float freq_scale);

/// Full causal GQA attention, for global layers.
[[nodiscard]] TensorNode gqa_attention_full(const TensorNode& q, const TensorNode& k,
                                            const TensorNode& v, std::size_t n_q_heads,
                                            std::size_t n_kv_heads, std::size_t d_head,
                                            float scale);

/// Sliding-window causal GQA attention, for local layers. Inference only.
[[nodiscard]] TensorNode gqa_attention_windowed(const TensorNode& q, const TensorNode& k,
                                                const TensorNode& v, std::size_t n_q_heads,
                                                std::size_t n_kv_heads, std::size_t d_head,
                                                std::size_t window, float scale);

/// GQA attention against the KV cache rows [k_start, k_end).
///
/// The single-query decode path avoids every per-head temporary; the prefill
/// path copies each head into contiguous matrices so sgemm can be used, and
/// masks causally relative to `k_start`.
[[nodiscard]] Mat gqa_attention_cached(const Mat& q_data, const Mat& k_cache, const Mat& v_cache,
                                       std::size_t k_start, std::size_t k_end,
                                       std::size_t n_q_heads, std::size_t n_kv_heads,
                                       std::size_t d_head, float scale);

/// Multiply a node's data by a scalar, with backward.
[[nodiscard]] TensorNode scale_tensor(const TensorNode& x, float factor);

/// Pick the next token from a logits row.
///
/// Order of operations: repetition penalty, temperature, softmax, top-k,
/// top-p, then either argmax (temperature 0) or an inverse-CDF draw.
///
/// The repetition penalty applies at every temperature. `seen` holds only
/// generated tokens, never the prompt, so chat-template tokens are never
/// penalized, and the window is capped at the last 64 tokens (llama.cpp's
/// default) so common function words cannot accumulate unbounded penalties.
[[nodiscard]] std::size_t sample_token(const Mat& logits, std::size_t row,
                                       const SamplingParams& params,
                                       const std::vector<std::size_t>& seen, LcgRng& rng);

// =============================================================================
// Model
// =============================================================================

class Gemma3Model : public Trainable {
   public:
    /// [vocab_size, hidden_size]. A small f32 placeholder in the inference
    /// paths -- the real table lives in `embed_bf16`.
    TensorNode embed_tokens = TensorNode::leaf(Mat::zeros(0, 0));
    /// BF16 embedding table, for row lookups without a 2.7 GB f32 allocation.
    std::optional<MatBf16> embed_bf16;
    std::vector<Gemma3Block> layers;
    RmsNorm2 norm;
    /// [hidden_size, vocab_size], weight-tied to `embed_tokens`.
    Linear2 lm_head;
    Config4 config;

    Gemma3Model(Config4 cfg, InitRng& rng);

    /// Zero-placeholder weights, for inference.
    ///
    /// Skips ~12.75 GB of random f32 weights that a subsequent load would
    /// immediately overwrite. Not usable for training -- gradients need real
    /// f32 weights.
    [[nodiscard]] static Gemma3Model new_for_inference(Config4 cfg);

    /// Free every CPU-side weight after a Metal upload: Q4_K blocks, BF16 bits
    /// and f32 weights across all projections, the embedding table and
    /// lm_head. Saves ~1.8 GB of layer weights plus ~1.3 GB of embeddings.
    void clear_cpu_weights();

    /// Quantize all large projections to INT4 and free their float storage.
    /// Inference only.
    void quantize_all_weights();

    /// token_ids -> logits [T, vocab_size].
    ///
    /// Gemma 3 scales the looked-up embeddings by sqrt(hidden_size).
    [[nodiscard]] TensorNode forward(const std::vector<std::size_t>& token_ids) const;

    /// Look up embedding rows for `token_ids`, scaled by sqrt(hidden_size).
    [[nodiscard]] Mat embed_rows(const std::vector<std::size_t>& token_ids) const;

    /// Forward through all blocks using the KV cache, returning logits for the
    /// final token only.
    [[nodiscard]] Mat forward_cached(const std::vector<std::size_t>& token_ids,
                                     Gemma3KvCache& cache) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
    [[nodiscard]] TensorNode forward_tokens(
        const std::vector<std::size_t>& token_ids) const override;
    [[nodiscard]] TensorNode loss_tokens(
        const std::vector<std::size_t>& token_ids,
        const std::vector<std::size_t>& targets) const override;

    Gemma3Model(Config4 cfg, InferenceInit);
};

/// Convert f32 values to BF16 bits, releasing the f32 storage as it goes.
[[nodiscard]] std::vector<std::uint16_t> f32s_to_bf16_and_drop(std::vector<float> f32s);

/// Ask the allocator to return free pages to the OS (macOS only).
void release_memory_to_os();

/// Print the process resident set size, tagged with `label`.
void print_rss(const char* label);

}  // namespace rt

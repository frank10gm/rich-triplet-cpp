#pragma once

// =============================================================================
// GPT-OSS -- grouped attention with YaRN RoPE and a mixture-of-experts FFN
// =============================================================================
//
// Against the GPT-2 model in `transformer2`:
//   - RMSNorm instead of LayerNorm, and RoPE instead of learned positions
//   - grouped-query attention: 64 Q heads share 8 K/V heads
//   - YaRN RoPE scaling, for contexts far beyond the trained length
//   - alternating full and sliding-window attention layers
//   - a mixture of experts in place of the dense FFN

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "rt/nn2.hpp"
#include "rt/quant.hpp"
#include "rt/result.hpp"
#include "rt/sampling.hpp"
#include "rt/tensor_node.hpp"

namespace rt {

// =============================================================================
// Config3
// =============================================================================

struct Config3 {
    /// Vocabulary size (201088 for GPT-OSS, a tiktoken vocabulary).
    std::size_t vocab_size = 0;
    /// Embedding and hidden dimension.
    std::size_t hidden_size = 0;
    std::size_t num_hidden_layers = 0;
    /// Q attention heads.
    std::size_t num_attention_heads = 0;
    /// K/V attention heads -- fewer than Q heads, which is what makes it GQA.
    std::size_t num_key_value_heads = 0;
    /// FFN hidden dimension per expert.
    std::size_t intermediate_size = 0;
    /// Experts per layer.
    std::size_t num_local_experts = 0;
    /// Experts activated per token.
    std::size_t experts_per_token = 0;
    std::size_t max_position_embeddings = 0;
    float rope_theta = 150000.0f;
    float rms_norm_eps = 1e-5f;
    /// Clamp on the SwiGLU gate before SiLU. GPT-OSS uses 7.0 to stop the gate
    /// saturating early in training; infinity disables it.
    float swiglu_limit = 7.0f;
    /// Window for the local-attention layers. GPT-OSS alternates: even layers
    /// attend to everything, odd layers to the last `sliding_window` tokens.
    std::optional<std::size_t> sliding_window;

    [[nodiscard]] static Config3 gpt_oss_20b();
    [[nodiscard]] static Config3 gpt_oss_120b();

    /// Per-head Q/K/V width: hidden_size / num_attention_heads.
    [[nodiscard]] std::size_t d_head() const;
};

// =============================================================================
// KV cache
// =============================================================================
//
// Generating token T without a cache re-runs the whole forward pass over all T
// tokens, which is O(T^2) overall. With one, the prompt is run once and each
// new token only attends against the stored K/V, which is O(T).
//
// The cache holds plain `Mat` values: generation never backpropagates.

struct LayerKvCache {
    /// [max_seq_len, n_kv_heads * d_head]
    Mat k;
    Mat v;
    /// Tokens written so far, and the next write position.
    std::size_t seq_len = 0;

    LayerKvCache(std::size_t n_kv_heads, std::size_t d_head, std::size_t max_seq_len);

    /// Append [n_new, n_kv_heads * d_head] rows.
    void append(const Mat& new_k, const Mat& new_v);

    [[nodiscard]] Mat k_filled() const;
    [[nodiscard]] Mat v_filled() const;
    /// The last `window` rows, or all of them when fewer are filled.
    [[nodiscard]] Mat k_last(std::size_t window) const;
    [[nodiscard]] Mat v_last(std::size_t window) const;
};

struct KvCache {
    std::vector<LayerKvCache> layers;

    explicit KvCache(const Config3& config);

    /// Reset every layer, starting a new sequence.
    void clear();
};

// =============================================================================
// GptOssAttention
// =============================================================================

class GptOssAttention : public Module2 {
   public:
    Linear2 q_proj;  // hidden -> n_q_heads * d_head
    Linear2 k_proj;  // hidden -> n_kv_heads * d_head
    Linear2 v_proj;  // hidden -> n_kv_heads * d_head
    Linear2 o_proj;  // n_q_heads * d_head -> hidden
    std::size_t n_q_heads = 0;
    std::size_t n_kv_heads = 0;
    std::size_t d_head = 0;
    float rope_theta = 0.0f;
    /// YaRN scaling, needed past the trained context. Without it the position
    /// is used directly.
    bool use_yarn = false;
    std::size_t original_ctx = 4096;
    std::size_t max_ctx = 0;
    /// Local window. Unset means full causal attention; set means a token
    /// attends only to the previous `window` positions.
    std::optional<std::size_t> sliding_window;

    GptOssAttention(const Config3& config, InitRng& rng,
                    std::optional<std::size_t> sliding_window);

    /// x [T, hidden] -> [T, hidden].
    [[nodiscard]] TensorNode forward(const TensorNode& x) const;

    /// Cached forward, with `x` usually a single decode token: project,
    /// rotate at the cache offset, append, attend over the cache, project out.
    [[nodiscard]] TensorNode forward_cached(const TensorNode& x, LayerKvCache& cache) const;

    /// Apply RoPE (YaRN-scaled when enabled) to every head of a
    /// [T, n_heads * d_head] tensor, with row `r` at position `seq_offset + r`.
    ///
    /// This is the interleaved pairing -- dimension `2i` rotates with `2i + 1`.
    [[nodiscard]] TensorNode apply_rope_to_all_heads(const TensorNode& x, std::size_t n_heads,
                                                     std::size_t t, std::size_t d_head,
                                                     std::size_t seq_offset = 0) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
};

// =============================================================================
// MoELayer
// =============================================================================
//
//   router_logits    = x @ W_router
//   weights, indices = top_k(softmax(router_logits), experts_per_token)
//   output           = sum_i weights[i] * expert_i(x)
//
// GPT-OSS-20b has 32 experts and activates 4, so a token touches about an
// eighth of the FFN parameters. The 120b model has 128 experts and still
// activates 4, which is around 3%. Capacity scales with the total while the
// per-token cost stays flat.

class MoELayer : public Module2 {
   public:
    /// hidden -> num_experts
    Linear2 router;
    std::vector<SwiGluMlp2> experts;
    std::size_t num_experts = 0;
    std::size_t experts_per_token = 0;

    MoELayer(const Config3& config, InitRng& rng);

    /// x [T, hidden] -> [T, hidden]. Per token: softmax the router, take the
    /// top-k experts, renormalize their weights, and sum their outputs.
    [[nodiscard]] TensorNode forward(const TensorNode& x) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
};

// =============================================================================
// GptOssBlock
// =============================================================================
//
//   x = x + Attention(RMSNorm(x))
//   x = x + MoE(RMSNorm(x))

class GptOssBlock : public Module2 {
   public:
    RmsNorm2 input_layernorm;
    GptOssAttention self_attn;
    RmsNorm2 post_attention_layernorm;
    MoELayer mlp;

    /// `layer_idx` picks the attention type: even layers attend to everything,
    /// odd layers to a local window.
    GptOssBlock(const Config3& config, std::size_t layer_idx, InitRng& rng);

    [[nodiscard]] TensorNode forward(const TensorNode& x) const;
    [[nodiscard]] TensorNode forward_cached(const TensorNode& x, LayerKvCache& cache) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
};

// =============================================================================
// GptOssModel
// =============================================================================

/// What `quantize_all_linear_weights` compressed.
struct Q4QuantStats {
    std::size_t n_tensors = 0;
    std::size_t f32_bytes = 0;
    std::size_t q4_bytes = 0;
    float compression_ratio = 0.0f;
};

class GptOssModel : public Trainable {
   public:
    // Declaration order is initialization order and every constructor draws
    // from the same InitRng, so these must appear in the order the reference
    // builds them.
    /// [vocab_size, hidden_size]. Position comes from RoPE, not an embedding.
    TensorNode embed_tokens;
    std::vector<GptOssBlock> layers;
    RmsNorm2 norm;
    /// hidden -> vocab_size
    Linear2 lm_head;
    Config3 config;

    GptOssModel(Config3 config, InitRng& rng);

    /// token_ids -> logits [T, vocab_size].
    [[nodiscard]] TensorNode forward(const std::vector<std::size_t>& token_ids) const;

    /// One forward pass per sequence. `Mat` is 2-D, so batching here means
    /// looping rather than a [B, T, D] tensor op -- the same work, and the same
    /// caller-facing shape.
    [[nodiscard]] std::vector<TensorNode> forward_batch(
        const std::vector<std::vector<std::size_t>>& sequences) const;

    /// Mean cross-entropy, as a 1x1 scalar node.
    [[nodiscard]] TensorNode loss(const std::vector<std::size_t>& token_ids,
                                  const std::vector<std::size_t>& targets) const;

    /// Mean loss over a batch of (input, target) pairs, as a 1x1 scalar node.
    ///
    /// Each sequence is run separately -- padding them into one tensor would
    /// need a mask -- and backward pushes 1/B into every per-sequence loss.
    [[nodiscard]] TensorNode loss_batch(
        const std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>>& batch)
        const;

    /// Perplexity of a token sequence: `exp(mean negative log-likelihood)`.
    ///
    /// A sliding window of `context_len` tokens (0 means the model's maximum)
    /// predicts the token just past it, advancing by half a window each time so
    /// most predictions see a reasonable amount of context.
    [[nodiscard]] float perplexity(const std::vector<std::size_t>& token_ids,
                                   std::size_t context_len) const;

#if RT_FEATURE_MMAP_LOADING
    /// Load every `*.safetensors` shard in `dir` through mmap.
    ///
    /// The kernel pages each shard in on demand and can evict those pages under
    /// pressure, so peak memory stays near the parsed tensors rather than
    /// parsed tensors plus a full file buffer.
    [[nodiscard]] Result<void> load_weights_from_dir_mmap(const std::string& dir);

    /// Load a single shard through mmap, returning how many tensors it held.
    [[nodiscard]] Result<std::size_t> load_shard_mmap(const std::string& path);
#endif

    /// The most likely next token for a prompt.
    [[nodiscard]] std::size_t predict_next(const std::vector<std::size_t>& token_ids) const;

    /// Generate with the KV cache. `temperature <= 0` is greedy.
    [[nodiscard]] std::vector<std::size_t> generate_cached(
        const std::vector<std::size_t>& token_ids, std::size_t max_new, float temperature) const;

    /// Same, calling `callback` per token instead of buffering.
    void generate_cached_streaming(const std::vector<std::size_t>& token_ids, std::size_t max_new,
                                   float temperature,
                                   const std::function<void(std::size_t)>& callback) const;

    /// Generate under full `SamplingParams` (top-k, top-p, penalties, seed).
    [[nodiscard]] std::vector<std::size_t> generate_with_params(
        const std::vector<std::size_t>& token_ids, std::size_t max_new,
        const SamplingParams& params) const;

    /// Same, calling `callback` per token instead of buffering.
    ///
    /// Unlike `generate_with_params`, the penalty window starts out holding the
    /// prompt, so prompt tokens are discouraged from being repeated.
    void generate_with_params_streaming(const std::vector<std::size_t>& token_ids,
                                        std::size_t max_new, const SamplingParams& params,
                                        const std::function<void(std::size_t)>& callback) const;

    /// Point `lm_head.weight` at `embed_tokens`.
    ///
    /// The embedding maps a token to a hidden vector and the head maps a hidden
    /// vector back to a per-token logit -- two views of one representation.
    /// Sharing them saves vocab_size * hidden_size parameters (~580M at 20b
    /// scale) and keeps the two consistent.
    void tie_weights();

    /// Quantize every weight matrix to 4-bit and store the dequantized
    /// approximation back, returning what it saved.
    ///
    /// This is static quantization: slightly lossy, but paid once. For dynamic
    /// quantization, keep the `Q4Mat` and call `matmul_q4_t` per forward pass.
    [[nodiscard]] Q4QuantStats quantize_all_linear_weights() const;

    /// Load every `*.safetensors` shard in `dir`.
    [[nodiscard]] Result<void> load_weights_from_dir(const std::string& dir);

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
    [[nodiscard]] TensorNode forward_tokens(
        const std::vector<std::size_t>& token_ids) const override;
    [[nodiscard]] TensorNode loss_tokens(
        const std::vector<std::size_t>& token_ids,
        const std::vector<std::size_t>& targets) const override;
};

/// The full sampler: penalties, then temperature, top-k, top-p, and a draw.
///
/// The repetition penalty *divides* the logit rather than subtracting, which
/// shrinks it toward zero whether it started positive or negative.
[[nodiscard]] std::size_t sample_token_full(const Mat& logits, std::size_t pos,
                                            const SamplingParams& params,
                                            const std::vector<std::size_t>& seen_ids,
                                            LcgRng53& rng);

// =============================================================================
// safetensors
// =============================================================================
//
// The file layout is:
//   [8 bytes: header length, u64 LE]
//   [header: UTF-8 JSON]
//   [raw tensor data, packed]
//
// The JSON maps each tensor name to its dtype, shape and byte range:
//   {"name": {"dtype": "BF16", "shape": [r, c], "data_offsets": [start, end]}}

struct SafeTensor {
    std::string name;
    std::vector<std::size_t> shape;
    /// Always f32, unless the BF16 conversion was skipped.
    std::vector<float> data;
    /// Raw BF16 bits, set only when the source dtype was BF16.
    std::optional<std::vector<std::uint16_t>> bf16_data;
};

/// Parse every tensor out of a safetensors blob.
[[nodiscard]] Result<std::vector<SafeTensor>> parse_safetensors(
    const std::vector<std::uint8_t>& bytes);

/// Same, but leave BF16 tensors unconverted.
///
/// A BF16 tensor's f32 form is twice its size, and for a multi-gigabyte shard
/// that dominates peak memory. Callers that only need the raw bits -- the
/// embedding table, say -- take this path.
[[nodiscard]] Result<std::vector<SafeTensor>> parse_safetensors_skip_bf16_f32(
    const std::vector<std::uint8_t>& bytes);

/// One tensor's header entry, for streaming a shard without loading it.
struct SafeTensorEntry {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t byte_start = 0;
    std::size_t byte_end = 0;
};

/// Read just the header: the data-section offset and every tensor's entry.
[[nodiscard]] Result<std::pair<std::size_t, std::vector<SafeTensorEntry>>>
parse_safetensors_header(const std::string& path);

/// Read one tensor's bytes from an open shard, converting to f32.
[[nodiscard]] Result<SafeTensor> read_safetensor_from_file(const std::string& path,
                                                           std::size_t data_offset,
                                                           const SafeTensorEntry& entry,
                                                           bool skip_bf16_to_f32);

/// Apply parsed tensors to a model by name.
void load_into_model(GptOssModel& model, const std::vector<SafeTensor>& tensors);

}  // namespace rt

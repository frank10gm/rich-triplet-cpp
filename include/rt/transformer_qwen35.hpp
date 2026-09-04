#pragma once

// =============================================================================
// Qwen 3.5 -- inference-only architecture
// =============================================================================
//
// Implements the Qwen 3.5 dense models (0.8B, 4B, 9B) for text-only inference.
// Weights load from GGUF files.
//
// Qwen 3.5 is a *hybrid*: about 75% of layers use Gated DeltaNet (linear
// attention with a recurrent state) and the rest use standard softmax
// attention.
//
// | Feature               | Gemma 3 (transformer4) | Qwen 3.5                 |
// |-----------------------|------------------------|--------------------------|
// | Attention             | softmax GQA everywhere | hybrid DeltaNet + GQA    |
// | Activation            | gelu_tanh              | silu                     |
// | Norms per block       | 4                      | 2 (pre-norm)             |
// | RMSNorm variant       | (1 + gamma)            | (1 + gamma)              |
// | Embedding scaling     | sqrt(hidden)           | none                     |
// | RoPE                  | full head_dim          | partial (25%)            |
// | Full-attn output gate | no                     | yes (sigmoid)            |
// | DeltaNet layers       | n/a                    | conv1d + recurrent state |

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "rt/nn2.hpp"
#include "rt/result.hpp"
#include "rt/sampling.hpp"
#include "rt/tensor_node.hpp"

namespace rt {

// =============================================================================
// ConfigQwen35
// =============================================================================

struct ConfigQwen35 {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t num_hidden_layers = 0;

    // Full-attention layers
    std::size_t num_attention_heads = 0;  // Q heads
    std::size_t num_key_value_heads = 0;  // KV heads
    std::size_t head_dim = 0;

    // DeltaNet layers
    std::size_t linear_num_key_heads = 0;
    std::size_t linear_num_value_heads = 0;
    std::size_t linear_key_head_dim = 0;
    std::size_t linear_value_head_dim = 0;
    std::size_t linear_conv_kernel_dim = 0;

    std::size_t intermediate_size = 0;
    float rms_norm_eps = 1e-6f;

    // RoPE, for the full-attention layers only.
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 0.25f;  // 0.25 -> only the first 64 of 256 dims

    /// Every `full_attention_interval`-th layer uses softmax attention:
    /// with 4, that is layers 3, 7, 11, ...
    std::size_t full_attention_interval = 4;
    std::size_t max_position_embeddings = 262144;
    std::size_t eos_token_id = 0;
    bool tie_word_embeddings = true;

    [[nodiscard]] static ConfigQwen35 qwen35_0_8b();
    [[nodiscard]] static ConfigQwen35 qwen35_4b();
    [[nodiscard]] static ConfigQwen35 qwen35_9b();

    [[nodiscard]] bool is_full_attention_layer(std::size_t layer_idx) const {
        return (layer_idx + 1) % full_attention_interval == 0;
    }

    /// Total DeltaNet QKV projection width: key_dim * 2 + value_dim.
    [[nodiscard]] std::size_t deltanet_qkv_dim() const {
        return linear_num_key_heads * linear_key_head_dim * 2 +
               linear_num_value_heads * linear_value_head_dim;
    }

    /// The rotated prefix of each head, for partial RoPE.
    [[nodiscard]] std::size_t rope_dim() const {
        return static_cast<std::size_t>(static_cast<float>(head_dim) * partial_rotary_factor);
    }
};

// =============================================================================
// Caches
// =============================================================================

/// A DeltaNet layer's state: the recurrent matrix plus the conv1d history.
struct DeltaNetState {
    /// Recurrent state, [num_v_heads * key_head_dim, value_head_dim]. Head `h`
    /// owns rows [h*kd, (h+1)*kd) and all `vd` columns.
    std::vector<float> state;
    /// Conv1d history, [qkv_dim, kernel_size - 1].
    std::vector<float> conv_state;
    std::size_t num_v_heads = 0;
    std::size_t key_head_dim = 0;
    std::size_t value_head_dim = 0;
    std::size_t conv_dim = 0;
    std::size_t conv_kernel = 0;

    explicit DeltaNetState(const ConfigQwen35& cfg);

    /// Head `h`'s [kd, vd] state matrix, row-major.
    [[nodiscard]] std::span<float> head_state(std::size_t h) {
        const std::size_t sz = key_head_dim * value_head_dim;
        return {state.data() + h * sz, sz};
    }
    [[nodiscard]] std::span<const float> head_state(std::size_t h) const {
        const std::size_t sz = key_head_dim * value_head_dim;
        return {state.data() + h * sz, sz};
    }
};

/// A full-attention layer's KV cache.
struct FullAttnKvCache {
    Mat k;  // [max_tokens, nkv * head_dim]
    Mat v;
    std::size_t seq_len = 0;

    FullAttnKvCache(std::size_t nkv, std::size_t head_dim, std::size_t max_tokens)
        : k(Mat::zeros(max_tokens, nkv * head_dim)),
          v(Mat::zeros(max_tokens, nkv * head_dim)) {}

    void append(const Mat& new_k, const Mat& new_v);
};

/// Per-layer cache: DeltaNet state or a KV cache, depending on the layer type.
using LayerCache = std::variant<DeltaNetState, FullAttnKvCache>;

struct Qwen35Cache {
    std::vector<LayerCache> layers;

    Qwen35Cache(const ConfigQwen35& cfg, std::size_t max_tokens);
};

// =============================================================================
// Layers
// =============================================================================

/// Gated DeltaNet: linear attention carrying a recurrent state.
///
/// Per token: project to QKV / z / a / b, run a causal depthwise conv1d with
/// SiLU, split and L2-normalize Q and K, expand the K heads to match the V
/// heads, then update the per-head state with the delta rule
///
///   S  <- S * exp(g)
///   S  <- S + outer(k, (v - S^T k) * beta)
///   out = S^T q
///
/// and finish with a gated RMSNorm against SiLU(z).
class Qwen35DeltaNet {
   public:
    Linear2 in_proj_qkv;  // [hidden, qkv_dim]
    Linear2 in_proj_z;    // [hidden, value_dim] -- output gate
    Linear2 in_proj_a;    // [hidden, num_v_heads] -- decay gate
    Linear2 in_proj_b;    // [hidden, num_v_heads] -- write gate
    Linear2 out_proj;     // [value_dim, hidden]
    /// Depthwise conv1d weights, [qkv_dim, kernel_size], stored flat.
    std::vector<float> conv1d_weight;
    /// Log of the base decay rate, one per value head.
    std::vector<float> a_log;
    /// Decay-gate bias, one per value head.
    std::vector<float> dt_bias;
    /// Output RMSNorm weight (zero-centered), [value_head_dim].
    std::vector<float> norm_weight;

    std::size_t num_k_heads = 0;
    std::size_t num_v_heads = 0;
    std::size_t key_head_dim = 0;
    std::size_t value_head_dim = 0;
    std::size_t conv_kernel = 0;
    std::size_t qkv_dim = 0;
    std::size_t value_dim = 0;

    [[nodiscard]] static Qwen35DeltaNet new_for_inference(const ConfigQwen35& cfg);

    /// Single-token decode: x [1, hidden] -> [1, hidden].
    [[nodiscard]] TensorNode forward_cached(const TensorNode& x, DeltaNetState& state) const;

    /// Batched prefill: x [T, hidden] -> [T, hidden]. Projections run as GEMM;
    /// the conv1d and recurrent update stay sequential.
    [[nodiscard]] TensorNode forward_prefill(const TensorNode& x, DeltaNetState& state) const;

    /// Causal depthwise conv1d over the stored history, followed by SiLU.
    /// Advances `state.conv_state` by one step.
    [[nodiscard]] std::vector<float> apply_conv1d(std::span<const float> input,
                                                  DeltaNetState& state) const;

   private:
    explicit Qwen35DeltaNet(const ConfigQwen35& cfg);
};

/// Softmax GQA with a sigmoid output gate and partial RoPE.
///
/// `q_proj` emits query and gate together: each head contributes `head_dim`
/// query values followed by `head_dim` gate values.
class Qwen35FullAttention {
   public:
    Linear2 q_proj;  // [hidden, n_heads * head_dim * 2] -- query and gate
    Linear2 k_proj;  // [hidden, n_kv_heads * head_dim]
    Linear2 v_proj;  // [hidden, n_kv_heads * head_dim]
    Linear2 o_proj;  // [n_heads * head_dim, hidden]
    RmsNorm2 q_norm;
    RmsNorm2 k_norm;
    std::size_t n_q_heads = 0;
    std::size_t n_kv_heads = 0;
    std::size_t head_dim = 0;
    float rope_theta = 0.0f;
    /// The rotated prefix of each head.
    std::size_t rope_dim = 0;

    [[nodiscard]] static Qwen35FullAttention new_for_inference(const ConfigQwen35& cfg);

    /// Single-token decode: x [1, hidden] -> [1, hidden].
    [[nodiscard]] TensorNode forward_cached(const TensorNode& x, FullAttnKvCache& cache) const;

    /// Batched causal prefill: x [T, hidden] -> [T, hidden].
    [[nodiscard]] TensorNode forward_prefill(const TensorNode& x, FullAttnKvCache& cache) const;

   private:
    explicit Qwen35FullAttention(const ConfigQwen35& cfg);
};

/// SwiGLU with SiLU gating.
class Qwen35Mlp {
   public:
    Linear2 gate_proj;  // [hidden, intermediate]
    Linear2 up_proj;    // [hidden, intermediate]
    Linear2 down_proj;  // [intermediate, hidden]

    [[nodiscard]] static Qwen35Mlp new_for_inference(const ConfigQwen35& cfg);

    [[nodiscard]] TensorNode forward(const TensorNode& x) const;

   private:
    explicit Qwen35Mlp(const ConfigQwen35& cfg);
};

using TokenMixer = std::variant<Qwen35DeltaNet, Qwen35FullAttention>;

/// One block: pre-norm -> token mixer -> residual, pre-norm -> MLP -> residual.
class Qwen35Block {
   public:
    RmsNorm2 input_layernorm;
    TokenMixer token_mixer;
    RmsNorm2 post_attention_layernorm;
    Qwen35Mlp mlp;

    [[nodiscard]] static Qwen35Block new_for_inference(const ConfigQwen35& cfg,
                                                       std::size_t layer_idx);

    /// x [1, hidden] -> [1, hidden].
    [[nodiscard]] TensorNode forward_cached(const TensorNode& x, LayerCache& cache) const;

    /// x [T, hidden] -> [T, hidden].
    [[nodiscard]] TensorNode forward_prefill(const TensorNode& x, LayerCache& cache) const;

   private:
    Qwen35Block(const ConfigQwen35& cfg, std::size_t layer_idx);
};

// =============================================================================
// Model
// =============================================================================

class Qwen35Model {
   public:
    TensorNode embed_tokens = TensorNode::leaf(Mat::zeros(0, 0));
    std::optional<MatBf16> embed_bf16;
    std::vector<Qwen35Block> layers;
    RmsNorm2 norm;
    Linear2 lm_head;
    ConfigQwen35 config;

    [[nodiscard]] static Qwen35Model new_for_inference(ConfigQwen35 cfg);

    /// Look up embedding rows. Unlike Gemma 3, Qwen 3.5 applies no scaling.
    [[nodiscard]] Mat embed_rows(const std::vector<std::size_t>& token_ids) const;

    /// Run the prompt through every block in batched prefill mode, returning
    /// logits for the final token.
    [[nodiscard]] Mat prefill(const std::vector<std::size_t>& token_ids,
                              Qwen35Cache& cache) const;

    /// Run one token through every block, returning its logits.
    [[nodiscard]] Mat decode_step(std::size_t token_id, Qwen35Cache& cache) const;

    /// Generate `max_new` tokens, calling `callback` as each one is produced.
    ///
    /// Prefill runs the whole prompt in batched mode; decode then runs one
    /// token per step. With Metal built in, decode moves to the GPU once the
    /// DeltaNet state and KV cache have been handed over.
    void generate_cached_streaming(const std::vector<std::size_t>& token_ids, std::size_t max_new,
                                   float temperature, std::size_t top_k, float top_p,
                                   float repetition_penalty, std::uint64_t seed, bool debug,
                                   const std::function<void(std::size_t)>& callback);

    /// Load weights from a GGUF file.
    ///
    /// llama.cpp names the DeltaNet tensors differently from HuggingFace:
    /// `attn_qkv` is the fused QKV projection, `attn_gate` the output gate,
    /// `post_attention_norm` the post-attention layernorm, and `ssm_a` carries
    /// no `.weight` suffix. Norm gammas arrive as `1 + weight` and have the 1
    /// subtracted, except the DeltaNet output norm, which is stored raw.
    [[nodiscard]] Result<void> load_weights_from_gguf(const std::string& path);

    /// Load weights from a directory of HuggingFace `.safetensors` shards.
    ///
    /// Unlike GGUF, safetensors stores norm gammas raw, so nothing is
    /// subtracted on the way in.
    [[nodiscard]] Result<void> load_weights_from_dir(const std::string& dir);

   private:
    explicit Qwen35Model(ConfigQwen35 cfg);

    /// One token's embedding row. Qwen 3.5 applies no scaling.
    [[nodiscard]] std::vector<float> embed_token(std::size_t tok) const;

    /// Route one safetensor to its model field. Returns whether it matched.
    bool apply_safetensor(struct SafeTensor t);
};

// =============================================================================
// Helpers (exposed for testing)
// =============================================================================

[[nodiscard]] float qwen_sigmoid(float x);
/// `log(1 + exp(x))`, passing large x through to avoid overflow.
[[nodiscard]] float qwen_softplus(float x);
[[nodiscard]] float qwen_silu(float x);

/// L2-normalize each head's slice in place.
void l2_normalize_heads(std::span<float> x, std::size_t n_heads, std::size_t dim);

/// Per-head RMSNorm in place, using the zero-centered (1 + gamma) variant.
void apply_per_head_norm_raw(std::span<float> x, const RmsNorm2& norm, std::size_t n_heads,
                             std::size_t dim);

/// Partial RoPE: rotate only the first `rope_dim` dimensions of each head,
/// pairing `i` with `i + rope_dim/2` (NeoX half-split).
void apply_partial_rope(std::span<float> x, std::size_t n_heads, std::size_t head_dim,
                        std::size_t rope_dim, float theta, std::size_t pos);

/// GQA attention for a single query token over cache rows [k_start, k_end).
[[nodiscard]] std::vector<float> qwen_gqa_attention_cached(std::span<const float> q,
                                                           const Mat& k_cache, const Mat& v_cache,
                                                           std::size_t k_start, std::size_t k_end,
                                                           std::size_t nq, std::size_t nkv,
                                                           std::size_t d, float scale);

}  // namespace rt

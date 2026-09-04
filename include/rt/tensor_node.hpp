#pragma once

// =============================================================================
// TensorNode -- one node in the computation graph, holding a full matrix
// =============================================================================
//
// ## Why a tensor-level graph
//
// A scalar autograd engine makes every number its own node: a Linear layer with
// shape [64, 64] creates 4096 nodes for the weights alone, and a small model's
// forward pass runs to ~400,000 nodes.
//
// Here there is one node per *operation*. A matmul of [16,32] x [32,32] is a
// single node rather than 16,384, and a whole model graph is ~50 nodes:
//
//   embed -> ln -> q_proj -> k_proj -> v_proj -> scores -> softmax ->
//   attn_out -> o_proj -> residual -> ln -> fc1 -> gelu -> fc2 -> residual ->
//   ln_final -> lm_head -> loss
//
// The math is the same chain rule, applied to whole matrices at once -- a
// vector-Jacobian product (VJP) rather than a scalar derivative.
//
// ## The key backward rules
//
//   matmul: C = A @ B      dA = dC @ B.T,   dB = A.T @ dC
//   add:    C = A + B      dA = dC,         dB = dC
//   mul:    C = A * B      dA = dC * B,     dB = dC * A
//   softmax: S = softmax(X)  dX[i] = S[i] * (dS[i] - dot(dS[i], S[i]))
//   layernorm: closed form using the stored mu, sigma and X-hat
//   cross-entropy: fused with softmax -- d(logits) = (softmax - one_hot) / T
//
// ## Ownership
//
// Nodes are reference-counted and share mutable state, so a node can be an
// input to several others and accumulate gradient from all of them. Backward
// closures capture the nodes they touch, which forms reference cycles; call
// `free_graph()` to break them when the graph is no longer needed.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rt/mat.hpp"
#include "rt/result.hpp"

namespace rt {

class TensorNode {
   public:
    /// Create a leaf node (a parameter or input -- no backward function).
    [[nodiscard]] static TensorNode leaf(Mat data);

    /// Read the forward data.
    [[nodiscard]] const Mat& data() const { return p_->data; }

    /// Read the gradient.
    [[nodiscard]] const Mat& grad() const { return p_->grad; }

    /// Zero the gradient tensor (call before each backward pass).
    void zero_grad() const;

    /// Directly set the data (used by the optimizer to apply updates).
    void set_data(Mat data) const { p_->data = std::move(data); }

    /// Directly set gradient values (used for gradient clipping).
    void set_grad(Mat grad) const { p_->grad = std::move(grad); }

    /// Accumulate into the gradient. This is what every backward rule does:
    /// a node feeding several consumers must sum their contributions.
    void grad_add(const Mat& g) const { p_->grad.add_assign(g); }

    /// Register a backward function and predecessors on an existing leaf.
    /// Used by layers that build fused backward nodes.
    void set_backward(std::function<void()> f, std::vector<TensorNode> prev) const;

    /// Invoke this node's backward function, if it has one.
    void call_backward_fn() const;

    /// Free the entire autograd graph reachable from this node.
    ///
    /// Backward closures capture the output node itself, so every op creates a
    /// reference cycle that plain reference counting can never collect. This
    /// walks every reachable node and drops the closure, the gradient, the data
    /// and the predecessor links, leaving empty 0x0 matrices behind.
    ///
    /// Leaf nodes (model parameters, and the output leaves of fused linears)
    /// are skipped -- they must survive, and they hold no cycle.
    void free_graph() const;

    /// Seed this node's gradient with ones. Used to kick off a manual backward.
    void seed_grad_ones() const;

    /// Identity of the underlying node, for visited-sets and equality checks.
    [[nodiscard]] const void* id() const { return p_.get(); }

    [[nodiscard]] std::string debug_string() const;

    // =========================================================================
    // Operations -- each returns a new node and registers the backward rule
    // =========================================================================

    /// C = A @ B, [M,K] x [K,N] -> [M,N].
    /// Backward: dA += dC @ B.T, dB += A.T @ dC.
    [[nodiscard]] TensorNode matmul(const TensorNode& b) const;

    /// C = A + B, element-wise, same shape. Gradient passes through to both.
    [[nodiscard]] TensorNode add(const TensorNode& b) const;

    /// C = A + bias_row -- broadcast a [1, cols] bias over every row.
    /// Backward: dA += dC, d_bias += sum_rows(dC), since the bias participates
    /// in every row.
    [[nodiscard]] TensorNode add_bias(const TensorNode& bias) const;

    /// GELU(x) = x * sigmoid(1.702 * x), element-wise.
    [[nodiscard]] TensorNode gelu() const;

    /// GELU with the PyTorch tanh approximation (`gelu_pytorch_tanh`, used by
    /// Gemma 3): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))).
    [[nodiscard]] TensorNode gelu_tanh() const;

    /// Row-wise softmax over [T, V], computed max-shifted for stability.
    [[nodiscard]] TensorNode softmax() const;

    /// Layer normalization: Y = (X - mu) / sigma * gamma + beta, per row.
    [[nodiscard]] TensorNode layer_norm(const TensorNode& gamma, const TensorNode& beta) const;

    /// RMS normalization: x / sqrt(mean(x^2) + eps) * gamma, per row.
    ///
    /// Compared with LayerNorm there is no beta and no mean subtraction --
    /// simpler and slightly faster. Used by LLaMA, Mistral, GPT-OSS and most
    /// post-2022 models.
    [[nodiscard]] TensorNode rms_norm(const TensorNode& gamma, float eps) const;

    /// Gemma 3 RMS normalization, which scales by `(1 + gamma)`.
    ///
    /// Gemma 3 stores gamma initialized to zeros rather than ones, and every
    /// norm layer in the model uses this form.
    [[nodiscard]] TensorNode rms_norm_gemma3(const TensorNode& gamma, float eps) const;

    /// SiLU (Sigmoid Linear Unit): x * sigmoid(x). The activation inside SwiGLU.
    [[nodiscard]] TensorNode silu() const;

    /// Element-wise clamp to [min_val, max_val].
    /// Gradient passes through where the input was strictly inside the range.
    [[nodiscard]] TensorNode clamp(float min_val, float max_val) const;

    /// Element-wise multiplication of two same-shape nodes (SwiGLU's
    /// `SiLU(gate) * up`). Backward: dA = dC * B, dB = dC * A.
    [[nodiscard]] TensorNode mul_elem_node(const TensorNode& other) const;

    /// Apply Rotary Position Embeddings to a [T, d_head] tensor.
    ///
    /// RoPE encodes position by rotating dimension pairs, so the attention
    /// score between positions i and j depends only on `i - j`. For pair
    /// (2i, 2i+1) at position t:
    ///   angle    = t / theta^(2i / d_head)
    ///   x'[2i]   = x[2i]   * cos - x[2i+1] * sin
    ///   x'[2i+1] = x[2i+1] * cos + x[2i]   * sin
    ///
    /// Backward is rotation by -angle, since RoPE is orthogonal.
    [[nodiscard]] TensorNode rope_apply(std::size_t seq_offset, float theta) const;

    /// RoPE with YaRN frequency scaling, for contexts beyond the trained length.
    ///
    /// YaRN rescales the effective position per frequency band: low frequencies
    /// are linearly interpolated (`pos / scale`), high frequencies are left
    /// alone so they stay in the trained range, and the middle blends between
    /// the two. A global `mscale` factor keeps attention scores calibrated.
    [[nodiscard]] TensorNode rope_apply_yarn(std::size_t seq_offset, float theta,
                                             std::size_t original_ctx, std::size_t max_ctx,
                                             float beta_fast, float beta_slow) const;

    /// Causal self-attention over Q, K, V, fusing scores, mask, softmax and the
    /// weighted sum into one node.
    ///
    ///   scores  = Q @ K.T / sqrt(d_head), masked to j <= i
    ///   weights = softmax(scores)
    ///   output  = weights @ V
    [[nodiscard]] static TensorNode causal_attention(const TensorNode& q, const TensorNode& k,
                                                     const TensorNode& v, std::size_t d_head);

    /// Grouped Multi-Query Attention with a causal mask.
    ///
    /// Q has `n_q_heads` heads but K and V share only `n_kv_heads`; each group
    /// of `n_q_heads / n_kv_heads` query heads reads one KV head, shrinking the
    /// KV cache by that factor.
    ///
    ///   q: [T, n_q_heads * d_head]
    ///   k: [T, n_kv_heads * d_head]
    ///   v: [T, n_kv_heads * d_head]  ->  [T, n_q_heads * d_head]
    [[nodiscard]] static TensorNode gqa_attention(const TensorNode& q, const TensorNode& k,
                                                  const TensorNode& v, std::size_t n_q_heads,
                                                  std::size_t n_kv_heads, std::size_t d_head);

    /// Same semantics as `gqa_attention`, but the score and output matmuls go
    /// through `NDArray::bmm` instead of a per-head loop.
    [[nodiscard]] static TensorNode batched_gqa_attention(const TensorNode& q,
                                                          const TensorNode& k,
                                                          const TensorNode& v,
                                                          std::size_t n_q_heads,
                                                          std::size_t n_kv_heads,
                                                          std::size_t d_head);

    /// Flash Attention -- causal self-attention in O(T) memory instead of O(T^2).
    ///
    /// Standard attention materializes the full [T, T] score matrix: 16 MB per
    /// head at T=2048, and the backward pass needs it again. Flash Attention
    /// tiles the computation and keeps a running softmax normalizer, so the
    /// T x T matrix never exists. Backward recomputes the weights from the
    /// stored (l, m) state -- about 2x the compute for O(T) memory.
    [[nodiscard]] static TensorNode flash_attention(const TensorNode& q, const TensorNode& k,
                                                    const TensorNode& v, std::size_t d_head);

    // =========================================================================
    // Backward pass
    // =========================================================================

    /// Run the full backward pass from this node, which must be a 1x1 scalar.
    ///
    /// Builds a topological order of all ancestors, seeds this node's gradient
    /// with ones, then walks in reverse order calling each backward function.
    void backward() const;

   private:
    struct NodeData {
        /// The forward-pass value (the matrix this node computed).
        Mat data;
        /// Gradient of the loss w.r.t. `data`, same shape, accumulated during
        /// backward.
        Mat grad;
        /// How to propagate `grad` back to this node's inputs.
        /// Empty for leaf nodes (parameters, inputs).
        std::function<void()> backward_fn;
        /// Input nodes, used to build the topological order.
        std::vector<TensorNode> prev;
    };

    explicit TensorNode(std::shared_ptr<NodeData> p) : p_(std::move(p)) {}

    std::shared_ptr<NodeData> p_;
};

// =============================================================================
// Checkpoint save/load -- binary format for TensorNode weights
// =============================================================================
//
//   [magic: u32 = 0x4358504B "CXPK"]
//   [version: u32 = 1]
//   [n_tensors: u32]
//   for each tensor:
//     [name_len: u32][name: UTF-8][rows: u32][cols: u32][data: rows*cols f32 LE]

inline constexpr std::uint32_t CKPT_MAGIC = 0x4358504B;  // "CXPK"
inline constexpr std::uint32_t CKPT_VERSION = 1;

/// Save named tensors to a binary checkpoint file. Tensors are identified by
/// name, so the save order need not match the load order.
[[nodiscard]] Result<void> save_checkpoint(
    const std::string& path,
    const std::vector<std::pair<std::string, TensorNode>>& tensors);

/// Load a checkpoint, returning (name, Mat) pairs. Matching names to model
/// parameters is the caller's job.
[[nodiscard]] Result<std::vector<std::pair<std::string, Mat>>> load_checkpoint(
    const std::string& path);

/// Restore model parameters from a checkpoint, applying tensors to `params` by
/// position (the index order must match the order used when saving).
[[nodiscard]] Result<void> restore_checkpoint(const std::string& path,
                                              const std::vector<TensorNode>& params);

}  // namespace rt

#pragma once

// =============================================================================
// Tensor-level neural network layers (nn2)
// =============================================================================
//
// The same layers as `nn.hpp`, but operating on whole matrices via TensorNode
// rather than vectors of scalar Values.
//
// Scalar form:
//   Linear::forward(const std::vector<Value>&) -> std::vector<Value>
//   a manual dot-product loop, creating out*in*2 + out nodes per call.
//
// Tensor form:
//   Linear2::forward(const TensorNode&) -> TensorNode
//   a single matmul, creating 2 nodes total.
//
// Each parameter is a `TensorNode::leaf` -- no backward function, just data and
// a gradient accumulator. The optimizer reads `grad()` and writes `set_data()`.

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "rt/init_rng.hpp"
#include "rt/quant.hpp"
#include "rt/tensor_node.hpp"

namespace rt {

// =============================================================================
// Memory management helpers
// =============================================================================

/// Tell the kernel a region's pages can be reclaimed.
///
/// On macOS, freed MALLOC_LARGE regions stay physically resident until they are
/// marked reusable, so a model that frees gigabytes of weights keeps showing
/// them in RSS. No-op on other platforms.
void mark_pages_reusable(const void* data, std::size_t bytes);

template <typename T>
void mark_pages_reusable(const std::vector<T>& data) {
    mark_pages_reusable(data.data(), data.size() * sizeof(T));
}

// =============================================================================
// Module2 -- anything with learnable parameters
// =============================================================================

class Module2 {
   public:
    virtual ~Module2() = default;

    /// All learnable parameters of this module.
    [[nodiscard]] virtual std::vector<TensorNode> parameters() const = 0;

    /// Zero all parameter gradients.
    void zero_grad() const {
        for (const TensorNode& p : parameters()) {
            p.zero_grad();
        }
    }
};

// =============================================================================
// Trainable -- anything the training loop can optimize
// =============================================================================

/// A model the training loop can drive. Both the GPT-2 and GPT-OSS models
/// implement it, so one loop works for either architecture.
class Trainable : public Module2 {
   public:
    /// Compute logits: token ids -> [T, vocab_size].
    [[nodiscard]] virtual TensorNode forward_tokens(const std::vector<std::size_t>& token_ids) const = 0;

    /// Cross-entropy loss as a scalar node with backward wired up.
    [[nodiscard]] virtual TensorNode loss_tokens(const std::vector<std::size_t>& token_ids,
                                                 const std::vector<std::size_t>& targets) const = 0;

    /// Mean cross-entropy over a batch of B sequences.
    ///
    /// Runs `loss_tokens` per sequence, averages the scalar losses, then calls
    /// backward on each individual loss scaled by 1/B so the gradients
    /// accumulate into the shared parameters correctly.
    ///
    /// The returned node is a plain scalar leaf holding the mean loss. Do NOT
    /// call `backward()` on it -- backward has already run internally.
    [[nodiscard]] virtual TensorNode loss_batch_tokens(
        const std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>>& batch)
        const;
};

// =============================================================================
// Linear layer
// =============================================================================
//
// output = input @ weight.T + bias
//
//   input:    [T, in_features]
//   weight:   [out_features, in_features]
//   bias:     [1, out_features]
//   output:   [T, out_features]
//
// The weight is stored as [out, in] -- one row per output neuron -- because
// that is the shape the gradient math wants (dW = dOut.T @ input). Computing
// the output therefore needs the transpose.

class Linear2 : public Module2 {
   public:
    /// [out_features, in_features]
    TensorNode weight = TensorNode::leaf(Mat::zeros(0, 0));
    /// [1, out_features]
    TensorNode bias = TensorNode::leaf(Mat::zeros(0, 0));
    std::size_t in_features = 0;
    std::size_t out_features = 0;

    /// INT4 quantized weight, set by `quantize()`. When present, forward uses
    /// `matmul_q4_t` instead of the f32 weight.
    std::optional<Q4Mat> q4_weight;
    /// Q4_K native packed weight (GGUF Q4_K_M), dequantized on the fly:
    /// ~3.5x less RAM than BF16.
    std::optional<Q4KMat> q4k_weight;
    /// BF16 weight storage (inference only), dequantized on the fly:
    /// 2x less RAM than f32, lossless.
    std::optional<MatBf16> bf16_weight;

    Linear2(std::size_t in_features, std::size_t out_features, InitRng& rng);

    /// A Linear layer with no bias -- the bias stays fixed at zero and is not a
    /// parameter. Used by architectures like Gemma 3.
    [[nodiscard]] static Linear2 new_no_bias(std::size_t in_features, std::size_t out_features,
                                             InitRng& rng);

    /// A Linear layer with zero-sized placeholder weights and no RNG.
    ///
    /// For inference-only paths where weights arrive from GGUF or safetensors.
    /// Skipping the random initialization saves ~12 GB of peak RAM for
    /// Gemma3-4b (34 layers x ~375 MB each). The bias is zero-sized, so
    /// `fused_linear` skips the bias addition entirely.
    [[nodiscard]] static Linear2 new_no_bias_zeros(std::size_t in_features,
                                                   std::size_t out_features);

    /// Quantize the weight to 4-bit and store it. The f32 weight is left
    /// untouched, so gradients still flow for further training.
    void quantize();

    /// Quantize to INT4 and free the f32 weight. Inference only -- do not call
    /// this if you still need backward passes.
    void quantize_and_free_f32();

    /// Quantize a BF16 weight to INT4 and free all float storage.
    /// Falls back to `quantize_and_free_f32` when there is no BF16 weight.
    void quantize_bf16_and_free();

    /// Quantize a BF16 weight to Q4_K and free the BF16 data.
    void quantize_bf16_to_q4k();

    /// Free all CPU-side weight data (Q4_K blocks, BF16 bits, f32 weight).
    ///
    /// Call once the weights are uploaded to GPU buffers, which hold their own
    /// copy. On macOS this also madvises the pages so the kernel reclaims them
    /// immediately rather than leaving them resident.
    void clear_weight_data();

    /// Store weights as BF16 for inference (lossless, 2x less RAM), clearing
    /// the f32 weight. Do not call this if you need backward passes.
    void load_bf16(std::vector<std::uint16_t> bits, std::size_t rows, std::size_t cols);

    /// Same as `load_bf16`, but adopts already-shared storage without copying.
    void load_bf16_shared(std::shared_ptr<std::vector<std::uint16_t>> data, std::size_t rows,
                          std::size_t cols);

    /// input: [T, in_features] -> output: [T, out_features]
    [[nodiscard]] TensorNode forward(const TensorNode& input) const { return fused_linear(input); }

    [[nodiscard]] std::vector<TensorNode> parameters() const override { return {weight, bias}; }

   private:
    Linear2() = default;

    /// Fused linear: `output = input @ weight.T + bias`, as a single node that
    /// tracks both input and weight so no separate transpose node is needed.
    ///
    /// Forward dispatch priority: Q4 > Q4_K > BF16 > f32.
    [[nodiscard]] TensorNode fused_linear(const TensorNode& input) const;
};

// =============================================================================
// LayerNorm2
// =============================================================================

class LayerNorm2 : public Module2 {
   public:
    TensorNode gamma;  // [1, d_model]
    TensorNode beta;   // [1, d_model]
    std::size_t d_model;

    explicit LayerNorm2(std::size_t d_model)
        : gamma(TensorNode::leaf(Mat::ones(1, d_model))),
          beta(TensorNode::leaf(Mat::zeros(1, d_model))),
          d_model(d_model) {}

    /// x: [T, d_model] -> normalized: [T, d_model]
    [[nodiscard]] TensorNode forward(const TensorNode& x) const {
        return x.layer_norm(gamma, beta);
    }

    [[nodiscard]] std::vector<TensorNode> parameters() const override { return {gamma, beta}; }
};

// =============================================================================
// Mlp2 -- feed-forward network
// =============================================================================
//
// x -> Linear(d_model -> 4*d_model) -> GELU -> Linear(4*d_model -> d_model)

class Mlp2 : public Module2 {
   public:
    Linear2 fc1;
    Linear2 fc2;

    Mlp2(std::size_t d_model, InitRng& rng)
        : fc1(d_model, 4 * d_model, rng), fc2(4 * d_model, d_model, rng) {}

    /// x: [T, d_model] -> output: [T, d_model]
    [[nodiscard]] TensorNode forward(const TensorNode& x) const {
        return fc2.forward(fc1.forward(x).gelu());
    }

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
};

// =============================================================================
// RmsNorm2 -- RMS layer normalization (GPT-OSS, LLaMA, Mistral)
// =============================================================================
//
// RMSNorm(x) = x / RMS(x) * gamma
//
// Simpler than LayerNorm: no mean subtraction and no beta. Empirically just as
// effective, and slightly faster.

class RmsNorm2 : public Module2 {
   public:
    TensorNode gamma;  // [1, d_model], initialized to ones
    std::size_t d_model;
    float eps;

    explicit RmsNorm2(std::size_t d_model, float eps = 1e-5f)
        : gamma(TensorNode::leaf(Mat::ones(1, d_model))), d_model(d_model), eps(eps) {}

    /// x: [T, d_model] -> normalized: [T, d_model]
    [[nodiscard]] TensorNode forward(const TensorNode& x) const { return x.rms_norm(gamma, eps); }

    /// Gemma 3 variant, which scales by `(1 + gamma)`. Every norm layer in
    /// Gemma 3 uses it; gamma is stored zero-initialized as a trained offset.
    [[nodiscard]] TensorNode forward_gemma3(const TensorNode& x) const {
        return x.rms_norm_gemma3(gamma, eps);
    }

    [[nodiscard]] std::vector<TensorNode> parameters() const override { return {gamma}; }
};

// =============================================================================
// SwiGluMlp2 -- SwiGLU feed-forward network (GPT-OSS, LLaMA, PaLM)
// =============================================================================
//
// Standard FFN (GPT-2):  x -> Linear -> GELU -> Linear
//
// SwiGLU FFN:
//   gate   = x @ W_gate           [T, intermediate_size]
//   up     = x @ W_up             [T, intermediate_size]
//   hidden = SiLU(gate) * up      element-wise
//   out    = hidden @ W_down      [T, d_model]
//
// The gating lets the network suppress or amplify each feature dimension
// individually -- more expressive than a single activation. No bias in the
// projections.

class SwiGluMlp2 : public Module2 {
   public:
    Linear2 gate_proj;  // d_model -> intermediate_size
    Linear2 up_proj;    // d_model -> intermediate_size
    Linear2 down_proj;  // intermediate_size -> d_model
    /// Clamp the gate pre-activation to [-clamp, clamp] before SiLU.
    /// GPT-OSS uses 7.0; infinity disables it (the default).
    float swiglu_clamp = std::numeric_limits<float>::infinity();

    SwiGluMlp2(std::size_t d_model, std::size_t intermediate_size, InitRng& rng)
        : gate_proj(d_model, intermediate_size, rng),
          up_proj(d_model, intermediate_size, rng),
          down_proj(intermediate_size, d_model, rng) {}

    SwiGluMlp2(std::size_t d_model, std::size_t intermediate_size, float clamp, InitRng& rng)
        : SwiGluMlp2(d_model, intermediate_size, rng) {
        swiglu_clamp = clamp;
    }

    /// x: [T, d_model] -> output: [T, d_model]
    [[nodiscard]] TensorNode forward(const TensorNode& x) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
};

// =============================================================================
// Dropout2
// =============================================================================
//
// During training each element is independently zeroed with probability p, and
// the survivors are scaled by 1/(1-p) so the expected value is unchanged.
// During inference nothing is dropped.
//
// Randomly disabling neurons forces redundant representations, trains a
// different sub-network per mini-batch, and reduces co-adaptation.
//
//   p = 0.0  no dropout
//   p = 0.1  mild; good for transformer residual streams
//   p = 0.5  aggressive; common in fully-connected classifiers
//
//   Forward:  mask = Bernoulli(1-p); y = x * mask / (1-p)
//   Backward: dx = dy * mask / (1-p)   (the same mask)
//
// The mask comes from an LCG seeded by an internal counter, so each forward
// call gets a fresh mask without the caller managing PRNG state.

class Dropout2 : public Module2 {
   public:
    /// Drop probability: 0 keeps everything, 1 would drop everything.
    float p;

    explicit Dropout2(float p);

    /// x: [T, D] -> output: [T, D]. With `training == false` this is identity.
    [[nodiscard]] TensorNode forward(const TensorNode& x, bool training) const;

    /// Dropout has no learnable parameters.
    [[nodiscard]] std::vector<TensorNode> parameters() const override { return {}; }

   private:
    /// LCG seed counter, advanced atomically on each forward call.
    mutable std::atomic<std::uint64_t> seed_{12345};
};

}  // namespace rt

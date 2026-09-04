#pragma once

// =============================================================================
// Scalar neural-network layers
// =============================================================================
//
// The teaching layers, built on the scalar `Value` engine. Same math as `nn2`,
// but one graph node per number instead of one per operation -- readable, and
// far too slow for a real model.

#include <cstddef>
#include <vector>

#include "rt/autograd.hpp"
#include "rt/init_rng.hpp"

namespace rt {

class Module {
   public:
    virtual ~Module() = default;

    /// All learnable parameters (weights and biases). The optimizer iterates
    /// over these to apply updates.
    [[nodiscard]] virtual std::vector<Value> parameters() const = 0;

    /// Zero every parameter gradient. Without this, gradients accumulate
    /// across batches -- occasionally wanted, usually not.
    void zero_grad() const {
        for (const Value& p : parameters()) {
            p.zero_grad();
        }
    }
};

// =============================================================================
// Linear
// =============================================================================
//
// `out[j] = dot(weight[j], input) + bias[j]`
//
// Weights start at N(0, 0.02), the GPT-2 initialization. Zeros would leave
// every neuron computing the same thing with the same gradient, and large
// values would make activations -- and then gradients -- explode.

class Linear : public Module {
   public:
    /// weight[j][i] is the connection strength from input i to output j.
    std::vector<std::vector<Value>> weight;
    /// bias[j] is the offset added to output j. Starts at zero.
    std::vector<Value> bias;
    std::size_t in_features = 0;
    std::size_t out_features = 0;

    Linear(std::size_t in_features, std::size_t out_features, InitRng& rng);

    /// One input vector of length `in_features` -> `out_features` outputs.
    [[nodiscard]] std::vector<Value> forward(const std::vector<Value>& input) const;

    [[nodiscard]] std::vector<Value> parameters() const override;
};

// =============================================================================
// Activations
// =============================================================================
//
// Without a non-linearity, stacked linear layers collapse into one linear map.
//
// ReLU hard-gates at 0: the gradient is exactly 0 or 1, so a neuron whose input
// stays negative never recovers. GELU gates softly -- `x * sigmoid(1.702 x)` --
// which is why modern transformers use it.

[[nodiscard]] std::vector<Value> relu(const std::vector<Value>& xs);

/// GELU(x) = x * sigmoid(1.702 * x), the sigmoid approximation.
[[nodiscard]] std::vector<Value> gelu(const std::vector<Value>& xs);

/// Softmax over a vector of logits.
///
/// The max is subtracted first for numerical stability; it is taken as a plain
/// float, outside the graph, since the shift cancels out of the result.
[[nodiscard]] std::vector<Value> softmax(const std::vector<Value>& xs);

// =============================================================================
// LayerNorm
// =============================================================================
//
// Per feature vector: subtract the mean, divide by the standard deviation, then
// apply a learnable scale and shift.
//
//   y = gamma * (x - mu) / sqrt(var + eps) + beta
//
// gamma and beta let the network undo the normalization when it needs to.
// Normalizing across features rather than the batch is what makes this work at
// batch size 1, which is the inference case.

class LayerNorm : public Module {
   public:
    /// Learnable scale, one per feature. Starts at 1 -- pure normalization.
    std::vector<Value> gamma;
    /// Learnable shift, one per feature. Starts at 0 -- no shift.
    std::vector<Value> beta;
    std::size_t d_model = 0;

    explicit LayerNorm(std::size_t d_model);

    [[nodiscard]] std::vector<Value> forward(const std::vector<Value>& xs) const;

    [[nodiscard]] std::vector<Value> parameters() const override;

   private:
    /// Guards against dividing by zero when the variance is tiny.
    float eps_ = 1e-5f;
};

// =============================================================================
// Mlp
// =============================================================================
//
// x -> Linear(d_model -> 4*d_model) -> GELU -> Linear(4*d_model -> d_model)
//
// Attention routes information between positions; the FFN transforms it at each
// position. The 4x expansion is empirical, and shared by GPT-2, GPT-3 and LLaMA.

class Mlp : public Module {
   public:
    Linear fc1;  // d_model -> 4 * d_model
    Linear fc2;  // 4 * d_model -> d_model

    Mlp(std::size_t d_model, InitRng& rng);

    [[nodiscard]] std::vector<Value> forward(const std::vector<Value>& x) const;

    [[nodiscard]] std::vector<Value> parameters() const override;
};

}  // namespace rt

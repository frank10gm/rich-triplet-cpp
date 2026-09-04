#pragma once

// =============================================================================
// Value -- scalar reverse-mode automatic differentiation
// =============================================================================
//
// The teaching engine: every number is its own node in the computation graph,
// and every operation records how to propagate gradient back to its inputs.
//
// Nodes are reference-counted and share mutable state, so one value can feed
// several operations and accumulate gradient from all of them. Backward
// closures capture the nodes they touch, which forms reference cycles -- fine
// for the small graphs this engine is meant for, but the reason the tensor
// engine (`TensorNode`) exists for real models: a d_model=32 forward pass here
// builds roughly 400,000 nodes.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rt {

class Value {
   public:
    /// A leaf node: an input or a weight, with no inputs of its own.
    explicit Value(float val);

    /// A leaf node with a debugging label.
    [[nodiscard]] static Value with_label(float val, const std::string& label);

    [[nodiscard]] float val() const { return p_->val; }
    [[nodiscard]] float grad() const { return p_->grad; }

    /// Zero the gradient. Call before each backward pass.
    void zero_grad() const { p_->grad = 0.0f; }

    /// Set the scalar value -- how the optimizer applies a weight update.
    void set_val(float val) const { p_->val = val; }

    /// Set the gradient -- used for gradient clipping.
    void set_grad(float grad) const { p_->grad = grad; }

    /// Attach a debugging label, returning the same node.
    Value label(const std::string& s) const;

    /// Identity of the underlying node, for visited-sets.
    [[nodiscard]] const void* id() const { return p_.get(); }

    [[nodiscard]] std::string debug_string() const;

    // -------------------------------------------------------------------------
    // Operations -- each records how to differentiate itself
    // -------------------------------------------------------------------------

    /// z = x + y. Both inputs receive the upstream gradient unchanged, which is
    /// exactly why residual connections let gradient flow unimpeded.
    [[nodiscard]] Value add(const Value& other) const;

    /// z = x * y. Each input's gradient is scaled by the *other* value: if y is
    /// large then x moves z a lot, so x's gradient is amplified by y.
    [[nodiscard]] Value mul(const Value& other) const;

    /// z = x^n for a constant n. `pow(-1)` is the reciprocal, `pow(2)` squares.
    /// dL/dx = dL/dz * n * x^(n-1).
    [[nodiscard]] Value pow(float n) const;

    /// z = exp(x). Its own derivative, which is why it shows up everywhere
    /// from softmax to GELU.
    [[nodiscard]] Value exp() const;

    /// z = ln(x). dL/dx = dL/dz / x. Used by cross-entropy.
    [[nodiscard]] Value ln() const;

    /// z = max(0, x). Gradient is blocked entirely where the forward value was
    /// negative -- the neuron is "dead" for that input.
    [[nodiscard]] Value relu() const;

    /// z = tanh(x), range (-1, 1). dL/dx = dL/dz * (1 - tanh(x)^2).
    [[nodiscard]] Value tanh() const;

    // Expressed in terms of add/mul/pow, so their backward rules come for free.
    [[nodiscard]] Value neg() const;
    [[nodiscard]] Value sub(const Value& other) const;
    [[nodiscard]] Value div(const Value& other) const;

    /// Run the full backward pass from this node, normally the loss.
    ///
    /// Builds a topological order of all ancestors, seeds this node's gradient
    /// to 1 (dL/dL = 1), then visits in reverse order calling each rule.
    void backward() const;

   private:
    struct ValueData {
        /// The scalar computed in the forward pass.
        float val = 0.0f;
        /// Accumulated dL/d(this), filled in during backward.
        float grad = 0.0f;
        /// Debugging label, e.g. "w1" or "loss".
        std::string label;
        /// How to propagate gradient to this node's inputs.
        /// Empty for leaves, which have no inputs.
        std::function<void()> backward_fn;
        /// The nodes that produced this value, for the topological order.
        std::vector<Value> prev;
    };

    explicit Value(std::shared_ptr<ValueData> p) : p_(std::move(p)) {}

    std::shared_ptr<ValueData> p_;
};

}  // namespace rt

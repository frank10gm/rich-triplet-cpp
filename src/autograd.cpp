#include "rt/autograd.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <unordered_set>

namespace rt {

Value::Value(float val) : p_(std::make_shared<ValueData>()) { p_->val = val; }

Value Value::with_label(float val, const std::string& label) {
    Value v(val);
    v.p_->label = label;
    return v;
}

Value Value::label(const std::string& s) const {
    p_->label = s;
    return *this;
}

std::string Value::debug_string() const {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "Value(val=%.4f, grad=%.4f, label=\"%s\")",
                  static_cast<double>(p_->val), static_cast<double>(p_->grad),
                  p_->label.c_str());
    return buf;
}

// Each operation below follows the same shape: compute the forward value,
// build the output node, wire a closure that adds this operation's derivative
// into each input's gradient, and record the inputs for the topological order.
//
// The closures capture the forward values they need rather than re-reading
// them at backward time, so a `set_val` between the two passes cannot change
// what the derivative was computed against.

Value Value::add(const Value& other) const {
    Value out(val() + other.val());
    Value self_c = *this, other_c = other, out_c = out;
    out.p_->backward_fn = [self_c, other_c, out_c] {
        const float upstream = out_c.p_->grad;
        self_c.p_->grad += upstream;
        other_c.p_->grad += upstream;
    };
    out.p_->prev = {*this, other};
    return out;
}

Value Value::mul(const Value& other) const {
    const float self_val = val();
    const float other_val = other.val();
    Value out(self_val * other_val);
    Value self_c = *this, other_c = other, out_c = out;
    out.p_->backward_fn = [self_c, other_c, out_c, self_val, other_val] {
        const float upstream = out_c.p_->grad;
        // Each input is scaled by the *other* operand's value.
        self_c.p_->grad += upstream * other_val;
        other_c.p_->grad += upstream * self_val;
    };
    out.p_->prev = {*this, other};
    return out;
}

Value Value::pow(float n) const {
    const float x = val();
    Value out(std::pow(x, n));
    Value self_c = *this, out_c = out;
    out.p_->backward_fn = [self_c, out_c, n, x] {
        self_c.p_->grad += out_c.p_->grad * n * std::pow(x, n - 1.0f);
    };
    out.p_->prev = {*this};
    return out;
}

Value Value::exp() const {
    const float out_val = std::exp(val());
    Value out(out_val);
    Value self_c = *this, out_c = out;
    out.p_->backward_fn = [self_c, out_c, out_val] {
        // d/dx exp(x) = exp(x), which is the forward value.
        self_c.p_->grad += out_c.p_->grad * out_val;
    };
    out.p_->prev = {*this};
    return out;
}

Value Value::ln() const {
    const float x = val();
    assert(x > 0.0f && "ln of a non-positive value");
    Value out(std::log(x));
    Value self_c = *this, out_c = out;
    out.p_->backward_fn = [self_c, out_c, x] {
        // Multiply by the reciprocal rather than dividing: `u * (1/x)` and
        // `u / x` differ in the last ulp, and the reference multiplies.
        self_c.p_->grad += out_c.p_->grad * (1.0f / x);
    };
    out.p_->prev = {*this};
    return out;
}

Value Value::relu() const {
    const float x = val();
    Value out(std::max(x, 0.0f));
    Value self_c = *this, out_c = out;
    out.p_->backward_fn = [self_c, out_c, x] {
        self_c.p_->grad += out_c.p_->grad * (x > 0.0f ? 1.0f : 0.0f);
    };
    out.p_->prev = {*this};
    return out;
}

Value Value::tanh() const {
    const float t = std::tanh(val());
    Value out(t);
    Value self_c = *this, out_c = out;
    out.p_->backward_fn = [self_c, out_c, t] {
        self_c.p_->grad += out_c.p_->grad * (1.0f - t * t);
    };
    out.p_->prev = {*this};
    return out;
}

Value Value::neg() const { return mul(Value(-1.0f)); }

Value Value::sub(const Value& other) const { return add(other.neg()); }

Value Value::div(const Value& other) const { return mul(other.pow(-1.0f)); }

void Value::backward() const {
    std::vector<Value> topo;
    std::unordered_set<const void*> visited;

    // Depth-first: a node is pushed only after all its inputs, so reversing the
    // result visits every consumer before its producers.
    const auto build_topo = [&](auto&& self, const Value& v) -> void {
        if (!visited.insert(v.id()).second) {
            return;
        }
        for (const Value& child : v.p_->prev) {
            self(self, child);
        }
        topo.push_back(v);
    };
    build_topo(build_topo, *this);

    p_->grad = 1.0f;  // dL/dL = 1

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if (it->p_->backward_fn) {
            it->p_->backward_fn();
        }
    }
}

}  // namespace rt

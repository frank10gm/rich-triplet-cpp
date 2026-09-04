#include "rt/nn.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

namespace rt {

// =============================================================================
// Linear
// =============================================================================

Linear::Linear(std::size_t in_features, std::size_t out_features, InitRng& rng)
    : in_features(in_features), out_features(out_features) {
    constexpr float std_dev = 0.02f;  // GPT-2 initialization

    weight.reserve(out_features);
    for (std::size_t j = 0; j < out_features; ++j) {
        std::vector<Value> row;
        row.reserve(in_features);
        for (float w : rng.normal_vec(in_features, std_dev)) {
            row.emplace_back(w);
        }
        weight.push_back(std::move(row));
    }

    bias.reserve(out_features);
    for (std::size_t j = 0; j < out_features; ++j) {
        bias.emplace_back(0.0f);
    }
}

std::vector<Value> Linear::forward(const std::vector<Value>& input) const {
    assert(input.size() == in_features && "Linear forward: input length != in_features");

    std::vector<Value> out;
    out.reserve(out_features);
    for (std::size_t j = 0; j < out_features; ++j) {
        // Left-folded dot product, then the bias.
        Value dot = weight[j][0].mul(input[0]);
        for (std::size_t i = 1; i < in_features; ++i) {
            dot = dot.add(weight[j][i].mul(input[i]));
        }
        out.push_back(dot.add(bias[j]));
    }
    return out;
}

std::vector<Value> Linear::parameters() const {
    std::vector<Value> params;
    for (const auto& row : weight) {
        params.insert(params.end(), row.begin(), row.end());
    }
    params.insert(params.end(), bias.begin(), bias.end());
    return params;
}

// =============================================================================
// Activations
// =============================================================================

std::vector<Value> relu(const std::vector<Value>& xs) {
    std::vector<Value> out;
    out.reserve(xs.size());
    for (const Value& x : xs) {
        out.push_back(x.relu());
    }
    return out;
}

std::vector<Value> gelu(const std::vector<Value>& xs) {
    std::vector<Value> out;
    out.reserve(xs.size());
    for (const Value& x : xs) {
        // sigmoid(1.702 x) = 1 / (1 + exp(-1.702 x)), built from graph ops so
        // the derivative comes from the chain rule rather than a closed form.
        const Value z = Value(1.702f).mul(x);
        const Value one(1.0f);
        const Value sigmoid_z = one.div(one.add(z.neg().exp()));
        out.push_back(x.mul(sigmoid_z));
    }
    return out;
}

std::vector<Value> softmax(const std::vector<Value>& xs) {
    // The max is a plain float, outside the graph: shifting by a constant does
    // not change softmax, so it needs no gradient.
    float max_val = -std::numeric_limits<float>::infinity();
    for (const Value& x : xs) {
        max_val = std::max(max_val, x.val());
    }
    const Value max_node(max_val);

    std::vector<Value> exps;
    exps.reserve(xs.size());
    for (const Value& x : xs) {
        exps.push_back(x.sub(max_node).exp());
    }

    Value sum = exps[0];
    for (std::size_t i = 1; i < exps.size(); ++i) {
        sum = sum.add(exps[i]);
    }

    std::vector<Value> out;
    out.reserve(exps.size());
    for (const Value& e : exps) {
        out.push_back(e.div(sum));
    }
    return out;
}

// =============================================================================
// LayerNorm
// =============================================================================

LayerNorm::LayerNorm(std::size_t d_model) : d_model(d_model) {
    gamma.reserve(d_model);
    beta.reserve(d_model);
    for (std::size_t i = 0; i < d_model; ++i) {
        gamma.emplace_back(1.0f);
        beta.emplace_back(0.0f);
    }
}

std::vector<Value> LayerNorm::forward(const std::vector<Value>& xs) const {
    assert(xs.size() == d_model);
    const float n = static_cast<float>(d_model);

    Value sum = xs[0];
    for (std::size_t i = 1; i < xs.size(); ++i) {
        sum = sum.add(xs[i]);
    }
    const Value mean = sum.mul(Value(1.0f / n));

    Value var_sum = xs[0].sub(mean).mul(xs[0].sub(mean));
    for (std::size_t i = 1; i < xs.size(); ++i) {
        const Value diff = xs[i].sub(mean);
        var_sum = var_sum.add(diff.mul(diff));
    }
    const Value variance = var_sum.mul(Value(1.0f / n));

    const Value std_dev = variance.add(Value(eps_)).pow(0.5f);

    std::vector<Value> out;
    out.reserve(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const Value x_hat = xs[i].sub(mean).div(std_dev);
        out.push_back(gamma[i].mul(x_hat).add(beta[i]));
    }
    return out;
}

std::vector<Value> LayerNorm::parameters() const {
    std::vector<Value> params = gamma;
    params.insert(params.end(), beta.begin(), beta.end());
    return params;
}

// =============================================================================
// Mlp
// =============================================================================

Mlp::Mlp(std::size_t d_model, InitRng& rng)
    : fc1(d_model, 4 * d_model, rng), fc2(4 * d_model, d_model, rng) {}

std::vector<Value> Mlp::forward(const std::vector<Value>& x) const {
    return fc2.forward(gelu(fc1.forward(x)));
}

std::vector<Value> Mlp::parameters() const {
    std::vector<Value> p = fc1.parameters();
    const std::vector<Value> p2 = fc2.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    return p;
}

}  // namespace rt

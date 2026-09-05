#include "rt/tensor_node.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numbers>
#include <unordered_set>

#include "rt/ndarray.hpp"

namespace rt {

TensorNode TensorNode::leaf(Mat data) {
    auto node = std::make_shared<NodeData>();
    node->grad = Mat::zeros(data.rows, data.cols);
    node->data = std::move(data);
    return TensorNode(std::move(node));
}

void TensorNode::zero_grad() const { p_->grad = Mat::zeros(p_->grad.rows, p_->grad.cols); }

void TensorNode::set_backward(std::function<void()> f, std::vector<TensorNode> prev) const {
    p_->backward_fn = std::move(f);
    p_->prev = std::move(prev);
}

void TensorNode::call_backward_fn() const {
    if (p_->backward_fn) {
        p_->backward_fn();
    }
}

void TensorNode::free_graph() const {
    std::vector<TensorNode> stack{*this};
    while (!stack.empty()) {
        TensorNode node = stack.back();
        stack.pop_back();
        NodeData& inner = *node.p_;
        // Skip leaves: either model parameters that must be preserved, or the
        // output leaves of fused linears, which plain refcounting can free.
        // A node visited twice also lands here, since the first visit cleared it.
        if (!inner.backward_fn && inner.prev.empty()) {
            continue;
        }
        inner.backward_fn = nullptr;
        inner.grad = Mat::zeros(0, 0);
        inner.data = Mat::zeros(0, 0);
        std::vector<TensorNode> prev = std::move(inner.prev);
        inner.prev.clear();
        stack.insert(stack.end(), std::make_move_iterator(prev.begin()),
                     std::make_move_iterator(prev.end()));
    }
}

void TensorNode::seed_grad_ones() const { p_->grad = Mat::ones(p_->data.rows, p_->data.cols); }

std::string TensorNode::debug_string() const {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "TensorNode(shape=[%zu,%zu], grad_norm=%.4f)", p_->data.rows,
                  p_->data.cols, static_cast<double>(p_->grad.norm()));
    return buf;
}

// =============================================================================
// Operations
// =============================================================================

TensorNode TensorNode::matmul(const TensorNode& b) const {
    TensorNode out = TensorNode::leaf(data().matmul(b.data()));

    TensorNode self_c = *this, b_c = b, out_c = out;
    out.set_backward(
        [self_c, b_c, out_c] {
            const Mat& dout = out_c.grad();

            // dA += dC @ B.T
            self_c.grad_add(dout.matmul(b_c.data().transpose()));
            // dB += A.T @ dC
            b_c.grad_add(self_c.data().transpose().matmul(dout));
        },
        {*this, b});
    return out;
}

TensorNode TensorNode::add(const TensorNode& b) const {
    TensorNode out = TensorNode::leaf(data().add(b.data()));

    TensorNode self_c = *this, b_c = b, out_c = out;
    out.set_backward(
        [self_c, b_c, out_c] {
            const Mat dout = out_c.grad();
            self_c.grad_add(dout);
            b_c.grad_add(dout);
        },
        {*this, b});
    return out;
}

TensorNode TensorNode::add_bias(const TensorNode& bias) const {
    assert(bias.data().rows == 1);
    assert(bias.data().cols == data().cols);

    const Mat& a = data();
    const Mat& b = bias.data();
    TensorNode out = TensorNode::leaf(Mat::from_fn(
        a.rows, a.cols, [&](std::size_t r, std::size_t c) { return a.at(r, c) + b.at(0, c); }));

    TensorNode self_c = *this, bias_c = bias, out_c = out;
    out.set_backward(
        [self_c, bias_c, out_c] {
            const Mat& dout = out_c.grad();
            // dA += dC
            self_c.grad_add(dout);
            // d_bias += sum over rows of dC -> [1, cols]
            bias_c.grad_add(dout.sum_rows());
        },
        {*this, bias});
    return out;
}

TensorNode TensorNode::gelu() const {
    const Mat& x = data();

    // sigmoid(1.702 x), reused by the forward value.
    const Mat sigmoid_vals = Mat::from_fn(x.rows, x.cols, [&](std::size_t r, std::size_t c) {
        return 1.0f / (1.0f + std::exp(-1.702f * x.at(r, c)));
    });
    TensorNode out = TensorNode::leaf(Mat::from_fn(
        x.rows, x.cols,
        [&](std::size_t r, std::size_t c) { return x.at(r, c) * sigmoid_vals.at(r, c); }));

    TensorNode self_c = *this, out_c = out;
    out.set_backward(
        [self_c, out_c] {
            const Mat& dout = out_c.grad();
            const Mat& x_data = self_c.data();
            // d/dx [x * sigma(1.702x)] = sigma + x * sigma * (1 - sigma) * 1.702
            const Mat dx =
                Mat::from_fn(x_data.rows, x_data.cols, [&](std::size_t r, std::size_t c) {
                    const float xv = x_data.at(r, c);
                    const float s = 1.0f / (1.0f + std::exp(-1.702f * xv));
                    return dout.at(r, c) * (s + xv * s * (1.0f - s) * 1.702f);
                });
            self_c.grad_add(dx);
        },
        {*this});
    return out;
}

namespace {
constexpr float kSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float kGeluCoeff = 0.044715f;
}  // namespace

TensorNode TensorNode::gelu_tanh() const {
    const Mat& x = data();

    TensorNode out = TensorNode::leaf(
        Mat::from_fn(x.rows, x.cols, [&](std::size_t r, std::size_t c) {
            const float xv = x.at(r, c);
            const float inner = kSqrt2OverPi * (xv + kGeluCoeff * xv * xv * xv);
            return xv * 0.5f * (1.0f + std::tanh(inner));
        }));

    TensorNode self_c = *this, out_c = out;
    out.set_backward(
        [self_c, out_c] {
            const Mat& dout = out_c.grad();
            const Mat& x_data = self_c.data();
            const Mat dx =
                Mat::from_fn(x_data.rows, x_data.cols, [&](std::size_t r, std::size_t c) {
                    const float xv = x_data.at(r, c);
                    const float x3 = xv * xv * xv;
                    const float inner = kSqrt2OverPi * (xv + kGeluCoeff * x3);
                    const float t = std::tanh(inner);
                    const float sech2 = 1.0f - t * t;
                    const float dg = 0.5f * (1.0f + t) +
                                     xv * 0.5f * sech2 * kSqrt2OverPi *
                                         (1.0f + 3.0f * kGeluCoeff * xv * xv);
                    return dout.at(r, c) * dg;
                });
            self_c.grad_add(dx);
        },
        {*this});
    return out;
}

TensorNode TensorNode::softmax() const {
    const Mat& x = data();
    const std::size_t t = x.rows, v = x.cols;

    // Numerically stable: subtract the row max before exp.
    Mat s_data = Mat::zeros(t, v);
    for (std::size_t r = 0; r < t; ++r) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < v; ++c) {
            row_max = std::max(row_max, x.at(r, c));
        }
        float row_sum = 0.0f;
        for (std::size_t c = 0; c < v; ++c) {
            const float e = std::exp(x.at(r, c) - row_max);
            s_data.at_mut(r, c) = e;
            row_sum += e;
        }
        for (std::size_t c = 0; c < v; ++c) {
            s_data.at_mut(r, c) /= row_sum;
        }
    }

    TensorNode out = TensorNode::leaf(std::move(s_data));
    TensorNode self_c = *this, out_c = out;
    out.set_backward(
        [self_c, out_c] {
            const Mat& ds = out_c.grad();  // upstream gradient
            const Mat& s = out_c.data();   // forward softmax values
            const std::size_t t = s.rows, v = s.cols;

            Mat dx = Mat::zeros(t, v);
            for (std::size_t r = 0; r < t; ++r) {
                float dot = 0.0f;
                for (std::size_t c = 0; c < v; ++c) {
                    dot += ds.at(r, c) * s.at(r, c);
                }
                for (std::size_t c = 0; c < v; ++c) {
                    dx.at_mut(r, c) = s.at(r, c) * (ds.at(r, c) - dot);
                }
            }
            self_c.grad_add(dx);
        },
        {*this});
    return out;
}

TensorNode TensorNode::layer_norm(const TensorNode& gamma, const TensorNode& beta) const {
    const Mat& x = data();
    const Mat& g = gamma.data();
    const Mat& b = beta.data();
    const std::size_t t = x.rows, d = x.cols;
    constexpr float eps = 1e-5f;
    const float inv_d = 1.0f / static_cast<float>(d);

    // Forward: per-row mean and variance, then normalize.
    auto var = std::make_shared<std::vector<float>>(t, 0.0f);
    auto x_hat = std::make_shared<Mat>(Mat::zeros(t, d));

    for (std::size_t r = 0; r < t; ++r) {
        float mean = 0.0f;
        for (std::size_t c = 0; c < d; ++c) {
            mean += x.at(r, c);
        }
        mean *= inv_d;
        float v = 0.0f;
        for (std::size_t c = 0; c < d; ++c) {
            const float diff = x.at(r, c) - mean;
            v += diff * diff;
        }
        (*var)[r] = v * inv_d;
        const float inv_std = 1.0f / std::sqrt((*var)[r] + eps);
        for (std::size_t c = 0; c < d; ++c) {
            x_hat->at_mut(r, c) = (x.at(r, c) - mean) * inv_std;
        }
    }

    // Y = gamma * X_hat + beta, broadcast over rows.
    TensorNode out = TensorNode::leaf(
        Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
            return x_hat->at(r, c) * g.at(0, c) + b.at(0, c);
        }));

    TensorNode self_c = *this, gamma_c = gamma, beta_c = beta, out_c = out;
    out.set_backward(
        [self_c, gamma_c, beta_c, out_c, x_hat, var, t, d, inv_d] {
            const Mat& dy = out_c.grad();
            const Mat& g = gamma_c.data();

            // dgamma = sum_rows(dY * X_hat)
            Mat dg = Mat::zeros(1, d);
            for (std::size_t c = 0; c < d; ++c) {
                for (std::size_t r = 0; r < t; ++r) {
                    dg.at_mut(0, c) += dy.at(r, c) * x_hat->at(r, c);
                }
            }
            gamma_c.grad_add(dg);

            // dbeta = sum_rows(dY)
            beta_c.grad_add(dy.sum_rows());

            // dX per row
            Mat dx = Mat::zeros(t, d);
            std::vector<float> d_row(d);
            for (std::size_t r = 0; r < t; ++r) {
                const float inv_std = 1.0f / std::sqrt((*var)[r] + eps);
                float mean_d = 0.0f, mean_dxh = 0.0f;
                for (std::size_t c = 0; c < d; ++c) {
                    d_row[c] = dy.at(r, c) * g.at(0, c);
                    mean_d += d_row[c];
                    mean_dxh += d_row[c] * x_hat->at(r, c);
                }
                mean_d *= inv_d;
                mean_dxh *= inv_d;
                for (std::size_t c = 0; c < d; ++c) {
                    dx.at_mut(r, c) =
                        inv_std * (d_row[c] - mean_d - x_hat->at(r, c) * mean_dxh);
                }
            }
            self_c.grad_add(dx);
        },
        {*this, gamma, beta});
    return out;
}

namespace {
/// Shared body of `rms_norm` and `rms_norm_gemma3`. `scale_of(gamma_value)`
/// turns a stored gamma into the multiplier actually applied: identity for the
/// standard form, `1 + g` for Gemma 3.
template <typename ScaleOf>
TensorNode rms_norm_impl(const TensorNode& self, const TensorNode& gamma, float eps,
                         ScaleOf scale_of) {
    const Mat& x = self.data();
    const Mat& g = gamma.data();
    const std::size_t t = x.rows, d = x.cols;
    const float inv_d = 1.0f / static_cast<float>(d);

    auto rms_inv = std::make_shared<std::vector<float>>(t, 0.0f);
    auto x_hat = std::make_shared<Mat>(Mat::zeros(t, d));

    for (std::size_t r = 0; r < t; ++r) {
        float mean_sq = 0.0f;
        for (std::size_t c = 0; c < d; ++c) {
            mean_sq += x.at(r, c) * x.at(r, c);
        }
        mean_sq *= inv_d;
        (*rms_inv)[r] = 1.0f / std::sqrt(mean_sq + eps);
        for (std::size_t c = 0; c < d; ++c) {
            x_hat->at_mut(r, c) = x.at(r, c) * (*rms_inv)[r];
        }
    }

    TensorNode out = TensorNode::leaf(Mat::from_fn(
        t, d, [&](std::size_t r, std::size_t c) { return x_hat->at(r, c) * scale_of(g.at(0, c)); }));

    TensorNode self_c = self, gamma_c = gamma, out_c = out;
    out.set_backward(
        [self_c, gamma_c, out_c, x_hat, rms_inv, t, d, inv_d, scale_of] {
            const Mat& dout = out_c.grad();
            const Mat& g = gamma_c.data();

            // dgamma = sum_t(dout[t] * x_hat[t]); the (1 + g) form has the same
            // derivative w.r.t. gamma, since d/dg[(1+g) * x_hat] = x_hat.
            Mat dg = Mat::zeros(1, d);
            for (std::size_t c = 0; c < d; ++c) {
                for (std::size_t r = 0; r < t; ++r) {
                    dg.at_mut(0, c) += dout.at(r, c) * x_hat->at(r, c);
                }
            }
            gamma_c.grad_add(dg);

            // dx[t] = r[t] * (D[t] - x_hat[t] * mean(D[t] * x_hat[t]))
            // with D[t,i] = dout[t,i] * scale(gamma[i])
            Mat dx = Mat::zeros(t, d);
            std::vector<float> d_row(d);
            for (std::size_t r = 0; r < t; ++r) {
                float mean_dxh = 0.0f;
                for (std::size_t c = 0; c < d; ++c) {
                    d_row[c] = dout.at(r, c) * scale_of(g.at(0, c));
                    mean_dxh += d_row[c] * x_hat->at(r, c);
                }
                mean_dxh *= inv_d;
                for (std::size_t c = 0; c < d; ++c) {
                    dx.at_mut(r, c) = (*rms_inv)[r] * (d_row[c] - x_hat->at(r, c) * mean_dxh);
                }
            }
            self_c.grad_add(dx);
        },
        {self, gamma});
    return out;
}
}  // namespace

TensorNode TensorNode::rms_norm(const TensorNode& gamma, float eps) const {
    return rms_norm_impl(*this, gamma, eps, [](float g) { return g; });
}

TensorNode TensorNode::rms_norm_gemma3(const TensorNode& gamma, float eps) const {
    return rms_norm_impl(*this, gamma, eps, [](float g) { return 1.0f + g; });
}

TensorNode TensorNode::silu() const {
    const Mat& x = data();

    // Compute the sigmoid, then multiply -- `x / (1 + e)` would round
    // differently in the last ulp and drift from the reference.
    const Mat sig = Mat::from_fn(x.rows, x.cols, [&](std::size_t r, std::size_t c) {
        return 1.0f / (1.0f + std::exp(-x.at(r, c)));
    });

    TensorNode out = TensorNode::leaf(Mat::from_fn(
        x.rows, x.cols,
        [&](std::size_t r, std::size_t c) { return x.at(r, c) * sig.at(r, c); }));

    TensorNode self_c = *this, out_c = out;
    out.set_backward(
        [self_c, out_c] {
            const Mat& dout = out_c.grad();
            const Mat& x_data = self_c.data();
            const Mat dx =
                Mat::from_fn(x_data.rows, x_data.cols, [&](std::size_t r, std::size_t c) {
                    const float xv = x_data.at(r, c);
                    const float s = 1.0f / (1.0f + std::exp(-xv));
                    // d/dx[x * sigma] = sigma + x * sigma * (1 - sigma)
                    return dout.at(r, c) * (s + xv * s * (1.0f - s));
                });
            self_c.grad_add(dx);
        },
        {*this});
    return out;
}

TensorNode TensorNode::clamp(float min_val, float max_val) const {
    const Mat& x = data();
    TensorNode out = TensorNode::leaf(Mat::from_fn(
        x.rows, x.cols,
        [&](std::size_t r, std::size_t c) { return std::clamp(x.at(r, c), min_val, max_val); }));

    TensorNode self_c = *this, out_c = out;
    out.set_backward(
        [self_c, out_c, min_val, max_val] {
            const Mat& dout = out_c.grad();
            const Mat& x_data = self_c.data();
            const Mat dx =
                Mat::from_fn(x_data.rows, x_data.cols, [&](std::size_t r, std::size_t c) {
                    const float v = x_data.at(r, c);
                    return (v > min_val && v < max_val) ? dout.at(r, c) : 0.0f;
                });
            self_c.grad_add(dx);
        },
        {*this});
    return out;
}

TensorNode TensorNode::mul_elem_node(const TensorNode& other) const {
    assert(data().rows == other.data().rows && data().cols == other.data().cols &&
           "mul_elem_node: shape mismatch");

    TensorNode out = TensorNode::leaf(data().mul_elem(other.data()));

    TensorNode self_c = *this, other_c = other, out_c = out;
    out.set_backward(
        [self_c, other_c, out_c] {
            const Mat& dout = out_c.grad();
            // dA = dC * B, dB = dC * A -- computed before either is updated.
            const Mat da = dout.mul_elem(other_c.data());
            const Mat db = dout.mul_elem(self_c.data());
            self_c.grad_add(da);
            other_c.grad_add(db);
        },
        {*this, other});
    return out;
}

namespace {
/// Forward rotation shared by both RoPE variants: pairs (2i, 2i+1) rotate by
/// the precomputed angle for that (row, pair).
Mat rope_forward(const Mat& x, const Mat& angles) {
    return Mat::from_fn(x.rows, x.cols, [&](std::size_t row, std::size_t col) {
        const float angle = angles.at(row, col / 2);
        const float cos_a = std::cos(angle), sin_a = std::sin(angle);
        if (col % 2 == 0) {
            return x.at(row, col) * cos_a - x.at(row, col + 1) * sin_a;
        }
        return x.at(row, col) * cos_a + x.at(row, col - 1) * sin_a;
    });
}

/// Backward is the inverse rotation, by -angle.
Mat rope_backward(const Mat& dout, const Mat& angles, std::size_t t, std::size_t d) {
    return Mat::from_fn(t, d, [&](std::size_t row, std::size_t col) {
        const float angle = angles.at(row, col / 2);
        const float cos_a = std::cos(angle), sin_a = std::sin(angle);
        if (col % 2 == 0) {
            return dout.at(row, col) * cos_a + dout.at(row, col + 1) * sin_a;
        }
        return dout.at(row, col) * cos_a - dout.at(row, col - 1) * sin_a;
    });
}
}  // namespace

TensorNode TensorNode::rope_apply(std::size_t seq_offset, float theta) const {
    const Mat& x = data();
    const std::size_t t = x.rows, d = x.cols;
    assert(d % 2 == 0 && "rope_apply: d_head must be even");

    // Angle table, reused in backward.
    auto angles = std::make_shared<Mat>(
        Mat::from_fn(t, d / 2, [&](std::size_t row, std::size_t pair) {
            const float pos = static_cast<float>(seq_offset + row);
            return pos / std::pow(theta, 2.0f * static_cast<float>(pair) / static_cast<float>(d));
        }));

    TensorNode out = TensorNode::leaf(rope_forward(x, *angles));

    TensorNode self_c = *this, out_c = out;
    out.set_backward(
        [self_c, out_c, angles, t, d] {
            self_c.grad_add(rope_backward(out_c.grad(), *angles, t, d));
        },
        {*this});
    return out;
}

TensorNode TensorNode::rope_apply_yarn(std::size_t seq_offset, float theta,
                                       std::size_t original_ctx, std::size_t max_ctx,
                                       float beta_fast, float beta_slow) const {
    const Mat& x = data();
    const std::size_t t = x.rows, d = x.cols;
    assert(d % 2 == 0 && "rope_apply_yarn: d_head must be even");

    const float scale = static_cast<float>(max_ctx) / static_cast<float>(original_ctx);

    // Attention temperature correction, YaRN paper eq. 12: keeps the softmax
    // well-calibrated at long range.
    const float mscale = 0.1f * std::log(scale) + 1.0f;

    // Per-pair effective angles. Frequency pair i has omega = 1/theta^(2i/d);
    // how fast it rotates over the trained context decides how much the
    // position is interpolated.
    auto angles = std::make_shared<Mat>(
        Mat::from_fn(t, d / 2, [&](std::size_t row, std::size_t pair) {
            const float pos = static_cast<float>(seq_offset + row);
            const float omega =
                1.0f / std::pow(theta, 2.0f * static_cast<float>(pair) / static_cast<float>(d));

            // Cycles this frequency completes per trained context.
            const float cycles_per_ctx = static_cast<float>(original_ctx) * omega /
                                         (2.0f * std::numbers::pi_v<float>);

            // Ramp: 0 = fully interpolated (slow), 1 = untouched (fast).
            float ramp;
            if (cycles_per_ctx < beta_slow) {
                ramp = 0.0f;
            } else if (cycles_per_ctx > beta_fast) {
                ramp = 1.0f;
            } else {
                ramp = (cycles_per_ctx - beta_slow) / (beta_fast - beta_slow);
            }

            const float effective_pos = (1.0f - ramp) * (pos / scale) + ramp * pos;
            return effective_pos * omega * mscale;
        }));

    TensorNode out = TensorNode::leaf(rope_forward(x, *angles));

    TensorNode self_c = *this, out_c = out;
    out.set_backward(
        [self_c, out_c, angles, t, d] {
            self_c.grad_add(rope_backward(out_c.grad(), *angles, t, d));
        },
        {*this});
    return out;
}

namespace {
/// Causal softmax over a [T, T] score matrix: mask j > i, then softmax rows.
Mat causal_softmax(Mat scores) {
    const std::size_t t = scores.rows;
    for (std::size_t i = 0; i < t; ++i) {
        for (std::size_t j = i + 1; j < t; ++j) {
            scores.at_mut(i, j) = -1e9f;
        }
    }
    Mat w = Mat::zeros(t, t);
    for (std::size_t r = 0; r < t; ++r) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < t; ++c) {
            row_max = std::max(row_max, scores.at(r, c));
        }
        float row_sum = 0.0f;
        for (std::size_t c = 0; c < t; ++c) {
            const float e = std::exp(scores.at(r, c) - row_max);
            w.at_mut(r, c) = e;
            row_sum += e;
        }
        for (std::size_t c = 0; c < t; ++c) {
            w.at_mut(r, c) /= row_sum;
        }
    }
    return w;
}

/// Softmax VJP restricted to the causal region: dS[i,j] = W[i,j] * (dW[i,j] -
/// dot) for j <= i, zero above the diagonal.
Mat causal_softmax_backward(const Mat& dw, const Mat& w, float scale) {
    const std::size_t t = w.rows;
    Mat dscores = Mat::zeros(t, t);
    for (std::size_t r = 0; r < t; ++r) {
        float dot = 0.0f;
        for (std::size_t c = 0; c <= r; ++c) {
            dot += dw.at(r, c) * w.at(r, c);
        }
        for (std::size_t c = 0; c <= r; ++c) {
            dscores.at_mut(r, c) = w.at(r, c) * (dw.at(r, c) - dot);
        }
    }
    return dscores.scale(scale);
}

/// Extract head `h`'s [T, d_head] block from a [T, H*d_head] matrix.
Mat head_slice(const Mat& m, std::size_t h, std::size_t d_head) {
    return Mat::from_fn(m.rows, d_head, [&](std::size_t r, std::size_t c) {
        return m.at(r, h * d_head + c);
    });
}

/// Accumulate a [T, d_head] block back into head `h` of a [T, H*d_head] matrix.
void head_accumulate(Mat& dst, const Mat& src, std::size_t h, std::size_t d_head) {
    for (std::size_t r = 0; r < src.rows; ++r) {
        for (std::size_t c = 0; c < d_head; ++c) {
            dst.at_mut(r, h * d_head + c) += src.at(r, c);
        }
    }
}
}  // namespace

TensorNode TensorNode::causal_attention(const TensorNode& q, const TensorNode& k,
                                        const TensorNode& v, std::size_t d_head) {
    const Mat& q_d = q.data();
    const Mat& k_d = k.data();
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));

    auto weights = std::make_shared<Mat>(
        causal_softmax(q_d.matmul(k_d.transpose()).scale(scale)));

    TensorNode out = TensorNode::leaf(weights->matmul(v.data()));

    TensorNode q_c = q, k_c = k, v_c = v, out_c = out;
    out.set_backward(
        [q_c, k_c, v_c, out_c, weights, scale] {
            const Mat& dout = out_c.grad();
            const Mat& w = *weights;

            // dV = W.T @ dOut
            v_c.grad_add(w.transpose().matmul(dout));

            // d_weights = dOut @ V.T, then back through the causal softmax.
            const Mat dw = dout.matmul(v_c.data().transpose());
            const Mat dscores = causal_softmax_backward(dw, w, scale);

            // dQ += dscores @ K, dK += dscores.T @ Q
            q_c.grad_add(dscores.matmul(k_c.data()));
            k_c.grad_add(dscores.transpose().matmul(q_c.data()));
        },
        {q, k, v});
    return out;
}

namespace {
/// Backward shared by `gqa_attention` and `batched_gqa_attention`: both store
/// one [T, T] weight matrix per query head and propagate identically.
void gqa_backward(const TensorNode& q_c, const TensorNode& k_c, const TensorNode& v_c,
                  const TensorNode& out_c, const std::vector<Mat>& all_weights,
                  std::size_t n_q_heads, std::size_t n_kv_heads, std::size_t d_head,
                  std::size_t t, float scale) {
    const Mat& dout = out_c.grad();
    const std::size_t group_size = n_q_heads / n_kv_heads;

    Mat dq_data = Mat::zeros(t, n_q_heads * d_head);
    Mat dk_data = Mat::zeros(t, n_kv_heads * d_head);
    Mat dv_data = Mat::zeros(t, n_kv_heads * d_head);

    for (std::size_t qh = 0; qh < n_q_heads; ++qh) {
        const std::size_t kvh = qh / group_size;
        const Mat& w = all_weights[qh];

        const Mat dout_h = head_slice(dout, qh, d_head);
        const Mat q_h = head_slice(q_c.data(), qh, d_head);
        const Mat k_h = head_slice(k_c.data(), kvh, d_head);
        const Mat v_h = head_slice(v_c.data(), kvh, d_head);

        // dV_kvh += W.T @ dOut_h
        head_accumulate(dv_data, w.transpose().matmul(dout_h), kvh, d_head);

        // dW = dOut_h @ V_h.T, then back through the causal softmax.
        const Mat dscores = causal_softmax_backward(dout_h.matmul(v_h.transpose()), w, scale);

        // dQ_h += dScores @ K_h,  dK_kvh += dScores.T @ Q_h
        head_accumulate(dq_data, dscores.matmul(k_h), qh, d_head);
        head_accumulate(dk_data, dscores.transpose().matmul(q_h), kvh, d_head);
    }

    q_c.grad_add(dq_data);
    k_c.grad_add(dk_data);
    v_c.grad_add(dv_data);
}
}  // namespace

TensorNode TensorNode::gqa_attention(const TensorNode& q, const TensorNode& k, const TensorNode& v,
                                     std::size_t n_q_heads, std::size_t n_kv_heads,
                                     std::size_t d_head) {
    const Mat& q_data = q.data();
    const Mat& k_data = k.data();
    const Mat& v_data = v.data();
    const std::size_t t = q_data.rows;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    const std::size_t group_size = n_q_heads / n_kv_heads;

    assert(q_data.cols == n_q_heads * d_head);
    assert(k_data.cols == n_kv_heads * d_head);
    assert(v_data.cols == n_kv_heads * d_head);

    // Forward: attention per query head, keeping the weights for backward.
    Mat out_data = Mat::zeros(t, n_q_heads * d_head);
    auto all_weights = std::make_shared<std::vector<Mat>>(n_q_heads);

    for (std::size_t qh = 0; qh < n_q_heads; ++qh) {
        const std::size_t kvh = qh / group_size;
        const Mat q_h = head_slice(q_data, qh, d_head);
        const Mat k_h = head_slice(k_data, kvh, d_head);
        const Mat v_h = head_slice(v_data, kvh, d_head);

        Mat w = causal_softmax(q_h.matmul(k_h.transpose()).scale(scale));
        const Mat out_h = w.matmul(v_h);
        for (std::size_t r = 0; r < t; ++r) {
            for (std::size_t c = 0; c < d_head; ++c) {
                out_data.at_mut(r, qh * d_head + c) = out_h.at(r, c);
            }
        }
        (*all_weights)[qh] = std::move(w);
    }

    TensorNode out = TensorNode::leaf(std::move(out_data));
    TensorNode q_c = q, k_c = k, v_c = v, out_c = out;
    out.set_backward(
        [q_c, k_c, v_c, out_c, all_weights, n_q_heads, n_kv_heads, d_head, t, scale] {
            gqa_backward(q_c, k_c, v_c, out_c, *all_weights, n_q_heads, n_kv_heads, d_head, t,
                         scale);
        },
        {q, k, v});
    return out;
}

TensorNode TensorNode::batched_gqa_attention(const TensorNode& q, const TensorNode& k,
                                             const TensorNode& v, std::size_t n_q_heads,
                                             std::size_t n_kv_heads, std::size_t d_head) {
    const Mat& q_data = q.data();
    const Mat& k_data = k.data();
    const Mat& v_data = v.data();
    const std::size_t t = q_data.rows;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    const std::size_t group_size = n_q_heads / n_kv_heads;

    assert(q_data.cols == n_q_heads * d_head);
    assert(k_data.cols == n_kv_heads * d_head);
    assert(v_data.cols == n_kv_heads * d_head);

    // Reshape [T, H*D] -> [T, H, D] -> permute -> [H, T, D].
    const NDArray q_nd = NDArray::from_mat(q_data).reshape({t, n_q_heads, d_head}).permute({1, 0, 2});
    const NDArray k_nd =
        NDArray::from_mat(k_data).reshape({t, n_kv_heads, d_head}).permute({1, 0, 2});
    const NDArray v_nd =
        NDArray::from_mat(v_data).reshape({t, n_kv_heads, d_head}).permute({1, 0, 2});

    // Expand the KV heads: each serves `group_size` query heads.
    const NDArray k_exp = NDArray::from_fn(
        {n_q_heads, t, d_head},
        [&](const std::vector<std::size_t>& idx) { return k_nd.at({idx[0] / group_size, idx[1], idx[2]}); });
    const NDArray v_exp = NDArray::from_fn(
        {n_q_heads, t, d_head},
        [&](const std::vector<std::size_t>& idx) { return v_nd.at({idx[0] / group_size, idx[1], idx[2]}); });

    // scores [n_q, T, T] = Q [n_q,T,D] @ K^T [n_q,D,T] * scale
    NDArray scores = q_nd.bmm(k_exp.permute({0, 2, 1})).scale(scale);

    // Causal mask.
    for (std::size_t h = 0; h < n_q_heads; ++h) {
        for (std::size_t i = 0; i < t; ++i) {
            for (std::size_t j = i + 1; j < t; ++j) {
                scores.at_mut({h, i, j}) = -1e9f;
            }
        }
    }

    const NDArray weights = scores.softmax(2);
    const NDArray out_nd = weights.bmm(v_exp);

    // [n_q, T, D] -> [T, n_q, D] -> [T, n_q*D]
    TensorNode out = TensorNode::leaf(
        out_nd.permute({1, 0, 2}).reshape({t, n_q_heads * d_head}).into_mat());

    // Pull the per-head [T, T] weights out for the backward closure.
    auto all_weights = std::make_shared<std::vector<Mat>>();
    all_weights->reserve(n_q_heads);
    for (std::size_t h = 0; h < n_q_heads; ++h) {
        all_weights->push_back(Mat::from_fn(
            t, t, [&](std::size_t r, std::size_t c) { return weights.at({h, r, c}); }));
    }

    TensorNode q_c = q, k_c = k, v_c = v, out_c = out;
    out.set_backward(
        [q_c, k_c, v_c, out_c, all_weights, n_q_heads, n_kv_heads, d_head, t, scale] {
            gqa_backward(q_c, k_c, v_c, out_c, *all_weights, n_q_heads, n_kv_heads, d_head, t,
                         scale);
        },
        {q, k, v});
    return out;
}

TensorNode TensorNode::flash_attention(const TensorNode& q, const TensorNode& k,
                                       const TensorNode& v, std::size_t d_head) {
    // Query and key/value tile sizes. They shrink automatically for small T.
    constexpr std::size_t BLOCK_R = 64;
    constexpr std::size_t BLOCK_C = 64;

    const Mat& q_data = q.data();
    const Mat& k_data = k.data();
    const Mat& v_data = v.data();
    const std::size_t t = q_data.rows;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));

    assert(q_data.cols == d_head);
    assert(k_data.cols == d_head);
    assert(v_data.cols == d_head);

    Mat out_data = Mat::zeros(t, d_head);
    // Online-softmax state per row: l = running normalizer, m = running max.
    auto l_global = std::make_shared<std::vector<float>>(t, 0.0f);
    auto m_global =
        std::make_shared<std::vector<float>>(t, -std::numeric_limits<float>::infinity());

    for (std::size_t q_start = 0; q_start < t; q_start += BLOCK_R) {
        const std::size_t q_end = std::min(q_start + BLOCK_R, t);
        const std::size_t br = q_end - q_start;

        std::vector<float> acc(br * d_head, 0.0f);
        std::vector<float> m_blk(br, -std::numeric_limits<float>::infinity());
        std::vector<float> l_blk(br, 0.0f);

        // Causal: only key positions <= the current query can attend, and the
        // latest query in this tile is q_end - 1, so kv tiles stop at q_end.
        for (std::size_t kv_start = 0; kv_start < q_end; kv_start += BLOCK_C) {
            const std::size_t kv_end = std::min(std::min(kv_start + BLOCK_C, t), q_end);
            const std::size_t bc = kv_end - kv_start;

            // Score tile [br, bc].
            std::vector<float> s(br * bc);
            for (std::size_t qi = 0; qi < br; ++qi) {
                const std::size_t global_qi = q_start + qi;
                for (std::size_t ki = 0; ki < bc; ++ki) {
                    const std::size_t global_ki = kv_start + ki;
                    if (global_ki > global_qi) {
                        s[qi * bc + ki] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    float dot = 0.0f;
                    for (std::size_t d = 0; d < d_head; ++d) {
                        dot += q_data.at(global_qi, d) * k_data.at(global_ki, d);
                    }
                    s[qi * bc + ki] = dot * scale;
                }
            }

            // Online softmax update per query row.
            for (std::size_t qi = 0; qi < br; ++qi) {
                float tile_max = -std::numeric_limits<float>::infinity();
                for (std::size_t ki = 0; ki < bc; ++ki) {
                    tile_max = std::max(tile_max, s[qi * bc + ki]);
                }
                const float m_new = std::max(m_blk[qi], tile_max);

                // Rescale the running accumulator to the new max.
                const float rescale = std::exp(m_blk[qi] - m_new);
                for (std::size_t d = 0; d < d_head; ++d) {
                    acc[qi * d_head + d] *= rescale;
                }
                l_blk[qi] *= rescale;

                for (std::size_t ki = 0; ki < bc; ++ki) {
                    const float p = std::exp(s[qi * bc + ki] - m_new);
                    l_blk[qi] += p;
                    const std::size_t global_ki = kv_start + ki;
                    for (std::size_t d = 0; d < d_head; ++d) {
                        acc[qi * d_head + d] += p * v_data.at(global_ki, d);
                    }
                }
                m_blk[qi] = m_new;
            }
        }

        // Normalize and write out this query tile.
        for (std::size_t qi = 0; qi < br; ++qi) {
            const std::size_t global_qi = q_start + qi;
            const float inv_l = 1.0f / l_blk[qi];
            for (std::size_t d = 0; d < d_head; ++d) {
                out_data.at_mut(global_qi, d) = acc[qi * d_head + d] * inv_l;
            }
            (*l_global)[global_qi] = l_blk[qi];
            (*m_global)[global_qi] = m_blk[qi];
        }
    }

    TensorNode out = TensorNode::leaf(std::move(out_data));
    TensorNode q_c = q, k_c = k, v_c = v, out_c = out;

    // Backward recomputes the softmax weights from the stored (l, m) instead of
    // keeping a T x T matrix.
    out.set_backward(
        [q_c, k_c, v_c, out_c, l_global, m_global, t, d_head, scale] {
            const Mat& dout = out_c.grad();
            const Mat& q_d = q_c.data();
            const Mat& k_d = k_c.data();
            const Mat& v_d = v_c.data();
            const Mat& out_d = out_c.data();

            Mat dq = Mat::zeros(t, d_head);
            Mat dk = Mat::zeros(t, d_head);
            Mat dv = Mat::zeros(t, d_head);

            for (std::size_t q_start = 0; q_start < t; q_start += BLOCK_R) {
                const std::size_t q_end = std::min(q_start + BLOCK_R, t);
                const std::size_t br = q_end - q_start;

                for (std::size_t kv_start = 0; kv_start < q_end; kv_start += BLOCK_C) {
                    const std::size_t kv_end = std::min(std::min(kv_start + BLOCK_C, t), q_end);
                    const std::size_t bc = kv_end - kv_start;

                    // Recompute the softmax weights p[qi, ki].
                    std::vector<float> p(br * bc, 0.0f);
                    for (std::size_t qi = 0; qi < br; ++qi) {
                        const std::size_t global_qi = q_start + qi;
                        const float m_i = (*m_global)[global_qi];
                        for (std::size_t ki = 0; ki < bc; ++ki) {
                            const std::size_t global_ki = kv_start + ki;
                            if (global_ki > global_qi) {
                                continue;  // stays 0
                            }
                            float dot = 0.0f;
                            for (std::size_t d = 0; d < d_head; ++d) {
                                dot += q_d.at(global_qi, d) * k_d.at(global_ki, d);
                            }
                            p[qi * bc + ki] =
                                std::exp(dot * scale - m_i) / (*l_global)[global_qi];
                        }
                    }

                    // dV += P.T @ dOut_tile
                    for (std::size_t ki = 0; ki < bc; ++ki) {
                        const std::size_t global_ki = kv_start + ki;
                        for (std::size_t d = 0; d < d_head; ++d) {
                            float sum = 0.0f;
                            for (std::size_t qi = 0; qi < br; ++qi) {
                                sum += p[qi * bc + ki] * dout.at(q_start + qi, d);
                            }
                            dv.at_mut(global_ki, d) += sum;
                        }
                    }

                    // dP = dOut_tile @ V.T
                    std::vector<float> dp(br * bc, 0.0f);
                    for (std::size_t qi = 0; qi < br; ++qi) {
                        for (std::size_t ki = 0; ki < bc; ++ki) {
                            const std::size_t global_ki = kv_start + ki;
                            float sum = 0.0f;
                            for (std::size_t d = 0; d < d_head; ++d) {
                                sum += dout.at(q_start + qi, d) * v_d.at(global_ki, d);
                            }
                            dp[qi * bc + ki] = sum;
                        }
                    }

                    // Softmax backward: dS = P * (dP - Di), where
                    // Di = sum_j P[qi,j] * dP[qi,j] = dOut[qi] . out[qi],
                    // since out = sum_j P * V.
                    std::vector<float> ds(br * bc, 0.0f);
                    for (std::size_t qi = 0; qi < br; ++qi) {
                        const std::size_t global_qi = q_start + qi;
                        float di = 0.0f;
                        for (std::size_t d = 0; d < d_head; ++d) {
                            di += dout.at(global_qi, d) * out_d.at(global_qi, d);
                        }
                        for (std::size_t ki = 0; ki < bc; ++ki) {
                            if (kv_start + ki > global_qi) {
                                continue;
                            }
                            ds[qi * bc + ki] = p[qi * bc + ki] * (dp[qi * bc + ki] - di) * scale;
                        }
                    }

                    // dQ += dS @ K_tile
                    for (std::size_t qi = 0; qi < br; ++qi) {
                        const std::size_t global_qi = q_start + qi;
                        for (std::size_t d = 0; d < d_head; ++d) {
                            float sum = 0.0f;
                            for (std::size_t ki = 0; ki < bc; ++ki) {
                                sum += ds[qi * bc + ki] * k_d.at(kv_start + ki, d);
                            }
                            dq.at_mut(global_qi, d) += sum;
                        }
                    }

                    // dK += dS.T @ Q_tile
                    for (std::size_t ki = 0; ki < bc; ++ki) {
                        const std::size_t global_ki = kv_start + ki;
                        for (std::size_t d = 0; d < d_head; ++d) {
                            float sum = 0.0f;
                            for (std::size_t qi = 0; qi < br; ++qi) {
                                sum += ds[qi * bc + ki] * q_d.at(q_start + qi, d);
                            }
                            dk.at_mut(global_ki, d) += sum;
                        }
                    }
                }
            }

            q_c.grad_add(dq);
            k_c.grad_add(dk);
            v_c.grad_add(dv);
        },
        {q, k, v});
    return out;
}

// =============================================================================
// Backward pass
// =============================================================================

void TensorNode::backward() const {
    // Build the topological order over all ancestors.
    std::vector<TensorNode> topo;
    std::unordered_set<const void*> visited;

    const auto build = [&](auto&& self, const TensorNode& v) -> void {
        if (!visited.insert(v.id()).second) {
            return;
        }
        for (const TensorNode& prev : v.p_->prev) {
            self(self, prev);
        }
        topo.push_back(v);
    };
    build(build, *this);

    // Seed: the gradient of the loss with respect to itself is 1.
    assert(p_->data.numel() == 1 && "backward() must be called on a 1x1 scalar");
    p_->grad = Mat::ones(1, 1);

    // Walk in reverse topological order.
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        it->call_backward_fn();
    }
}

// =============================================================================
// Checkpoint save/load
// =============================================================================

namespace {
void put_u32(std::vector<std::uint8_t>& buf, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        buf.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    }
}

void put_f32(std::vector<std::uint8_t>& buf, float v) {
    put_u32(buf, std::bit_cast<std::uint32_t>(v));
}

[[nodiscard]] Result<std::uint32_t> take_u32(const std::vector<std::uint8_t>& b, std::size_t& pos) {
    if (pos + 4 > b.size()) {
        return err("unexpected EOF reading u32");
    }
    std::uint32_t v = 0;
    for (int i = 3; i >= 0; --i) {
        v = (v << 8) | b[pos + static_cast<std::size_t>(i)];
    }
    pos += 4;
    return v;
}

[[nodiscard]] Result<float> take_f32(const std::vector<std::uint8_t>& b, std::size_t& pos) {
    if (pos + 4 > b.size()) {
        return err("unexpected EOF reading f32");
    }
    RT_TRY(bits, take_u32(b, pos));
    return std::bit_cast<float>(bits);
}
}  // namespace

Result<void> save_checkpoint(const std::string& path,
                             const std::vector<std::pair<std::string, TensorNode>>& tensors) {
    std::vector<std::uint8_t> buf;

    put_u32(buf, CKPT_MAGIC);
    put_u32(buf, CKPT_VERSION);
    put_u32(buf, static_cast<std::uint32_t>(tensors.size()));

    for (const auto& [name, node] : tensors) {
        put_u32(buf, static_cast<std::uint32_t>(name.size()));
        buf.insert(buf.end(), name.begin(), name.end());
        const Mat& data = node.data();
        put_u32(buf, static_cast<std::uint32_t>(data.rows));
        put_u32(buf, static_cast<std::uint32_t>(data.cols));
        for (float v : data.data) {
            put_f32(buf, v);
        }
    }

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return err("save_checkpoint: cannot create " + path);
    }
    f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!f) {
        return err("save_checkpoint: write failed");
    }
    return {};
}

Result<std::vector<std::pair<std::string, Mat>>> load_checkpoint(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return err("load_checkpoint: cannot read " + path);
    }
    const auto size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::uint8_t> bytes(size);
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));

    std::size_t pos = 0;
    RT_TRY(magic, take_u32(bytes, pos));
    RT_TRY(version, take_u32(bytes, pos));
    if (magic != CKPT_MAGIC) {
        char msg[80];
        std::snprintf(msg, sizeof(msg), "load_checkpoint: bad magic 0x%08X (expected 0x%08X)",
                      magic, CKPT_MAGIC);
        return err(msg);
    }
    if (version != CKPT_VERSION) {
        return err("load_checkpoint: unsupported version " + std::to_string(version));
    }

    RT_TRY(n_tensors, take_u32(bytes, pos));
    std::vector<std::pair<std::string, Mat>> tensors;
    tensors.reserve(n_tensors);

    for (std::uint32_t i = 0; i < n_tensors; ++i) {
        RT_TRY(name_len, take_u32(bytes, pos));
        if (pos + name_len > bytes.size()) {
            return err("unexpected EOF reading name");
        }
        std::string name(reinterpret_cast<const char*>(bytes.data() + pos), name_len);
        pos += name_len;

        RT_TRY(rows, take_u32(bytes, pos));
        RT_TRY(cols, take_u32(bytes, pos));
        std::vector<float> data;
        data.reserve(static_cast<std::size_t>(rows) * cols);
        for (std::size_t e = 0; e < static_cast<std::size_t>(rows) * cols; ++e) {
            RT_TRY(v, take_f32(bytes, pos));
            data.push_back(v);
        }
        tensors.emplace_back(std::move(name), Mat(std::move(data), rows, cols));
    }

    return tensors;
}

Result<void> restore_checkpoint(const std::string& path, const std::vector<TensorNode>& params) {
    RT_TRY(tensors, load_checkpoint(path));
    if (tensors.size() != params.size()) {
        return err("restore_checkpoint: checkpoint has " + std::to_string(tensors.size()) +
                   " tensors but model has " + std::to_string(params.size()) + " parameters");
    }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& [name, mat] = tensors[i];
        const Mat& p_data = params[i].data();
        if (mat.rows != p_data.rows || mat.cols != p_data.cols) {
            return err("restore_checkpoint: tensor " + std::to_string(i) + " ('" + name +
                       "') shape [" + std::to_string(mat.rows) + "," + std::to_string(mat.cols) +
                       "] does not match parameter shape [" + std::to_string(p_data.rows) + "," +
                       std::to_string(p_data.cols) + "]");
        }
        params[i].set_data(mat);
    }
    return {};
}

}  // namespace rt

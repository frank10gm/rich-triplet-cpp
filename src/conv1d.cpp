#include "rt/conv1d.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "rt/mat.hpp"

namespace rt {

namespace {

/// Add a per-output-channel bias to every row, if there is one.
void add_bias_rows(Mat& out, std::span<const float> bias) {
    if (bias.empty()) {
        return;
    }
    assert(bias.size() == out.cols && "conv1d: bias length != output channels");
    for (std::size_t r = 0; r < out.rows; ++r) {
        float* row = out.row_mut(r).data();
        for (std::size_t c = 0; c < out.cols; ++c) {
            row[c] += bias[c];
        }
    }
}

}  // namespace

// =============================================================================
// Output lengths
// =============================================================================

std::size_t conv1d_out_len(std::size_t t_in, std::size_t kernel, std::size_t dilation,
                           std::size_t padding) {
    assert(kernel >= 1 && "conv1d_out_len: kernel must be >= 1");
    assert(dilation >= 1 && "conv1d_out_len: dilation must be >= 1");
    // PyTorch's formula at stride 1: T + 2p - d*(k-1).
    const std::size_t reach = dilation * (kernel - 1);
    const std::size_t padded = t_in + 2 * padding;
    return padded > reach ? padded - reach : 0;
}

std::size_t conv_transpose1d_out_len(std::size_t t_in, std::size_t kernel, std::size_t stride,
                                     std::size_t padding, std::size_t output_padding) {
    assert(kernel >= 1 && "conv_transpose1d_out_len: kernel must be >= 1");
    assert(stride >= 1 && "conv_transpose1d_out_len: stride must be >= 1");
    if (t_in == 0) {
        return 0;
    }
    const std::size_t grown = (t_in - 1) * stride + kernel + output_padding;
    const std::size_t trim = 2 * padding;
    return grown > trim ? grown - trim : 0;
}

// =============================================================================
// Convolutions
// =============================================================================

Mat conv1d_pointwise(const Mat& x, const Mat& weight, std::span<const float> bias) {
    assert(x.cols == weight.cols && "conv1d_pointwise: x.cols != weight.cols (Cin mismatch)");
    // x [T, Cin] @ weight^T [Cin, Cout] -- a kernel-size-1 convolution is a
    // dense channel mix with no sliding window at all.
    Mat out = x.matmul_bt(weight);
    add_bias_rows(out, bias);
    return out;
}

Mat conv1d_depthwise(const Mat& x, const Mat& weight, std::span<const float> bias,
                     std::size_t dilation, std::size_t padding) {
    const std::size_t channels = x.cols;
    const std::size_t kernel = weight.cols;
    assert(weight.rows == channels && "conv1d_depthwise: weight.rows != x.cols (channel mismatch)");

    const std::size_t t_out = conv1d_out_len(x.rows, kernel, dilation, padding);
    Mat out = Mat::zeros(t_out, channels);

    // Tap-major loop order. `weight` is [C, K], so a single tap's coefficients
    // are strided by K; hoisting them into a contiguous scratch column once per
    // tap leaves the inner loop reading three contiguous arrays, which
    // vectorizes. The accumulation is order-independent, so this is free.
    std::vector<float> tap(channels);
    for (std::size_t k = 0; k < kernel; ++k) {
        for (std::size_t c = 0; c < channels; ++c) {
            tap[c] = weight.at(c, k);
        }
        for (std::size_t t = 0; t < t_out; ++t) {
            const std::size_t shifted = t + k * dilation;
            if (shifted < padding) {
                continue;  // left zero pad
            }
            const std::size_t src = shifted - padding;
            if (src >= x.rows) {
                continue;  // right zero pad
            }
            const float* xrow = x.row(src).data();
            float* orow = out.row_mut(t).data();
            for (std::size_t c = 0; c < channels; ++c) {
                orow[c] += xrow[c] * tap[c];
            }
        }
    }

    add_bias_rows(out, bias);
    return out;
}

Mat conv1d_dense(const Mat& x, const Mat& weight, std::size_t out_channels, std::size_t kernel,
                 std::span<const float> bias, std::size_t dilation, std::size_t padding) {
    assert(weight.rows == out_channels && "conv1d_dense: weight.rows != out_channels");
    assert(weight.cols == x.cols * kernel && "conv1d_dense: weight.cols != Cin * kernel");

    if (kernel == 1 && dilation == 1 && padding == 0) {
        return conv1d_pointwise(x, weight, bias);
    }

    const std::size_t c_in = x.cols;
    const std::size_t t_out = conv1d_out_len(x.rows, kernel, dilation, padding);
    Mat out = Mat::zeros(t_out, out_channels);

    for (std::size_t t = 0; t < t_out; ++t) {
        float* orow = out.row_mut(t).data();
        for (std::size_t k = 0; k < kernel; ++k) {
            const std::size_t shifted = t + k * dilation;
            if (shifted < padding) {
                continue;
            }
            const std::size_t src = shifted - padding;
            if (src >= x.rows) {
                continue;
            }
            const float* xrow = x.row(src).data();
            for (std::size_t co = 0; co < out_channels; ++co) {
                const float* wrow = weight.row(co).data();
                float acc = 0.0f;
                for (std::size_t ci = 0; ci < c_in; ++ci) {
                    acc += xrow[ci] * wrow[ci * kernel + k];
                }
                orow[co] += acc;
            }
        }
    }

    add_bias_rows(out, bias);
    return out;
}

Mat conv_transpose1d(const Mat& x, const Mat& weight, std::size_t out_channels,
                     std::size_t kernel, std::span<const float> bias, std::size_t stride,
                     std::size_t padding, std::size_t output_padding) {
    assert(weight.rows == x.cols && "conv_transpose1d: weight.rows != x.cols (Cin mismatch)");
    assert(weight.cols == out_channels * kernel &&
           "conv_transpose1d: weight.cols != out_channels * kernel");
    assert(stride >= 1 && "conv_transpose1d: stride must be >= 1");
    assert((bias.empty() || bias.size() == out_channels) &&
           "conv_transpose1d: bias length != out_channels");

    const std::size_t t_out =
        conv_transpose1d_out_len(x.rows, kernel, stride, padding, output_padding);

    // A codec decoder block picks kernel = 2*stride, padding = ceil(stride/2),
    // output_padding = stride % 2, which makes the output exactly `stride`
    // times longer. Every padding or output-padding mistake breaks that
    // multiple, so check it here: a failed assertion beats bisecting a
    // 4-block decoder by listening to the result.
    if (kernel == 2 * stride && padding == (stride + 1) / 2 && output_padding == stride % 2) {
        assert(t_out == x.rows * stride &&
               "conv_transpose1d: codec block must multiply length by exactly stride");
    }

    // Every (t, co, k) product in one gemm. `weight` is [Cin, Cout*K], so
    // column `co * K + k` of the result holds the contribution that tap `k` of
    // output channel `co` makes from input frame `t`.
    const Mat prod = x.matmul(weight);

    Mat out = Mat::zeros(t_out, out_channels);
    for (std::size_t t = 0; t < x.rows; ++t) {
        const float* prow = prod.row(t).data();
        for (std::size_t k = 0; k < kernel; ++k) {
            // Scatter form: input frame t writes to t*stride + k, then padding
            // trims from the front. No kernel flip -- see the header.
            const std::size_t placed = t * stride + k;
            if (placed < padding) {
                continue;
            }
            const std::size_t dst = placed - padding;
            if (dst >= t_out) {
                continue;
            }
            float* orow = out.row_mut(dst).data();
            for (std::size_t co = 0; co < out_channels; ++co) {
                orow[co] += prow[co * kernel + k];
            }
        }
    }

    add_bias_rows(out, bias);
    return out;
}

// =============================================================================
// Activations
// =============================================================================

void snake1d_inplace(Mat& x, std::span<const float> alpha) {
    assert(alpha.size() == x.cols && "snake1d: alpha length != x.cols");

    // Reciprocal hoisted per channel, and the epsilon added before inverting
    // rather than used as a guard afterwards -- trained alphas come close
    // enough to zero that the two differ.
    std::vector<float> recip(x.cols);
    for (std::size_t c = 0; c < x.cols; ++c) {
        recip[c] = 1.0f / (alpha[c] + 1e-9f);
    }

    for (std::size_t r = 0; r < x.rows; ++r) {
        float* row = x.row_mut(r).data();
        for (std::size_t c = 0; c < x.cols; ++c) {
            const float s = std::sin(alpha[c] * row[c]);
            row[c] += s * s * recip[c];
        }
    }
}

Mat snake1d(const Mat& x, std::span<const float> alpha) {
    Mat out = x;
    snake1d_inplace(out, alpha);
    return out;
}

// =============================================================================
// Weight normalization
// =============================================================================

std::vector<float> weight_norm_combine(std::span<const float> g, std::span<const float> v) {
    assert(!g.empty() && "weight_norm_combine: g is empty");
    assert(!v.empty() && "weight_norm_combine: v is empty");
    assert(v.size() % g.size() == 0 && "weight_norm_combine: v length not divisible by g length");

    // Axis 0 of the *stored* tensor, whatever that axis means for this layer:
    // output channels for Conv1d, input channels for ConvTranspose1d.
    const std::size_t groups = g.size();
    const std::size_t inner = v.size() / groups;

    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < groups; ++i) {
        const float* vr = v.data() + i * inner;
        float sq = 0.0f;
        for (std::size_t j = 0; j < inner; ++j) {
            sq += vr[j] * vr[j];
        }
        const float len = std::sqrt(sq);
        const float scale = len > 0.0f ? g[i] / len : 0.0f;
        float* orow = out.data() + i * inner;
        for (std::size_t j = 0; j < inner; ++j) {
            orow[j] = vr[j] * scale;
        }
    }
    return out;
}

}  // namespace rt

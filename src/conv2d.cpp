#include "rt/conv2d.hpp"

#include <algorithm>
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
    assert(bias.size() == out.cols && "conv2d: bias length != output channels");
    for (std::size_t r = 0; r < out.rows; ++r) {
        float* row = out.row_mut(r).data();
        for (std::size_t c = 0; c < out.cols; ++c) {
            row[c] += bias[c];
        }
    }
}

}  // namespace

// =============================================================================
// Output sizes
// =============================================================================

std::size_t conv2d_out_size(std::size_t in, std::size_t kernel, std::size_t dilation,
                            std::size_t padding, std::size_t stride) {
    assert(kernel >= 1 && "conv2d_out_size: kernel must be >= 1");
    assert(dilation >= 1 && "conv2d_out_size: dilation must be >= 1");
    assert(stride >= 1 && "conv2d_out_size: stride must be >= 1");
    const std::size_t reach = dilation * (kernel - 1);
    const std::size_t padded = in + 2 * padding;
    return padded > reach ? (padded - reach - 1) / stride + 1 : 0;
}

// =============================================================================
// Convolutions
// =============================================================================

Mat conv2d_pointwise(const Mat& x, const Mat& weight, std::span<const float> bias) {
    assert(x.cols == weight.cols && "conv2d_pointwise: x.cols != weight.cols (Cin mismatch)");
    Mat out = x.matmul_bt(weight);
    add_bias_rows(out, bias);
    return out;
}

Mat conv2d_dense(const Mat& x, std::size_t h, std::size_t w, const Mat& weight,
                 std::size_t out_channels, std::size_t kernel_h, std::size_t kernel_w,
                 std::span<const float> bias, std::size_t stride, std::size_t padding,
                 std::size_t dilation) {
    assert(x.rows == h * w && "conv2d_dense: x.rows != h * w");
    assert(weight.rows == out_channels && "conv2d_dense: weight.rows != out_channels");
    assert(weight.cols == x.cols * kernel_h * kernel_w &&
           "conv2d_dense: weight.cols != Cin * KH * KW");

    if (kernel_h == 1 && kernel_w == 1 && dilation == 1 && padding == 0 && stride == 1) {
        return conv2d_pointwise(x, weight, bias);
    }

    const std::size_t c_in = x.cols;
    const std::size_t out_h = conv2d_out_size(h, kernel_h, dilation, padding, stride);
    const std::size_t out_w = conv2d_out_size(w, kernel_w, dilation, padding, stride);
    const std::size_t n_out = out_h * out_w;

    // The "same" case a decoder lives in. Getting this wrong crops the image by
    // a couple of pixels per layer, which is invisible in isolation.
    assert((!(stride == 1 && dilation == 1 && kernel_h == 2 * padding + 1 &&
              kernel_w == 2 * padding + 1) ||
            (out_h == h && out_w == w)) &&
           "conv2d_dense: odd kernel with matching padding must preserve size");

    if (n_out == 0) {
        return Mat::zeros(0, out_channels);
    }

    const std::size_t patch = c_in * kernel_h * kernel_w;

    // im2col + gemm once the problem is big enough to pay for the scratch
    // buffer, on the same threshold as the 1-D version.
    constexpr std::size_t kGemmThreshold = 1 << 16;  // output elements x taps
    if (n_out * out_channels * patch >= kGemmThreshold) {
        Mat out = Mat::zeros(n_out, out_channels);

        // Tile over output pixels so the patch matrix stays a few megabytes
        // rather than `n_out * patch` floats -- gigabytes at decoder shapes.
        const std::size_t tile =
            std::max<std::size_t>(1, (1u << 21) / std::max<std::size_t>(patch, 1));
        Mat cols = Mat::zeros(std::min(tile, n_out), patch);

        for (std::size_t base = 0; base < n_out; base += tile) {
            const std::size_t rows = std::min(tile, n_out - base);
            if (cols.rows != rows) {
                cols = Mat::zeros(rows, patch);
            } else {
                std::fill(cols.data.begin(), cols.data.end(), 0.0f);
            }
            for (std::size_t r = 0; r < rows; ++r) {
                const std::size_t p = base + r;
                const std::size_t oy = p / out_w;
                const std::size_t ox = p % out_w;
                float* crow = cols.row_mut(r).data();
                for (std::size_t ky = 0; ky < kernel_h; ++ky) {
                    const std::size_t shifted_y = oy * stride + ky * dilation;
                    if (shifted_y < padding) {
                        continue;  // top zero pad
                    }
                    const std::size_t sy = shifted_y - padding;
                    if (sy >= h) {
                        continue;  // bottom zero pad
                    }
                    for (std::size_t kx = 0; kx < kernel_w; ++kx) {
                        const std::size_t shifted_x = ox * stride + kx * dilation;
                        if (shifted_x < padding) {
                            continue;  // left zero pad
                        }
                        const std::size_t sx = shifted_x - padding;
                        if (sx >= w) {
                            continue;  // right zero pad
                        }
                        const float* xrow = x.row(sy * w + sx).data();
                        const std::size_t tap = ky * kernel_w + kx;
                        // Column `ci * KH * KW + ky * KW + kx` mirrors the
                        // weight's layout, so the gemm repacks neither operand.
                        for (std::size_t ci = 0; ci < c_in; ++ci) {
                            crow[ci * kernel_h * kernel_w + tap] = xrow[ci];
                        }
                    }
                }
            }
            const Mat block = cols.matmul_bt(weight);
            for (std::size_t r = 0; r < rows; ++r) {
                std::copy(block.row(r).begin(), block.row(r).end(),
                          out.row_mut(base + r).begin());
            }
        }
        add_bias_rows(out, bias);
        return out;
    }

    Mat out = Mat::zeros(n_out, out_channels);
    for (std::size_t oy = 0; oy < out_h; ++oy) {
        for (std::size_t ox = 0; ox < out_w; ++ox) {
            float* orow = out.row_mut(oy * out_w + ox).data();
            for (std::size_t ky = 0; ky < kernel_h; ++ky) {
                const std::size_t shifted_y = oy * stride + ky * dilation;
                if (shifted_y < padding) {
                    continue;
                }
                const std::size_t sy = shifted_y - padding;
                if (sy >= h) {
                    continue;
                }
                for (std::size_t kx = 0; kx < kernel_w; ++kx) {
                    const std::size_t shifted_x = ox * stride + kx * dilation;
                    if (shifted_x < padding) {
                        continue;
                    }
                    const std::size_t sx = shifted_x - padding;
                    if (sx >= w) {
                        continue;
                    }
                    const float* xrow = x.row(sy * w + sx).data();
                    const std::size_t tap = ky * kernel_w + kx;
                    for (std::size_t co = 0; co < out_channels; ++co) {
                        const float* wrow = weight.row(co).data();
                        float acc = 0.0f;
                        for (std::size_t ci = 0; ci < c_in; ++ci) {
                            acc += xrow[ci] * wrow[ci * kernel_h * kernel_w + tap];
                        }
                        orow[co] += acc;
                    }
                }
            }
        }
    }

    add_bias_rows(out, bias);
    return out;
}

// =============================================================================
// Resampling
// =============================================================================

Mat upsample_nearest2d(const Mat& x, std::size_t h, std::size_t w, std::size_t factor) {
    assert(x.rows == h * w && "upsample_nearest2d: x.rows != h * w");
    assert(factor >= 1 && "upsample_nearest2d: factor must be >= 1");
    if (factor == 1) {
        return x;
    }

    const std::size_t out_h = h * factor;
    const std::size_t out_w = w * factor;
    const std::size_t channels = x.cols;
    Mat out = Mat::zeros(out_h * out_w, channels);

    for (std::size_t oy = 0; oy < out_h; ++oy) {
        const std::size_t sy = oy / factor;
        for (std::size_t ox = 0; ox < out_w; ++ox) {
            const std::size_t sx = ox / factor;
            const float* src = x.row(sy * w + sx).data();
            float* dst = out.row_mut(oy * out_w + ox).data();
            std::copy(src, src + channels, dst);
        }
    }
    return out;
}

// =============================================================================
// Normalization and activations
// =============================================================================

void group_norm_inplace(Mat& x, std::size_t groups, std::span<const float> weight,
                        std::span<const float> bias, float eps) {
    const std::size_t channels = x.cols;
    assert(groups >= 1 && "group_norm: groups must be >= 1");
    assert(channels % groups == 0 && "group_norm: channels not divisible by groups");
    assert((weight.empty() || weight.size() == channels) && "group_norm: weight length != C");
    assert((bias.empty() || bias.size() == channels) && "group_norm: bias length != C");
    if (x.rows == 0) {
        return;
    }

    const std::size_t per_group = channels / groups;
    const std::size_t n = x.rows * per_group;
    const auto count = static_cast<double>(n);

    for (std::size_t g = 0; g < groups; ++g) {
        const std::size_t lo = g * per_group;

        // Two passes rather than sum-of-squares. A decoder's last level reduces
        // over sixteen million values, and the one-pass form loses the variance
        // in the cancellation when the mean is far from zero.
        double sum = 0.0;
        for (std::size_t r = 0; r < x.rows; ++r) {
            const float* row = x.row(r).data();
            for (std::size_t c = 0; c < per_group; ++c) {
                sum += static_cast<double>(row[lo + c]);
            }
        }
        const double mean = sum / count;

        double sq = 0.0;
        for (std::size_t r = 0; r < x.rows; ++r) {
            const float* row = x.row(r).data();
            for (std::size_t c = 0; c < per_group; ++c) {
                const double d = static_cast<double>(row[lo + c]) - mean;
                sq += d * d;
            }
        }
        // Biased variance, as PyTorch uses for normalization layers.
        const auto inv_std = static_cast<float>(1.0 / std::sqrt(sq / count + eps));
        const auto mean_f = static_cast<float>(mean);

        for (std::size_t r = 0; r < x.rows; ++r) {
            float* row = x.row_mut(r).data();
            for (std::size_t c = 0; c < per_group; ++c) {
                const std::size_t ch = lo + c;
                float v = (row[ch] - mean_f) * inv_std;
                if (!weight.empty()) {
                    v *= weight[ch];
                }
                if (!bias.empty()) {
                    v += bias[ch];
                }
                row[ch] = v;
            }
        }
    }
}

Mat group_norm(const Mat& x, std::size_t groups, std::span<const float> weight,
               std::span<const float> bias, float eps) {
    Mat out = x;
    group_norm_inplace(out, groups, weight, bias, eps);
    return out;
}

void silu_inplace(Mat& x) {
    for (float& v : x.data) {
        v = v / (1.0f + std::exp(-v));
    }
}

void quick_gelu_inplace(Mat& x) {
    for (float& v : x.data) {
        v = v / (1.0f + std::exp(-1.702f * v));
    }
}

void gelu_tanh_inplace(Mat& x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    for (float& v : x.data) {
        const float c = v * v * v;
        v = 0.5f * v * (1.0f + std::tanh(kSqrt2OverPi * (v + 0.044715f * c)));
    }
}

}  // namespace rt

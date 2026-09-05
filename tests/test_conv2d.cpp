#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "rt/conv2d.hpp"
#include "rt/mat.hpp"
#include "test_helpers.hpp"

using namespace rt;

namespace {

/// Absolute comparison with an explicit tolerance. `test_helpers.hpp` fixes
/// its tolerance at 1e-3, which is far too loose for checking an activation
/// against its own definition.
[[nodiscard]] bool approx(float a, float b, float tol) {
    return std::fabs(a - b) < tol;
}

/// Relative comparison, for the same reason `test_conv1d.cpp` needs one:
/// im2col + gemm and the direct loop sum the same products in different orders.
[[nodiscard]] bool approx_rel(float a, float b, float tol = 1e-5f) {
    const float scale = std::max({1.0f, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) / scale < tol;
}

[[nodiscard]] bool mats_close(const Mat& a, const Mat& b, float tol = 1e-5f) {
    if (a.rows != b.rows || a.cols != b.cols) {
        return false;
    }
    for (std::size_t i = 0; i < a.data.size(); ++i) {
        if (!approx_rel(a.data[i], b.data[i], tol)) {
            return false;
        }
    }
    return true;
}

/// Fill a matrix with a deterministic spread of small signed values.
[[nodiscard]] Mat ramp(std::size_t rows, std::size_t cols, float scale = 0.01f,
                       float shift = 0.5f) {
    return Mat::from_fn(rows, cols, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * cols + c) * scale - shift;
    });
}

/// A spread that is not monotone, so a transposed spatial index cannot pass by
/// accident on a smoothly varying input.
[[nodiscard]] Mat wobble(std::size_t rows, std::size_t cols) {
    return Mat::from_fn(rows, cols, [](std::size_t r, std::size_t c) {
        const auto i = static_cast<float>(r * 31 + c * 17);
        return std::sin(i * 0.37f) * 0.8f + std::cos(i * 0.11f) * 0.3f;
    });
}

/// Reference dense 2-D convolution, written out index by index.
[[nodiscard]] Mat ref_conv2d(const Mat& x, std::size_t h, std::size_t w, const Mat& weight,
                             std::size_t c_out, std::size_t kh, std::size_t kw,
                             std::size_t stride, std::size_t padding, std::size_t dilation) {
    const std::size_t c_in = x.cols;
    const std::size_t out_h = conv2d_out_size(h, kh, dilation, padding, stride);
    const std::size_t out_w = conv2d_out_size(w, kw, dilation, padding, stride);
    Mat out = Mat::zeros(out_h * out_w, c_out);
    for (std::size_t oy = 0; oy < out_h; ++oy) {
        for (std::size_t ox = 0; ox < out_w; ++ox) {
            for (std::size_t co = 0; co < c_out; ++co) {
                float acc = 0.0f;
                for (std::size_t ci = 0; ci < c_in; ++ci) {
                    for (std::size_t ky = 0; ky < kh; ++ky) {
                        for (std::size_t kx = 0; kx < kw; ++kx) {
                            const long sy = static_cast<long>(oy * stride + ky * dilation) -
                                            static_cast<long>(padding);
                            const long sx = static_cast<long>(ox * stride + kx * dilation) -
                                            static_cast<long>(padding);
                            if (sy < 0 || sy >= static_cast<long>(h) || sx < 0 ||
                                sx >= static_cast<long>(w)) {
                                continue;
                            }
                            const std::size_t p =
                                static_cast<std::size_t>(sy) * w + static_cast<std::size_t>(sx);
                            acc += x.at(p, ci) * weight.at(co, ci * kh * kw + ky * kw + kx);
                        }
                    }
                }
                out.at_mut(oy * out_w + ox, co) = acc;
            }
        }
    }
    return out;
}

/// Reference GroupNorm, reducing over channels and space in two explicit passes.
[[nodiscard]] Mat ref_group_norm(const Mat& x, std::size_t groups, std::span<const float> weight,
                                 std::span<const float> bias, float eps) {
    const std::size_t c = x.cols;
    const std::size_t per = c / groups;
    Mat out = x;
    for (std::size_t g = 0; g < groups; ++g) {
        double sum = 0.0;
        std::size_t n = 0;
        for (std::size_t r = 0; r < x.rows; ++r) {
            for (std::size_t k = 0; k < per; ++k) {
                sum += x.at(r, g * per + k);
                ++n;
            }
        }
        const double mean = sum / static_cast<double>(n);
        double var = 0.0;
        for (std::size_t r = 0; r < x.rows; ++r) {
            for (std::size_t k = 0; k < per; ++k) {
                const double d = x.at(r, g * per + k) - mean;
                var += d * d;
            }
        }
        var /= static_cast<double>(n);
        for (std::size_t r = 0; r < x.rows; ++r) {
            for (std::size_t k = 0; k < per; ++k) {
                const std::size_t ch = g * per + k;
                double v = (x.at(r, ch) - mean) / std::sqrt(var + eps);
                if (!weight.empty()) {
                    v *= weight[ch];
                }
                if (!bias.empty()) {
                    v += bias[ch];
                }
                out.at_mut(r, ch) = static_cast<float>(v);
            }
        }
    }
    return out;
}

}  // namespace

// =============================================================================
// Output sizes
// =============================================================================

TEST_CASE("conv2d_out_size matches PyTorch's formula", "[conv2d]") {
    // "same" padding for odd kernels
    REQUIRE(conv2d_out_size(32, 3, 1, 1, 1) == 32);
    REQUIRE(conv2d_out_size(32, 5, 1, 2, 1) == 32);
    REQUIRE(conv2d_out_size(32, 1, 1, 0, 1) == 32);
    // valid padding shrinks by the reach
    REQUIRE(conv2d_out_size(32, 3, 1, 0, 1) == 30);
    // stride halves, rounding up
    REQUIRE(conv2d_out_size(32, 3, 1, 1, 2) == 16);
    REQUIRE(conv2d_out_size(33, 3, 1, 1, 2) == 17);
    // dilation extends the reach
    REQUIRE(conv2d_out_size(32, 3, 2, 0, 1) == 28);
    // a kernel that does not fit reports zero rather than wrapping
    REQUIRE(conv2d_out_size(2, 5, 1, 0, 1) == 0);
}

// =============================================================================
// Pointwise
// =============================================================================

TEST_CASE("conv2d_pointwise is exactly a matmul against the transposed weight", "[conv2d]") {
    const Mat x = ramp(6 * 5, 8);
    const Mat w = ramp(4, 8, 0.03f, 0.2f);
    const Mat expected = x.matmul_bt(w);
    const Mat got = conv2d_pointwise(x, w, {});
    REQUIRE(mats_close(got, expected, 1e-6f));
}

TEST_CASE("a 1x1 conv2d_dense takes the pointwise path and agrees with it", "[conv2d]") {
    const Mat x = wobble(7 * 9, 6);
    const Mat w = ramp(5, 6, 0.05f, 0.3f);
    const std::vector<float> bias{0.1f, -0.2f, 0.3f, -0.4f, 0.5f};
    const Mat got = conv2d_dense(x, 7, 9, w, 5, 1, 1, bias, 1, 0, 1);
    const Mat expected = conv2d_pointwise(x, w, bias);
    REQUIRE(got.rows == 63);
    REQUIRE(got.cols == 5);
    REQUIRE(mats_close(got, expected, 1e-6f));
}

// =============================================================================
// Dense convolution, against the index-by-index reference
// =============================================================================

TEST_CASE("3x3 padding 1 preserves the spatial extent", "[conv2d]") {
    const std::size_t h = 9;
    const std::size_t w = 7;
    const Mat x = wobble(h * w, 4);
    const Mat weight = ramp(6, 4 * 3 * 3, 0.004f, 0.07f);
    const Mat got = conv2d_dense(x, h, w, weight, 6, 3, 3, {}, 1, 1, 1);
    REQUIRE(got.rows == h * w);
    REQUIRE(got.cols == 6);
    REQUIRE(mats_close(got, ref_conv2d(x, h, w, weight, 6, 3, 3, 1, 1, 1)));
}

TEST_CASE("both conv2d_dense paths agree with the reference and each other", "[conv2d]") {
    // Small enough for the direct loop, then the same convolution at a size
    // that crosses the im2col threshold. Both must match the reference.
    struct Case {
        std::size_t h, w, c_in, c_out;
    };
    const Case small{5, 4, 3, 2};
    const Case large{28, 24, 16, 24};

    for (const Case& c : {small, large}) {
        const Mat x = wobble(c.h * c.w, c.c_in);
        const Mat weight = ramp(c.c_out, c.c_in * 3 * 3, 0.002f, 0.05f);
        const Mat got = conv2d_dense(x, c.h, c.w, weight, c.c_out, 3, 3, {}, 1, 1, 1);
        const Mat expected = ref_conv2d(x, c.h, c.w, weight, c.c_out, 3, 3, 1, 1, 1);
        REQUIRE(mats_close(got, expected));
    }
}

TEST_CASE("asymmetric height and width do not transpose the spatial index", "[conv2d]") {
    // h != w and a non-symmetric input: any swap of the two axes changes the
    // answer, which a square test would silently accept.
    const std::size_t h = 11;
    const std::size_t w = 4;
    const Mat x = wobble(h * w, 3);
    const Mat weight = ramp(2, 3 * 3 * 3, 0.01f, 0.15f);
    const Mat got = conv2d_dense(x, h, w, weight, 2, 3, 3, {}, 1, 1, 1);
    const Mat expected = ref_conv2d(x, h, w, weight, 2, 3, 3, 1, 1, 1);
    REQUIRE(got.rows == h * w);
    REQUIRE(mats_close(got, expected));

    // The transposed problem is genuinely different.
    const Mat swapped = conv2d_dense(x, w, h, weight, 2, 3, 3, {}, 1, 1, 1);
    REQUIRE_FALSE(mats_close(got, swapped));
}

TEST_CASE("a non-square kernel keeps its axes straight", "[conv2d]") {
    const std::size_t h = 8;
    const std::size_t w = 9;
    const Mat x = wobble(h * w, 3);
    const Mat weight = ramp(4, 3 * 1 * 5, 0.01f, 0.2f);
    const Mat got = conv2d_dense(x, h, w, weight, 4, 1, 5, {}, 1, 0, 1);
    REQUIRE(got.rows == conv2d_out_size(h, 1, 1, 0, 1) * conv2d_out_size(w, 5, 1, 0, 1));
    REQUIRE(mats_close(got, ref_conv2d(x, h, w, weight, 4, 1, 5, 1, 0, 1)));
}

TEST_CASE("stride 2 halves both axes", "[conv2d]") {
    const std::size_t h = 16;
    const std::size_t w = 12;
    const Mat x = wobble(h * w, 8);
    const Mat weight = ramp(8, 8 * 3 * 3, 0.003f, 0.05f);
    const Mat got = conv2d_dense(x, h, w, weight, 8, 3, 3, {}, 2, 1, 1);
    REQUIRE(got.rows == 8 * 6);
    REQUIRE(mats_close(got, ref_conv2d(x, h, w, weight, 8, 3, 3, 2, 1, 1)));
}

TEST_CASE("dilation spreads the taps without changing the output layout", "[conv2d]") {
    const std::size_t h = 12;
    const std::size_t w = 10;
    const Mat x = wobble(h * w, 4);
    const Mat weight = ramp(4, 4 * 3 * 3, 0.006f, 0.1f);
    const Mat got = conv2d_dense(x, h, w, weight, 4, 3, 3, {}, 1, 2, 2);
    REQUIRE(got.rows == h * w);  // dilation 2, padding 2, kernel 3 is also "same"
    REQUIRE(mats_close(got, ref_conv2d(x, h, w, weight, 4, 3, 3, 1, 2, 2)));
}

TEST_CASE("padding wider than the input still only reads real pixels", "[conv2d]") {
    const std::size_t h = 2;
    const std::size_t w = 3;
    const Mat x = wobble(h * w, 2);
    const Mat weight = ramp(3, 2 * 5 * 5, 0.02f, 0.25f);
    const Mat got = conv2d_dense(x, h, w, weight, 3, 5, 5, {}, 1, 4, 1);
    REQUIRE(got.rows == conv2d_out_size(h, 5, 1, 4, 1) * conv2d_out_size(w, 5, 1, 4, 1));
    REQUIRE(mats_close(got, ref_conv2d(x, h, w, weight, 3, 5, 5, 1, 4, 1)));
}

TEST_CASE("bias is added once per output channel", "[conv2d]") {
    const std::size_t h = 6;
    const std::size_t w = 6;
    const Mat x = wobble(h * w, 3);
    const Mat weight = ramp(4, 3 * 3 * 3, 0.01f, 0.1f);
    const std::vector<float> bias{1.0f, -2.0f, 0.5f, 3.0f};
    const Mat plain = conv2d_dense(x, h, w, weight, 4, 3, 3, {}, 1, 1, 1);
    const Mat biased = conv2d_dense(x, h, w, weight, 4, 3, 3, bias, 1, 1, 1);
    for (std::size_t r = 0; r < plain.rows; ++r) {
        for (std::size_t c = 0; c < plain.cols; ++c) {
            REQUIRE(approx(biased.at(r, c), plain.at(r, c) + bias[c], 1e-5f));
        }
    }
}

TEST_CASE("a delta kernel reproduces the input, shifted", "[conv2d]") {
    // One input channel, one output channel, a 3x3 kernel that is zero except
    // for the top-left tap. The result is the input translated by one pixel in
    // each axis -- which pins the sign of the padding offset.
    const std::size_t h = 5;
    const std::size_t w = 4;
    const Mat x = wobble(h * w, 1);
    Mat weight = Mat::zeros(1, 9);
    weight.at_mut(0, 0) = 1.0f;  // ky = 0, kx = 0
    const Mat got = conv2d_dense(x, h, w, weight, 1, 3, 3, {}, 1, 1, 1);
    REQUIRE(got.rows == h * w);
    for (std::size_t oy = 0; oy < h; ++oy) {
        for (std::size_t ox = 0; ox < w; ++ox) {
            // out[oy][ox] reads x[oy - 1][ox - 1]
            const float expected =
                (oy == 0 || ox == 0) ? 0.0f : x.at((oy - 1) * w + (ox - 1), 0);
            REQUIRE(approx(got.at(oy * w + ox, 0), expected, 1e-6f));
        }
    }
}

// =============================================================================
// Upsampling
// =============================================================================

TEST_CASE("upsample_nearest2d repeats each pixel into a block", "[conv2d]") {
    const std::size_t h = 3;
    const std::size_t w = 4;
    const Mat x = wobble(h * w, 5);
    const Mat up = upsample_nearest2d(x, h, w, 2);
    REQUIRE(up.rows == (h * 2) * (w * 2));
    REQUIRE(up.cols == 5);
    for (std::size_t oy = 0; oy < h * 2; ++oy) {
        for (std::size_t ox = 0; ox < w * 2; ++ox) {
            for (std::size_t c = 0; c < 5; ++c) {
                REQUIRE(approx(up.at(oy * (w * 2) + ox, c), x.at((oy / 2) * w + (ox / 2), c),
                               1e-7f));
            }
        }
    }
}

TEST_CASE("upsample_nearest2d by 1 is the identity", "[conv2d]") {
    const Mat x = wobble(4 * 4, 3);
    REQUIRE(mats_close(upsample_nearest2d(x, 4, 4, 1), x, 1e-7f));
}

// =============================================================================
// GroupNorm
// =============================================================================

TEST_CASE("group_norm agrees with a two-pass reference", "[conv2d][groupnorm]") {
    const Mat x = wobble(8 * 6, 16);
    std::vector<float> weight(16);
    std::vector<float> bias(16);
    for (std::size_t i = 0; i < 16; ++i) {
        weight[i] = 0.5f + 0.1f * static_cast<float>(i);
        bias[i] = -0.2f + 0.05f * static_cast<float>(i);
    }
    const Mat got = group_norm(x, 4, weight, bias, 1e-6f);
    const Mat expected = ref_group_norm(x, 4, weight, bias, 1e-6f);
    REQUIRE(mats_close(got, expected, 1e-5f));
}

TEST_CASE("group_norm statistics span space, not just channels", "[conv2d][groupnorm]") {
    // Every row is constant across its channels but the rows differ. LayerNorm
    // maps this to all zeros. GroupNorm must not: the variance it sees is the
    // variance *between* rows, which is nonzero.
    Mat x = Mat::zeros(6, 4);
    for (std::size_t r = 0; r < x.rows; ++r) {
        for (std::size_t c = 0; c < x.cols; ++c) {
            x.at_mut(r, c) = static_cast<float>(r) - 2.5f;
        }
    }
    const Mat got = group_norm(x, 1, {}, {}, 1e-6f);

    bool all_zero = true;
    for (const float v : got.data) {
        if (std::fabs(v) > 1e-3f) {
            all_zero = false;
        }
    }
    REQUIRE_FALSE(all_zero);
    REQUIRE(mats_close(got, ref_group_norm(x, 1, {}, {}, 1e-6f), 1e-5f));
}

TEST_CASE("group_norm with one group normalizes everything jointly", "[conv2d][groupnorm]") {
    const Mat x = wobble(5 * 5, 8);
    const Mat got = group_norm(x, 1, {}, {}, 1e-6f);
    double sum = 0.0;
    double sq = 0.0;
    for (const float v : got.data) {
        sum += v;
        sq += static_cast<double>(v) * v;
    }
    const auto n = static_cast<double>(got.data.size());
    REQUIRE(std::fabs(sum / n) < 1e-4);
    REQUIRE(std::fabs(sq / n - 1.0) < 1e-3);
}

TEST_CASE("group_norm with one group per channel normalizes each channel over space",
          "[conv2d][groupnorm]") {
    const std::size_t channels = 6;
    const Mat x = wobble(7 * 3, channels);
    const Mat got = group_norm(x, channels, {}, {}, 1e-6f);
    for (std::size_t c = 0; c < channels; ++c) {
        double sum = 0.0;
        double sq = 0.0;
        for (std::size_t r = 0; r < got.rows; ++r) {
            sum += got.at(r, c);
            sq += static_cast<double>(got.at(r, c)) * got.at(r, c);
        }
        const auto n = static_cast<double>(got.rows);
        REQUIRE(std::fabs(sum / n) < 1e-4);
        REQUIRE(std::fabs(sq / n - 1.0) < 1e-3);
    }
}

TEST_CASE("group_norm groups do not leak into each other", "[conv2d][groupnorm]") {
    // Scaling one group must leave the other's output untouched.
    Mat x = wobble(4 * 4, 8);
    const Mat base = group_norm(x, 2, {}, {}, 1e-6f);
    for (std::size_t r = 0; r < x.rows; ++r) {
        for (std::size_t c = 4; c < 8; ++c) {
            x.at_mut(r, c) *= 10.0f;
        }
    }
    const Mat perturbed = group_norm(x, 2, {}, {}, 1e-6f);
    for (std::size_t r = 0; r < x.rows; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            REQUIRE(approx(base.at(r, c), perturbed.at(r, c), 1e-5f));
        }
    }
}

TEST_CASE("group_norm affine parameters are per channel", "[conv2d][groupnorm]") {
    const Mat x = wobble(4 * 4, 4);
    const std::vector<float> weight{2.0f, 2.0f, 2.0f, 2.0f};
    const std::vector<float> bias{1.0f, 1.0f, 1.0f, 1.0f};
    const Mat plain = group_norm(x, 2, {}, {}, 1e-6f);
    const Mat affine = group_norm(x, 2, weight, bias, 1e-6f);
    for (std::size_t i = 0; i < plain.data.size(); ++i) {
        REQUIRE(approx(affine.data[i], plain.data[i] * 2.0f + 1.0f, 1e-5f));
    }
}

TEST_CASE("group_norm eps is not negligible at the default", "[conv2d][groupnorm]") {
    // 1e-6 against 1e-5 is a visible difference on a low-variance group, which
    // is why the default is spelled out rather than inherited.
    Mat x = Mat::zeros(4, 4);
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            x.at_mut(r, c) = 1e-3f * static_cast<float>(r * 4 + c);
        }
    }
    const Mat a = group_norm(x, 1, {}, {}, 1e-6f);
    const Mat b = group_norm(x, 1, {}, {}, 1e-5f);
    REQUIRE_FALSE(mats_close(a, b, 1e-4f));
}

// =============================================================================
// Activations
// =============================================================================

TEST_CASE("silu, quick_gelu and gelu_tanh match their definitions", "[conv2d][activation]") {
    const std::vector<float> xs{-3.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 3.0f};
    Mat m(std::vector<float>(xs), 1, xs.size());

    Mat silu = m;
    silu_inplace(silu);
    Mat qg = m;
    quick_gelu_inplace(qg);
    Mat gt = m;
    gelu_tanh_inplace(gt);

    for (std::size_t i = 0; i < xs.size(); ++i) {
        const float x = xs[i];
        REQUIRE(approx(silu.data[i], x / (1.0f + std::exp(-x)), 1e-6f));
        REQUIRE(approx(qg.data[i], x / (1.0f + std::exp(-1.702f * x)), 1e-6f));
        const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        REQUIRE(approx(gt.data[i], 0.5f * x * (1.0f + std::tanh(inner)), 1e-6f));
    }
}

TEST_CASE("quick_gelu is not interchangeable with gelu_tanh", "[conv2d][activation]") {
    // They differ by up to a few percent in the middle of the range, which is
    // enough to move a CLIP text embedding.
    Mat a(std::vector<float>{1.0f}, 1, 1);
    Mat b = a;
    quick_gelu_inplace(a);
    gelu_tanh_inplace(b);
    REQUIRE(std::fabs(a.data[0] - b.data[0]) > 1e-3f);
}

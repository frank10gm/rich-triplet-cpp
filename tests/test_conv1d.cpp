#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include "rt/conv1d.hpp"
#include "rt/init_rng.hpp"
#include "rt/mat.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

namespace {

/// Fill a matrix with a deterministic spread of small signed values.
[[nodiscard]] Mat ramp(std::size_t rows, std::size_t cols, float scale = 0.01f, float shift = 0.5f) {
    return Mat::from_fn(rows, cols, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * cols + c) * scale - shift;
    });
}

/// Reference dense convolution, written out index by index.
[[nodiscard]] Mat ref_conv1d_dense(const Mat& x, const Mat& w, std::size_t c_out,
                                   std::size_t kernel, std::size_t dilation,
                                   std::size_t padding) {
    const std::size_t c_in = x.cols;
    const std::size_t t_out = x.rows + 2 * padding - dilation * (kernel - 1);
    Mat out = Mat::zeros(t_out, c_out);
    for (std::size_t t = 0; t < t_out; ++t) {
        for (std::size_t co = 0; co < c_out; ++co) {
            float acc = 0.0f;
            for (std::size_t ci = 0; ci < c_in; ++ci) {
                for (std::size_t k = 0; k < kernel; ++k) {
                    const long src = static_cast<long>(t + k * dilation) - static_cast<long>(padding);
                    if (src < 0 || src >= static_cast<long>(x.rows)) {
                        continue;
                    }
                    acc += x.at(static_cast<std::size_t>(src), ci) * w.at(co, ci * kernel + k);
                }
            }
            out.at_mut(t, co) = acc;
        }
    }
    return out;
}

/// Reference transposed convolution in scatter form, written out index by index.
[[nodiscard]] Mat ref_conv_transpose1d(const Mat& x, const Mat& w, std::size_t c_out,
                                       std::size_t kernel, std::size_t stride,
                                       std::size_t padding, std::size_t output_padding) {
    const std::size_t c_in = x.cols;
    const std::size_t t_out = (x.rows - 1) * stride + kernel + output_padding - 2 * padding;
    Mat out = Mat::zeros(t_out, c_out);
    for (std::size_t t = 0; t < x.rows; ++t) {
        for (std::size_t k = 0; k < kernel; ++k) {
            const long dst = static_cast<long>(t * stride + k) - static_cast<long>(padding);
            if (dst < 0 || dst >= static_cast<long>(t_out)) {
                continue;
            }
            for (std::size_t co = 0; co < c_out; ++co) {
                float acc = 0.0f;
                for (std::size_t ci = 0; ci < c_in; ++ci) {
                    acc += x.at(t, ci) * w.at(ci, co * kernel + k);
                }
                out.at_mut(static_cast<std::size_t>(dst), co) += acc;
            }
        }
    }
    return out;
}

}  // namespace

// =============================================================================
// Output lengths
// =============================================================================

TEST_CASE("conv1d_out_len matches PyTorch's stride-1 formula", "[conv1d]") {
    REQUIRE(conv1d_out_len(10, 1, 1, 0) == 10);
    REQUIRE(conv1d_out_len(10, 7, 1, 0) == 4);
    REQUIRE(conv1d_out_len(10, 7, 1, 3) == 10);  // pad = (k-1)/2 preserves length
    REQUIRE(conv1d_out_len(10, 3, 2, 2) == 10);  // pad = d*(k-1)/2 preserves length
}

TEST_CASE("conv1d_out_len preserves length for every SNAC residual dilation", "[conv1d]") {
    // ResidualUnit: kernel 7, padding = ((7-1)*dilation)/2 = 3*dilation.
    // Exact length preservation is what makes the reference's centre-crop
    // branch dead code, so the whole decoder can assume it.
    for (const std::size_t dilation : {std::size_t{1}, std::size_t{3}, std::size_t{9}}) {
        REQUIRE(conv1d_out_len(64, 7, dilation, 3 * dilation) == 64);
    }
}

TEST_CASE("conv1d_out_len reports 0 when the kernel does not fit", "[conv1d]") {
    REQUIRE(conv1d_out_len(4, 7, 1, 0) == 0);
    REQUIRE(conv1d_out_len(6, 7, 1, 0) == 0);  // padded == reach, not one more
}

TEST_CASE("conv_transpose1d_out_len collapses to T*stride for codec blocks", "[conv1d]") {
    // kernel = 2*stride, padding = ceil(stride/2), output_padding = stride % 2.
    for (const std::size_t stride : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                                     std::size_t{4}, std::size_t{8}}) {
        const std::size_t kernel = 2 * stride;
        const std::size_t padding = (stride + 1) / 2;
        const std::size_t output_padding = stride % 2;
        for (const std::size_t t : {std::size_t{1}, std::size_t{4}, std::size_t{37}}) {
            REQUIRE(conv_transpose1d_out_len(t, kernel, stride, padding, output_padding) ==
                    t * stride);
        }
    }
}

TEST_CASE("conv_transpose1d_out_len matches PyTorch outside the codec form", "[conv1d]") {
    REQUIRE(conv_transpose1d_out_len(4, 3, 1, 0, 0) == 6);   // (4-1)*1 + 3
    REQUIRE(conv_transpose1d_out_len(4, 4, 2, 0, 0) == 10);  // (4-1)*2 + 4
    REQUIRE(conv_transpose1d_out_len(0, 4, 2, 0, 0) == 0);
}

// =============================================================================
// Pointwise
// =============================================================================

TEST_CASE("conv1d_pointwise is a dense channel mix", "[conv1d]") {
    // x = [1 2], weight = [[1 0], [0 1], [1 1]] -> out = [1 2 3]
    const Mat x({1.0f, 2.0f}, 1, 2);
    const Mat w({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, 3, 2);
    const Mat out = conv1d_pointwise(x, w, {});
    REQUIRE(out.rows == 1);
    REQUIRE(out.cols == 3);
    REQUIRE(approx(out.at(0, 0), 1.0f));
    REQUIRE(approx(out.at(0, 1), 2.0f));
    REQUIRE(approx(out.at(0, 2), 3.0f));
}

TEST_CASE("conv1d_pointwise adds a per-output-channel bias", "[conv1d]") {
    const Mat x({1.0f, 2.0f}, 1, 2);
    const Mat w({1.0f, 0.0f, 0.0f, 1.0f}, 2, 2);
    const std::vector<float> bias{10.0f, -10.0f};
    const Mat out = conv1d_pointwise(x, w, bias);
    REQUIRE(approx(out.at(0, 0), 11.0f));
    REQUIRE(approx(out.at(0, 1), -8.0f));
}

TEST_CASE("conv1d_dense delegates to pointwise at kernel 1", "[conv1d]") {
    const Mat x = ramp(9, 5);
    const Mat w = ramp(7, 5, 0.03f, 0.2f);
    const std::vector<float> bias{0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f};
    const Mat dense = conv1d_dense(x, w, 7, 1, bias, 1, 0);
    const Mat point = conv1d_pointwise(x, w, bias);
    REQUIRE(dense.rows == point.rows);
    REQUIRE(dense.cols == point.cols);
    for (std::size_t i = 0; i < dense.data.size(); ++i) {
        REQUIRE(approx(dense.data[i], point.data[i]));
    }
}

// =============================================================================
// Depthwise
// =============================================================================

TEST_CASE("conv1d_depthwise does not mix channels", "[conv1d]") {
    // Channel 0 gets an identity tap, channel 1 a doubling tap. If the two
    // ever mixed, the outputs would not stay proportional to their own input.
    const Mat x({1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f}, 3, 2);
    const Mat w({1.0f, 2.0f}, 2, 1);  // [C=2, K=1]
    const Mat out = conv1d_depthwise(x, w, {}, 1, 0);
    REQUIRE(out.rows == 3);
    REQUIRE(out.cols == 2);
    for (std::size_t t = 0; t < 3; ++t) {
        REQUIRE(approx(out.at(t, 0), x.at(t, 0)));
        REQUIRE(approx(out.at(t, 1), x.at(t, 1) * 2.0f));
    }
}

TEST_CASE("conv1d_depthwise matches a per-channel dense convolution", "[conv1d]") {
    constexpr std::size_t channels = 6;
    constexpr std::size_t kernel = 7;
    for (const std::size_t dilation : {std::size_t{1}, std::size_t{3}, std::size_t{9}}) {
        const std::size_t padding = 3 * dilation;
        const Mat x = ramp(24, channels, 0.017f, 0.4f);
        const Mat w = ramp(channels, kernel, 0.031f, 0.3f);
        const Mat got = conv1d_depthwise(x, w, {}, dilation, padding);
        REQUIRE(got.rows == 24);
        REQUIRE(got.cols == channels);

        // Same thing as a dense convolution with a block-diagonal weight.
        Mat dense_w = Mat::zeros(channels, channels * kernel);
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t k = 0; k < kernel; ++k) {
                dense_w.at_mut(c, c * kernel + k) = w.at(c, k);
            }
        }
        const Mat want = ref_conv1d_dense(x, dense_w, channels, kernel, dilation, padding);
        for (std::size_t i = 0; i < got.data.size(); ++i) {
            REQUIRE(approx(got.data[i], want.data[i]));
        }
    }
}

// =============================================================================
// Dense
// =============================================================================

TEST_CASE("conv1d_dense matches the reference loop", "[conv1d]") {
    const Mat x = ramp(20, 4, 0.023f, 0.35f);
    const Mat w = ramp(3, 4 * 7, 0.011f, 0.25f);
    const std::vector<float> bias{0.5f, -0.25f, 0.125f};
    const Mat got = conv1d_dense(x, w, 3, 7, bias, 1, 3);
    Mat want = ref_conv1d_dense(x, w, 3, 7, 1, 3);
    for (std::size_t r = 0; r < want.rows; ++r) {
        for (std::size_t c = 0; c < want.cols; ++c) {
            want.at_mut(r, c) += bias[c];
        }
    }
    REQUIRE(got.rows == want.rows);
    REQUIRE(got.cols == want.cols);
    for (std::size_t i = 0; i < got.data.size(); ++i) {
        REQUIRE(approx(got.data[i], want.data[i]));
    }
}

TEST_CASE("conv1d_dense shapes match SNAC's final projection", "[conv1d]") {
    // decoder.model.7: WNConv1d(64 -> 1, kernel 7, padding 3).
    const Mat x = ramp(128, 64, 0.001f, 0.06f);
    const Mat w = ramp(1, 64 * 7, 0.0005f, 0.01f);
    const std::vector<float> bias{0.02f};
    const Mat out = conv1d_dense(x, w, 1, 7, bias, 1, 3);
    REQUIRE(out.rows == 128);
    REQUIRE(out.cols == 1);
}

// =============================================================================
// Transposed
// =============================================================================

TEST_CASE("conv_transpose1d lays an impulse response down unflipped", "[conv1d]") {
    // A single input frame of 1.0 must reproduce the kernel in forward order.
    // The gather formulation of the same operation would emit it reversed, so
    // this is the test that catches a flipped kernel.
    constexpr std::size_t kernel = 3;
    const Mat x({1.0f}, 1, 1);
    const Mat w({7.0f, 8.0f, 9.0f}, 1, kernel);  // [Cin=1, Cout*K = 1*3]
    const Mat out = conv_transpose1d(x, w, 1, kernel, {}, 1, 0, 0);
    REQUIRE(out.rows == 3);
    REQUIRE(out.cols == 1);
    REQUIRE(approx(out.at(0, 0), 7.0f));
    REQUIRE(approx(out.at(1, 0), 8.0f));
    REQUIRE(approx(out.at(2, 0), 9.0f));
}

TEST_CASE("conv_transpose1d overlaps strided impulses additively", "[conv1d]") {
    // Two frames, stride 2, kernel 4: frame 0 writes positions 0..3 and frame 1
    // writes 2..5, so 2 and 3 receive a contribution from both.
    constexpr std::size_t kernel = 4;
    const Mat x({1.0f, 1.0f}, 2, 1);
    const Mat w({1.0f, 2.0f, 4.0f, 8.0f}, 1, kernel);
    const Mat out = conv_transpose1d(x, w, 1, kernel, {}, 2, 0, 0);
    REQUIRE(out.rows == 6);
    REQUIRE(approx(out.at(0, 0), 1.0f));
    REQUIRE(approx(out.at(1, 0), 2.0f));
    REQUIRE(approx(out.at(2, 0), 4.0f + 1.0f));
    REQUIRE(approx(out.at(3, 0), 8.0f + 2.0f));
    REQUIRE(approx(out.at(4, 0), 4.0f));
    REQUIRE(approx(out.at(5, 0), 8.0f));
}

TEST_CASE("conv_transpose1d matches the reference loop", "[conv1d]") {
    for (const std::size_t stride : {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        const std::size_t kernel = 2 * stride;
        const std::size_t padding = (stride + 1) / 2;
        constexpr std::size_t c_in = 6;
        constexpr std::size_t c_out = 3;
        const Mat x = ramp(5, c_in, 0.019f, 0.3f);
        const Mat w = ramp(c_in, c_out * kernel, 0.007f, 0.2f);
        const Mat got = conv_transpose1d(x, w, c_out, kernel, {}, stride, padding, stride % 2);
        const Mat want = ref_conv_transpose1d(x, w, c_out, kernel, stride, padding, stride % 2);
        REQUIRE(got.rows == want.rows);
        REQUIRE(got.cols == want.cols);
        for (std::size_t i = 0; i < got.data.size(); ++i) {
            REQUIRE(approx(got.data[i], want.data[i]));
        }
    }
}

TEST_CASE("conv_transpose1d biases per output channel, not per input", "[conv1d]") {
    // Cin != Cout for every SNAC upsampling block, so a bias applied on the
    // wrong axis is a length mismatch rather than a silent error.
    constexpr std::size_t stride = 2;
    constexpr std::size_t kernel = 2 * stride;
    const Mat x = Mat::zeros(3, 4);  // Cin = 4
    const Mat w = Mat::zeros(4, 2 * kernel);
    const std::vector<float> bias{1.0f, -1.0f};  // Cout = 2
    const Mat out = conv_transpose1d(x, w, 2, kernel, bias, stride, (stride + 1) / 2, stride % 2);
    REQUIRE(out.rows == 6);
    REQUIRE(out.cols == 2);
    for (std::size_t t = 0; t < out.rows; ++t) {
        REQUIRE(approx(out.at(t, 0), 1.0f));
        REQUIRE(approx(out.at(t, 1), -1.0f));
    }
}

TEST_CASE("SNAC's decoder rates upsample 4 code frames to exactly 2048 samples", "[conv1d]") {
    // The end-to-end length invariant: SNAC 24 kHz uses decoder_rates
    // [8, 8, 4, 2] for a total of 512x, and one Orpheus 7-token frame carries
    // 4 frames at the finest codebook rate -- so 4 * 512 = 2048 samples, which
    // is 85.33 ms at 24 kHz. Everything about the frame accounting downstream
    // rests on this, so pin it here with real convolutions rather than
    // arithmetic.
    InitRng rng(7);
    Mat h = Mat::from_fn(4, 32, [&](std::size_t, std::size_t) { return rng.next_normal() * 0.1f; });

    std::size_t channels = 32;
    for (const std::size_t stride : {std::size_t{8}, std::size_t{8}, std::size_t{4},
                                     std::size_t{2}}) {
        const std::size_t kernel = 2 * stride;
        const std::size_t out_channels = channels / 2;
        Mat w = Mat::from_fn(channels, out_channels * kernel,
                             [&](std::size_t, std::size_t) { return rng.next_normal() * 0.05f; });
        h = conv_transpose1d(h, w, out_channels, kernel, {}, stride, (stride + 1) / 2,
                             stride % 2);
        channels = out_channels;

        // Each residual unit must preserve length exactly, at every dilation.
        for (const std::size_t dilation : {std::size_t{1}, std::size_t{3}, std::size_t{9}}) {
            const std::size_t before = h.rows;
            Mat dw = Mat::from_fn(channels, 7,
                                  [&](std::size_t, std::size_t) { return rng.next_normal() * 0.1f; });
            const Mat filtered = conv1d_depthwise(h, dw, {}, dilation, 3 * dilation);
            REQUIRE(filtered.rows == before);
        }
    }

    REQUIRE(h.rows == 4 * 512);
    REQUIRE(h.rows == 2048);
    REQUIRE(channels == 2);
}

// =============================================================================
// Snake
// =============================================================================

TEST_CASE("snake1d matches x + sin^2(alpha*x)/alpha", "[conv1d]") {
    const Mat x({0.5f, -1.25f, 2.0f, 0.0f}, 2, 2);
    const std::vector<float> alpha{1.0f, 0.5f};
    const Mat out = snake1d(x, alpha);
    for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            const float a = alpha[c];
            const float s = std::sin(a * x.at(r, c));
            REQUIRE(approx(out.at(r, c), x.at(r, c) + s * s / (a + 1e-9f)));
        }
    }
}

TEST_CASE("snake1d applies alpha per channel", "[conv1d]") {
    // Same value in both channels, different alpha -> different output. A
    // scalar alpha, or one indexed by row, would make these equal.
    const Mat x({1.0f, 1.0f}, 1, 2);
    const std::vector<float> alpha{1.0f, 2.0f};
    const Mat out = snake1d(x, alpha);
    REQUIRE(!approx(out.at(0, 0), out.at(0, 1)));
}

TEST_CASE("snake1d survives alpha at and below zero", "[conv1d]") {
    // Trained alphas pass close to zero; the epsilon lives inside the
    // reciprocal so this must stay finite rather than blowing up.
    const Mat x({1.0f, 1.0f, 1.0f}, 1, 3);
    const std::vector<float> alpha{0.0f, -1.0f, 1e-8f};
    const Mat out = snake1d(x, alpha);
    for (std::size_t c = 0; c < 3; ++c) {
        REQUIRE(std::isfinite(out.at(0, c)));
    }
    // alpha == 0 gives sin(0) == 0 exactly, so the value passes through.
    REQUIRE(approx(out.at(0, 0), 1.0f));
    // sin is odd and squared, so negating alpha only changes the divisor sign.
    REQUIRE(approx(out.at(0, 1), 1.0f - std::sin(1.0f) * std::sin(1.0f)));
}

TEST_CASE("snake1d_inplace agrees with the copying form", "[conv1d]") {
    const Mat x = ramp(7, 5, 0.07f, 0.9f);
    const std::vector<float> alpha{0.5f, 1.0f, 1.5f, -0.5f, 2.0f};
    const Mat copied = snake1d(x, alpha);
    Mat in_place = x;
    snake1d_inplace(in_place, alpha);
    for (std::size_t i = 0; i < copied.data.size(); ++i) {
        REQUIRE(approx(copied.data[i], in_place.data[i]));
    }
}

// =============================================================================
// Weight normalization
// =============================================================================

TEST_CASE("weight_norm_combine rescales each group to its magnitude", "[conv1d]") {
    // ||[3, 4]|| == 5, so g == 2 scales it by 2/5.
    const std::vector<float> g{2.0f};
    const std::vector<float> v{3.0f, 4.0f};
    const std::vector<float> w = weight_norm_combine(g, v);
    REQUIRE(w.size() == 2);
    REQUIRE(approx(w[0], 1.2f));
    REQUIRE(approx(w[1], 1.6f));
}

TEST_CASE("weight_norm_combine normalizes every group independently", "[conv1d]") {
    const std::vector<float> g{1.0f, 10.0f};
    const std::vector<float> v{3.0f, 4.0f, 0.0f, 5.0f};
    const std::vector<float> w = weight_norm_combine(g, v);
    REQUIRE(approx(w[0], 0.6f));
    REQUIRE(approx(w[1], 0.8f));
    REQUIRE(approx(w[2], 0.0f));
    REQUIRE(approx(w[3], 10.0f));
}

TEST_CASE("weight_norm_combine groups by axis 0 of the stored tensor", "[conv1d]") {
    // The trap: Conv1d stores [Cout, Cin, K] and ConvTranspose1d stores
    // [Cin, Cout, K], so the same flat buffer normalizes over a different axis
    // depending on the layer. Deriving the group count from g rather than
    // assuming "output channels" handles both -- these two calls share a `v`
    // and must give genuinely different answers.
    const std::vector<float> v{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // Conv1d-style: g per output channel, 2 groups of 3.
    const std::vector<float> as_conv = weight_norm_combine(std::vector<float>{1.0f, 1.0f}, v);
    // ConvTranspose1d-style: g per input channel, 3 groups of 2.
    const std::vector<float> as_convt =
        weight_norm_combine(std::vector<float>{1.0f, 1.0f, 1.0f}, v);

    REQUIRE(as_conv.size() == v.size());
    REQUIRE(as_convt.size() == v.size());
    // 1/sqrt(3) vs 1/sqrt(2) -- picking the wrong axis is a real numeric error,
    // not a relabelling.
    REQUIRE(approx(as_conv[0], 1.0f / std::sqrt(3.0f)));
    REQUIRE(approx(as_convt[0], 1.0f / std::sqrt(2.0f)));
}

TEST_CASE("weight_norm_combine handles SNAC's transposed-conv shapes", "[conv1d]") {
    // decoder.model.2.block.1: v is [1024, 512, 16] and g is [1024, 1, 1], so
    // the magnitude is per input channel. Scaled down here, same structure.
    constexpr std::size_t c_in = 8;
    constexpr std::size_t c_out = 4;
    constexpr std::size_t kernel = 16;
    std::vector<float> g(c_in);
    for (std::size_t i = 0; i < c_in; ++i) {
        g[i] = 1.0f + static_cast<float>(i);
    }
    std::vector<float> v(c_in * c_out * kernel, 0.0f);
    for (std::size_t i = 0; i < c_in; ++i) {
        v[i * c_out * kernel] = 2.0f;  // one nonzero per group, norm == 2
    }
    const std::vector<float> w = weight_norm_combine(g, v);
    REQUIRE(w.size() == v.size());
    for (std::size_t i = 0; i < c_in; ++i) {
        // g[i] * 2 / 2 == g[i]
        REQUIRE(approx(w[i * c_out * kernel], g[i]));
    }
}

TEST_CASE("weight_norm_combine yields zeros for a zero direction", "[conv1d]") {
    const std::vector<float> g{5.0f};
    const std::vector<float> v{0.0f, 0.0f, 0.0f};
    const std::vector<float> w = weight_norm_combine(g, v);
    for (const float value : w) {
        REQUIRE(value == 0.0f);
        REQUIRE(std::isfinite(value));
    }
}

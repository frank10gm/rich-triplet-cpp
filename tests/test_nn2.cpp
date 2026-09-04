#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

#include "rt/nn2.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

namespace {

/// Central-difference gradient of `f` with respect to one element of `param`.
template <typename F>
float numerical_grad_param(const TensorNode& param, std::size_t r, std::size_t c, const F& f) {
    constexpr float h = 1e-3f;
    const Mat original = param.data();

    Mat plus = original;
    plus.at_mut(r, c) += h;
    param.set_data(plus);
    const float f_plus = f();

    Mat minus = original;
    minus.at_mut(r, c) -= h;
    param.set_data(minus);
    const float f_minus = f();

    param.set_data(original);
    return (f_plus - f_minus) / (2.0f * h);
}

/// Wrap `out` in a scalar sum node so `backward()` has a 1x1 seed to start from.
TensorNode sum_to_scalar(const TensorNode& out) {
    const TensorNode loss = TensorNode::leaf(Mat({out.data().sum()}, 1, 1));
    const std::size_t rows = out.data().rows, cols = out.data().cols;
    loss.set_backward(
        [out, rows, cols] {
            out.grad_add(Mat::ones(rows, cols));
            out.call_backward_fn();
        },
        {out});
    return loss;
}

}  // namespace

// -----------------------------------------------------------------------------
// InitRng
// -----------------------------------------------------------------------------

TEST_CASE("InitRng is deterministic and roughly standard-normal", "[nn2]") {
    InitRng a(7), b(7);
    for (int i = 0; i < 16; ++i) {
        REQUIRE(a.next_f32() == b.next_f32());
    }

    InitRng rng(0);
    const std::vector<float> v = rng.normal_vec(4096, 1.0f);
    float mean = 0.0f;
    for (float x : v) {
        mean += x;
    }
    mean /= static_cast<float>(v.size());
    float var = 0.0f;
    for (float x : v) {
        var += (x - mean) * (x - mean);
    }
    var /= static_cast<float>(v.size());

    INFO("mean=" << mean << " var=" << var);
    REQUIRE(std::fabs(mean) < 0.1f);
    REQUIRE(std::fabs(std::sqrt(var) - 1.0f) < 0.1f);
}

// -----------------------------------------------------------------------------
// Linear2
// -----------------------------------------------------------------------------

TEST_CASE("linear2 output shape", "[nn2]") {
    InitRng rng(0);
    const Linear2 layer(4, 6, rng);
    const TensorNode out = layer.forward(TensorNode::leaf(Mat::zeros(3, 4)));  // [T=3, in=4]
    REQUIRE(out.data().rows == 3);
    REQUIRE(out.data().cols == 6);
}

TEST_CASE("linear2 bias starts at zero", "[nn2]") {
    InitRng rng(0);
    const Linear2 layer(4, 3, rng);
    for (float v : layer.bias.data().data) {
        REQUIRE(v == 0.0f);
    }
}

TEST_CASE("linear2 weight gradient", "[nn2]") {
    InitRng rng(42);
    const Linear2 layer(3, 2, rng);
    const Mat x_data({1.0f, -0.5f, 0.3f}, 1, 3);

    const TensorNode out = layer.forward(TensorNode::leaf(x_data));
    out.seed_grad_ones();
    out.call_backward_fn();
    const Mat analytical = layer.weight.grad();

    const float num = numerical_grad_param(layer.weight, 0, 0, [&] {
        return layer.forward(TensorNode::leaf(x_data)).data().sum();
    });

    INFO("dW[0,0]: analytical=" << analytical.at(0, 0) << " numerical=" << num);
    REQUIRE(approx(analytical.at(0, 0), num));
}

TEST_CASE("linear2 input and bias gradients", "[nn2]") {
    InitRng rng(11);
    const Linear2 layer(3, 2, rng);
    const Mat x_data({0.7f, -0.2f, 1.1f}, 1, 3);

    const TensorNode x = TensorNode::leaf(x_data);
    const TensorNode out = layer.forward(x);
    out.seed_grad_ones();
    out.call_backward_fn();

    // dInput = dOut @ W, with dOut all ones -> column sums of W.
    for (std::size_t c = 0; c < 3; ++c) {
        float want = 0.0f;
        for (std::size_t o = 0; o < 2; ++o) {
            want += layer.weight.data().at(o, c);
        }
        INFO("dInput[" << c << "]");
        REQUIRE(approx(x.grad().at(0, c), want));
    }
    // d_bias = sum over rows of ones = 1 per output.
    for (std::size_t o = 0; o < 2; ++o) {
        REQUIRE(approx(layer.bias.grad().at(0, o), 1.0f));
    }
}

TEST_CASE("linear2 exposes weight and bias as parameters", "[nn2]") {
    InitRng rng(0);
    const Linear2 layer(4, 3, rng);
    REQUIRE(layer.parameters().size() == 2);
    REQUIRE(layer.parameters()[0].data().rows == 3);
    REQUIRE(layer.parameters()[0].data().cols == 4);
    REQUIRE(layer.parameters()[1].data().rows == 1);
    REQUIRE(layer.parameters()[1].data().cols == 3);
}

TEST_CASE("linear2 new_no_bias_zeros allocates nothing", "[nn2]") {
    const Linear2 layer = Linear2::new_no_bias_zeros(2560, 10240);
    REQUIRE(layer.in_features == 2560);
    REQUIRE(layer.out_features == 10240);
    REQUIRE(layer.weight.data().numel() == 0);
    REQUIRE(layer.bias.data().numel() == 0);
}

TEST_CASE("linear2 quantized paths match the f32 forward", "[nn2]") {
    // Each quantized format should reproduce the f32 output to within its own
    // quantization error, and none of them should keep an f32 weight around.
    // Note that copying a Linear2 shares its TensorNode storage, so each case
    // builds its own layer from the same seed rather than copying.
    constexpr std::size_t in = 256, out_f = 8;
    const Mat x = Mat::from_fn(1, in, [&](std::size_t, std::size_t c) {
        return static_cast<float>(c) / static_cast<float>(in) - 0.5f;
    });
    const auto fresh_layer = [] {
        InitRng rng(5);
        return Linear2(in, out_f, rng);
    };

    const Linear2 base = fresh_layer();
    const Mat reference = base.forward(TensorNode::leaf(x)).data();
    const MatBf16 base_bf16 = base.weight.data().to_bf16();

    const auto require_close = [&](const Mat& got, float tol, const char* what) {
        for (std::size_t c = 0; c < out_f; ++c) {
            INFO(what << " output " << c);
            REQUIRE(std::fabs(got.at(0, c) - reference.at(0, c)) < tol);
        }
    };

    SECTION("bf16") {
        Linear2 l = fresh_layer();
        l.load_bf16(*base_bf16.data, out_f, in);
        REQUIRE(l.weight.data().numel() == 0);
        require_close(l.forward(TensorNode::leaf(x)).data(), 0.05f, "bf16");
    }

    SECTION("q4") {
        Linear2 l = fresh_layer();
        l.quantize_and_free_f32();
        REQUIRE(l.q4_weight.has_value());
        REQUIRE(l.weight.data().numel() == 0);
        require_close(l.forward(TensorNode::leaf(x)).data(), 0.5f, "q4");
    }

    SECTION("q4k from bf16") {
        Linear2 l = fresh_layer();
        l.load_bf16(*base_bf16.data, out_f, in);
        l.quantize_bf16_to_q4k();
        REQUIRE(l.q4k_weight.has_value());
        REQUIRE_FALSE(l.bf16_weight.has_value());
        require_close(l.forward(TensorNode::leaf(x)).data(), 0.5f, "q4k");
    }

    SECTION("quantize keeps the f32 weight for training") {
        Linear2 l = fresh_layer();
        l.quantize();
        REQUIRE(l.q4_weight.has_value());
        REQUIRE(l.weight.data().numel() == in * out_f);
    }
}

TEST_CASE("linear2 clear_weight_data drops every representation", "[nn2]") {
    InitRng rng(5);
    Linear2 l(256, 4, rng);
    l.quantize();
    l.clear_weight_data();
    REQUIRE_FALSE(l.q4_weight.has_value());
    REQUIRE_FALSE(l.q4k_weight.has_value());
    REQUIRE_FALSE(l.bf16_weight.has_value());
    REQUIRE(l.weight.data().numel() == 0);
}

// -----------------------------------------------------------------------------
// LayerNorm2 / RmsNorm2
// -----------------------------------------------------------------------------

TEST_CASE("layernorm2 output has zero mean and unit std", "[nn2]") {
    const LayerNorm2 ln(4);
    const Mat out = ln.forward(TensorNode::leaf(Mat({1., 2., 3., 4.}, 1, 4))).data();

    float mean = 0.0f;
    for (float v : out.data) {
        mean += v;
    }
    mean /= 4.0f;
    INFO("LN mean = " << mean);
    REQUIRE(std::fabs(mean) < 1e-5f);

    float var = 0.0f;
    for (float v : out.data) {
        var += (v - mean) * (v - mean);
    }
    const float std_dev = std::sqrt(var / 4.0f);
    INFO("LN std = " << std_dev);
    REQUIRE(std::fabs(std_dev - 1.0f) < 1e-4f);
}

TEST_CASE("rmsnorm2 forward and gemma3 variant", "[nn2]") {
    const RmsNorm2 norm(4, 1e-6f);
    const Mat x({1.0f, 2.0f, 3.0f, 4.0f}, 1, 4);

    // gamma is initialized to ones, so the standard form is a plain RMS scale.
    const Mat out = norm.forward(TensorNode::leaf(x)).data();
    float mean_sq = 0.0f;
    for (float v : x.data) {
        mean_sq += v * v;
    }
    const float inv_rms = 1.0f / std::sqrt(mean_sq / 4.0f + 1e-6f);
    for (std::size_t c = 0; c < 4; ++c) {
        REQUIRE(approx(out.at(0, c), x.at(0, c) * inv_rms));
    }

    // The Gemma 3 form scales by (1 + gamma), so ones-gamma doubles it.
    const Mat gemma = norm.forward_gemma3(TensorNode::leaf(x)).data();
    for (std::size_t c = 0; c < 4; ++c) {
        REQUIRE(approx(gemma.at(0, c), 2.0f * out.at(0, c)));
    }

    REQUIRE(norm.parameters().size() == 1);
}

// -----------------------------------------------------------------------------
// Dropout2
// -----------------------------------------------------------------------------

TEST_CASE("dropout is identity at inference", "[nn2]") {
    const Dropout2 drop(0.5f);
    const TensorNode x = TensorNode::leaf(Mat::from_fn(4, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c);
    }));
    const TensorNode out = drop.forward(x, false);
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 8; ++c) {
            REQUIRE(x.data().at(r, c) == out.data().at(r, c));
        }
    }
}

TEST_CASE("dropout zeros roughly p of the elements in training", "[nn2]") {
    const Dropout2 drop(0.5f);
    const TensorNode out =
        drop.forward(TensorNode::leaf(Mat::ones(8, 16)), true);
    const auto zeros = static_cast<std::size_t>(
        std::count(out.data().data.begin(), out.data().data.end(), 0.0f));
    // p=0.5 over 128 elements: expect ~64. The bounds are deliberately wide.
    INFO("zeros = " << zeros << "/128");
    REQUIRE(zeros > 20);
    REQUIRE(zeros < 108);
}

TEST_CASE("dropout with p=0 is identity even in training", "[nn2]") {
    const Dropout2 drop(0.0f);
    const TensorNode x = TensorNode::leaf(Mat::from_fn(3, 4, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c) * 0.1f;
    }));
    const TensorNode out = drop.forward(x, true);
    for (std::size_t i = 0; i < x.data().data.size(); ++i) {
        REQUIRE(x.data().data[i] == out.data().data[i]);
    }
}

TEST_CASE("dropout scales the survivors by 1/(1-p)", "[nn2]") {
    const Dropout2 drop(0.5f);
    const TensorNode out = drop.forward(TensorNode::leaf(Mat::ones(4, 4)), true);
    for (float v : out.data().data) {
        INFO("dropout output " << v);
        REQUIRE((v == 0.0f || std::fabs(v - 2.0f) < 1e-5f));
    }
}

TEST_CASE("dropout backward produces finite gradients", "[nn2]") {
    const Dropout2 drop(0.3f);
    const TensorNode x = TensorNode::leaf(Mat::from_fn(2, 4, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c) * 0.5f;
    }));
    sum_to_scalar(drop.forward(x, true)).backward();
    for (float v : x.grad().data) {
        REQUIRE(std::isfinite(v));
    }
}

// -----------------------------------------------------------------------------
// Mlp2 / SwiGluMlp2
// -----------------------------------------------------------------------------

TEST_CASE("mlp2 output shape", "[nn2]") {
    InitRng rng(0);
    const Mlp2 mlp(8, rng);
    const TensorNode out = mlp.forward(TensorNode::leaf(Mat::zeros(5, 8)));  // [T=5, d=8]
    REQUIRE(out.data().rows == 5);
    REQUIRE(out.data().cols == 8);
    REQUIRE(mlp.parameters().size() == 4);  // two Linears, weight + bias each
}

TEST_CASE("mlp2 backward produces finite gradients", "[nn2]") {
    InitRng rng(3);
    const Mlp2 mlp(4, rng);
    const TensorNode x = TensorNode::leaf(Mat({0.5f, -0.3f, 1.2f, -0.8f}, 1, 4));

    sum_to_scalar(mlp.forward(x)).backward();

    for (const TensorNode& p : mlp.parameters()) {
        for (float v : p.grad().data) {
            REQUIRE(std::isfinite(v));
        }
    }
}

TEST_CASE("swiglu mlp shape, clamp and gradients", "[nn2]") {
    InitRng rng(9);
    const SwiGluMlp2 mlp(8, 16, rng);
    const TensorNode x = TensorNode::leaf(Mat::from_fn(3, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.1f - 0.5f;
    }));

    const TensorNode out = mlp.forward(x);
    REQUIRE(out.data().rows == 3);
    REQUIRE(out.data().cols == 8);
    REQUIRE(mlp.parameters().size() == 6);  // three Linears

    sum_to_scalar(out).backward();
    for (const TensorNode& p : mlp.parameters()) {
        for (float v : p.grad().data) {
            REQUIRE(std::isfinite(v));
        }
    }

    // A finite clamp bounds the gate pre-activation; with a very small limit the
    // gate saturates and the output shrinks toward SiLU(+/-limit) * up.
    InitRng rng2(9);
    const SwiGluMlp2 clamped(8, 16, 0.01f, rng2);
    REQUIRE(clamped.swiglu_clamp == 0.01f);
    const Mat clamped_out = clamped.forward(x).data();
    const Mat unclamped_out = mlp.forward(x).data();
    bool differs = false;
    for (std::size_t i = 0; i < clamped_out.data.size(); ++i) {
        differs = differs || std::fabs(clamped_out.data[i] - unclamped_out.data[i]) > 1e-6f;
    }
    REQUIRE(differs);
}

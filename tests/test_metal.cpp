#include <catch2/catch_test_macros.hpp>

// The whole file is inert unless the Metal backend is compiled in.
#if RT_FEATURE_METAL

#include <cmath>

#include "rt/metal_ops.hpp"

using namespace rt;

namespace {

/// Largest absolute difference between two same-shaped matrices.
float max_err(const Mat& a, const Mat& b) {
    REQUIRE(a.rows == b.rows);
    REQUIRE(a.cols == b.cols);
    float m = 0.0f;
    for (std::size_t i = 0; i < a.data.size(); ++i) {
        m = std::max(m, std::fabs(a.data[i] - b.data[i]));
    }
    return m;
}

/// A deterministic matrix, sized past METAL_THRESHOLD so the GPU path is taken.
Mat make_mat(std::size_t rows, std::size_t cols, float k) {
    return Mat::from_fn(rows, cols, [&](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * cols + c) * k);
    });
}

}  // namespace

TEST_CASE("metal matmul matches the CPU result", "[metal]") {
    const Mat a = make_mat(64, 64, 0.01f);
    const Mat b = make_mat(64, 64, 0.013f);
    // The tiled kernel accumulates in the same order as the scalar loop, so
    // this is exact rather than merely close.
    REQUIRE(max_err(metal_matmul(a, b), a.matmul_bt(b.transpose())) == 0.0f);
}

TEST_CASE("metal matmul falls back below the threshold", "[metal]") {
    // 8x8x8 is far under METAL_THRESHOLD, so this runs on the CPU.
    const Mat a = make_mat(8, 8, 0.05f);
    const Mat b = make_mat(8, 8, 0.07f);
    REQUIRE(max_err(metal_matmul(a, b), a.matmul_bt(b.transpose())) < 1e-5f);
}

TEST_CASE("metal batched matmul", "[metal]") {
    const Mat a = make_mat(64, 64, 0.01f);
    const Mat b = make_mat(64, 64, 0.013f);

    const std::vector<Mat> out = metal_matmul_batched({{a, b}, {b, a}, {a, a}});
    REQUIRE(out.size() == 3);
    REQUIRE(max_err(out[0], a.matmul_bt(b.transpose())) == 0.0f);
    REQUIRE(max_err(out[1], b.matmul_bt(a.transpose())) == 0.0f);
    REQUIRE(max_err(out[2], a.matmul_bt(a.transpose())) == 0.0f);

    // Degenerate batches take the single-matmul path.
    REQUIRE(metal_matmul_batched({}).empty());
    REQUIRE(metal_matmul_batched({{a, b}}).size() == 1);
}

TEST_CASE("metal q4 matmul", "[metal]") {
    const Mat w = Mat::from_fn(512, 256, [](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * 256 + c) * 0.002f) * 0.4f;
    });
    const Q4Mat q4 = Q4Mat::quantize(w);
    const Mat a = Mat::from_fn(8, 256, [](std::size_t r, std::size_t c) {
        return (static_cast<float>(c) / 256.0f - 0.5f) * static_cast<float>(r + 1);
    });

    // The GPU reduces in a different order than the CPU, so the two agree to
    // float rounding rather than exactly.
    const float err = max_err(metal_matmul_q4_t(a, q4), q4.matmul_q4_t(a));
    INFO("q4 max error " << err);
    REQUIRE(err < 1e-3f);
}

TEST_CASE("metal q4k gemv", "[metal]") {
    const Mat w = Mat::from_fn(512, 256, [](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * 256 + c) * 0.002f) * 0.4f;
    });
    const Q4KMat q4k = Q4KMat::quantize(w);

    Mat dq = Mat::zeros(512, 256);
    for (std::size_t j = 0; j < 512; ++j) {
        q4k.dequantize_row_into(j, dq.row_mut(j));
    }

    const Mat a = Mat::from_fn(1, 256, [](std::size_t, std::size_t c) {
        return static_cast<float>(c) / 256.0f - 0.5f;
    });
    const float err = max_err(metal_gemv_q4k_t(a, q4k), a.matmul_bt(dq));
    INFO("q4k max error " << err);
    REQUIRE(err < 1e-4f);
}

TEST_CASE("metal bf16 gemv", "[metal]") {
    const Mat w = Mat::from_fn(512, 256, [](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * 256 + c) * 0.002f) * 0.4f;
    });
    const MatBf16 bf = w.to_bf16();

    const Mat a = Mat::from_fn(1, 256, [](std::size_t, std::size_t c) {
        return static_cast<float>(c) / 256.0f - 0.5f;
    });
    const float err = max_err(metal_gemv_bf16_t(a, bf), a.matmul_bt(bf.to_f32()));
    INFO("bf16 max error " << err);
    REQUIRE(err < 1e-4f);
}

TEST_CASE("metal weight buffers are cached across calls", "[metal]") {
    // A second call with the same weights reuses the uploaded buffer, so the
    // result must be identical, not merely close.
    const Mat w = Mat::from_fn(256, 256, [](std::size_t r, std::size_t c) {
        return std::cos(static_cast<float>(r * 256 + c) * 0.003f);
    });
    const MatBf16 bf = w.to_bf16();
    const Mat a = Mat::from_fn(1, 256, [](std::size_t, std::size_t c) {
        return static_cast<float>(c) * 0.001f;
    });

    const Mat first = metal_gemv_bf16_t(a, bf);
    const Mat second = metal_gemv_bf16_t(a, bf);
    REQUIRE(first.data == second.data);
}

#endif  // RT_FEATURE_METAL

#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "rt/bf16.hpp"
#include "rt/mat.hpp"

using namespace rt;

TEST_CASE("f16 roundtrip", "[bf16]") {
    // f32 -> f16 -> f32 round-trip.
    for (float v : {0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, -65504.0f, 0.001f, 100.0f}) {
        const std::uint16_t bits = f32_to_f16(v);
        const float back = f16_to_f32(bits);
        const float err = std::fabs(back - v);
        const float tol = std::fabs(v) * 0.002f + 1e-6f;  // ~0.1% relative tolerance
        INFO("f16 round-trip for " << v << ": got " << back << " (err " << err << ")");
        REQUIRE(err < tol);
    }
}

TEST_CASE("bf16 roundtrip values", "[bf16]") {
    for (float v : {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 16.0f, -8.0f, 0.125f}) {
        const std::uint16_t bits = f32_to_bf16(v);
        const float back = bf16_to_f32(bits);
        INFO("bf16 roundtrip: " << v << " -> " << back);
        REQUIRE(std::fabs(back - v) < 1e-2f);
    }
}

TEST_CASE("bf16 precision loss stays small", "[bf16]") {
    const float v = 3.14159f;
    const float back = bf16_to_f32(f32_to_bf16(v));
    REQUIRE(std::fabs(back - v) < 0.01f);
}

TEST_CASE("mat to_bf16 preserves shape", "[bf16]") {
    const Mat m = Mat::from_fn(4, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.1f;
    });
    const MatBf16 bf = m.to_bf16();
    REQUIRE(bf.rows == 4);
    REQUIRE(bf.cols == 8);
    REQUIRE(bf.data->size() == 32);
}

TEST_CASE("mat bf16 roundtrip", "[bf16]") {
    const Mat m = Mat::from_fn(3, 5, [](std::size_t r, std::size_t c) {
        return (static_cast<float>(r) - 1.5f) * (static_cast<float>(c) + 0.5f);
    });
    const Mat back = m.to_bf16().to_f32();
    REQUIRE(back.rows == m.rows);
    REQUIRE(back.cols == m.cols);
    for (std::size_t r = 0; r < m.rows; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            INFO("bf16 Mat roundtrip [" << r << "," << c << "]");
            REQUIRE(std::fabs(back.at(r, c) - m.at(r, c)) < 0.05f);
        }
    }
}

TEST_CASE("mat from_bf16_bytes", "[bf16]") {
    const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::uint8_t> bytes;
    for (float v : values) {
        const std::uint16_t bits = f32_to_bf16(v);
        bytes.push_back(static_cast<std::uint8_t>(bits & 0xff));
        bytes.push_back(static_cast<std::uint8_t>(bits >> 8));
    }
    const Mat m = Mat::from_bf16_bytes(bytes, 2, 2);
    REQUIRE(m.rows == 2);
    REQUIRE(m.cols == 2);
    for (std::size_t i = 0; i < 4; ++i) {
        INFO("from_bf16_bytes index " << i);
        REQUIRE(std::fabs(m.at(i / 2, i % 2) - values[i]) < 0.01f);
    }
}

TEST_CASE("bf16 compression ratio", "[bf16]") {
    const Mat m = Mat::from_fn(8, 8, [](std::size_t, std::size_t) { return 1.0f; });
    const MatBf16 bf = m.to_bf16();
    REQUIRE(MatBf16::compression_ratio() == 2.0f);
    REQUIRE(bf.size_bytes() == m.numel() * 2);
}

namespace {
/// Reference for `matmul_by_t`: dequantize the BF16 weights and use the plain
/// path, so BF16 rounding error is shared and only the matmul is under test.
Mat reference_by_t(const Mat& input, const MatBf16& w) {
#if RT_FEATURE_BLAS
    return input.matmul_bt(w.to_f32());
#else
    return input.matmul(w.to_f32().transpose());
#endif
}
}  // namespace

TEST_CASE("bf16 matmul_by_t decode matches reference", "[bf16]") {
    // Decode path: M=1, N large (simulates lm_head with small batch).
    const std::size_t n = 128;  // vocab-like dimension
    const std::size_t k = 16;   // hidden-like dimension
    const MatBf16 w = Mat::from_fn(n, k, [&](std::size_t r, std::size_t c) {
                          return static_cast<float>(r * k + c) * 0.01f - 0.3f;
                      }).to_bf16();
    const Mat input = Mat::from_fn(
        1, k, [](std::size_t, std::size_t c) { return static_cast<float>(c) * 0.05f - 0.1f; });

    const Mat reference = reference_by_t(input, w);
    const Mat result = w.matmul_by_t(input);

    REQUIRE(result.rows == 1);
    REQUIRE(result.cols == n);
    for (std::size_t j = 0; j < n; ++j) {
        INFO("matmul_by_t decode [" << j << "]");
        REQUIRE(std::fabs(result.at(0, j) - reference.at(0, j)) < 1e-4f);
    }
}

TEST_CASE("bf16 matmul_by_t prefill matches reference", "[bf16]") {
    // Prefill path: M>4 triggers the chunked sgemm branch.
    const std::size_t n = 32, k = 16, m = 8;
    const MatBf16 w = Mat::from_fn(n, k, [](std::size_t r, std::size_t c) {
                          return static_cast<float>(r + c) * 0.05f - 0.5f;
                      }).to_bf16();
    const Mat input = Mat::from_fn(m, k, [](std::size_t r, std::size_t c) {
        return (static_cast<float>(r) - 0.5f) * (static_cast<float>(c) * 0.1f + 0.1f);
    });

    const Mat reference = reference_by_t(input, w);
    const Mat result = w.matmul_by_t(input);

    REQUIRE(result.rows == m);
    REQUIRE(result.cols == n);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            INFO("matmul_by_t prefill [" << r << "," << c << "]");
            REQUIRE(std::fabs(result.at(r, c) - reference.at(r, c)) < 1e-4f);
        }
    }
}

TEST_CASE("bf16 matmul_by_t decode large N single chunk", "[bf16]") {
    // K=64 -> chunk = 8MB/256 = 32768 > N, so one loop iteration covers all N.
    // This validates the strided-ldc write path.
    const std::size_t n = 4096, k = 64;
    const MatBf16 w = Mat::from_fn(n, k, [&](std::size_t r, std::size_t c) {
                          return static_cast<float>(r * k + c) * 0.001f - 0.5f;
                      }).to_bf16();
    const Mat input = Mat::from_fn(
        1, k, [](std::size_t, std::size_t c) { return static_cast<float>(c) * 0.05f - 0.1f; });

    const Mat reference = reference_by_t(input, w);
    const Mat result = w.matmul_by_t(input);

    REQUIRE(result.rows == 1);
    REQUIRE(result.cols == n);
    for (std::size_t j = 0; j < n; ++j) {
        INFO("chunked BF16 decode [" << j << "]");
        REQUIRE(std::fabs(result.at(0, j) - reference.at(0, j)) < 5e-3f);
    }
}

TEST_CASE("bf16 matmul_by_t decode multi chunk", "[bf16]") {
    // K=2048 -> chunk = 8MB/8192 = 1024; N=4096 gives 4 chunks.
    const std::size_t k = 2048, n = 4096;
    const MatBf16 w = Mat::from_fn(n, k, [](std::size_t r, std::size_t c) {
                          return static_cast<float>(r) * 0.001f +
                                 static_cast<float>(c) * 0.0001f - 0.5f;
                      }).to_bf16();
    const Mat input = Mat::from_fn(1, k, [&](std::size_t, std::size_t c) {
        return static_cast<float>(c) * (1.0f / static_cast<float>(k)) - 0.5f;
    });

    const Mat reference = reference_by_t(input, w);
    const Mat result = w.matmul_by_t(input);

    REQUIRE(result.rows == 1);
    REQUIRE(result.cols == n);
    float max_err = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
        max_err = std::max(max_err, std::fabs(result.at(0, j) - reference.at(0, j)));
    }
    INFO("multi-chunk BF16 decode max error " << max_err);
    REQUIRE(max_err < 1e-2f);
}

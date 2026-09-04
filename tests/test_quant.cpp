#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

#include "rt/quant.hpp"

using namespace rt;

namespace {
/// Reference for the fused Q4/Q4K paths: dequantize, then plain matmul.
Mat reference_by_t(const Mat& input, const Mat& dq) {
#if RT_FEATURE_BLAS
    return input.matmul_bt(dq);
#else
    return input.matmul(dq.transpose());
#endif
}
}  // namespace

// -----------------------------------------------------------------------------
// Q4
// -----------------------------------------------------------------------------

TEST_CASE("q4 quantize/dequantize roundtrip", "[quant][q4]") {
    const Mat m = Mat::from_fn(4, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) / 31.0f * 2.0f - 1.0f;
    });
    const Mat m2 = Q4Mat::quantize(m).dequantize();
    REQUIRE(m2.rows == 4);
    REQUIRE(m2.cols == 8);
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 8; ++c) {
            INFO("q4 round-trip at [" << r << "," << c << "]");
            REQUIRE(std::fabs(m2.at(r, c) - m.at(r, c)) < 0.15f);
        }
    }
}

TEST_CASE("q4 compression ratio", "[quant][q4]") {
    const Mat m = Mat::from_fn(32, 32, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 32 + c);
    });
    REQUIRE(Q4Mat::quantize(m).compression_ratio() > 4.0f);
}

TEST_CASE("q4 zeros stay zero", "[quant][q4]") {
    const Mat m2 = Q4Mat::quantize(Mat::zeros(4, 8)).dequantize();
    REQUIRE(std::all_of(m2.data.begin(), m2.data.end(), [](float v) { return v == 0.0f; }));
}

TEST_CASE("q4 fused matmul matches dequant", "[quant][q4]") {
    const Mat a = Mat::from_fn(3, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.05f + 0.1f;
    });
    const Mat w = Mat::from_fn(4, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.03f - 0.2f;
    });
    const Q4Mat q = Q4Mat::quantize(w);

    const Mat ref_out = a.matmul(q.dequantize().transpose());
    const Mat fused_out = q.matmul_q4_t(a);

    REQUIRE(fused_out.rows == ref_out.rows);
    REQUIRE(fused_out.cols == ref_out.cols);
    for (std::size_t r = 0; r < ref_out.rows; ++r) {
        for (std::size_t c = 0; c < ref_out.cols; ++c) {
            INFO("q4 matmul [" << r << "," << c << "]");
            REQUIRE(std::fabs(fused_out.at(r, c) - ref_out.at(r, c)) < 1e-4f);
        }
    }
}

TEST_CASE("q4 handles non-multiple of block size", "[quant][q4]") {
    const Mat m = Mat::from_fn(1, 10, [](std::size_t, std::size_t c) {
        return static_cast<float>(c) * 0.1f - 0.5f;
    });
    const Mat m2 = Q4Mat::quantize(m).dequantize();
    for (std::size_t c = 0; c < 10; ++c) {
        INFO("small Q4 error at col " << c);
        REQUIRE(std::fabs(m2.at(0, c) - m.at(0, c)) < 0.15f);
    }
}

TEST_CASE("q4 chunked decode large N", "[quant][q4]") {
    // K=256 -> chunk = 8MB/1024 = 8192; N=4096 fits in one chunk.
    const std::size_t n = 4096, k = 256;
    const Mat weight = Mat::from_fn(n, k, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r + c) * 0.005f - 0.5f;
    });
    const Q4Mat q4w = Q4Mat::quantize(weight);
    const Mat input = Mat::from_fn(1, k, [](std::size_t, std::size_t c) {
        return static_cast<float>(c) * 0.01f - 0.5f;
    });

    const Mat reference = reference_by_t(input, q4w.dequantize());
    const Mat result = q4w.matmul_q4_t_blas(input);

    REQUIRE(result.rows == 1);
    REQUIRE(result.cols == n);
    float max_err = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
        max_err = std::max(max_err, std::fabs(result.at(0, j) - reference.at(0, j)));
    }
    // Q4 has quantization error; tolerate up to ~5% of absmax.
    INFO("chunked Q4 decode large-N max error " << max_err);
    REQUIRE(max_err < 0.05f);
}

TEST_CASE("q4 chunked decode multi chunk", "[quant][q4]") {
    // K=2048 -> chunk = 1024; N=4096 gives 4 chunks.
    const std::size_t k = 2048, n = 4096;
    const Mat weight = Mat::from_fn(n, k, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * k + c) * (1.0f / static_cast<float>(n * k)) * 2.0f - 1.0f;
    });
    const Q4Mat q4w = Q4Mat::quantize(weight);
    const Mat input = Mat::from_fn(1, k, [&](std::size_t, std::size_t c) {
        return static_cast<float>(c) / static_cast<float>(k) - 0.5f;
    });

    const Mat reference = reference_by_t(input, q4w.dequantize());
    const Mat result = q4w.matmul_q4_t_blas(input);

    REQUIRE(result.rows == 1);
    REQUIRE(result.cols == n);
    float max_err = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
        max_err = std::max(max_err, std::fabs(result.at(0, j) - reference.at(0, j)));
    }
    INFO("multi-chunk Q4 decode max error " << max_err);
    REQUIRE(max_err < 0.05f);
}

// -----------------------------------------------------------------------------
// Q4_K
// -----------------------------------------------------------------------------

TEST_CASE("q4k quantize/dequantize roundtrip", "[quant][q4k]") {
    // cols must be a multiple of 256.
    const Mat m = Mat::from_fn(4, 256, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 256 + c) / 1024.0f * 2.0f - 1.0f;
    });
    const Q4KMat q = Q4KMat::quantize(m);

    REQUIRE(q.blocks.size() == 4 * 144);  // 4 rows x 1 block/row
    REQUIRE(q.rows == 4);
    REQUIRE(q.cols == 256);

    std::vector<float> buf(256);
    float max_err = 0.0f;
    for (std::size_t r = 0; r < 4; ++r) {
        q.dequantize_row_into(r, buf);
        for (std::size_t c = 0; c < 256; ++c) {
            max_err = std::max(max_err, std::fabs(buf[c] - m.at(r, c)));
        }
    }
    INFO("Q4K round-trip max error " << max_err);
    REQUIRE(max_err < 0.15f);
}

TEST_CASE("q4k quantize zeros", "[quant][q4k]") {
    const Q4KMat q = Q4KMat::quantize(Mat::zeros(2, 256));
    std::vector<float> buf(256);
    for (std::size_t r = 0; r < 2; ++r) {
        q.dequantize_row_into(r, buf);
        for (std::size_t c = 0; c < 256; ++c) {
            INFO("Q4K zeros: row " << r << " col " << c);
            REQUIRE(std::fabs(buf[c]) < 1e-6f);
        }
    }
}

TEST_CASE("q4k quantize large matrix", "[quant][q4k]") {
    // lm_head-like dimensions, at a smaller scale.
    const Mat m = Mat::from_fn(16, 512, [](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * 512 + c) * 0.001f);
    });
    const Q4KMat q = Q4KMat::quantize(m);
    REQUIRE(q.rows == 16);
    REQUIRE(q.cols == 512);

    std::vector<float> buf(512);
    float max_err = 0.0f;
    for (std::size_t r = 0; r < 16; ++r) {
        q.dequantize_row_into(r, buf);
        for (std::size_t c = 0; c < 512; ++c) {
            max_err = std::max(max_err, std::fabs(buf[c] - m.at(r, c)));
        }
    }
    INFO("Q4K large matrix round-trip max error " << max_err);
    REQUIRE(max_err < 0.15f);
}

TEST_CASE("q4k quantize_from_bf16 matches quantize", "[quant][q4k]") {
    // Quantizing via the BF16 scratch path must agree with quantizing the
    // dequantized f32 matrix -- the BF16 rounding is the only difference.
    const Mat m = Mat::from_fn(8, 256, [](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * 256 + c) * 0.01f) * 0.7f;
    });
    const MatBf16 bf = m.to_bf16();
    const Q4KMat from_bf = Q4KMat::quantize_from_bf16(bf);
    const Q4KMat direct = Q4KMat::quantize(bf.to_f32());
    REQUIRE(from_bf.blocks == direct.blocks);
}

TEST_CASE("q4k from_q4mat matches requantized Q4", "[quant][q4k]") {
    const Mat m = Mat::from_fn(4, 256, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 256 + c) / 1024.0f - 0.5f;
    });
    const Q4Mat q4 = Q4Mat::quantize(m);
    const Q4KMat via_q4 = Q4KMat::from_q4mat(q4);
    const Q4KMat direct = Q4KMat::quantize(q4.dequantize());
    REQUIRE(via_q4.blocks == direct.blocks);
}

TEST_CASE("q4k dot_row matches dequantized dot", "[quant][q4k]") {
    const std::size_t n = 8, k = 512;
    const Mat w = Mat::from_fn(n, k, [](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * 512 + c) * 0.003f);
    });
    const Q4KMat q = Q4KMat::quantize(w);
    const Mat a = Mat::from_fn(1, k, [&](std::size_t, std::size_t c) {
        return static_cast<float>(c) / static_cast<float>(k) - 0.5f;
    });

    std::vector<float> buf(k);
    for (std::size_t j = 0; j < n; ++j) {
        q.dequantize_row_into(j, buf);
        float want = 0.0f;
        for (std::size_t p = 0; p < k; ++p) {
            want += a.data[p] * buf[p];
        }
        INFO("q4k dot_row " << j);
        REQUIRE(std::fabs(q.dot_row(j, a.data) - want) < 1e-3f);
    }
}

TEST_CASE("q4k gemv decode matches dequantized reference", "[quant][q4k]") {
    // M=1 takes the SDOT/NEON GEMV path; the Q8 activation quantization adds
    // error on top of Q4_K's, so the tolerance is looser than the sgemm path.
    const std::size_t n = 2048, k = 256;
    const Mat w = Mat::from_fn(n, k, [](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * 256 + c) * 0.002f) * 0.5f;
    });
    const Q4KMat q = Q4KMat::quantize(w);
    const Mat input = Mat::from_fn(1, k, [&](std::size_t, std::size_t c) {
        return static_cast<float>(c) / static_cast<float>(k) - 0.5f;
    });

    Mat dq = Mat::zeros(n, k);
    for (std::size_t j = 0; j < n; ++j) {
        q.dequantize_row_into(j, dq.row_mut(j));
    }
    const Mat reference = reference_by_t(input, dq);
    const Mat result = q.matmul_q4k_t_blas(input);

    REQUIRE(result.rows == 1);
    REQUIRE(result.cols == n);
    float max_err = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
        max_err = std::max(max_err, std::fabs(result.at(0, j) - reference.at(0, j)));
    }
    INFO("q4k gemv max error " << max_err);
    REQUIRE(max_err < 0.05f);
}

TEST_CASE("q4k prefill sgemm matches dequantized reference", "[quant][q4k]") {
    // N >= 4096 exercises the multi-threaded chunked-sgemm branch.
    const std::size_t n = 4096, k = 256, m = 6;
    const Mat w = Mat::from_fn(n, k, [](std::size_t r, std::size_t c) {
        return std::cos(static_cast<float>(r * 256 + c) * 0.0015f) * 0.4f;
    });
    const Q4KMat q = Q4KMat::quantize(w);
    const Mat input = Mat::from_fn(m, k, [&](std::size_t r, std::size_t c) {
        return (static_cast<float>(r) - 2.5f) * (static_cast<float>(c) / static_cast<float>(k));
    });

    Mat dq = Mat::zeros(n, k);
    for (std::size_t j = 0; j < n; ++j) {
        q.dequantize_row_into(j, dq.row_mut(j));
    }
    const Mat reference = reference_by_t(input, dq);
    const Mat result = q.matmul_q4k_t_blas(input);

    REQUIRE(result.rows == m);
    REQUIRE(result.cols == n);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            INFO("q4k prefill [" << r << "," << c << "]");
            REQUIRE(std::fabs(result.at(r, c) - reference.at(r, c)) < 1e-2f);
        }
    }
}

TEST_CASE("q4k scalar matmul matches dequantized reference", "[quant][q4k]") {
    const std::size_t n = 12, k = 256, m = 3;
    const Mat w = Mat::from_fn(n, k, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 256 + c) * 0.0005f - 0.4f;
    });
    const Q4KMat q = Q4KMat::quantize(w);
    const Mat input = Mat::from_fn(m, k, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r + 1) * (static_cast<float>(c) * 0.001f - 0.1f);
    });

    Mat dq = Mat::zeros(n, k);
    for (std::size_t j = 0; j < n; ++j) {
        q.dequantize_row_into(j, dq.row_mut(j));
    }
    const Mat reference = input.matmul(dq.transpose());
    const Mat result = q.matmul_q4k_t(input);

    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            INFO("q4k scalar matmul [" << r << "," << c << "]");
            REQUIRE(std::fabs(result.at(r, c) - reference.at(r, c)) < 1e-3f);
        }
    }
}

// -----------------------------------------------------------------------------
// Q8
// -----------------------------------------------------------------------------

TEST_CASE("q8 quantize preserves shape", "[quant][q8]") {
    const Mat m = Mat::from_fn(8, 16, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 16 + c) * 0.1f - 1.0f;
    });
    const Q8Mat q = Q8Mat::quantize(m);
    REQUIRE(q.rows == 8);
    REQUIRE(q.cols == 16);
    REQUIRE(q.packed.size() == 8 * 16);
}

TEST_CASE("q8 roundtrip is accurate", "[quant][q8]") {
    const Mat m = Mat::from_fn(4, 8, [](std::size_t r, std::size_t c) {
        return (static_cast<float>(r) - 1.5f) * (static_cast<float>(c) + 0.5f);
    });
    const Mat back = Q8Mat::quantize(m).dequantize();
    REQUIRE(back.rows == m.rows);
    REQUIRE(back.cols == m.cols);

    float absmax = 0.0f;
    for (float v : m.data) {
        absmax = std::max(absmax, std::fabs(v));
    }
    const float max_err = absmax / 254.0f + 1e-5f;  // Q8 error bound per block
    for (std::size_t r = 0; r < m.rows; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            INFO("Q8 roundtrip [" << r << "," << c << "]");
            REQUIRE(std::fabs(back.at(r, c) - m.at(r, c)) <= max_err);
        }
    }
}

TEST_CASE("q8 uses less memory than f32", "[quant][q8]") {
    const Mat m = Mat::from_fn(32, 64, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 64 + c) * 0.01f;
    });
    const Q8Mat q = Q8Mat::quantize(m);
    REQUIRE(q.size_bytes() < 32 * 64 * 4);
    REQUIRE(q.compression_ratio() > 2.0f);
}

TEST_CASE("q8 matmul matches f32", "[quant][q8]") {
    const Mat a = Mat::from_fn(3, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.1f - 1.0f;
    });
    const Mat w = Mat::from_fn(4, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.05f - 0.5f;
    });
    const Q8Mat q = Q8Mat::quantize(w);
    const Mat reference = a.matmul(q.dequantize().transpose());
    const Mat fused = q.matmul_q8_t(a);
    for (std::size_t r = 0; r < reference.rows; ++r) {
        for (std::size_t c = 0; c < reference.cols; ++c) {
            INFO("q8 matmul [" << r << "," << c << "]");
            REQUIRE(std::fabs(fused.at(r, c) - reference.at(r, c)) < 1e-4f);
        }
    }
}

TEST_CASE("bf16 special values", "[bf16]") {
    REQUIRE(bf16_to_f32(f32_to_bf16(0.0f)) == 0.0f);
    const float inf = bf16_to_f32(f32_to_bf16(std::numeric_limits<float>::infinity()));
    REQUIRE(std::isinf(inf));
    REQUIRE(inf > 0.0f);
    const float neg_inf = bf16_to_f32(f32_to_bf16(-std::numeric_limits<float>::infinity()));
    REQUIRE(std::isinf(neg_inf));
    REQUIRE(neg_inf < 0.0f);
}

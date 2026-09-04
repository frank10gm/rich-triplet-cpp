#include "rt/mat.hpp"

#include <cmath>
#include <numeric>

#include "rt/parallel.hpp"

#if RT_FEATURE_BLAS
#include <Accelerate/Accelerate.h>
#endif

#if RT_FEATURE_METAL
#include "rt/metal_ops.hpp"
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rt {

// =============================================================================
// Mat
// =============================================================================

#if RT_FEATURE_BLAS
/// BLAS-accelerated matmul via `cblas_sgemm`.
///
/// `cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)`
/// computes C = alpha * op(A) @ op(B) + beta * C. With RowMajor / NoTrans /
/// NoTrans, alpha=1, beta=0 it is a plain multiply, and lda/ldb/ldc are just
/// the column counts of A/B/C.
static Mat matmul_blas(const Mat& a, const Mat& b) {
    const std::size_t m = a.rows, k = a.cols, n = b.cols;
    Mat out = Mat::zeros(m, n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f,
                a.data.data(), static_cast<int>(k),
                b.data.data(), static_cast<int>(n),
                0.0f,
                out.data.data(), static_cast<int>(n));
    return out;
}
#endif

Mat Mat::matmul(const Mat& b) const {
    assert(cols == b.rows && "matmul shape mismatch");

#if RT_FEATURE_BLAS
    return matmul_blas(*this, b);
#elif RT_FEATURE_METAL
    return metal_matmul(*this, b);
#elif RT_FEATURE_PARALLEL
    return matmul_parallel(b, 0);
#else
    const std::size_t m = rows, k = cols, n = b.cols;
    Mat out = Mat::zeros(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const float a_ip = at(i, p);
            for (std::size_t j = 0; j < n; ++j) {
                out.at_mut(i, j) += a_ip * b.at(p, j);
            }
        }
    }
    return out;
#endif
}

Mat Mat::matmul_bt(const Mat& b) const {
    // self: [M,K], b: [N,K] -> out: [M,N]
    const std::size_t m = rows, k = cols, n = b.rows;
    assert(k == b.cols && "matmul_bt shape mismatch");
    Mat out = Mat::zeros(m, n);
#if RT_FEATURE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f,
                data.data(), static_cast<int>(k),
                b.data.data(), static_cast<int>(k),  // ldb = K (B is [N,K] row-major)
                0.0f,
                out.data.data(), static_cast<int>(n));
#else
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (std::size_t p = 0; p < k; ++p) {
                acc += data[i * k + p] * b.data[j * k + p];
            }
            out.at_mut(i, j) = acc;
        }
    }
#endif
    return out;
}

// =============================================================================
// Parallel matmul -- multi-threaded CPU version of matmul()
// =============================================================================
//
// Output rows are split across N threads. Thread i writes rows
// [i*chunk, (i+1)*chunk) of `out`; the ranges are disjoint, so no
// synchronization is needed during computation and `scoped_spawn` joins
// everyone before the result escapes.
Mat Mat::matmul_parallel(const Mat& b, std::size_t n_threads) const {
    assert(cols == b.rows && "matmul_parallel shape mismatch");
    const std::size_t m = rows, k = cols, n = b.cols;

    n_threads = (n_threads == 0)
                    ? hardware_threads()
                    : std::max<std::size_t>(1, std::min(n_threads, m));

    Mat out = Mat::zeros(m, n);

    const float* a_ptr = data.data();
    const float* b_ptr = b.data.data();
    float* out_ptr = out.data.data();

    const std::size_t chunk = div_ceil(m, n_threads);
    scoped_spawn(n_threads, [&](std::size_t tid) {
        const std::size_t row_start = tid * chunk;
        if (row_start >= m) {
            return;
        }
        const std::size_t row_end = std::min(row_start + chunk, m);
        // Each thread owns a disjoint range of output rows.
        float* out_rows = out_ptr + row_start * n;
        for (std::size_t i = 0; i < row_end - row_start; ++i) {
            const std::size_t global_i = row_start + i;
            for (std::size_t p = 0; p < k; ++p) {
                const float a_ip = a_ptr[global_i * k + p];
                for (std::size_t j = 0; j < n; ++j) {
                    out_rows[i * n + j] += a_ip * b_ptr[p * n + j];
                }
            }
        }
    });

    return out;
}

Mat Mat::transpose() const {
    return Mat::from_fn(cols, rows, [this](std::size_t r, std::size_t c) { return at(c, r); });
}

Mat Mat::add(const Mat& other) const {
    assert(rows == other.rows && cols == other.cols);
    std::vector<float> out(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        out[i] = data[i] + other.data[i];
    }
    return Mat(std::move(out), rows, cols);
}

void Mat::add_assign(const Mat& other) {
    assert(data.size() == other.data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
}

Mat Mat::mul_elem(const Mat& other) const {
    assert(rows == other.rows && cols == other.cols);
    std::vector<float> out(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        out[i] = data[i] * other.data[i];
    }
    return Mat(std::move(out), rows, cols);
}

Mat Mat::scale(float s) const {
    std::vector<float> out(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        out[i] = data[i] * s;
    }
    return Mat(std::move(out), rows, cols);
}

float Mat::sum() const {
    float acc = 0.0f;
    for (float x : data) {
        acc += x;
    }
    return acc;
}

Mat Mat::sum_rows() const {
    Mat out = Mat::zeros(1, cols);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            out.at_mut(0, c) += at(r, c);
        }
    }
    return out;
}

Mat Mat::row_mean() const {
    const float inv_n = 1.0f / static_cast<float>(cols);
    return Mat::from_fn(rows, 1, [&](std::size_t r, std::size_t) {
        float acc = 0.0f;
        for (std::size_t c = 0; c < cols; ++c) {
            acc += data[r * cols + c];
        }
        return acc * inv_n;
    });
}

Mat Mat::add_col_broadcast(const Mat& col) const {
    assert(col.cols == 1 && col.rows == rows);
    return Mat::from_fn(rows, cols,
                        [&](std::size_t r, std::size_t c) { return at(r, c) + col.at(r, 0); });
}

Mat Mat::sub_col_broadcast(const Mat& col) const {
    assert(col.cols == 1 && col.rows == rows);
    return Mat::from_fn(rows, cols,
                        [&](std::size_t r, std::size_t c) { return at(r, c) - col.at(r, 0); });
}

Mat Mat::mul_col_broadcast(const Mat& col) const {
    assert(col.cols == 1 && col.rows == rows);
    return Mat::from_fn(rows, cols,
                        [&](std::size_t r, std::size_t c) { return at(r, c) * col.at(r, 0); });
}

Mat Mat::mul_row_broadcast(const Mat& row) const {
    assert(row.rows == 1 && row.cols == cols);
    return Mat::from_fn(rows, cols,
                        [&](std::size_t r, std::size_t c) { return at(r, c) * row.at(0, c); });
}

float Mat::norm() const {
    float acc = 0.0f;
    for (float x : data) {
        acc += x * x;
    }
    return std::sqrt(acc);
}

// =============================================================================
// MatBf16
// =============================================================================

Mat MatBf16::to_f32() const {
    std::vector<float> out(data->size());
    for (std::size_t i = 0; i < data->size(); ++i) {
        out[i] = bf16_to_f32((*data)[i]);
    }
    return Mat(std::move(out), rows, cols);
}

#if defined(__aarch64__)
/// NEON-accelerated BF16 dot product.
///
/// Processes 16 elements per iteration: load u16, zero-extend to u32, shift
/// left 16 -> f32 bit pattern, FMA with the activation. Four independent
/// accumulators hide FMA latency.
static float dot_bf16_neon(const std::uint16_t* bf16_ptr, const float* a_ptr, std::size_t k) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    std::size_t p = 0;

    // Main loop: 16 elements per iteration.
    for (; p + 16 <= k; p += 16) {
        const uint16x4_t b0 = vld1_u16(bf16_ptr + p);
        const uint16x4_t b1 = vld1_u16(bf16_ptr + p + 4);
        const uint16x4_t b2 = vld1_u16(bf16_ptr + p + 8);
        const uint16x4_t b3 = vld1_u16(bf16_ptr + p + 12);

        const float32x4_t f0 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(b0), 16));
        const float32x4_t f1 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(b1), 16));
        const float32x4_t f2 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(b2), 16));
        const float32x4_t f3 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(b3), 16));

        acc0 = vfmaq_f32(acc0, f0, vld1q_f32(a_ptr + p));
        acc1 = vfmaq_f32(acc1, f1, vld1q_f32(a_ptr + p + 4));
        acc2 = vfmaq_f32(acc2, f2, vld1q_f32(a_ptr + p + 8));
        acc3 = vfmaq_f32(acc3, f3, vld1q_f32(a_ptr + p + 12));
    }

    // Tail: 4 elements at a time.
    for (; p + 4 <= k; p += 4) {
        const uint16x4_t b = vld1_u16(bf16_ptr + p);
        const float32x4_t f = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(b), 16));
        acc0 = vfmaq_f32(acc0, f, vld1q_f32(a_ptr + p));
    }

    // Reduce 4 accumulators -> scalar.
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float result = vaddvq_f32(acc0);

    // Scalar remainder.
    for (; p < k; ++p) {
        result += bf16_to_f32(bf16_ptr[p]) * a_ptr[p];
    }
    return result;
}
#endif

float MatBf16::dot_row(std::size_t row, std::span<const float> a) const {
    const std::size_t k = cols;
    const std::size_t base = row * k;
#if defined(__aarch64__)
    return dot_bf16_neon(data->data() + base, a.data(), k);
#else
    float acc = 0.0f;
    for (std::size_t p = 0; p < k; ++p) {
        acc += bf16_to_f32((*data)[base + p]) * a[p];
    }
    return acc;
#endif
}

Mat MatBf16::gemv_mt(const Mat& a) const {
    const std::size_t k = cols;
    const std::size_t n = rows;
    assert(a.rows == 1 && a.cols == k);

    Mat out = Mat::zeros(1, n);

    const std::size_t n_threads = std::max<std::size_t>(1, std::min(hardware_threads(), n / 128));
    const std::span<const float> a_data{a.data.data(), k};

    if (n_threads > 1) {
        float* out_ptr = out.data.data();
        const std::size_t chunk = div_ceil(n, n_threads);
        scoped_spawn(n_threads, [&](std::size_t tid) {
            const std::size_t j0 = tid * chunk;
            if (j0 >= n) {
                return;
            }
            const std::size_t j1 = std::min(j0 + chunk, n);
            // Each thread writes a disjoint range [j0, j1).
            for (std::size_t j = j0; j < j1; ++j) {
                out_ptr[j] = dot_row(j, a_data);
            }
        });
    } else {
        for (std::size_t j = 0; j < n; ++j) {
            out.data[j] = dot_row(j, a_data);
        }
    }

    return out;
}

Mat MatBf16::matmul_by_t(const Mat& a) const {
    const std::size_t m = a.rows;
    const std::size_t k = a.cols;
    const std::size_t n = rows;
    assert(k == cols && "matmul_by_t BF16: input cols != weight cols");

    // GEMV fast path for decode (M=1): fused BF16 dot + multi-threading.
    // Bypasses dequant-to-scratch + sgemm entirely -- reads BF16 directly.
    if (m == 1) {
        return gemv_mt(a);
    }

#if RT_FEATURE_BLAS
    // Chunked SGEMM for all M values (prefill & decode): dequantize CHUNK
    // weight rows -> f32, one sgemm per chunk, ~8 MB scratch instead of a full
    // N x K dequant. Passing &out.data[j0] with ldc = n makes sgemm write
    // C[i,j] straight to out[i, j0 + j].
    const std::size_t chunk =
        std::min(n, std::max<std::size_t>(64, (8 * 1024 * 1024) / (k * 4)));
    Mat out = Mat::zeros(m, n);
    std::vector<float> chunk_buf(chunk * k);

    for (std::size_t j0 = 0; j0 < n;) {
        const std::size_t j1 = std::min(j0 + chunk, n);
        const std::size_t actual = j1 - j0;

        for (std::size_t ji = 0; ji < actual; ++ji) {
            const std::size_t src = (j0 + ji) * k;
            const std::size_t dst = ji * k;
            for (std::size_t c = 0; c < k; ++c) {
                chunk_buf[dst + c] = bf16_to_f32((*data)[src + c]);
            }
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(m), static_cast<int>(actual), static_cast<int>(k),
                    1.0f,
                    a.data.data(), static_cast<int>(k),
                    chunk_buf.data(), static_cast<int>(k),
                    0.0f,
                    out.data.data() + j0, static_cast<int>(n));
        j0 = j1;
    }
    return out;
#else
    // Non-BLAS fallback: scalar row-by-row.
    Mat out = Mat::zeros(m, n);
    std::vector<float> row_f32(k);
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t base = j * k;
        for (std::size_t c = 0; c < k; ++c) {
            row_f32[c] = bf16_to_f32((*data)[base + c]);
        }
        for (std::size_t i = 0; i < m; ++i) {
            float dot = 0.0f;
            for (std::size_t c = 0; c < k; ++c) {
                dot += a.data[i * k + c] * row_f32[c];
            }
            out.at_mut(i, j) = dot;
        }
    }
    return out;
#endif
}

// -----------------------------------------------------------------------------
// BF16 interop
// -----------------------------------------------------------------------------

MatBf16 Mat::to_bf16() const {
    std::vector<std::uint16_t> bits(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        bits[i] = f32_to_bf16(data[i]);
    }
    return MatBf16(std::move(bits), rows, cols);
}

Mat Mat::from_bf16_bytes(std::span<const std::uint8_t> bytes, std::size_t rows,
                         std::size_t cols) {
    assert(bytes.size() == rows * cols * 2 && "from_bf16_bytes: wrong byte count");
    std::vector<float> data(rows * cols);
    for (std::size_t i = 0; i < data.size(); ++i) {
        const std::uint16_t bits =
            static_cast<std::uint16_t>(bytes[2 * i]) |
            (static_cast<std::uint16_t>(bytes[2 * i + 1]) << 8);
        data[i] = bf16_to_f32(bits);
    }
    return Mat(std::move(data), rows, cols);
}

Mat Mat::from_bf16(const MatBf16& src) { return src.to_f32(); }

MatBf16 mat_to_bf16(const Mat& m) {
    std::vector<std::uint16_t> bits(m.data.size());
    for (std::size_t i = 0; i < m.data.size(); ++i) {
        bits[i] = f32_to_bf16(m.data[i]);
    }
    return MatBf16(std::move(bits), m.rows, m.cols);
}

Mat mat_from_bf16(const MatBf16& m) { return m.to_f32(); }

}  // namespace rt

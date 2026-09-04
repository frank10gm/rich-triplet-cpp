#include <algorithm>
#include <cmath>

#include "rt/quant.hpp"
#include "rt/parallel.hpp"

#if RT_FEATURE_BLAS
#include <Accelerate/Accelerate.h>
#endif

namespace rt {

// =============================================================================
// Q4Mat -- 4-bit block quantization
// =============================================================================

Q4Mat Q4Mat::quantize(const Mat& mat) {
    const std::size_t n = mat.rows * mat.cols;
    const std::size_t n_blocks = div_ceil(n, Q4_BLOCK_SIZE);
    const std::size_t n_packed = div_ceil(n, 2);

    Q4Mat q;
    q.rows = mat.rows;
    q.cols = mat.cols;
    q.packed.assign(n_packed, 0);
    q.scales.assign(n_blocks, 0.0f);

    for (std::size_t block = 0; block < n_blocks; ++block) {
        const std::size_t start = block * Q4_BLOCK_SIZE;
        const std::size_t end = std::min(start + Q4_BLOCK_SIZE, n);

        float absmax = 0.0f;
        for (std::size_t k = start; k < end; ++k) {
            absmax = std::max(absmax, std::fabs(mat.data[k]));
        }

        const float scale = (absmax == 0.0f) ? 1.0f : absmax / 7.0f;
        q.scales[block] = scale;

        for (std::size_t k = start; k < end; ++k) {
            const float r = std::clamp(std::round(mat.data[k] / scale), -7.0f, 7.0f);
            const auto qv = static_cast<std::int8_t>(r);
            // Low 4 bits preserve the sign bit for i4.
            const auto nibble = static_cast<std::uint8_t>(qv & 0x0f);
            if (k % 2 == 0) {
                q.packed[k / 2] |= nibble;  // low nibble
            } else {
                q.packed[k / 2] |= static_cast<std::uint8_t>(nibble << 4);  // high nibble
            }
        }
    }

    return q;
}

namespace {
/// Read the 4-bit two's-complement value at flat index `k` and sign-extend it.
[[nodiscard]] inline std::int8_t q4_nibble(const std::vector<std::uint8_t>& packed,
                                           std::size_t k) {
    const std::uint8_t nibble =
        (k % 2 == 0) ? (packed[k / 2] & 0x0f) : ((packed[k / 2] >> 4) & 0x0f);
    return nibble >= 8 ? static_cast<std::int8_t>(static_cast<int>(nibble) - 16)
                       : static_cast<std::int8_t>(nibble);
}
}  // namespace

Mat Q4Mat::dequantize() const {
    const std::size_t n = rows * cols;
    std::vector<float> data(n);
    for (std::size_t k = 0; k < n; ++k) {
        data[k] = static_cast<float>(q4_nibble(packed, k)) * scales[k / Q4_BLOCK_SIZE];
    }
    return Mat(std::move(data), rows, cols);
}

void Q4Mat::dequantize_row_into(std::size_t j, std::span<float> buf) const {
    const std::size_t k = cols;
    const std::size_t row_start = j * k;
    std::size_t block = row_start / Q4_BLOCK_SIZE;
    std::size_t block_end = (block + 1) * Q4_BLOCK_SIZE;

    for (std::size_t p = 0; p < k; ++p) {
        const std::size_t flat = row_start + p;
        if (flat >= block_end) {
            ++block;
            block_end += Q4_BLOCK_SIZE;
        }
        buf[p] = static_cast<float>(q4_nibble(packed, flat)) * scales[block];
    }
}

Mat Q4Mat::matmul_q4_t(const Mat& a) const {
    // self is [N,K] row-major, a is [M,K], out is [M,N].
    const std::size_t m = a.rows, k = a.cols, nn = rows;
    assert(k == cols && "matmul_q4_t: a.cols != q4.cols");

    Mat out = Mat::zeros(m, nn);

    for (std::size_t j = 0; j < nn; ++j) {
        const std::size_t row_start_elem = j * k;
        for (std::size_t i = 0; i < m; ++i) {
            float acc = 0.0f;
            for (std::size_t p = 0; p < k; ++p) {
                const std::size_t flat_idx = row_start_elem + p;
                const float w = static_cast<float>(q4_nibble(packed, flat_idx)) *
                                scales[flat_idx / Q4_BLOCK_SIZE];
                acc += a.at(i, p) * w;
            }
            out.at_mut(i, j) = acc;
        }
    }

    return out;
}

Mat Q4Mat::matmul_q4_t_blas(const Mat& a) const {
#if !RT_FEATURE_BLAS
    return matmul_q4_t(a);
#else
    const std::size_t m = a.rows, k = a.cols, n = rows;
    assert(k == cols && "matmul_q4_t_blas: a.cols != q4.cols");

    // Prefill: dequantize the full weight matrix and call sgemm once.
    if (m > 4) {
        return a.matmul_bt(dequantize());  // A[M,K] @ W^T -> [M,N]
    }

    // Decode path (M <= 4): chunked SGEMM. Dequantize CHUNK weight rows into a
    // contiguous f32 scratch buffer, then one sgemm per chunk. Passing
    // &out.data[j0] with ldc = n makes sgemm write C[i,j] to out[i, j0+j].
    // Chunk targets ~8 MB of scratch (fits in Apple Silicon L2).
    const std::size_t chunk =
        std::min(n, std::max<std::size_t>(64, (8 * 1024 * 1024) / (k * 4)));
    Mat out = Mat::zeros(m, n);
    std::vector<float> chunk_buf(chunk * k);

    for (std::size_t j0 = 0; j0 < n;) {
        const std::size_t j1 = std::min(j0 + chunk, n);
        const std::size_t actual = j1 - j0;

        for (std::size_t ji = 0; ji < actual; ++ji) {
            dequantize_row_into(j0 + ji, std::span<float>(chunk_buf).subspan(ji * k, k));
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
#endif
}

// =============================================================================
// Q8Mat -- 8-bit symmetric block quantization
// =============================================================================

Q8Mat Q8Mat::quantize(const Mat& mat) {
    const std::size_t n = mat.rows * mat.cols;
    const std::size_t n_blocks = div_ceil(n, Q8_BLOCK_SIZE);

    Q8Mat q;
    q.rows = mat.rows;
    q.cols = mat.cols;
    q.packed.assign(n, 0);
    q.scales.assign(n_blocks, 0.0f);

    for (std::size_t block = 0; block < n_blocks; ++block) {
        const std::size_t start = block * Q8_BLOCK_SIZE;
        const std::size_t end = std::min(start + Q8_BLOCK_SIZE, n);

        float absmax = 0.0f;
        for (std::size_t k = start; k < end; ++k) {
            absmax = std::max(absmax, std::fabs(mat.data[k]));
        }

        const float scale = (absmax == 0.0f) ? 1.0f : absmax / 127.0f;
        q.scales[block] = scale;

        for (std::size_t k = start; k < end; ++k) {
            q.packed[k] = static_cast<std::int8_t>(
                std::clamp(std::round(mat.data[k] / scale), -127.0f, 127.0f));
        }
    }

    return q;
}

Mat Q8Mat::dequantize() const {
    const std::size_t n = rows * cols;
    std::vector<float> data(n);
    for (std::size_t k = 0; k < n; ++k) {
        data[k] = static_cast<float>(packed[k]) * scales[k / Q8_BLOCK_SIZE];
    }
    return Mat(std::move(data), rows, cols);
}

Mat Q8Mat::matmul_q8_t(const Mat& a) const {
    const std::size_t m = a.rows, k = a.cols, nn = rows;
    assert(k == cols && "matmul_q8_t: a.cols != q8.cols");

    Mat out = Mat::zeros(m, nn);
    for (std::size_t j = 0; j < nn; ++j) {
        const std::size_t row_start = j * k;
        for (std::size_t i = 0; i < m; ++i) {
            float acc = 0.0f;
            for (std::size_t p = 0; p < k; ++p) {
                const std::size_t flat = row_start + p;
                const float w =
                    static_cast<float>(packed[flat]) * scales[flat / Q8_BLOCK_SIZE];
                acc += a.at(i, p) * w;
            }
            out.at_mut(i, j) = acc;
        }
    }
    return out;
}

}  // namespace rt

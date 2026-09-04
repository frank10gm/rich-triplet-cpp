#pragma once

// =============================================================================
// Mat / MatBf16 -- dense f32 and bf16 matrix storage
// =============================================================================
//
// `Mat` is a flat 2-D matrix in row-major order: the storage type every
// forward/backward rule operates on. `MatBf16` is the compact half-size
// weight format; it dequantizes to f32 on the fly for computation.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "rt/bf16.hpp"

namespace rt {

/// A flat 2-D matrix stored in row-major order.
struct Mat {
    std::vector<float> data;
    std::size_t rows = 0;
    std::size_t cols = 0;

    Mat() = default;

    Mat(std::vector<float> d, std::size_t r, std::size_t c)
        : data(std::move(d)), rows(r), cols(c) {
        assert(data.size() == r * c && "Mat: data len != rows*cols");
    }

    [[nodiscard]] static Mat zeros(std::size_t rows, std::size_t cols) {
        return Mat(std::vector<float>(rows * cols, 0.0f), rows, cols);
    }

    [[nodiscard]] static Mat ones(std::size_t rows, std::size_t cols) {
        return Mat(std::vector<float>(rows * cols, 1.0f), rows, cols);
    }

    template <typename F>
    [[nodiscard]] static Mat from_fn(std::size_t rows, std::size_t cols, F&& f) {
        std::vector<float> data;
        data.reserve(rows * cols);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                data.push_back(f(r, c));
            }
        }
        return Mat(std::move(data), rows, cols);
    }

    [[nodiscard]] float at(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }
    [[nodiscard]] float& at_mut(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }

    [[nodiscard]] std::size_t numel() const { return rows * cols; }

    /// Row `r` as a contiguous span.
    [[nodiscard]] std::span<const float> row(std::size_t r) const {
        return {data.data() + r * cols, cols};
    }
    [[nodiscard]] std::span<float> row_mut(std::size_t r) {
        return {data.data() + r * cols, cols};
    }

    // -------------------------------------------------------------------------
    // Core linear algebra -- these are the hot paths
    // -------------------------------------------------------------------------

    /// C = A @ B    [M,K] x [K,N] -> [M,N]
    ///
    /// Dispatch priority (highest first):
    ///   1. RT_FEATURE_BLAS     -> cblas_sgemm (4-8x vs the plain triple loop)
    ///   2. RT_FEATURE_METAL    -> Apple Metal GPU compute shader
    ///   3. RT_FEATURE_PARALLEL -> multi-threaded scalar (~N_CPU x speedup)
    ///   4. default             -> single-threaded scalar
    [[nodiscard]] Mat matmul(const Mat& b) const;

    /// `self @ b^T` without materializing the transpose. `b` is [N,K].
    [[nodiscard]] Mat matmul_bt(const Mat& b) const;

    /// C = A @ B -- multi-threaded. `n_threads == 0` uses the CPU count.
    [[nodiscard]] Mat matmul_parallel(const Mat& b, std::size_t n_threads) const;

    /// A.T -- transpose: [M,N] -> [N,M]
    [[nodiscard]] Mat transpose() const;

    /// Element-wise addition (same shape).
    [[nodiscard]] Mat add(const Mat& other) const;

    /// In-place element-wise addition: self += other
    void add_assign(const Mat& other);

    /// Element-wise multiplication (same shape).
    [[nodiscard]] Mat mul_elem(const Mat& other) const;

    /// Scale every element by a scalar.
    [[nodiscard]] Mat scale(float s) const;

    /// Element-wise map.
    template <typename F>
    [[nodiscard]] Mat map(F&& f) const {
        std::vector<float> out;
        out.reserve(data.size());
        for (float x : data) {
            out.push_back(f(x));
        }
        return Mat(std::move(out), rows, cols);
    }

    /// Sum all elements.
    [[nodiscard]] float sum() const;

    /// Sum along rows -> shape [1, cols]. out[c] = sum_r self[r,c]
    [[nodiscard]] Mat sum_rows() const;

    /// Row-wise mean -> shape [rows, 1].
    [[nodiscard]] Mat row_mean() const;

    /// Broadcast-add a [rows,1] column vector to each column of self.
    [[nodiscard]] Mat add_col_broadcast(const Mat& col) const;

    /// Broadcast-subtract a [rows,1] column vector from each column.
    [[nodiscard]] Mat sub_col_broadcast(const Mat& col) const;

    /// Broadcast-multiply element-wise by a [rows,1] column vector.
    [[nodiscard]] Mat mul_col_broadcast(const Mat& col) const;

    /// Broadcast-multiply element-wise by a [1,cols] row vector.
    [[nodiscard]] Mat mul_row_broadcast(const Mat& row) const;

    /// L2 norm of all elements: sqrt(sum(x^2))
    [[nodiscard]] float norm() const;

    // -------------------------------------------------------------------------
    // BF16 interop
    // -------------------------------------------------------------------------

    /// Convert this `Mat` to compact BF16 storage.
    [[nodiscard]] struct MatBf16 to_bf16() const;

    /// Build a `Mat` from raw BF16 bytes (as stored in .safetensors BF16 shards).
    /// `bytes` must be `rows * cols * 2` bytes, little-endian BF16.
    [[nodiscard]] static Mat from_bf16_bytes(std::span<const std::uint8_t> bytes,
                                             std::size_t rows, std::size_t cols);

    /// Construct a `Mat` from a `MatBf16` (dequantize on load).
    [[nodiscard]] static Mat from_bf16(const struct MatBf16& src);
};

/// A matrix stored in BF16 format for compact in-memory storage.
///
/// `data` is reference-counted so that weight-tied tensors (embed_tokens and
/// lm_head) can share the same allocation without cloning 1.3 GB of BF16 bits.
struct MatBf16 {
    std::shared_ptr<std::vector<std::uint16_t>> data;  // bf16 bits, one per element
    std::size_t rows = 0;
    std::size_t cols = 0;

    MatBf16() : data(std::make_shared<std::vector<std::uint16_t>>()) {}

    MatBf16(std::vector<std::uint16_t> bits, std::size_t r, std::size_t c)
        : data(std::make_shared<std::vector<std::uint16_t>>(std::move(bits))),
          rows(r),
          cols(c) {}

    MatBf16(std::shared_ptr<std::vector<std::uint16_t>> bits, std::size_t r, std::size_t c)
        : data(std::move(bits)), rows(r), cols(c) {}

    /// Convert this `MatBf16` to a full f32 `Mat`.
    [[nodiscard]] Mat to_f32() const;

    /// Size in bytes (2 bytes per element).
    [[nodiscard]] std::size_t size_bytes() const { return data->size() * 2; }

    /// Compression ratio vs f32 (always 2.0x).
    [[nodiscard]] static constexpr float compression_ratio() { return 2.0f; }

    [[nodiscard]] float at(std::size_t r, std::size_t c) const {
        return bf16_to_f32((*data)[r * cols + c]);
    }

    /// Compute `a [M,K] @ self^T` where `self` is [N,K] BF16.
    ///
    /// **Decode path (M == 1)** -- fused BF16 dot + multi-threading, reading
    /// BF16 directly with no scratch buffer.
    ///
    /// **Otherwise** -- dequantizes CHUNK weight rows at a time into a scratch
    /// buffer and calls one `sgemm` per chunk. This cuts BLAS call overhead
    /// from N calls (262 K for lm_head) to N/CHUNK, while keeping scratch at
    /// ~8 MB (comfortably inside Apple-Silicon L2).
    [[nodiscard]] Mat matmul_by_t(const Mat& a) const;

    /// Fused BF16 dot product of weight row `row` against activation vector.
    /// On aarch64 this uses NEON intrinsics for ~4x throughput over scalar.
    [[nodiscard]] float dot_row(std::size_t row, std::span<const float> a) const;

   private:
    /// Multi-threaded GEMV: `a[1,K] @ self^T[N,K] -> [1,N]`.
    [[nodiscard]] Mat gemv_mt(const Mat& a) const;
};

/// Convert an f32 `Mat` to `MatBf16`.
[[nodiscard]] MatBf16 mat_to_bf16(const Mat& m);

/// Convert a `MatBf16` to an f32 `Mat`.
[[nodiscard]] Mat mat_from_bf16(const MatBf16& m);

}  // namespace rt

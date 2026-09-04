#pragma once

// =============================================================================
// Tensor -- an N-dimensional array with explicit strides
// =============================================================================
//
// Data is stored flat in row-major order; the shape says how many dimensions
// there are and how big each is, and the strides say how many elements to skip
// when stepping one position along a dimension.
//
//   strides for shape [D, R, C] = [R*C, C, 1]
//
// This is the teaching-oriented tensor: correctness and clarity over speed.
// The inference paths use `Mat` and `NDArray` instead.

#include <cstddef>
#include <string>
#include <vector>

namespace rt {

struct Tensor {
    /// The numbers, flat and row-major.
    std::vector<float> data;
    /// Sizes per dimension, e.g. [2, 3] is 2 rows of 3 columns.
    std::vector<std::size_t> shape;
    /// Elements to skip per dimension, computed from the shape.
    std::vector<std::size_t> strides;

    Tensor() = default;

    /// Build from raw data and a shape. The data length must equal the shape
    /// product.
    Tensor(std::vector<float> data, std::vector<std::size_t> shape);

    /// Row-major strides for a shape.
    [[nodiscard]] static std::vector<std::size_t> compute_strides(
        const std::vector<std::size_t>& shape);

    [[nodiscard]] static Tensor zeros(std::vector<std::size_t> shape);
    [[nodiscard]] static Tensor ones(std::vector<std::size_t> shape);
    /// A 1-D tensor holding [0, n) -- useful for position indices.
    [[nodiscard]] static Tensor arange(std::size_t n);

    [[nodiscard]] std::size_t numel() const { return data.size(); }
    [[nodiscard]] std::size_t ndim() const { return shape.size(); }

    /// Flatten a multi-dimensional index through the strides.
    [[nodiscard]] std::size_t flat_index(const std::vector<std::size_t>& indices) const;
    [[nodiscard]] float get(const std::vector<std::size_t>& indices) const;
    void set(const std::vector<std::size_t>& indices, float value);

    // -------------------------------------------------------------------------
    // Element-wise operations
    // -------------------------------------------------------------------------

    [[nodiscard]] Tensor add(const Tensor& other) const;
    [[nodiscard]] Tensor sub(const Tensor& other) const;
    /// Hadamard product -- element-wise, not matrix multiplication.
    [[nodiscard]] Tensor mul(const Tensor& other) const;
    [[nodiscard]] Tensor scale(float s) const;

    template <typename F>
    [[nodiscard]] Tensor map(F&& f) const {
        std::vector<float> out(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            out[i] = f(data[i]);
        }
        return Tensor(std::move(out), shape);
    }

    // -------------------------------------------------------------------------
    // Reductions
    // -------------------------------------------------------------------------

    [[nodiscard]] float sum_all() const;

    /// Sum along the last dimension, dropping it: [M, N] -> [M].
    /// A fully collapsed result is reported as shape [1].
    [[nodiscard]] Tensor sum_last_dim() const;

    /// Max along the last dimension -- softmax's numerical-stability step.
    [[nodiscard]] Tensor max_last_dim() const;

    // -------------------------------------------------------------------------
    // Matrix multiplication
    // -------------------------------------------------------------------------

    /// A [M, K] x B [K, N] -> C [M, N], where
    /// `C[i,j] = sum_k A[i,k] * B[k,j]`.
    ///
    /// Every linear layer is a matmul, and attention is two of them. This naive
    /// O(M*N*K) loop is written for clarity; the fast paths live in `Mat`.
    [[nodiscard]] Tensor matmul(const Tensor& other) const;

    /// Swap rows and columns of a 2-D tensor.
    [[nodiscard]] Tensor transpose() const;

    // -------------------------------------------------------------------------
    // Shape manipulation
    // -------------------------------------------------------------------------

    /// Reinterpret with a new shape; the element count must not change.
    [[nodiscard]] Tensor reshape(std::vector<std::size_t> new_shape) const;

    /// Print a one-line summary.
    void print_info(const std::string& name) const;
};

/// Softmax over the last dimension.
///
/// Converts raw scores into a probability distribution:
/// `softmax(x)[i] = exp(x[i]) / sum_j exp(x[j])`.
///
/// The row max is subtracted first: `exp` of a large number overflows f32, and
/// the constant cancels out, so the result is unchanged but stays finite.
[[nodiscard]] Tensor softmax(const Tensor& x);

}  // namespace rt

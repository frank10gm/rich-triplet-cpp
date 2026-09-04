#pragma once

// =============================================================================
// NDArray -- N-dimensional float array in row-major (C) order
// =============================================================================
//
// Views (`reshape`, `permute`, `slice`) share the underlying storage without
// copying. Mutation through `at_mut` triggers copy-on-write, so a view never
// silently writes through to an array someone else still holds.

#include <cstddef>
#include <memory>
#include <vector>

#include "rt/mat.hpp"

namespace rt {

/// Row-major (C-order) strides for a shape.
[[nodiscard]] std::vector<std::size_t> c_strides(const std::vector<std::size_t>& shape);

/// Copy of `v` with `val` inserted at position `axis`.
[[nodiscard]] std::vector<std::size_t> insert_at(const std::vector<std::size_t>& v,
                                                 std::size_t axis, std::size_t val);

/// Iterates over all multi-indices of a shape in row-major (C) order.
///
/// An empty shape yields exactly one item (the empty index, i.e. a scalar);
/// a shape with any zero dimension yields none.
class IndexIter {
   public:
    explicit IndexIter(const std::vector<std::size_t>& shape);

    /// Advance to the next multi-index. Returns false once exhausted; the
    /// current index stays valid until then.
    ///
    ///   for (IndexIter it(shape); it.valid(); it.next()) { use(it.index()); }
    [[nodiscard]] bool valid() const { return !done_; }
    void next();
    [[nodiscard]] const std::vector<std::size_t>& index() const { return current_; }

   private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> current_;
    bool done_ = false;
};

class NDArray {
   public:
    /// Flat storage shared across views.
    std::shared_ptr<std::vector<float>> data;
    /// Logical shape, e.g. [batch, heads, seq, d_head].
    std::vector<std::size_t> shape;
    /// Element strides (not byte strides): how many elements to skip when the
    /// corresponding index increases by 1.
    std::vector<std::size_t> strides;
    /// Element offset into `data` for this view's first element.
    std::size_t offset = 0;

    NDArray() : data(std::make_shared<std::vector<float>>()) {}

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    [[nodiscard]] static NDArray zeros(const std::vector<std::size_t>& shape);
    [[nodiscard]] static NDArray ones(const std::vector<std::size_t>& shape);

    /// Build from `f(multi_index) -> float`, iterating in C-order.
    template <typename F>
    [[nodiscard]] static NDArray from_fn(const std::vector<std::size_t>& shape, F&& f) {
        std::vector<float> data;
        data.reserve(numel_of(shape));
        for (IndexIter it(shape); it.valid(); it.next()) {
            data.push_back(f(it.index()));
        }
        return from_vec(std::move(data), shape);
    }

    /// Build from flat storage. The length must equal the shape product.
    [[nodiscard]] static NDArray from_vec(std::vector<float> data,
                                          const std::vector<std::size_t>& shape);

    /// Convert a 2-D `Mat` to shape [rows, cols].
    [[nodiscard]] static NDArray from_mat(const Mat& m);

    /// Convert a 2-D array back to a `Mat`.
    [[nodiscard]] Mat into_mat() const;

    // -------------------------------------------------------------------------
    // Shape / size accessors
    // -------------------------------------------------------------------------

    [[nodiscard]] std::size_t ndim() const { return shape.size(); }
    [[nodiscard]] std::size_t numel() const { return numel_of(shape); }
    [[nodiscard]] static std::size_t numel_of(const std::vector<std::size_t>& shape);

    // -------------------------------------------------------------------------
    // Indexing
    // -------------------------------------------------------------------------

    [[nodiscard]] std::size_t flat_index(const std::vector<std::size_t>& idx) const;
    [[nodiscard]] float at(const std::vector<std::size_t>& idx) const;
    /// Write access; copies the storage first if this view shares it.
    [[nodiscard]] float& at_mut(const std::vector<std::size_t>& idx);

    // -------------------------------------------------------------------------
    // Contiguity
    // -------------------------------------------------------------------------

    /// True when the array has C-order strides and `offset == 0`.
    [[nodiscard]] bool is_contiguous() const;
    /// A contiguous, offset-0 array. Shares storage when already contiguous.
    [[nodiscard]] NDArray contiguous() const;
    [[nodiscard]] NDArray clone_owned() const { return contiguous(); }

    // -------------------------------------------------------------------------
    // Views (zero-copy)
    // -------------------------------------------------------------------------

    /// Reshape without copying; makes the array contiguous first if needed.
    [[nodiscard]] NDArray reshape(const std::vector<std::size_t>& new_shape) const;
    /// Permute axes without copying. The result is typically non-contiguous.
    [[nodiscard]] NDArray permute(const std::vector<std::size_t>& axes) const;
    /// Fix `axis` at `index`, dropping that dimension.
    [[nodiscard]] NDArray slice(std::size_t axis, std::size_t index) const;
    /// Insert a size-1 dimension at `axis` (may equal `ndim()`).
    [[nodiscard]] NDArray expand_dims(std::size_t axis) const;

    // -------------------------------------------------------------------------
    // Elementwise operations
    // -------------------------------------------------------------------------

    [[nodiscard]] NDArray add(const NDArray& other) const;
    [[nodiscard]] NDArray mul(const NDArray& other) const;
    [[nodiscard]] NDArray scale(float s) const;
    [[nodiscard]] NDArray add_scalar(float s) const;

    template <typename F>
    [[nodiscard]] NDArray map(F&& f) const {
        NDArray out = NDArray::zeros(shape);
        for (IndexIter it(shape); it.valid(); it.next()) {
            out.at_mut(it.index()) = f(at(it.index()));
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Reductions
    // -------------------------------------------------------------------------

    /// Sum over `axis`, removing that dimension. `[B,T,D].reduce_sum(1)` -> `[B,D]`.
    [[nodiscard]] NDArray reduce_sum(std::size_t axis) const;

    /// Numerically stable softmax along `axis`.
    [[nodiscard]] NDArray softmax(std::size_t axis) const;

    // -------------------------------------------------------------------------
    // Batched matrix multiply
    // -------------------------------------------------------------------------

    /// `C[..., M, N] = A[..., M, K] @ B[..., K, N]`, over the last two
    /// dimensions. All leading (batch) dimensions must match exactly.
    [[nodiscard]] NDArray bmm(const NDArray& other) const;
};

}  // namespace rt

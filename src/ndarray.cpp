#include "rt/ndarray.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#if RT_FEATURE_METAL
#include "rt/metal_ops.hpp"
#endif

namespace rt {

std::vector<std::size_t> c_strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> s(shape.size());
    std::size_t acc = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        s[i] = acc;
        acc *= shape[i];
    }
    return s;
}

std::vector<std::size_t> insert_at(const std::vector<std::size_t>& v, std::size_t axis,
                                   std::size_t val) {
    std::vector<std::size_t> out;
    out.reserve(v.size() + 1);
    out.insert(out.end(), v.begin(), v.begin() + static_cast<std::ptrdiff_t>(axis));
    out.push_back(val);
    out.insert(out.end(), v.begin() + static_cast<std::ptrdiff_t>(axis), v.end());
    return out;
}

// ---------------------------------------------------------------------------
// IndexIter
// ---------------------------------------------------------------------------

IndexIter::IndexIter(const std::vector<std::size_t>& shape) : shape_(shape) {
    if (shape.empty()) {
        // Scalar: exactly one iteration, with an empty index.
        return;
    }
    current_.assign(shape.size(), 0);
    done_ = std::any_of(shape.begin(), shape.end(), [](std::size_t d) { return d == 0; });
}

void IndexIter::next() {
    if (done_) {
        return;
    }
    if (shape_.empty()) {
        done_ = true;
        return;
    }
    // Increment the last index, carrying over.
    for (std::size_t dim = shape_.size(); dim-- > 0;) {
        if (++current_[dim] < shape_[dim]) {
            return;
        }
        current_[dim] = 0;
        if (dim == 0) {
            done_ = true;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

std::size_t NDArray::numel_of(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (std::size_t d : shape) {
        n *= d;
    }
    return n;
}

NDArray NDArray::zeros(const std::vector<std::size_t>& shape) {
    return from_vec(std::vector<float>(numel_of(shape), 0.0f), shape);
}

NDArray NDArray::ones(const std::vector<std::size_t>& shape) {
    return from_vec(std::vector<float>(numel_of(shape), 1.0f), shape);
}

NDArray NDArray::from_vec(std::vector<float> data, const std::vector<std::size_t>& shape) {
    assert(data.size() == numel_of(shape) && "from_vec: data length != shape product");
    NDArray a;
    a.data = std::make_shared<std::vector<float>>(std::move(data));
    a.shape = shape;
    a.strides = c_strides(shape);
    a.offset = 0;
    return a;
}

NDArray NDArray::from_mat(const Mat& m) {
    NDArray a;
    a.data = std::make_shared<std::vector<float>>(m.data);
    a.shape = {m.rows, m.cols};
    a.strides = {m.cols, 1};
    a.offset = 0;
    return a;
}

Mat NDArray::into_mat() const {
    assert(ndim() == 2 && "into_mat: requires a 2-D array");
    const std::size_t rows = shape[0], cols = shape[1];
    std::vector<float> out;
    out.reserve(rows * cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            out.push_back(at({i, j}));
        }
    }
    return Mat(std::move(out), rows, cols);
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

std::size_t NDArray::flat_index(const std::vector<std::size_t>& idx) const {
    assert(idx.size() == ndim() && "flat_index: idx len != ndim");
    std::size_t f = offset;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        f += idx[i] * strides[i];
    }
    return f;
}

float NDArray::at(const std::vector<std::size_t>& idx) const { return (*data)[flat_index(idx)]; }

float& NDArray::at_mut(const std::vector<std::size_t>& idx) {
    const std::size_t fi = flat_index(idx);
    if (data.use_count() > 1) {
        // Copy-on-write: this view shares storage with someone else.
        data = std::make_shared<std::vector<float>>(*data);
    }
    return (*data)[fi];
}

// ---------------------------------------------------------------------------
// Contiguity
// ---------------------------------------------------------------------------

bool NDArray::is_contiguous() const { return offset == 0 && strides == c_strides(shape); }

NDArray NDArray::contiguous() const {
    if (is_contiguous()) {
        return *this;  // shares storage
    }
    std::vector<float> out;
    out.reserve(numel());
    for (IndexIter it(shape); it.valid(); it.next()) {
        out.push_back(at(it.index()));
    }
    return from_vec(std::move(out), shape);
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

NDArray NDArray::reshape(const std::vector<std::size_t>& new_shape) const {
    assert(numel_of(new_shape) == numel() && "reshape: element count changed");

    if (!is_contiguous()) {
        return contiguous().reshape(new_shape);
    }

    NDArray a;
    a.data = data;
    a.shape = new_shape;
    a.strides = c_strides(new_shape);
    a.offset = offset;
    return a;
}

NDArray NDArray::permute(const std::vector<std::size_t>& axes) const {
    assert(axes.size() == ndim() && "permute: axes len != ndim");

#ifndef NDEBUG
    std::vector<bool> seen(ndim(), false);
    for (std::size_t a : axes) {
        assert(a < ndim() && "permute: axis out of range");
        assert(!seen[a] && "permute: duplicate axis");
        seen[a] = true;
    }
#endif

    NDArray a;
    a.data = data;
    a.shape.reserve(axes.size());
    a.strides.reserve(axes.size());
    for (std::size_t ax : axes) {
        a.shape.push_back(shape[ax]);
        a.strides.push_back(strides[ax]);
    }
    a.offset = offset;
    return a;
}

NDArray NDArray::slice(std::size_t axis, std::size_t index) const {
    assert(axis < ndim() && "slice: axis out of range");
    assert(index < shape[axis] && "slice: index out of range");

    NDArray a;
    a.data = data;
    a.offset = offset + index * strides[axis];
    for (std::size_t i = 0; i < ndim(); ++i) {
        if (i != axis) {
            a.shape.push_back(shape[i]);
            a.strides.push_back(strides[i]);
        }
    }
    return a;
}

NDArray NDArray::expand_dims(std::size_t axis) const {
    assert(axis <= ndim() && "expand_dims: axis out of range");

    NDArray a;
    a.data = data;
    a.shape = shape;
    a.strides = strides;
    a.shape.insert(a.shape.begin() + static_cast<std::ptrdiff_t>(axis), 1);
    // A size-1 dimension is never stepped through, so its stride is irrelevant.
    a.strides.insert(a.strides.begin() + static_cast<std::ptrdiff_t>(axis), 0);
    a.offset = offset;
    return a;
}

// ---------------------------------------------------------------------------
// Elementwise operations
// ---------------------------------------------------------------------------

NDArray NDArray::add(const NDArray& other) const {
    assert(shape == other.shape && "add: shape mismatch");
    NDArray out = NDArray::zeros(shape);
    for (IndexIter it(shape); it.valid(); it.next()) {
        out.at_mut(it.index()) = at(it.index()) + other.at(it.index());
    }
    return out;
}

NDArray NDArray::mul(const NDArray& other) const {
    assert(shape == other.shape && "mul: shape mismatch");
    NDArray out = NDArray::zeros(shape);
    for (IndexIter it(shape); it.valid(); it.next()) {
        out.at_mut(it.index()) = at(it.index()) * other.at(it.index());
    }
    return out;
}

NDArray NDArray::scale(float s) const {
    NDArray out = NDArray::zeros(shape);
    for (IndexIter it(shape); it.valid(); it.next()) {
        out.at_mut(it.index()) = at(it.index()) * s;
    }
    return out;
}

NDArray NDArray::add_scalar(float s) const {
    NDArray out = NDArray::zeros(shape);
    for (IndexIter it(shape); it.valid(); it.next()) {
        out.at_mut(it.index()) = at(it.index()) + s;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

NDArray NDArray::reduce_sum(std::size_t axis) const {
    assert(axis < ndim() && "reduce_sum: axis out of range");

    std::vector<std::size_t> out_shape = shape;
    out_shape.erase(out_shape.begin() + static_cast<std::ptrdiff_t>(axis));
    NDArray out = NDArray::zeros(out_shape);

    for (IndexIter it(shape); it.valid(); it.next()) {
        std::vector<std::size_t> out_idx = it.index();
        out_idx.erase(out_idx.begin() + static_cast<std::ptrdiff_t>(axis));
        out.at_mut(out_idx) += at(it.index());
    }
    return out;
}

NDArray NDArray::softmax(std::size_t axis) const {
    assert(axis < ndim() && "softmax: axis out of range");

    NDArray out = clone_owned();
    const std::size_t n = shape[axis];

    // Every index except `axis` -- one softmax per fiber.
    std::vector<std::size_t> prefix_shape;
    for (std::size_t i = 0; i < ndim(); ++i) {
        if (i != axis) {
            prefix_shape.push_back(shape[i]);
        }
    }

    for (IndexIter it(prefix_shape); it.valid(); it.next()) {
        const auto& prefix = it.index();

        // Pass 1: row max, for numerical stability.
        float max_v = -std::numeric_limits<float>::infinity();
        for (std::size_t k = 0; k < n; ++k) {
            max_v = std::max(max_v, out.at(insert_at(prefix, axis, k)));
        }

        // Pass 2: exp(x - max), accumulating the sum.
        float sum = 0.0f;
        for (std::size_t k = 0; k < n; ++k) {
            const auto idx = insert_at(prefix, axis, k);
            const float e = std::exp(out.at(idx) - max_v);
            out.at_mut(idx) = e;
            sum += e;
        }

        // Pass 3: normalize.
        for (std::size_t k = 0; k < n; ++k) {
            out.at_mut(insert_at(prefix, axis, k)) /= sum;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Batched matrix multiply
// ---------------------------------------------------------------------------

NDArray NDArray::bmm(const NDArray& other) const {
    const std::size_t nd = ndim();
    assert(nd >= 2 && "bmm: requires at least 2-D arrays");
    assert(other.ndim() == nd && "bmm: ndim mismatch");

    const std::size_t m = shape[nd - 2];
    const std::size_t k = shape[nd - 1];
    const std::size_t n = other.shape[nd - 1];
    assert(k == other.shape[nd - 2] && "bmm: inner dims do not match");

    const std::vector<std::size_t> batch_shape(shape.begin(),
                                               shape.begin() + static_cast<std::ptrdiff_t>(nd - 2));
    assert(std::equal(batch_shape.begin(), batch_shape.end(), other.shape.begin()) &&
           "bmm: batch shape mismatch");

    std::vector<std::size_t> out_shape = batch_shape;
    out_shape.push_back(m);
    out_shape.push_back(n);

#if RT_FEATURE_METAL
    // Metal batched path: extract each batch slice as a contiguous Mat, dispatch
    // all slices in one GPU call, then pack the results back. Only worth the
    // dispatch when the per-slice problem is large enough.
    {
        constexpr std::size_t kMetalBmmThreshold = 32768;
        if (m * k * n >= kMetalBmmThreshold) {
            // Make both arrays contiguous so the flat slices are meaningful.
            const NDArray a_cont = contiguous();
            const NDArray b_cont = other.contiguous();

            std::vector<std::pair<Mat, Mat>> pairs;
            for (IndexIter it(batch_shape); it.valid(); it.next()) {
                const auto& bi = it.index();
                std::size_t a_off = a_cont.offset, b_off = b_cont.offset;
                for (std::size_t i = 0; i < bi.size(); ++i) {
                    a_off += bi[i] * a_cont.strides[i];
                    b_off += bi[i] * b_cont.strides[i];
                }
                pairs.emplace_back(
                    Mat(std::vector<float>(a_cont.data->begin() + static_cast<std::ptrdiff_t>(a_off),
                                           a_cont.data->begin() +
                                               static_cast<std::ptrdiff_t>(a_off + m * k)),
                        m, k),
                    Mat(std::vector<float>(b_cont.data->begin() + static_cast<std::ptrdiff_t>(b_off),
                                           b_cont.data->begin() +
                                               static_cast<std::ptrdiff_t>(b_off + k * n)),
                        k, n));
            }

            const std::vector<Mat> results = metal_matmul_batched(pairs);

            std::vector<float> flat;
            flat.reserve(numel_of(out_shape));
            for (const Mat& r : results) {
                flat.insert(flat.end(), r.data.begin(), r.data.end());
            }
            return from_vec(std::move(flat), out_shape);
        }
    }
#endif

    // CPU fallback: scalar loop over every (batch_idx, i, p, j).
    NDArray out = NDArray::zeros(out_shape);

    for (IndexIter it(batch_shape); it.valid(); it.next()) {
        const auto& batch_idx = it.index();
        std::vector<std::size_t> a_idx = batch_idx, b_idx = batch_idx, o_idx = batch_idx;
        a_idx.resize(nd);
        b_idx.resize(nd);
        o_idx.resize(nd);

        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t p = 0; p < k; ++p) {
                a_idx[nd - 2] = i;
                a_idx[nd - 1] = p;
                const float a_val = at(a_idx);

                for (std::size_t j = 0; j < n; ++j) {
                    b_idx[nd - 2] = p;
                    b_idx[nd - 1] = j;
                    o_idx[nd - 2] = i;
                    o_idx[nd - 1] = j;
                    out.at_mut(o_idx) += a_val * other.at(b_idx);
                }
            }
        }
    }
    return out;
}

}  // namespace rt

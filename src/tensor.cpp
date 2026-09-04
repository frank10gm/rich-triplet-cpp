#include "rt/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

namespace rt {

namespace {
[[nodiscard]] std::size_t shape_product(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (std::size_t d : shape) {
        n *= d;
    }
    return n;
}
}  // namespace

Tensor::Tensor(std::vector<float> d, std::vector<std::size_t> s)
    : data(std::move(d)), shape(std::move(s)) {
    assert(data.size() == shape_product(shape) && "data length does not match shape");
    strides = compute_strides(shape);
}

std::vector<std::size_t> Tensor::compute_strides(const std::vector<std::size_t>& shape) {
    // strides[last] = 1, and each earlier stride is the next one times the
    // next dimension's size.
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

Tensor Tensor::zeros(std::vector<std::size_t> shape) {
    const std::size_t n = shape_product(shape);
    return Tensor(std::vector<float>(n, 0.0f), std::move(shape));
}

Tensor Tensor::ones(std::vector<std::size_t> shape) {
    const std::size_t n = shape_product(shape);
    return Tensor(std::vector<float>(n, 1.0f), std::move(shape));
}

Tensor Tensor::arange(std::size_t n) {
    std::vector<float> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i);
    }
    return Tensor(std::move(data), {n});
}

std::size_t Tensor::flat_index(const std::vector<std::size_t>& indices) const {
    assert(indices.size() == ndim() && "wrong number of indices");
    std::size_t flat = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        flat += indices[i] * strides[i];
    }
    return flat;
}

float Tensor::get(const std::vector<std::size_t>& indices) const {
    return data[flat_index(indices)];
}

void Tensor::set(const std::vector<std::size_t>& indices, float value) {
    data[flat_index(indices)] = value;
}

namespace {
/// Element-wise binary op with a shape check.
template <typename Op>
[[nodiscard]] Tensor elementwise(const Tensor& a, const Tensor& b, Op op, const char* what) {
    assert(a.shape == b.shape && what);
    (void)what;
    std::vector<float> out(a.data.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = op(a.data[i], b.data[i]);
    }
    return Tensor(std::move(out), a.shape);
}
}  // namespace

Tensor Tensor::add(const Tensor& other) const {
    return elementwise(*this, other, [](float x, float y) { return x + y; },
                       "shape mismatch for add");
}

Tensor Tensor::sub(const Tensor& other) const {
    return elementwise(*this, other, [](float x, float y) { return x - y; },
                       "shape mismatch for sub");
}

Tensor Tensor::mul(const Tensor& other) const {
    return elementwise(*this, other, [](float x, float y) { return x * y; },
                       "shape mismatch for mul");
}

Tensor Tensor::scale(float s) const {
    return map([s](float x) { return x * s; });
}

float Tensor::sum_all() const {
    float acc = 0.0f;
    for (float x : data) {
        acc += x;
    }
    return acc;
}

namespace {
/// Shape with the last dimension dropped; a fully collapsed result becomes [1].
[[nodiscard]] std::vector<std::size_t> drop_last_dim(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> out(shape.begin(), shape.end() - 1);
    if (out.empty()) {
        out.push_back(1);
    }
    return out;
}

/// Reduce each contiguous run of `last` elements with `reduce`.
template <typename Reduce>
[[nodiscard]] Tensor reduce_last_dim(const Tensor& t, Reduce reduce) {
    assert(t.ndim() >= 1);
    const std::size_t last = t.shape[t.ndim() - 1];
    const std::size_t outer = t.data.size() / last;

    std::vector<float> out(outer);
    for (std::size_t i = 0; i < outer; ++i) {
        out[i] = reduce(t.data.data() + i * last, last);
    }
    return Tensor(std::move(out), drop_last_dim(t.shape));
}
}  // namespace

Tensor Tensor::sum_last_dim() const {
    return reduce_last_dim(*this, [](const float* row, std::size_t n) {
        float acc = 0.0f;
        for (std::size_t j = 0; j < n; ++j) {
            acc += row[j];
        }
        return acc;
    });
}

Tensor Tensor::max_last_dim() const {
    return reduce_last_dim(*this, [](const float* row, std::size_t n) {
        float best = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < n; ++j) {
            best = std::max(best, row[j]);
        }
        return best;
    });
}

Tensor Tensor::matmul(const Tensor& other) const {
    assert(ndim() == 2 && other.ndim() == 2 && "matmul requires 2-D tensors");
    const std::size_t m = shape[0];
    const std::size_t k = shape[1];
    const std::size_t n = other.shape[1];
    assert(k == other.shape[0] && "matmul inner dimensions must match");

    std::vector<float> result(m * n, 0.0f);
    // i over rows of A, p over the shared dimension, j over columns of B:
    // holding a_val across the inner loop keeps B's access sequential.
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const float a_val = data[i * k + p];
            for (std::size_t j = 0; j < n; ++j) {
                result[i * n + j] += a_val * other.data[p * n + j];
            }
        }
    }
    return Tensor(std::move(result), {m, n});
}

Tensor Tensor::transpose() const {
    assert(ndim() == 2 && "transpose requires a 2-D tensor");
    const std::size_t m = shape[0];
    const std::size_t n = shape[1];

    std::vector<float> result(m * n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[j * m + i] = data[i * n + j];
        }
    }
    return Tensor(std::move(result), {n, m});
}

Tensor Tensor::reshape(std::vector<std::size_t> new_shape) const {
    assert(numel() == shape_product(new_shape) && "reshape: element count must stay the same");
    return Tensor(data, std::move(new_shape));
}

void Tensor::print_info(const std::string& name) const {
    std::printf("%s: shape=[", name.c_str());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        std::printf("%zu%s", shape[i], i + 1 < shape.size() ? ", " : "");
    }
    std::printf("], numel=%zu, first_few=[", numel());
    const std::size_t show = std::min<std::size_t>(data.size(), 6);
    for (std::size_t i = 0; i < show; ++i) {
        std::printf("%g%s", static_cast<double>(data[i]), i + 1 < show ? ", " : "");
    }
    std::printf("]\n");
}

Tensor softmax(const Tensor& x) {
    const Tensor max = x.max_last_dim();

    const std::size_t last_dim = x.shape[x.ndim() - 1];
    const std::size_t outer = x.numel() / last_dim;

    // Shift by the row max, then exponentiate.
    std::vector<float> shifted = x.data;
    for (std::size_t i = 0; i < outer; ++i) {
        const float m = max.data[i];
        for (std::size_t j = 0; j < last_dim; ++j) {
            shifted[i * last_dim + j] -= m;
        }
    }
    const Tensor exp_x = Tensor(std::move(shifted), x.shape).map([](float v) {
        return std::exp(v);
    });

    // Normalize each row by its sum.
    const Tensor sum_exp = exp_x.sum_last_dim();
    std::vector<float> result = exp_x.data;
    for (std::size_t i = 0; i < outer; ++i) {
        const float s = sum_exp.data[i];
        for (std::size_t j = 0; j < last_dim; ++j) {
            result[i * last_dim + j] /= s;
        }
    }

    return Tensor(std::move(result), x.shape);
}

}  // namespace rt

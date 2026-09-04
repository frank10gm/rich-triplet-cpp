#pragma once

#include <cmath>
#include <cstddef>

#include "rt/mat.hpp"

namespace rt::testing {

inline bool approx(float a, float b) { return std::fabs(a - b) < 1e-3f; }

/// Numerical gradient check: perturb each element of `mat` by h, measure the
/// change in `f`, and form the central difference.
template <typename F>
[[nodiscard]] inline Mat numerical_grad(const F& f, const Mat& mat) {
    constexpr float h = 1e-3f;
    return Mat::from_fn(mat.rows, mat.cols, [&](std::size_t r, std::size_t c) {
        Mat plus = mat;
        plus.at_mut(r, c) += h;
        Mat minus = mat;
        minus.at_mut(r, c) -= h;
        return (f(plus) - f(minus)) / (2.0f * h);
    });
}

}  // namespace rt::testing

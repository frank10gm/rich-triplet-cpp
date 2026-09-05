#include "rt/qlinear.hpp"

#include <cassert>
#include <cstddef>
#include <utility>

#include "rt/mat.hpp"
#include "rt/nn2.hpp"

namespace rt {

QLinear QLinear::from_f32(Mat weight, std::vector<float> b) {
    QLinear l;
    l.out_features = weight.rows;
    l.in_features = weight.cols;
    l.f32 = std::move(weight);
    l.bias = std::move(b);
    return l;
}

QLinear QLinear::from_bf16(MatBf16 weight, std::vector<float> b) {
    QLinear l;
    l.out_features = weight.rows;
    l.in_features = weight.cols;
    l.bf16 = std::move(weight);
    l.bias = std::move(b);
    return l;
}

QLinear QLinear::from_q4k(Q4KMat weight, std::size_t out_features, std::size_t in_features,
                          std::vector<float> b) {
    QLinear l;
    l.out_features = out_features;
    l.in_features = in_features;
    l.q4k = std::move(weight);
    l.bias = std::move(b);
    return l;
}

bool QLinear::loaded() const { return q4k.has_value() || bf16.has_value() || f32.rows > 0; }

void add_bias_rows(Mat& x, std::span<const float> b) {
    if (b.empty()) {
        return;
    }
    assert(b.size() == x.cols && "QLinear: bias length != out_features");
    for (std::size_t r = 0; r < x.rows; ++r) {
        float* row = x.row_mut(r).data();
        for (std::size_t c = 0; c < x.cols; ++c) {
            row[c] += b[c];
        }
    }
}

Mat QLinear::forward(const Mat& x) const {
    assert(x.cols == in_features && "QLinear: input width != in_features");

    // Priority matches the memory cost of the format: whichever compact form
    // was loaded is the one that exists, and f32 is the fallback.
    Mat out = q4k.has_value()   ? q4k->matmul_q4k_t(x)
              : bf16.has_value() ? bf16->matmul_by_t(x)
                                 : x.matmul_bt(f32);
    add_bias_rows(out, bias);
    return out;
}

std::size_t QLinear::size_bytes() const {
    if (q4k.has_value()) {
        return q4k->size_bytes();
    }
    if (bf16.has_value()) {
        return bf16->size_bytes();
    }
    return f32.numel() * sizeof(float);
}

void QLinear::free_weight() {
    q4k.reset();
    bf16.reset();
    mark_pages_reusable(f32.data);
    f32 = Mat();
}

}  // namespace rt

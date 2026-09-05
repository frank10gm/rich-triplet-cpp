#pragma once

// =============================================================================
// QLinear -- an inference-only linear layer over whichever weight format loaded
// =============================================================================
//
// `Linear2` in `nn2.hpp` is the trainable layer: it carries `TensorNode`
// weights so gradients flow, and inference paths reach it by wrapping their
// activations in `TensorNode::leaf` and unwrapping the result. That works, and
// it is what the language models do -- but it also builds an autodiff graph
// that inference never walks, and some of those nodes hold reference cycles
// that plain refcounting cannot collect. `transformer6.cpp` carries the scars.
//
// The diffusion stack never trains, so it uses this instead: one weight, three
// possible storages, `Mat` in and `Mat` out, no graph at any point.
//
//   f32   -- Mat::matmul_bt, the VAE and anything small
//   BF16  -- MatBf16::matmul_by_t, half the memory and lossless
//   Q4_K  -- Q4KMat::matmul_q4k_t, what a 12B transformer has to use
//
// The weight is always [out_features, in_features] -- PyTorch's `Linear.weight`
// layout, so loading is a wrap -- and the multiply is always against its
// transpose.

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "rt/mat.hpp"
#include "rt/quant.hpp"
#include "rt/result.hpp"

namespace rt {

class QLinear {
   public:
    std::size_t in_features = 0;
    std::size_t out_features = 0;

    /// Exactly one of these is populated once a weight is loaded.
    Mat f32;
    std::optional<MatBf16> bf16;
    std::optional<Q4KMat> q4k;

    /// Empty means no bias, which is the common case in this stack: T5 has no
    /// biases at all, and FLUX has them only on its projections.
    std::vector<float> bias;

    QLinear() = default;

    [[nodiscard]] static QLinear from_f32(Mat weight, std::vector<float> bias = {});
    [[nodiscard]] static QLinear from_bf16(MatBf16 weight, std::vector<float> bias = {});
    [[nodiscard]] static QLinear from_q4k(Q4KMat weight, std::size_t out_features,
                                          std::size_t in_features, std::vector<float> bias = {});

    /// True once a weight of any format is present.
    [[nodiscard]] bool loaded() const;

    /// `x [T, in] -> [T, out]`, plus the bias if there is one.
    [[nodiscard]] Mat forward(const Mat& x) const;

    /// Bytes held by the weight, whichever format it is in.
    [[nodiscard]] std::size_t size_bytes() const;

    /// Release the weight. For after a GPU upload, which holds its own copy.
    void free_weight();
};

/// Add a row vector to every row of `x`, in place. Empty `b` is a no-op.
void add_bias_rows(Mat& x, std::span<const float> b);

}  // namespace rt

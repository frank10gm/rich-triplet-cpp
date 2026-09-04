#include "rt/nn2.hpp"

#include <cassert>
#include <cmath>

#if defined(__APPLE__)
#include <sys/mman.h>
#endif

namespace rt {

// =============================================================================
// Memory management helpers
// =============================================================================

void mark_pages_reusable(const void* data, std::size_t bytes) {
#if defined(__APPLE__)
    constexpr std::uintptr_t PAGE_SIZE = 16384;  // Apple Silicon
    const auto ptr = reinterpret_cast<std::uintptr_t>(data);
    // Round the start up and the end down, so only whole pages are marked.
    const std::uintptr_t start = (ptr + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    const std::uintptr_t end = (ptr + bytes) & ~(PAGE_SIZE - 1);
    if (end > start) {
        ::madvise(reinterpret_cast<void*>(start), end - start, MADV_FREE_REUSABLE);
    }
#else
    (void)data;
    (void)bytes;
#endif
}

// =============================================================================
// Trainable
// =============================================================================

TensorNode Trainable::loss_batch_tokens(
    const std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>>& batch) const {
    const std::size_t b = batch.size();
    assert(b > 0 && "loss_batch_tokens: empty batch");

    // Forward + backward per sequence, accumulating gradients scaled by 1/B.
    float total_loss = 0.0f;
    for (const auto& [inp, tgt] : batch) {
        const TensorNode loss_node = loss_tokens(inp, tgt);
        total_loss += loss_node.data().at(0, 0);

        // Scaling the upstream gradient by 1/B makes the accumulated gradient
        // across all B sequences equal the mean-batch gradient.
        loss_node.set_grad(Mat({1.0f / static_cast<float>(b)}, 1, 1));
        loss_node.backward();
    }

    return TensorNode::leaf(Mat({total_loss / static_cast<float>(b)}, 1, 1));
}

// =============================================================================
// Linear2
// =============================================================================

Linear2::Linear2(std::size_t in_features, std::size_t out_features, InitRng& rng)
    : weight(TensorNode::leaf(
          Mat(rng.normal_vec(out_features * in_features, 0.02f), out_features, in_features))),
      bias(TensorNode::leaf(Mat::zeros(1, out_features))),
      in_features(in_features),
      out_features(out_features) {}

Linear2 Linear2::new_no_bias(std::size_t in_features, std::size_t out_features, InitRng& rng) {
    // The bias is allocated but stays zero and is never updated.
    return Linear2(in_features, out_features, rng);
}

Linear2 Linear2::new_no_bias_zeros(std::size_t in_features, std::size_t out_features) {
    Linear2 l;
    l.in_features = in_features;
    l.out_features = out_features;
    return l;
}

void Linear2::quantize() { q4_weight = Q4Mat::quantize(weight.data()); }

void Linear2::quantize_and_free_f32() {
    q4_weight = Q4Mat::quantize(weight.data());
    // Drop the f32 weight to reclaim ~4x the memory.
    weight.set_data(Mat::zeros(0, 0));
}

void Linear2::quantize_bf16_and_free() {
    if (bf16_weight) {
        q4_weight = Q4Mat::quantize(bf16_weight->to_f32());
        bf16_weight.reset();
        weight.set_data(Mat::zeros(0, 0));
    } else {
        quantize_and_free_f32();
    }
}

void Linear2::quantize_bf16_to_q4k() {
    if (bf16_weight) {
        q4k_weight = Q4KMat::quantize(bf16_weight->to_f32());
        bf16_weight.reset();
        weight.set_data(Mat::zeros(0, 0));
    }
}

void Linear2::clear_weight_data() {
    if (q4k_weight) {
        mark_pages_reusable(q4k_weight->blocks);
    }
    q4k_weight.reset();
    if (bf16_weight) {
        // Only madvise when this is the last reference to the shared bits.
        if (bf16_weight->data.use_count() == 1) {
            mark_pages_reusable(*bf16_weight->data);
        }
    }
    bf16_weight.reset();
    q4_weight.reset();
    weight.set_data(Mat::zeros(0, 0));
}

void Linear2::load_bf16(std::vector<std::uint16_t> bits, std::size_t rows, std::size_t cols) {
    bf16_weight = MatBf16(std::move(bits), rows, cols);
    weight.set_data(Mat::zeros(0, 0));
}

void Linear2::load_bf16_shared(std::shared_ptr<std::vector<std::uint16_t>> data, std::size_t rows,
                               std::size_t cols) {
    bf16_weight = MatBf16(std::move(data), rows, cols);
    weight.set_data(Mat::zeros(0, 0));
}

namespace {
/// Broadcast-add a [1, cols] bias row into every row of `o`, in place.
void add_bias_rows(Mat& o, const Mat& b) {
    for (std::size_t r = 0; r < o.rows; ++r) {
        for (std::size_t c = 0; c < o.cols; ++c) {
            o.at_mut(r, c) += b.at(0, c);
        }
    }
}
}  // namespace

TensorNode Linear2::fused_linear(const TensorNode& input) const {
    const Mat& x = input.data();
    const Mat& b = bias.data();
    const bool has_bias = b.rows > 0 && b.cols > 0;

    Mat out_data;
    if (q4_weight) {
        assert(x.cols == q4_weight->cols && "Linear (q4): input cols != weight cols");
#if RT_FEATURE_BLAS
        out_data = q4_weight->matmul_q4_t_blas(x);
#else
        out_data = q4_weight->matmul_q4_t(x);
#endif
    } else if (q4k_weight) {
        assert(x.cols == q4k_weight->cols && "Linear (q4k): input cols != weight cols");
#if RT_FEATURE_BLAS
        out_data = q4k_weight->matmul_q4k_t_blas(x);
#else
        out_data = q4k_weight->matmul_q4k_t(x);
#endif
    } else if (bf16_weight) {
        assert(x.cols == bf16_weight->cols && "Linear (bf16): input cols != weight cols");
        out_data = bf16_weight->matmul_by_t(x);
    } else {
        const Mat& w = weight.data();
        assert(x.cols == w.cols && "Linear: input cols != weight cols");
        out_data = x.matmul(w.transpose());
    }
    if (has_bias) {
        add_bias_rows(out_data, b);
    }

    TensorNode out = TensorNode::leaf(std::move(out_data));

    // Quantized and BF16 weights are inference-only: the f32 weight has been
    // freed, and quantization is not differentiable anyway. Skipping the
    // backward closure for those paths saves one heap allocation and four
    // refcount bumps per linear call -- and decode runs ~180 linear layers per
    // token.
    if (q4_weight || q4k_weight || bf16_weight) {
        return out;
    }

    // f32 weight path: wire up the backward graph for training.
    TensorNode input_c = input, weight_c = weight, bias_c = bias, out_c = out;
    out.set_backward(
        [input_c, weight_c, bias_c, out_c] {
            const Mat& dout = out_c.grad();   // [T, out]
            const Mat& x = input_c.data();    // [T, in]
            const Mat& w = weight_c.data();   // [out, in]

            // dInput = dOut @ W          [T, in]
            input_c.grad_add(dout.matmul(w));
            // dW = dOut.T @ input        [out, in]
            weight_c.grad_add(dout.transpose().matmul(x));
            // d_bias = sum_rows(dOut)    [1, out]
            bias_c.grad_add(dout.sum_rows());
        },
        {input, weight, bias});
    return out;
}

// =============================================================================
// Mlp2 / SwiGluMlp2
// =============================================================================

std::vector<TensorNode> Mlp2::parameters() const {
    std::vector<TensorNode> p = fc1.parameters();
    const std::vector<TensorNode> p2 = fc2.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    return p;
}

TensorNode SwiGluMlp2::forward(const TensorNode& x) const {
    const TensorNode gate_pre = gate_proj.forward(x);
    // Clamp before SiLU when the limit is finite (GPT-OSS uses 7.0).
    const TensorNode gate = std::isfinite(swiglu_clamp)
                                ? gate_pre.clamp(-swiglu_clamp, swiglu_clamp).silu()
                                : gate_pre.silu();
    return down_proj.forward(gate.mul_elem_node(up_proj.forward(x)));
}

std::vector<TensorNode> SwiGluMlp2::parameters() const {
    std::vector<TensorNode> p = gate_proj.parameters();
    for (const auto* proj : {&up_proj, &down_proj}) {
        const std::vector<TensorNode> q = proj->parameters();
        p.insert(p.end(), q.begin(), q.end());
    }
    return p;
}

// =============================================================================
// Dropout2
// =============================================================================

Dropout2::Dropout2(float p) : p(p) { assert(p >= 0.0f && p < 1.0f && "dropout p must be in [0, 1)"); }

TensorNode Dropout2::forward(const TensorNode& x, bool training) const {
    if (!training || p == 0.0f) {
        return x;
    }

    const float scale = 1.0f / (1.0f - p);

    // One LCG draw per element, mixing the call counter with the position.
    const std::uint64_t seed0 = seed_.fetch_add(1, std::memory_order_relaxed);
    const Mat& xd = x.data();
    const Mat mask = Mat::from_fn(xd.rows, xd.cols, [&](std::size_t r, std::size_t c) {
        const std::uint64_t s =
            (seed0 + static_cast<std::uint64_t>(r * xd.cols + c)) * 6364136223846793005ull +
            1442695040888963407ull;
        const float u = static_cast<float>(s >> 33) / static_cast<float>(1ull << 31);
        return u > p ? scale : 0.0f;
    });

    TensorNode out = TensorNode::leaf(Mat::from_fn(
        xd.rows, xd.cols,
        [&](std::size_t r, std::size_t c) { return xd.at(r, c) * mask.at(r, c); }));

    TensorNode x_c = x, out_c = out;
    out.set_backward(
        [x_c, out_c, mask] {
            // dx = dout * mask, reusing the forward mask.
            const Mat& dout = out_c.grad();
            Mat dx = x_c.grad();
            for (std::size_t i = 0; i < dx.data.size(); ++i) {
                dx.data[i] += dout.data[i] * mask.data[i];
            }
            x_c.set_grad(std::move(dx));
            x_c.call_backward_fn();
        },
        {x});

    return out;
}

}  // namespace rt

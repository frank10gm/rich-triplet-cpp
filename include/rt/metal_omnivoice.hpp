#pragma once

// =============================================================================
// Metal full-graph forward pass (OmniVoice)
// =============================================================================
//
// The first Metal engine here for a text-to-speech model, and the first for a
// *prefill* shape rather than a decode one. The two Gemma and Qwen engines
// exist because a decode step is a chain of GEMVs, memory-bound and too small
// to dispatch one at a time. This one exists for the opposite reason:
// OmniVoice has no KV cache, so every step is a full-sequence forward pass over
// a few hundred positions -- large GEMMs, compute-bound, and exactly the shape
// a GPU is built for.
//
// ## Why it is worth it, measured rather than assumed
//
// Accelerate's sgemm reaches about 1.0 TFLOP/s on an M3 Pro for these shapes,
// which is a high bar -- the tiled matmul in `metal_ops.mm` manages 250 GF/s
// and would be a large regression. What clears the bar is `simdgroup_matrix`:
// a 64x64 tile per threadgroup built from sixteen 8x8 accumulators reaches
// 1.2-2.3 TFLOP/s depending on shape, since the matrix units do the work the
// tiled kernel does with scalar multiply-adds.
//
// The GEMMs are only 40% of the CPU runtime, though, so moving them alone
// would cap the win near 1.3x. The rest goes to three things the GPU gets for
// free:
//
//   * **BF16 weights are read natively.** The CPU path dequantizes every
//     weight to an f32 scratch buffer on every call -- 12% of its runtime, and
//     437 million conversions per forward pass. A Metal kernel reads `bfloat`
//     as an operand.
//   * **No zero-fill.** `Mat::zeros` costs 8% of the CPU runtime in `bzero`
//     for buffers that sgemm immediately overwrites. GPU scratch is allocated
//     once and reused.
//   * **The elementwise work moves too.** RMSNorm, RoPE, SiLU, the attention
//     softmax and the residual adds are another quarter of the runtime, all of
//     it memory-bound scalar loops.
//
// ## Structure
//
// One command buffer per forward pass: 28 layers of 11 dispatches plus 3, so
// about 311. Within a single compute encoder on Apple Silicon dispatches run in
// order and each one's writes are visible to the next through the unified L2,
// so there are no barriers.
//
// Sequence length is padded up to a multiple of 64 so the GEMM tiles divide
// evenly and need no bounds checks in their inner loop. The padding rows are
// zeroed once and carry through harmlessly -- every operator except attention
// is row-independent, and attention is dispatched over the true length.

#include <cstddef>
#include <memory>
#include <vector>

#include "rt/transformer6.hpp"

namespace rt {

/// Owns every GPU resource a forward pass needs: uploaded weights, activation
/// scratch, and the compiled pipelines.
class MetalOmniContext final : public OmniForward {
   public:
    /// The longest sequence the attention kernel can hold -- it keeps one
    /// query's scores in threadgroup memory. Forty seconds of audio plus its
    /// prompt, so not a limit generation reaches.
    static constexpr std::size_t kMaxTokens = 1024;

    /// Upload `lm`'s weights and allocate scratch for sequences up to
    /// `max_tokens`.
    ///
    /// **Consumes the model's projection weights.** Each is freed as it is
    /// uploaded, so peak memory never holds both copies -- and `lm` cannot run
    /// a forward pass of its own afterwards. It is still needed for its config,
    /// its embedding tables and its prompt layout.
    ///
    /// The embedding tables stay on the CPU: the text table alone is 310 MB and
    /// only `T` of its rows are ever read, so gathering on the CPU and
    /// uploading the result is cheaper than holding it twice.
    [[nodiscard]] static std::unique_ptr<MetalOmniContext> create(OmniLm& lm,
                                                                  std::size_t max_tokens);

    ~MetalOmniContext() override;

    MetalOmniContext(const MetalOmniContext&) = delete;
    MetalOmniContext& operator=(const MetalOmniContext&) = delete;

    /// Run the whole sequence and return audio logits, `[T, codebooks * vocab]`
    /// -- the same contract as `OmniLm::forward`.
    [[nodiscard]] Result<Mat> forward(const std::vector<OmniToken>& tokens) const override;

    /// Longest sequence the allocated scratch can take.
    [[nodiscard]] std::size_t max_tokens() const;

    /// Bytes held in GPU buffers.
    [[nodiscard]] std::size_t buffer_bytes() const;

   private:
    struct Impl;
    explicit MetalOmniContext(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace rt

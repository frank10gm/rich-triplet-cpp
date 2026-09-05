#pragma once

// =============================================================================
// MetalFluxContext -- the FLUX forward pass as one GPU command buffer
// =============================================================================
//
// The third full-graph engine in this project, after Gemma 3 and OmniVoice, and
// the one with the most work per dispatch: 57 blocks of a 12B transformer over
// four thousand tokens, encoded once and submitted once per denoising step.
//
// ## Weights stay quantized; activations do not
//
// A 12B model is 6.7 GB as Q4_K and 24 GB as bfloat, so the weights live on the
// GPU in their packed form and cannot be widened at rest. They are widened per
// use instead: before each GEMM, one dispatch dequantizes that weight into a
// shared bfloat scratch buffer, and the GEMM reads it from there.
//
// This sounds wasteful and is not. The largest weight is a single-stream
// block's fused `linear1` at 3072 -> 21504 -- 66 M elements to widen, against
// 575 GFLOP of GEMM to follow. The scratch buffer is 132 MB and is reused by
// every projection in the model.
//
// ## Shape constraints
//
// The GEMM kernel is a 64x64 register-blocked `simdgroup_matrix` tile with no
// bounds checks, so every projection width must be a multiple of 64 and every
// contracted dimension a multiple of 8. FLUX satisfies all of them naturally --
// 3072, 9216, 12288, 15360, 18432, 21504 -- and `create` checks rather than
// assumes, because a config that violates one produces silent corruption in
// the last partial tile rather than a fault.
//
// The token count is padded up to 64 with zero rows. Every operator except
// attention is row-independent, so the pad rows are harmless, and attention is
// dispatched over the true length.

#include <cstddef>
#include <memory>
#include <span>
#include <string>

#include "rt/flux.hpp"
#include "rt/mat.hpp"
#include "rt/result.hpp"

namespace rt {

class MetalFluxContext {
   public:
    /// Upload `model`'s weights and allocate scratch for a latent of at most
    /// `max_lat_h x max_lat_w` with at most `max_text_tokens` of prompt.
    ///
    /// **Consumes the model's weights.** Each is freed as it is uploaded, so
    /// peak memory never holds both copies -- and `model` cannot run a forward
    /// pass of its own afterwards. Its config is still read.
    [[nodiscard]] static Result<std::unique_ptr<MetalFluxContext>> create(
        FluxModel& model, std::size_t max_lat_h, std::size_t max_lat_w,
        std::size_t max_text_tokens);

    ~MetalFluxContext();

    MetalFluxContext(const MetalFluxContext&) = delete;
    MetalFluxContext& operator=(const MetalFluxContext&) = delete;

    /// One velocity prediction -- the same contract as `FluxModel::forward`.
    [[nodiscard]] Result<Mat> forward(const Mat& latent, std::size_t lat_h, std::size_t lat_w,
                                      const Mat& context, std::span<const float> pooled,
                                      float timestep, float guidance = 0.0f) const;

    /// Bytes held in GPU buffers.
    [[nodiscard]] std::size_t device_bytes() const;

    /// The largest latent the allocated scratch can take.
    [[nodiscard]] std::size_t max_latent_tokens() const;

    struct Impl;

   private:
    MetalFluxContext();
    std::unique_ptr<Impl> impl_;
};

/// Sample with the GPU engine. The CPU-side `flux_sample` with the loop body
/// swapped, so the schedule and the Euler step stay in one place.
[[nodiscard]] Result<Mat> flux_sample_metal(const MetalFluxContext& ctx, const FluxConfig& cfg,
                                            const Mat& context, std::span<const float> pooled,
                                            const FluxSampleParams& params);

}  // namespace rt

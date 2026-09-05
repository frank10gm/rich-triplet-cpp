#pragma once

// =============================================================================
// FLUX.1 -- a rectified-flow diffusion transformer
// =============================================================================
//
// The model that turns a prompt into an image, in the sense that it is where
// almost all the parameters and almost all the time go. It does not emit
// pixels: it emits a velocity field over a 16-channel latent, which an Euler
// solver integrates and `vae.hpp` decodes.
//
// 11.9B parameters, hidden width 3072, 24 heads of 128.
//
// ## The two block types
//
// **19 double-stream blocks.** Image and text are separate residual streams
// with separate weights, but a single joint attention over the concatenation of
// both. That is the "MM" in MMDiT: the modalities keep their own parameters and
// share only the attention.
//
// **38 single-stream blocks.** One stream over the concatenation, and -- the
// unusual part -- attention and MLP are computed *in parallel* from the same
// modulated input and concatenated before a single output projection, rather
// than run in sequence. It is the ViT-22B trick, and it makes one fused
// `linear1` do the work of a QKV projection and an MLP up-projection at once.
//
// ## Conditioning does not enter through cross-attention
//
// There is no cross-attention anywhere. The prompt enters as *tokens* in the
// text stream, and the timestep and the pooled CLIP vector enter through
// adaptive layer norm: a per-block linear map from the conditioning vector to
// a shift, a scale and a gate for each sub-layer. Every LayerNorm in the model
// is affine-free precisely because the affine part comes from there.
//
// ## Three-axis RoPE
//
// Position is `(t, h, w)` with per-axis head dimensions [16, 56, 56], summing
// to the 128 of a head. Image tokens carry their patch-grid coordinates; text
// tokens carry all zeros, which makes their rotation the identity. The pairs
// are adjacent -- `(x[2i], x[2i+1])` -- not split-half.
//
// ## schnell against dev
//
// schnell is timestep-distilled to four steps *and* guidance-distilled, so
// there is no classifier-free guidance and one forward pass per step. It also
// has no `guidance_in` embedding at all, so loading dev weights into a schnell
// config fails on a missing tensor -- which is the outcome worth having, since
// the reverse (running dev without guidance) silently produces washed-out
// images.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rt/mat.hpp"
#include "rt/qlinear.hpp"
#include "rt/result.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

struct FluxConfig {
    std::size_t in_channels = 16;
    std::size_t hidden_size = 3072;
    std::size_t n_heads = 24;
    std::size_t n_double_blocks = 19;
    std::size_t n_single_blocks = 38;
    float mlp_ratio = 4.0f;
    /// Width of the T5 sequence entering `txt_in`.
    std::size_t context_dim = 4096;
    /// Width of the pooled CLIP vector entering `vector_in`.
    std::size_t pooled_dim = 768;
    /// Per-axis RoPE dimensions over (t, h, w). Must sum to `head_dim()`.
    std::vector<std::size_t> axes_dim{16, 56, 56};
    float rope_theta = 10000.0f;
    /// dev has a distilled-guidance embedding; schnell does not.
    bool guidance_embed = false;
    /// Latent patch size. 2 everywhere in the FLUX family.
    std::size_t patch_size = 2;
    float qk_norm_eps = 1e-6f;
    float layer_norm_eps = 1e-6f;

    [[nodiscard]] static FluxConfig schnell();
    [[nodiscard]] static FluxConfig dev();

    [[nodiscard]] std::size_t head_dim() const { return hidden_size / n_heads; }
    /// `in_channels * patch_size^2` -- the width of one packed latent patch.
    [[nodiscard]] std::size_t patch_dim() const;
    [[nodiscard]] std::size_t mlp_hidden() const;
};

// =============================================================================
// Patching
// =============================================================================

/// Pack a latent into transformer tokens.
///
/// `z` is [lat_h * lat_w, channels] and the result is
/// [(lat_h/p) * (lat_w/p), channels * p * p], one row per patch.
///
/// The packing order within a patch is **channel-major**:
/// `c (h ph) (w pw) -> (h w) (c ph pw)`. Ordering it the other way is the kind
/// of mistake that survives review, because the model still trains its way to
/// something and the image comes out with a fine 2x2 scramble that reads as
/// noise rather than as a transposition.
[[nodiscard]] Mat flux_patchify(const Mat& z, std::size_t lat_h, std::size_t lat_w,
                                std::size_t patch);

/// The inverse of `flux_patchify`.
[[nodiscard]] Mat flux_unpatchify(const Mat& tokens, std::size_t lat_h, std::size_t lat_w,
                                  std::size_t channels, std::size_t patch);

/// Position ids for the packed tokens: `[n_tokens, 3]` holding (0, y, x).
[[nodiscard]] Mat flux_image_ids(std::size_t lat_h, std::size_t lat_w, std::size_t patch);

/// Check that the RoPE axes tile the head dimension in whole pairs.
///
/// Each axis rotates adjacent pairs inside its own slice, so an odd axis width
/// leaves one dimension of that slice unrotated and pushes every later axis off
/// its frequencies. The result is a model that runs, produces finite numbers,
/// and attends to the wrong positions.
[[nodiscard]] Result<void> flux_check_axes(const FluxConfig& cfg);

// =============================================================================
// Embeddings
// =============================================================================

/// Sinusoidal timestep embedding, **cosines first**.
///
/// `t` is scaled by `time_factor` (1000 in FLUX) before the frequencies are
/// applied. The concatenation order is `[cos, sin]`, which is the reverse of
/// the more common convention -- swapping it rotates every conditioning vector
/// by a quarter turn and the model produces coherent images of the wrong
/// timestep.
[[nodiscard]] std::vector<float> flux_timestep_embedding(float t, std::size_t dim,
                                                         float max_period = 10000.0f,
                                                         float time_factor = 1000.0f);

/// Rotate `x` in place by three-axis RoPE.
///
/// `x` is [T, n_heads * head_dim], `ids` is [T, 3]. Each axis owns a
/// contiguous slice of the head dimension, sized by `axes_dim`, and rotates
/// adjacent pairs within it.
void flux_apply_rope(Mat& x, const Mat& ids, std::size_t n_heads, std::size_t head_dim,
                     const std::vector<std::size_t>& axes_dim, float theta);

// =============================================================================
// Layers
// =============================================================================

/// One `(shift, scale, gate)` triple out of a modulation projection.
struct FluxModulation {
    std::vector<float> shift;
    std::vector<float> scale;
    std::vector<float> gate;
};

/// Per-head RMSNorm applied to Q and K before rotation.
struct FluxQkNorm {
    std::vector<float> query_scale;  // [head_dim]
    std::vector<float> key_scale;
};

struct FluxDoubleBlock {
    QLinear img_mod;  // hidden -> 6 * hidden
    QLinear img_qkv;  // hidden -> 3 * hidden
    QLinear img_proj;
    QLinear img_mlp_in;
    QLinear img_mlp_out;
    FluxQkNorm img_norm;

    QLinear txt_mod;
    QLinear txt_qkv;
    QLinear txt_proj;
    QLinear txt_mlp_in;
    QLinear txt_mlp_out;
    FluxQkNorm txt_norm;
};

struct FluxSingleBlock {
    QLinear modulation;  // hidden -> 3 * hidden
    /// hidden -> 3 * hidden + mlp_hidden, one projection doing QKV and the MLP
    /// up-projection together.
    QLinear linear1;
    /// hidden + mlp_hidden -> hidden
    QLinear linear2;
    FluxQkNorm norm;
};

// =============================================================================
// Model
// =============================================================================

class FluxModel {
   public:
    FluxConfig cfg;

    QLinear img_in;   // patch_dim -> hidden
    QLinear txt_in;   // context_dim -> hidden
    QLinear time_in_1;  // 256 -> hidden
    QLinear time_in_2;  // hidden -> hidden
    QLinear vector_in_1;  // pooled_dim -> hidden
    QLinear vector_in_2;
    /// dev only; empty on schnell.
    QLinear guidance_in_1;
    QLinear guidance_in_2;

    std::vector<FluxDoubleBlock> double_blocks;
    std::vector<FluxSingleBlock> single_blocks;

    QLinear final_mod;     // hidden -> 2 * hidden
    QLinear final_linear;  // hidden -> patch_dim

    /// Load from a GGUF checkpoint using the original FLUX tensor names, which
    /// is what the community Q4_K/Q5_K/Q8_0 conversions carry.
    [[nodiscard]] static Result<FluxModel> load_gguf(const std::string& path, FluxConfig cfg);

    /// One velocity prediction.
    ///
    /// `latent` is [lat_h * lat_w, in_channels] -- unpacked, as the sampler
    /// holds it. `context` is the T5 sequence [T_txt, context_dim] and `pooled`
    /// is the CLIP vector. `timestep` runs from 1 down to 0.
    ///
    /// Returns a velocity of the same shape as `latent`.
    [[nodiscard]] Result<Mat> forward(const Mat& latent, std::size_t lat_h, std::size_t lat_w,
                                      const Mat& context, std::span<const float> pooled,
                                      float timestep, float guidance = 0.0f) const;

    [[nodiscard]] std::size_t parameter_count() const;
    [[nodiscard]] std::size_t weight_bytes() const;
    void free_weights();
};

// =============================================================================
// Sampling
// =============================================================================

struct FluxSampleParams {
    std::size_t width = 1024;
    std::size_t height = 1024;
    std::size_t steps = 4;
    std::uint64_t seed = 0;
    /// Ignored unless the config has `guidance_embed`.
    float guidance = 3.5f;
    /// Timestep shift. schnell uses 1.0, which is no shift at all; dev shifts
    /// as a function of sequence length.
    float shift = 1.0f;
};

/// The rectified-flow schedule: `steps + 1` sigmas running from 1 down to 0.
///
/// With `shift == 1` this is a plain linear spacing, which is what schnell
/// wants. Larger values push the samples toward the noisy end, where a
/// many-step sampler needs the resolution.
[[nodiscard]] std::vector<float> flux_schedule(std::size_t steps, float shift);

/// Sample a latent by Euler integration of the velocity field.
///
/// Returns [lat_h * lat_w, in_channels], still in the model's scaling -- hand
/// it straight to `VaeDecoder::decode`, which applies the rescaling itself.
[[nodiscard]] Result<Mat> flux_sample(const FluxModel& model, const Mat& context,
                                      std::span<const float> pooled,
                                      const FluxSampleParams& params);

}  // namespace rt

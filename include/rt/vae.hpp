#pragma once

// =============================================================================
// AutoencoderKL -- the 16-channel image VAE, decoder only
// =============================================================================
//
// A diffusion transformer does not emit pixels. It emits latents, and something
// else has to turn those into an image. FLUX emits 16-channel latents at 1/8
// resolution, so this is the second half of that pipeline: latents in, RGB out.
//
// Only the decoder is implemented, for the same reason `snac.hpp` is decode
// only: text-to-image never runs the encoder.
//
// ## The stack
//
//   z [h/8 * w/8, 16]
//     -> conv_in            16 -> 512, 3x3
//     -> mid: Resnet(512), Attention(512), Resnet(512)
//     -> up 3: 3x Resnet(512 -> 512), upsample 2x
//     -> up 2: 3x Resnet(512 -> 512), upsample 2x
//     -> up 1: 3x Resnet(512 -> 256), upsample 2x
//     -> up 0: 3x Resnet(256 -> 128)
//     -> GroupNorm, SiLU, conv_out 128 -> 3, 3x3
//
// Note the ordering: diffusers stores the up blocks coarsest-last, so
// `decoder.up_blocks.0` is the *finest* level and runs last. Loading them in
// file order gives a decoder whose channel counts happen to line up for the
// first two levels and then fail an assertion on the third -- which is the good
// case, because reversing only the resnets and not the upsamplers produces a
// blurred image and no error at all.
//
// ## Latents are stored scaled
//
// The VAE was trained on a latent distribution the diffusion model does not
// use directly. Going back the other way needs both constants:
//
//   z = z_model / scaling_factor + shift_factor       (0.3611, 0.1159)
//
// SD 1.x and SDXL have a scaling factor and no shift, so the shift is the one
// that gets dropped when porting from older code. Dropping it does not break
// the image -- it desaturates it and adds a colour cast, which reads as a
// stylistic choice rather than as a bug.
//
// ## Memory
//
// The last upsampling level runs 128 channels at full resolution. At 1024x1024
// that is 537 MB for a single activation, and a residual unit holds three at
// once. `decode_tiled` splits the latent into overlapping tiles and blends the
// results, which caps the working set at the cost of some redundant
// convolution. At 512x512 the whole-image path is fine.

#include <cstddef>
#include <string>
#include <vector>

#include "rt/mat.hpp"
#include "rt/result.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

struct VaeConfig {
    /// Latent channels. 16 for the SD3/FLUX autoencoder, 4 for SD 1.x and SDXL.
    std::size_t latent_channels = 16;
    /// Channel width per level, **finest first**, matching diffusers'
    /// `block_out_channels`.
    std::vector<std::size_t> block_out_channels{128, 256, 512, 512};
    /// Residual blocks per level. The decoder uses `layers_per_block + 1`.
    std::size_t layers_per_block = 2;
    std::size_t norm_groups = 32;
    float norm_eps = 1e-6f;
    /// `z = z_model / scaling_factor + shift_factor`.
    float scaling_factor = 0.3611f;
    float shift_factor = 0.1159f;

    [[nodiscard]] static VaeConfig flux();

    /// Spatial factor between latent and pixel, `2^(levels - 1)`.
    [[nodiscard]] std::size_t downsample_factor() const;
};

// =============================================================================
// Layers
// =============================================================================

/// GroupNorm, SiLU, 3x3 conv, GroupNorm, SiLU, 3x3 conv, plus a residual.
///
/// The shortcut is a 1x1 convolution present **only** when the channel count
/// changes. Adding one unconditionally works numerically -- it would just be an
/// identity the loader has no weights for -- so the presence of the tensor is
/// what decides it.
struct VaeResnetBlock {
    std::size_t in_channels = 0;
    std::size_t out_channels = 0;

    std::vector<float> norm1_weight;
    std::vector<float> norm1_bias;
    Mat conv1_weight;  // [out, in * 9]
    std::vector<float> conv1_bias;

    std::vector<float> norm2_weight;
    std::vector<float> norm2_bias;
    Mat conv2_weight;  // [out, out * 9]
    std::vector<float> conv2_bias;

    /// Empty when `in_channels == out_channels`.
    Mat shortcut_weight;  // [out, in]
    std::vector<float> shortcut_bias;

    [[nodiscard]] Mat forward(const Mat& x, std::size_t h, std::size_t w,
                              const VaeConfig& cfg) const;
};

/// Single-head self-attention over the spatial positions.
///
/// In `conv2d.hpp`'s spatial-major layout the input is already one row per
/// position, so this is ordinary attention with no reshaping: GroupNorm, three
/// 1x1 projections, softmax over `h*w` keys at scale `1/sqrt(C)`, a 1x1 output
/// projection and a residual add.
///
/// diffusers stores the projections as `Linear` in current checkpoints and as
/// 1x1 `Conv2d` in older ones. Both flatten to the same [C, C] matrix, so the
/// loader accepts either name and this code does not care.
struct VaeAttentionBlock {
    std::size_t channels = 0;

    std::vector<float> norm_weight;
    std::vector<float> norm_bias;
    Mat q_weight;  // [C, C]
    std::vector<float> q_bias;
    Mat k_weight;
    std::vector<float> k_bias;
    Mat v_weight;
    std::vector<float> v_bias;
    Mat out_weight;
    std::vector<float> out_bias;

    [[nodiscard]] Mat forward(const Mat& x, const VaeConfig& cfg) const;
};

/// One decoder level: some residual blocks, then an optional 2x upsample
/// followed by a 3x3 convolution.
struct VaeUpBlock {
    std::vector<VaeResnetBlock> resnets;
    /// Empty on the finest level, which does not upsample.
    Mat upsample_weight;  // [C, C * 9]
    std::vector<float> upsample_bias;

    [[nodiscard]] bool upsamples() const { return upsample_weight.rows > 0; }
};

// =============================================================================
// Decoder
// =============================================================================

class VaeDecoder {
   public:
    VaeConfig cfg;

    Mat conv_in_weight;  // [C_mid, latent * 9]
    std::vector<float> conv_in_bias;

    VaeResnetBlock mid_resnet1;
    VaeAttentionBlock mid_attn;
    VaeResnetBlock mid_resnet2;

    /// Coarsest first -- the reverse of diffusers' storage order.
    std::vector<VaeUpBlock> up_blocks;

    std::vector<float> conv_out_norm_weight;
    std::vector<float> conv_out_norm_bias;
    Mat conv_out_weight;  // [3, C_fine * 9]
    std::vector<float> conv_out_bias;

    /// Load from a diffusers `ae.safetensors` / `diffusion_pytorch_model.safetensors`.
    ///
    /// Tensors are streamed one at a time rather than parsed in bulk: the file
    /// also holds the encoder, which is never used and is half its size.
    [[nodiscard]] static Result<VaeDecoder> load(const std::string& path, VaeConfig cfg);

    /// Decode a latent to pixels in roughly [-1, 1].
    ///
    /// `z` is [lat_h * lat_w, latent_channels] as the diffusion model emits it,
    /// still scaled -- this applies `scaling_factor` and `shift_factor` itself,
    /// so callers must not.
    ///
    /// Returns [(lat_h * f) * (lat_w * f), 3] where `f` is
    /// `cfg.downsample_factor()`.
    [[nodiscard]] Result<Mat> decode(const Mat& z, std::size_t lat_h, std::size_t lat_w) const;

    /// Decode in overlapping tiles, blending the seams.
    ///
    /// `tile` and `overlap` are in latent pixels. The whole-image path needs a
    /// few gigabytes at 1024x1024; this caps it at roughly
    /// `(tile + overlap)^2 * 64 * channels` bytes per tile.
    ///
    /// The blend is a linear ramp across the overlap. A hard cut leaves a seam
    /// that is invisible in flat regions and obvious across any edge, because
    /// the two tiles' GroupNorm statistics differ -- which is also why the
    /// overlap has to be a real fraction of the tile and not two pixels.
    [[nodiscard]] Result<Mat> decode_tiled(const Mat& z, std::size_t lat_h, std::size_t lat_w,
                                           std::size_t tile = 64,
                                           std::size_t overlap = 16) const;

    [[nodiscard]] std::size_t parameter_count() const;
};

}  // namespace rt

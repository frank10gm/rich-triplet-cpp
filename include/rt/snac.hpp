#pragma once

// =============================================================================
// SNAC -- Multi-Scale Neural Audio Codec, decoder only
// =============================================================================
//
// An autoregressive speech model does not emit audio. It emits codec tokens,
// and something else has to turn those into samples. Orpheus emits SNAC codes,
// so this is the second half of that pipeline: codes in, 24 kHz waveform out.
//
// Only the decoder is implemented. Encoding needs the analysis stack and the
// nearest-neighbour codebook search, neither of which text-to-speech uses.
//
// ## Residual vector quantization at three time scales
//
// A plain codec quantizes every frame at one rate. SNAC uses three codebooks
// running at 1/4, 1/2 and 1/1 of the frame rate, each coding what the previous
// one left behind. Coarse structure gets cheap slow codes and detail gets fast
// ones, so 7 codes cover 4 frames:
//
//   codebook 0, stride 4 -> 1 code per 4 frames   (~11.7 Hz)
//   codebook 1, stride 2 -> 2 codes per 4 frames  (~23.4 Hz)
//   codebook 2, stride 1 -> 4 codes per 4 frames  (~46.9 Hz)
//
// Reconstruction is `sum_i out_proj_i(codebook_i[code]) upsampled by stride_i`.
// The upsampling is a **repeat**, not a tile: stride 4 turns `[a, b]` into
// `[a, a, a, a, b, b, b, b]`. Interleaving instead produces a warble that is
// entirely plausible-sounding and completely wrong.
//
// ## The decoder stack
//
// Each frame carries 512 samples (`decoder_rates` [8, 8, 4, 2] multiply out to
// 512), so 4 frames is 2048 samples -- 85.33 ms at 24 kHz, which is exactly one
// Orpheus 7-token group. Every upsampling block multiplies the length by its
// stride exactly, and every residual unit preserves length exactly, so the
// output length is `n_frames * 512` with no slack. `conv1d.hpp` asserts both.
//
// The 24 kHz configuration sets `attn_window_size` to null, so unlike the
// 44 kHz model there is no local attention anywhere in the decoder -- it is
// convolutions and activations end to end.
//
// ## Noise injection is stochastic by design
//
// Each upsampling block ends with a `NoiseBlock`: `x + randn * conv(x)`. The
// noise is drawn per timestep and **shared across all channels** (PyTorch draws
// shape `[B, 1, T]`, not `[B, C, T]`), which correlates it the way the trained
// model expects. Drawing per channel instead is a plausible mistake that only
// makes the output slightly hissier -- unhearable without a reference.
//
// Because of this, decoding the same codes twice does not give the same
// samples. `SnacNoise::Zero` suppresses the draw entirely, which makes the
// decoder deterministic for testing at the cost of a little naturalness.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "rt/init_rng.hpp"
#include "rt/mat.hpp"
#include "rt/result.hpp"
#include "rt/torch_pickle.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

struct SnacConfig {
    std::size_t sampling_rate = 24000;
    /// Quantizer and decoder input width. `encoder_dim * 2^len(encoder_rates)`
    /// in the reference, which is 48 * 16 for the 24 kHz model.
    std::size_t latent_dim = 768;
    /// Channel count entering the first upsampling block; halves each block.
    std::size_t decoder_dim = 1024;
    /// Upsampling factors, coarsest first.
    std::vector<std::size_t> decoder_rates{8, 8, 4, 2};
    /// One entry per codebook, in the order they are applied.
    std::vector<std::size_t> vq_strides{4, 2, 1};
    std::size_t codebook_size = 4096;
    std::size_t codebook_dim = 8;
    /// Whether each upsampling block carries a `NoiseBlock`.
    bool noise = true;
    /// Whether the residual units and the input convolution are grouped.
    bool depthwise = true;

    /// `hubertsiuzdak/snac_24khz`, which is the codec Orpheus emits into.
    [[nodiscard]] static SnacConfig snac_24khz();

    /// Samples per frame -- the product of `decoder_rates`, 512 at 24 kHz.
    [[nodiscard]] std::size_t upsample_factor() const;

    /// Frames covered by one group of codes, which is the coarsest stride.
    [[nodiscard]] std::size_t frames_per_group() const;
};

/// How the `NoiseBlock`s draw their noise.
enum class SnacNoise {
    /// Suppress the draw. Makes decoding deterministic, for tests.
    Zero,
    /// Draw N(0,1) per timestep from a seeded LCG. What real output uses.
    Seeded,
};

// =============================================================================
// Layers
// =============================================================================

/// Snake, dilated depthwise convolution, Snake, pointwise convolution, plus a
/// skip connection.
///
/// The dilated convolution pads by `3 * dilation` against a kernel of 7, which
/// preserves length exactly -- so the reference's centre-crop of the skip
/// branch never triggers, and this can add the input directly.
struct SnacResidualUnit {
    std::vector<float> alpha1;
    /// [C, K] when depthwise, [C, C*K] otherwise.
    Mat conv1_weight;
    std::vector<float> conv1_bias;
    std::vector<float> alpha2;
    /// [C, C] pointwise.
    Mat conv2_weight;
    std::vector<float> conv2_bias;
    std::size_t dilation = 1;
    std::size_t kernel = 7;
    bool depthwise = true;

    [[nodiscard]] Mat forward(const Mat& x) const;
};

/// `x + noise * conv(x)`, with one noise sample per timestep shared across
/// channels. The convolution has no bias.
struct SnacNoiseBlock {
    /// [C, C] pointwise.
    Mat weight;

    [[nodiscard]] Mat forward(const Mat& x, SnacNoise mode, InitRng& rng) const;
};

/// Snake, transposed convolution, optional noise, then three residual units at
/// dilations 1, 3 and 9.
struct SnacDecoderBlock {
    std::vector<float> alpha;
    /// [Cin, Cout * K] -- PyTorch's input-channels-first transposed layout.
    Mat up_weight;
    /// Per **output** channel, so length `out_channels`.
    std::vector<float> up_bias;
    std::size_t out_channels = 0;
    std::size_t kernel = 0;
    std::size_t stride = 0;
    std::size_t padding = 0;
    std::size_t output_padding = 0;
    std::optional<SnacNoiseBlock> noise;
    std::vector<SnacResidualUnit> units;

    [[nodiscard]] Mat forward(const Mat& x, SnacNoise mode, InitRng& rng) const;
};

/// The residual vector quantizer's synthesis half.
struct SnacQuantizer {
    struct Level {
        /// [codebook_size, codebook_dim]
        Mat codebook;
        /// [latent_dim, codebook_dim] pointwise.
        Mat out_proj_weight;
        std::vector<float> out_proj_bias;
        std::size_t stride = 1;
    };
    std::vector<Level> levels;
    std::size_t latent_dim = 0;

    /// Codes to latents: `[T, latent_dim]`.
    ///
    /// `codes[i]` must hold `T / vq_strides[i]` entries, so the finest level
    /// fixes `T`. Every id must be below `codebook_size`.
    [[nodiscard]] Result<Mat> from_codes(
        const std::vector<std::vector<std::uint32_t>>& codes) const;
};

// =============================================================================
// Decoder
// =============================================================================

class SnacDecoder {
   public:
    SnacConfig config;
    SnacQuantizer quantizer;

    /// `decoder.model.0` -- [768, 7] depthwise, or [768, 768*7] dense.
    Mat in_conv_weight;
    std::vector<float> in_conv_bias;
    /// `decoder.model.1` -- [1024, 768] pointwise. Absent when not depthwise,
    /// because then `decoder.model.0` already widens the channels.
    std::optional<Mat> in_mix_weight;
    std::vector<float> in_mix_bias;

    std::vector<SnacDecoderBlock> blocks;

    std::vector<float> out_alpha;
    /// [1, C*7] dense, projecting to a single audio channel.
    Mat out_weight;
    std::vector<float> out_bias;

    /// Load the decoder and quantizer from a `torch.save`d checkpoint.
    ///
    /// Ignores every encoder tensor. Reconstructs each weight-normalized
    /// parameter from its stored magnitude and direction, taking the group
    /// count from the magnitude's own length so that the transposed
    /// convolutions -- whose magnitude is per *input* channel, unlike every
    /// other convolution here -- need no special case.
    [[nodiscard]] static Result<SnacDecoder> load(const std::string& path, SnacConfig cfg);

    /// Codes to mono f32 samples in [-1, 1].
    ///
    /// Output length is exactly `codes.back().size() * upsample_factor()`.
    /// `seed` is only read when `mode` is `SnacNoise::Seeded`.
    [[nodiscard]] Result<std::vector<float>> decode(
        const std::vector<std::vector<std::uint32_t>>& codes, SnacNoise mode,
        std::uint64_t seed) const;

    /// Total number of f32 weights held, for reporting.
    [[nodiscard]] std::size_t parameter_count() const;
};

// =============================================================================
// Helpers (exposed for testing)
// =============================================================================

/// Repeat every row of `x` `factor` times in place of tiling the whole matrix.
///
/// `[a; b]` with factor 3 becomes `[a; a; a; b; b; b]`, which is what
/// `repeat_interleave` does and what the quantizer's stride upsampling needs.
[[nodiscard]] Mat repeat_rows(const Mat& x, std::size_t factor);

/// Reconstruct a weight-normalized convolution weight as a `Mat` whose rows are
/// axis 0 of the stored tensor.
///
/// `prefix` names the module, e.g. `decoder.model.2.block.1`; the magnitude and
/// direction are read from `<prefix>.parametrizations.weight.original0` and
/// `...original1`.
[[nodiscard]] Result<Mat> load_weight_norm_mat(const TorchStateDict& sd,
                                               const std::string& prefix);

/// Read a `[1, C, 1]` Snake alpha as a flat vector of length C.
[[nodiscard]] Result<std::vector<float>> load_alpha(const TorchStateDict& sd,
                                                    const std::string& name);

}  // namespace rt

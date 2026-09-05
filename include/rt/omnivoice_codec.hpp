#pragma once

// =============================================================================
// OmniVoice audio codec -- decoder only
// =============================================================================
//
// The second half of OmniVoice: eight streams of codebook indices in, a 24 kHz
// waveform out. Structurally this is the same family as SNAC -- residual vector
// quantization feeding a stack of transposed convolutions with Snake
// activations -- so it reuses `conv1d.hpp` wholesale. The differences are worth
// naming.
//
// | | SNAC 24 kHz | OmniVoice |
// |---|---|---|
// | Codebooks | 3, at strides 4/2/1 | **8, all at the same rate** |
// | Codebook | 4096 x 8 | 1024 x 64 |
// | Upsampling | 8, 8, 4, 2 (512x) | **8, 5, 4, 2, 3 (960x)** |
// | Residual convs | depthwise | **dense** |
// | Weight storage | weight-norm `g` and `v` | already reconstructed |
// | Noise injection | yes, stochastic | **none -- fully deterministic** |
//
// Two of those matter in practice. The codebooks running at one rate means
// there is no stride bookkeeping and no interleave to get wrong: eight indices
// per frame, summed. And no noise block means decoding is deterministic, so a
// given set of codes always produces exactly the same samples -- unlike SNAC,
// where reproducibility had to be bought by suppressing the noise draw.
//
// The dense residual convolutions are why `conv1d_dense` needed an im2col path.
// A 7-tap dense convolution at 512 channels over a sequence upsampled toward
// 96 000 samples is tens of GFLOP per clip.
//
// ## Frame arithmetic
//
// `hop_length` is 960 and the upsampling ratios multiply to exactly that, so
// one frame is 960 samples: 25 Hz at 24 kHz. Eight codes per frame means 200
// tokens per second of audio.
//
// ## What is not implemented
//
// The checkpoint also carries an `acoustic_encoder`, an `encoder_semantic`
// stack and a 12-layer wav2vec2-style `semantic_model`. Those are the analysis
// half, needed to turn reference audio into codes for voice cloning. Synthesis
// needs none of them, so roughly 300 of the file's 486 tensors are skipped.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rt/gguf.hpp"
#include "rt/hubert.hpp"
#include "rt/mat.hpp"
#include "rt/result.hpp"

namespace rt {

/// The rate the semantic model runs at, whatever the codec's own is.
inline constexpr std::size_t kSemanticSampleRate = 16000;

// =============================================================================
// Config
// =============================================================================

struct OmniCodecConfig {
    std::size_t sample_rate = 24000;
    /// Samples per frame. Equals the product of `upsampling_ratios`.
    std::size_t hop_length = 960;
    /// Quantizer width, and the decoder's input before `fc2`.
    std::size_t latent_dim = 1024;
    /// Channels entering the first upsampling block; halves each block.
    std::size_t decoder_dim = 1024;
    /// Channels leaving the encoder's input convolution; doubles each block.
    std::size_t encoder_dim = 64;
    /// The decoder's input width after `fc2`, and the acoustic encoder's
    /// output width -- the same number because they are the two ends of one
    /// bottleneck.
    std::size_t decoder_in_dim = 256;
    /// Coarsest first.
    std::vector<std::size_t> upsampling_ratios{8, 5, 4, 2, 3};
    std::size_t n_codebooks = 8;
    std::size_t codebook_size = 1024;
    std::size_t codebook_dim = 64;

    [[nodiscard]] static OmniCodecConfig defaults();

    /// Product of `upsampling_ratios`; must equal `hop_length`.
    [[nodiscard]] std::size_t upsample_factor() const;

    /// The semantic path's width: whatever the acoustic path does not fill of
    /// the quantizer's input.
    [[nodiscard]] std::size_t semantic_dim() const {
        return latent_dim > decoder_in_dim ? latent_dim - decoder_in_dim : 0;
    }
};

// =============================================================================
// Layers
// =============================================================================

/// Snake, dense dilated convolution, Snake, pointwise convolution, plus a skip.
///
/// Padding is `3 * dilation` against a kernel of 7, which preserves length
/// exactly, so the skip lines up with no crop.
struct OmniResidualUnit {
    std::vector<float> alpha1;
    /// [C, C * 7]
    Mat conv1_weight;
    std::vector<float> conv1_bias;
    std::vector<float> alpha2;
    /// [C, C]
    Mat conv2_weight;
    std::vector<float> conv2_bias;
    std::size_t dilation = 1;

    [[nodiscard]] Mat forward(const Mat& x) const;
};

/// Snake, transposed convolution, then three residual units at dilations
/// 1, 3 and 9.
struct OmniDecoderBlock {
    std::vector<float> alpha;
    /// [Cin, Cout * K] -- transposed convolutions store input channels first.
    Mat up_weight;
    /// Per output channel.
    std::vector<float> up_bias;
    std::size_t out_channels = 0;
    std::size_t kernel = 0;
    std::size_t stride = 0;
    std::size_t padding = 0;
    std::size_t output_padding = 0;
    std::vector<OmniResidualUnit> units;

    [[nodiscard]] Mat forward(const Mat& x) const;
};

/// The residual vector quantizer's synthesis half.
///
/// Every codebook runs at the frame rate, so reconstruction is a plain sum with
/// no stride bookkeeping:
///
///     z = sum_i project_out_i(codebook_i[code_i])
struct OmniQuantizer {
    struct Level {
        /// [codebook_size, codebook_dim]
        Mat codebook;
        /// [latent_dim, codebook_dim] pointwise.
        Mat project_out_weight;
        std::vector<float> project_out_bias;
        /// [codebook_dim, latent_dim] pointwise. Empty in a decoder-only load;
        /// analysis needs it to get *into* the codebook's space.
        Mat project_in_weight;
        std::vector<float> project_in_bias;
    };
    std::vector<Level> levels;
    std::size_t latent_dim = 0;

    /// `codes[i]` holds one index per frame, the same count for every i.
    /// Returns [T, latent_dim].
    [[nodiscard]] Result<Mat> from_codes(
        const std::vector<std::vector<std::uint32_t>>& codes) const;

    /// The analysis direction: latents to codes, one vector per codebook.
    ///
    /// Residual quantization is greedy and sequential. Level 0 picks the
    /// nearest entry to the latent, its reconstruction is subtracted, and
    /// level 1 codes what is left. So the codebooks are not independent and
    /// cannot be searched in parallel -- each one only ever sees the error the
    /// ones before it could not represent, which is why the later codebooks
    /// carry finer detail and why unmasking them last is the right order.
    [[nodiscard]] Result<std::vector<std::vector<std::uint32_t>>> to_codes(
        const Mat& latents) const;
};

// =============================================================================
// Decoder
// =============================================================================

class OmniCodecDecoder {
   public:
    OmniCodecConfig config;
    OmniQuantizer quantizer;

    /// `fc2`: quantizer width down to the decoder's input width.
    Mat fc2_weight;
    std::vector<float> fc2_bias;

    /// `acoustic_decoder.conv1`: [decoder_dim, decoder_in_dim * 7].
    Mat in_conv_weight;
    std::vector<float> in_conv_bias;

    std::vector<OmniDecoderBlock> blocks;

    std::vector<float> out_alpha;
    /// `acoustic_decoder.conv2`: [1, C * 7].
    Mat out_weight;
    std::vector<float> out_bias;

    /// Load the decode-side tensors from an `omnivoice-tokenizer` GGUF.
    ///
    /// Ignores every analysis tensor.
    [[nodiscard]] static Result<OmniCodecDecoder> load(const std::string& path,
                                                       OmniCodecConfig cfg);

    /// Codes to mono f32 samples in [-1, 1].
    ///
    /// Output length is exactly `frames * hop_length`. Deterministic: there is
    /// no noise injection anywhere in this decoder.
    [[nodiscard]] Result<std::vector<float>> decode(
        const std::vector<std::vector<std::uint32_t>>& codes) const;

    [[nodiscard]] std::size_t parameter_count() const;
};


// =============================================================================
// Analysis
// =============================================================================
//
// The mirror of the decoder, and the half voice cloning needs: a waveform in,
// eight streams of codebook indices out. It is not a mirror of one stack but of
// two -- the codec quantizes the concatenation of an *acoustic* path and a
// *semantic* one, so that a code carries both how a voice sounds and what it
// said.
//
//   24 kHz samples --> acoustic encoder (DAC) ------> [T, 256] --+
//                  \                                             |--> fc --> RVQ
//                   -> 16 kHz --> HuBERT --> conv stack --> [T, 768] --+
//
// Both arrive at the same 25 Hz frame rate by different arithmetic, and that
// they agree is the load-bearing invariant: the acoustic path divides 24 kHz by
// 960, while the semantic path divides 16 kHz by 320 for 50 Hz and then keeps
// every other frame.

/// One downsampling stage: three residual units, Snake, then a strided
/// convolution that halves the length by `stride` and doubles the channels.
///
/// The residual units run *before* the downsample here, where the decoder runs
/// them after its upsample. That is not symmetry for its own sake -- it keeps
/// the expensive dilated convolutions on the shorter side of the stride in
/// both directions.
struct OmniEncoderBlock {
    std::vector<OmniResidualUnit> units;
    std::vector<float> alpha;
    /// [Cout, Cin * K]
    Mat down_weight;
    std::vector<float> down_bias;
    std::size_t out_channels = 0;
    std::size_t kernel = 0;
    std::size_t stride = 1;
    std::size_t padding = 0;

    [[nodiscard]] Mat forward(const Mat& x) const;
};

/// The acoustic half: raw 24 kHz samples to `hidden_size`-wide frames.
struct OmniAcousticEncoder {
    /// `acoustic_encoder.conv1`: [encoder_dim, 1 * 7].
    Mat in_weight;
    std::vector<float> in_bias;
    std::vector<OmniEncoderBlock> blocks;
    std::vector<float> out_alpha;
    /// `acoustic_encoder.conv2`: [acoustic_dim, C * 3].
    Mat out_weight;
    std::vector<float> out_bias;

    /// [T] samples to [T / hop_length, acoustic_dim].
    [[nodiscard]] Result<Mat> forward(std::span<const float> samples,
                                      std::size_t hop_length) const;
};

/// A residual unit in the semantic stack.
///
/// Same shape as the acoustic one and none of the same details: ELU rather
/// than Snake, kernel 3 rather than 7, and no biases at all.
struct OmniSemanticResidualUnit {
    /// [C, C * 3]
    Mat conv1_weight;
    /// [C, C]
    Mat conv2_weight;
    std::size_t dilation = 1;

    [[nodiscard]] Mat forward(const Mat& x) const;
};

/// Residual units, then a convolution. Both of OmniVoice's blocks run at
/// stride 1, so this stack reshapes the features without resampling them.
struct OmniSemanticBlock {
    std::vector<OmniSemanticResidualUnit> units;
    /// [Cout, Cin * K]
    Mat conv_weight;
    std::vector<float> conv_bias;
    std::size_t out_channels = 0;
    std::size_t kernel = 3;
    std::size_t stride = 1;
    std::size_t padding = 1;

    [[nodiscard]] Mat forward(const Mat& x) const;
};

/// The convolutional adapter between HuBERT's features and the quantizer.
struct OmniSemanticEncoder {
    /// `encoder_semantic.conv`: [C, C * 3], no bias.
    Mat in_weight;
    std::vector<OmniSemanticBlock> blocks;

    [[nodiscard]] Mat forward(const Mat& x) const;
};

/// Waveform to codes.
class OmniCodecEncoder {
   public:
    OmniCodecConfig config;
    HubertModel semantic;
    OmniSemanticEncoder semantic_adapter;
    OmniAcousticEncoder acoustic;

    /// `fc`: the concatenated [acoustic | semantic] width, mixed in place.
    Mat fc_weight;
    std::vector<float> fc_bias;

    OmniQuantizer quantizer;

    /// Load the analysis-side tensors from an `omnivoice-tokenizer` GGUF.
    ///
    /// Roughly the complement of what `OmniCodecDecoder::load` reads, plus the
    /// codebooks, which both halves need.
    [[nodiscard]] static Result<OmniCodecEncoder> load(const std::string& path,
                                                       OmniCodecConfig cfg);

    /// Mono 24 kHz samples to `n_codebooks` streams of indices.
    ///
    /// Trailing samples that do not complete a frame are dropped, which is
    /// what the reference does: a partial frame would put the two paths'
    /// lengths out of step and force a padding branch.
    [[nodiscard]] Result<std::vector<std::vector<std::uint32_t>>> encode(
        std::span<const float> samples) const;

    [[nodiscard]] std::size_t parameter_count() const;
};

}  // namespace rt

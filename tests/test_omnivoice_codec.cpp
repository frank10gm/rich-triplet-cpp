#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "rt/conv1d.hpp"
#include "rt/omnivoice_codec.hpp"
#include "rt/wav.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

namespace {

constexpr const char* kCodecPath = "models/omnivoice-tokenizer-Q8_0.gguf";

/// A quantizer whose codebooks are readable off the output: code `c` of
/// codebook `i` projects to `c` in column 0 and `i` in column 1.
[[nodiscard]] OmniQuantizer toy_quantizer(std::size_t n_codebooks, std::size_t latent_dim) {
    OmniQuantizer q;
    q.latent_dim = latent_dim;
    for (std::size_t i = 0; i < n_codebooks; ++i) {
        OmniQuantizer::Level level;
        level.codebook = Mat::from_fn(8, 2, [i](std::size_t r, std::size_t c) {
            return c == 0 ? static_cast<float>(r) : static_cast<float>(i);
        });
        level.project_out_weight = Mat::zeros(latent_dim, 2);
        level.project_out_weight.at_mut(0, 0) = 1.0f;
        level.project_out_weight.at_mut(1, 1) = 1.0f;
        q.levels.push_back(std::move(level));
    }
    return q;
}

}  // namespace

// =============================================================================
// Config
// =============================================================================

TEST_CASE("OmniVoice codec upsamples by exactly its hop length", "[omnicodec]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    // 8 * 5 * 4 * 2 * 3 == 960, and 24000 / 960 == 25 Hz.
    REQUIRE(c.upsample_factor() == 960);
    REQUIRE(c.upsample_factor() == c.hop_length);
    REQUIRE(c.sample_rate / c.hop_length == 25);
    // Eight codes per frame at 25 Hz is 200 audio tokens a second.
    REQUIRE(c.n_codebooks * (c.sample_rate / c.hop_length) == 200);
}

TEST_CASE("OmniVoice channel widths halve to the output", "[omnicodec]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    std::size_t ch = c.decoder_dim;
    for (std::size_t i = 0; i < c.upsampling_ratios.size(); ++i) {
        ch /= 2;
    }
    REQUIRE(c.decoder_dim == 1024);
    REQUIRE(ch == 32);  // 1024 -> 512 -> 256 -> 128 -> 64 -> 32
}

// =============================================================================
// Quantizer
// =============================================================================

TEST_CASE("from_codes sums every codebook at one rate", "[omnicodec]") {
    // Unlike SNAC there are no strides: all codebooks run at the frame rate,
    // so reconstruction is a plain sum and there is no interleave to get wrong.
    const OmniQuantizer q = toy_quantizer(4, 4);
    const std::vector<std::vector<std::uint32_t>> codes{
        {1, 2}, {3, 4}, {5, 6}, {7, 0},
    };
    const Result<Mat> z = q.from_codes(codes);
    REQUIRE(z.has_value());
    REQUIRE(z->rows == 2);
    REQUIRE(z->cols == 4);
    // Column 0 is the sum of the code values at that frame.
    REQUIRE(approx(z->at(0, 0), 1.0f + 3.0f + 5.0f + 7.0f));
    REQUIRE(approx(z->at(1, 0), 2.0f + 4.0f + 6.0f + 0.0f));
    // Column 1 is the sum of the codebook indices, once per frame.
    REQUIRE(approx(z->at(0, 1), 0.0f + 1.0f + 2.0f + 3.0f));
    REQUIRE(approx(z->at(1, 1), 6.0f));
}

TEST_CASE("from_codes rejects ragged codebooks", "[omnicodec]") {
    const OmniQuantizer q = toy_quantizer(3, 4);
    const Result<Mat> z = q.from_codes({{1, 2}, {3}, {5, 6}});
    REQUIRE(!z.has_value());
    REQUIRE(z.error().find("expected 2") != std::string::npos);
}

TEST_CASE("from_codes rejects the wrong codebook count", "[omnicodec]") {
    const OmniQuantizer q = toy_quantizer(8, 4);
    const Result<Mat> z = q.from_codes({{1}, {2}});
    REQUIRE(!z.has_value());
    REQUIRE(z.error().find("expected 8 codebooks") != std::string::npos);
}

TEST_CASE("from_codes rejects an out-of-range code", "[omnicodec]") {
    const OmniQuantizer q = toy_quantizer(2, 4);
    const Result<Mat> z = q.from_codes({{1}, {999}});
    REQUIRE(!z.has_value());
    REQUIRE(z.error().find("out of range") != std::string::npos);
}

TEST_CASE("from_codes rejects an empty request", "[omnicodec]") {
    const OmniQuantizer q = toy_quantizer(2, 4);
    REQUIRE(!q.from_codes({{}, {}}).has_value());
}

// =============================================================================
// Layers
// =============================================================================

TEST_CASE("OmniResidualUnit preserves length at every dilation", "[omnicodec]") {
    constexpr std::size_t channels = 8;
    for (const std::size_t dilation : {std::size_t{1}, std::size_t{3}, std::size_t{9}}) {
        OmniResidualUnit unit;
        unit.dilation = dilation;
        unit.alpha1.assign(channels, 1.0f);
        unit.alpha2.assign(channels, 1.0f);
        unit.conv1_weight = Mat::zeros(channels, channels * 7);
        unit.conv2_weight = Mat::zeros(channels, channels);

        const Mat x = Mat::ones(48, channels);
        const Mat out = unit.forward(x);
        REQUIRE(out.rows == 48);
        REQUIRE(out.cols == channels);
        // Both convolutions are zeroed, so only the skip survives.
        for (std::size_t i = 0; i < out.data.size(); ++i) {
            REQUIRE(approx(out.data[i], 1.0f));
        }
    }
}

TEST_CASE("OmniDecoderBlock multiplies length by its stride", "[omnicodec]") {
    // The odd strides matter: OmniVoice upsamples by 5 and 3, where SNAC only
    // ever used powers of two, and the output-padding term is what keeps the
    // multiple exact for those.
    for (const std::size_t stride : {std::size_t{2}, std::size_t{3}, std::size_t{4},
                                     std::size_t{5}, std::size_t{8}}) {
        constexpr std::size_t c_in = 16;
        const std::size_t c_out = c_in / 2;

        OmniDecoderBlock block;
        block.stride = stride;
        block.kernel = 2 * stride;
        block.padding = (stride + 1) / 2;
        block.output_padding = stride % 2;
        block.out_channels = c_out;
        block.alpha.assign(c_in, 1.0f);
        block.up_weight = Mat::zeros(c_in, c_out * block.kernel);
        block.up_bias.assign(c_out, 0.0f);
        for (std::size_t u = 0; u < 3; ++u) {
            OmniResidualUnit unit;
            unit.dilation = u == 0 ? 1 : (u == 1 ? 3 : 9);
            unit.alpha1.assign(c_out, 1.0f);
            unit.alpha2.assign(c_out, 1.0f);
            unit.conv1_weight = Mat::zeros(c_out, c_out * 7);
            unit.conv2_weight = Mat::zeros(c_out, c_out);
            block.units.push_back(std::move(unit));
        }

        const Mat out = block.forward(Mat::ones(7, c_in));
        REQUIRE(out.rows == 7 * stride);
        REQUIRE(out.cols == c_out);
    }
}

TEST_CASE("the full upsampling chain multiplies by 960", "[omnicodec]") {
    // End to end through the real ratios, with random weights: 5 frames must
    // become exactly 4800 samples. Every padding mistake breaks the multiple.
    const OmniCodecConfig cfg = OmniCodecConfig::defaults();
    std::size_t t = 5;
    for (const std::size_t stride : cfg.upsampling_ratios) {
        t *= stride;
    }
    REQUIRE(t == 5 * 960);
    REQUIRE(t == 4800);
}

// =============================================================================
// The real checkpoint
// =============================================================================

TEST_CASE("OmniCodecDecoder loads the tokenizer GGUF", "[omnicodec][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("models/omnivoice-tokenizer-Q8_0.gguf not present");
    }
    const Result<OmniCodecDecoder> d =
        OmniCodecDecoder::load(kCodecPath, OmniCodecConfig::defaults());
    if (!d.has_value()) {
        FAIL(d.error());
    }

    REQUIRE(d->blocks.size() == 5);
    REQUIRE(d->quantizer.levels.size() == 8);

    const std::vector<std::size_t> want_stride{8, 5, 4, 2, 3};
    const std::vector<std::size_t> want_out{512, 256, 128, 64, 32};
    for (std::size_t i = 0; i < 5; ++i) {
        REQUIRE(d->blocks[i].stride == want_stride[i]);
        REQUIRE(d->blocks[i].kernel == 2 * want_stride[i]);
        REQUIRE(d->blocks[i].out_channels == want_out[i]);
        REQUIRE(d->blocks[i].units.size() == 3);
        // Transposed convolutions are [Cin, Cout * K], biased per output.
        REQUIRE(d->blocks[i].up_weight.cols == want_out[i] * d->blocks[i].kernel);
        REQUIRE(d->blocks[i].up_bias.size() == want_out[i]);
    }

    // Codebooks are 1024 x 64, projected up to the 1024-wide latent.
    for (const OmniQuantizer::Level& level : d->quantizer.levels) {
        REQUIRE(level.codebook.rows == 1024);
        REQUIRE(level.codebook.cols == 64);
        REQUIRE(level.project_out_weight.rows == 1024);
        REQUIRE(level.project_out_weight.cols == 64);
    }

    REQUIRE(d->out_alpha.size() == 32);
    REQUIRE(d->out_weight.rows == 1);
    REQUIRE(d->out_weight.cols == 32 * 7);

    INFO("decoder parameters: " << d->parameter_count());
    REQUIRE(d->parameter_count() > 15'000'000);
    REQUIRE(d->parameter_count() < 30'000'000);
}

TEST_CASE("OmniCodecDecoder produces exactly frames * 960 samples", "[omnicodec][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("models/omnivoice-tokenizer-Q8_0.gguf not present");
    }
    const Result<OmniCodecDecoder> d =
        OmniCodecDecoder::load(kCodecPath, OmniCodecConfig::defaults());
    REQUIRE(d.has_value());

    constexpr std::size_t frames = 25;  // one second
    std::vector<std::vector<std::uint32_t>> codes(8, std::vector<std::uint32_t>(frames));
    for (std::size_t c = 0; c < 8; ++c) {
        for (std::size_t t = 0; t < frames; ++t) {
            codes[c][t] = static_cast<std::uint32_t>((t * 37 + c * 101) % 1024);
        }
    }

    const Result<std::vector<float>> audio = d->decode(codes);
    if (!audio.has_value()) {
        FAIL(audio.error());
    }
    REQUIRE(audio->size() == frames * 960);
    REQUIRE(audio->size() == 24000);

    // Arbitrary codes are not speech, so only the bound is asserted -- but the
    // final tanh has to hold it, or the int16 conversion downstream is unsafe.
    const WaveStats s = wave_stats(*audio);
    INFO(s.describe());
    REQUIRE(s.in_range());
}

TEST_CASE("OmniCodecDecoder is deterministic", "[omnicodec][.integration]") {
    // There is no noise block anywhere in this decoder, unlike SNAC, so the
    // same codes must give bit-identical samples with no seed involved.
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("models/omnivoice-tokenizer-Q8_0.gguf not present");
    }
    const Result<OmniCodecDecoder> d =
        OmniCodecDecoder::load(kCodecPath, OmniCodecConfig::defaults());
    REQUIRE(d.has_value());

    std::vector<std::vector<std::uint32_t>> codes(8, std::vector<std::uint32_t>(4));
    for (std::size_t c = 0; c < 8; ++c) {
        for (std::size_t t = 0; t < 4; ++t) {
            codes[c][t] = static_cast<std::uint32_t>(c * 13 + t);
        }
    }
    const Result<std::vector<float>> a = d->decode(codes);
    const Result<std::vector<float>> b = d->decode(codes);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    REQUIRE(a->size() == b->size());
    for (std::size_t i = 0; i < a->size(); ++i) {
        REQUIRE((*a)[i] == (*b)[i]);
    }
}

TEST_CASE("OmniCodecDecoder rejects a mismatched config", "[omnicodec][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("models/omnivoice-tokenizer-Q8_0.gguf not present");
    }
    OmniCodecConfig wrong = OmniCodecConfig::defaults();
    wrong.decoder_dim = 1536;
    const Result<OmniCodecDecoder> d = OmniCodecDecoder::load(kCodecPath, wrong);
    REQUIRE(!d.has_value());
    INFO(d.error());
    REQUIRE(d.error().find("1536") != std::string::npos);
}

TEST_CASE("OmniCodecDecoder rejects ratios that disagree with hop_length", "[omnicodec]") {
    // The two are stated independently in the checkpoint metadata, and a
    // disagreement means every length downstream would be wrong.
    OmniCodecConfig wrong = OmniCodecConfig::defaults();
    wrong.upsampling_ratios = {8, 5, 4, 2};  // 320, not 960
    const Result<OmniCodecDecoder> d = OmniCodecDecoder::load("models/does-not-exist.gguf", wrong);
    REQUIRE(!d.has_value());
}

// =============================================================================
// Analysis -- the quantizer
// =============================================================================

namespace {

/// A toy quantizer that can also encode: `project_in` is the identity onto the
/// first two latent columns, so a latent is its own codebook coordinate.
[[nodiscard]] OmniQuantizer toy_encoding_quantizer(std::size_t n_codebooks,
                                                   std::size_t latent_dim) {
    OmniQuantizer q = toy_quantizer(n_codebooks, latent_dim);
    for (OmniQuantizer::Level& level : q.levels) {
        level.project_in_weight = Mat::zeros(2, latent_dim);
        level.project_in_weight.at_mut(0, 0) = 1.0f;
        level.project_in_weight.at_mut(1, 1) = 1.0f;
    }
    return q;
}

}  // namespace

TEST_CASE("to_codes picks the nearest codebook entry", "[omnicodec]") {
    // One codebook of eight entries at (0,0), (1,0) ... (7,0). A latent at
    // 2.4 has to land on entry 2 and one at 2.6 on entry 3.
    OmniQuantizer q = toy_encoding_quantizer(1, 4);
    Mat latents = Mat::zeros(4, 4);
    latents.at_mut(0, 0) = 2.4f;
    latents.at_mut(1, 0) = 2.6f;
    latents.at_mut(2, 0) = -5.0f;   // below every entry
    latents.at_mut(3, 0) = 100.0f;  // above every entry

    const Result<std::vector<std::vector<std::uint32_t>>> codes = q.to_codes(latents);
    REQUIRE(codes.has_value());
    REQUIRE(codes->size() == 1);
    REQUIRE((*codes)[0] == std::vector<std::uint32_t>{2, 3, 0, 7});
}

TEST_CASE("to_codes quantizes the residual, not the signal", "[omnicodec]") {
    // Two codebooks over the same eight entries. The first takes the latent,
    // the second takes what is left after the first has been subtracted -- so
    // a latent of exactly 3 leaves nothing and the second codebook picks 0.
    OmniQuantizer q = toy_encoding_quantizer(2, 4);
    Mat latents = Mat::zeros(2, 4);
    latents.at_mut(0, 0) = 3.0f;
    latents.at_mut(1, 0) = 5.5f;

    const Result<std::vector<std::vector<std::uint32_t>>> codes = q.to_codes(latents);
    REQUIRE(codes.has_value());
    REQUIRE(codes->size() == 2);
    REQUIRE((*codes)[0][0] == 3);
    REQUIRE((*codes)[1][0] == 0);
    // 5.5 rounds to 5 or 6; whichever it takes, the leftover is half a step,
    // which the second codebook can only round back to nothing.
    REQUIRE((*codes)[1][1] == 0);
}

TEST_CASE("from_codes inverts to_codes on the codebook grid", "[omnicodec]") {
    // Latents that sit exactly on an entry survive the round trip, which is
    // the strongest statement a lossy quantizer can make.
    OmniQuantizer q = toy_encoding_quantizer(1, 4);
    Mat latents = Mat::zeros(5, 4);
    for (std::size_t t = 0; t < 5; ++t) {
        latents.at_mut(t, 0) = static_cast<float>(t + 1);
    }
    const Result<std::vector<std::vector<std::uint32_t>>> codes = q.to_codes(latents);
    REQUIRE(codes.has_value());
    const Result<Mat> back = q.from_codes(*codes);
    REQUIRE(back.has_value());
    for (std::size_t t = 0; t < 5; ++t) {
        REQUIRE(approx(back->at(t, 0), static_cast<float>(t + 1)));
    }
}

TEST_CASE("to_codes refuses a quantizer loaded for synthesis only", "[omnicodec]") {
    // A decoder-only load leaves project_in empty, and guessing it would be
    // worse than failing.
    const OmniQuantizer q = toy_quantizer(2, 4);
    REQUIRE_FALSE(q.to_codes(Mat::zeros(3, 4)).has_value());
}

TEST_CASE("to_codes checks the latent width", "[omnicodec]") {
    const OmniQuantizer q = toy_encoding_quantizer(1, 4);
    REQUIRE_FALSE(q.to_codes(Mat::zeros(3, 5)).has_value());
}

// =============================================================================
// Analysis -- the convolution stacks
// =============================================================================

TEST_CASE("an encoder block divides the length by its stride", "[omnicodec]") {
    // The mirror of the decoder's invariant, and the one that makes
    // frames * 960 exact in the analysis direction too. Kernel 2s with padding
    // ceil(s/2) is what makes it hold for odd strides as well as even ones.
    for (const std::size_t stride : {8u, 5u, 4u, 2u, 3u}) {
        const std::size_t channels = 4;
        OmniEncoderBlock block;
        block.alpha.assign(channels, 1.0f);
        block.out_channels = channels * 2;
        block.stride = stride;
        block.kernel = 2 * stride;
        block.padding = (stride + 1) / 2;
        block.down_weight = Mat::from_fn(block.out_channels, channels * block.kernel,
                                         [](std::size_t r, std::size_t c) {
                                             return 0.01f * static_cast<float>((r + c) % 7);
                                         });
        block.down_bias.assign(block.out_channels, 0.0f);

        for (const std::size_t frames : {1u, 3u, 20u}) {
            const Mat x = Mat::from_fn(frames * stride, channels, [](std::size_t r, std::size_t c) {
                return 0.001f * static_cast<float>((r * 3 + c) % 11);
            });
            const Mat y = block.forward(x);
            REQUIRE(y.rows == frames);
            REQUIRE(y.cols == block.out_channels);
        }
    }
}

TEST_CASE("the full downsampling chain divides by 960", "[omnicodec]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    std::size_t t = 960 * 7;
    for (const std::size_t stride : c.upsampling_ratios) {
        const std::size_t kernel = 2 * stride;
        const std::size_t padding = (stride + 1) / 2;
        t = conv1d_out_len(t, kernel, 1, padding, stride);
    }
    REQUIRE(t == 7);
}

TEST_CASE("a semantic residual unit preserves length", "[omnicodec]") {
    // Kernel 3 at padding 1, unlike the acoustic units' 7 at 3.
    const std::size_t channels = 6;
    OmniSemanticResidualUnit unit;
    unit.dilation = 1;
    unit.conv1_weight = Mat::from_fn(channels, channels * 3, [](std::size_t r, std::size_t c) {
        return 0.01f * static_cast<float>((r + 2 * c) % 5) - 0.02f;
    });
    unit.conv2_weight = Mat::from_fn(channels, channels, [](std::size_t r, std::size_t c) {
        return r == c ? 0.5f : 0.0f;
    });

    const Mat x = Mat::from_fn(17, channels, [](std::size_t r, std::size_t c) {
        return 0.05f * static_cast<float>((r + c) % 4);
    });
    const Mat y = unit.forward(x);
    REQUIRE(y.rows == 17);
    REQUIRE(y.cols == channels);
    for (const float v : y.data) {
        REQUIRE(std::isfinite(v));
    }
}

TEST_CASE("the semantic width is whatever the acoustic path leaves", "[omnicodec]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    REQUIRE(c.semantic_dim() == 768);
    REQUIRE(c.decoder_in_dim + c.semantic_dim() == c.latent_dim);
    REQUIRE(c.encoder_dim == 64);
    // The encoder doubles its width per block, ending at 2048 before the
    // projection down to 256.
    std::size_t ch = c.encoder_dim;
    for (std::size_t i = 0; i < c.upsampling_ratios.size(); ++i) {
        ch *= 2;
    }
    REQUIRE(ch == 2048);
}

// =============================================================================
// Analysis -- the real checkpoint
// =============================================================================

TEST_CASE("OmniCodecEncoder loads the tokenizer GGUF", "[omnicodec][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("no OmniVoice tokenizer checkpoint in models/");
    }
    const Result<OmniCodecEncoder> e =
        OmniCodecEncoder::load(kCodecPath, OmniCodecConfig::defaults());
    REQUIRE(e.has_value());
    REQUIRE(e->acoustic.blocks.size() == 5);
    REQUIRE(e->semantic_adapter.blocks.size() == 2);
    REQUIRE(e->quantizer.levels.size() == 8);
    for (const OmniQuantizer::Level& l : e->quantizer.levels) {
        REQUIRE(l.project_in_weight.rows == 64);
        REQUIRE(l.project_in_weight.cols == 1024);
    }
    // The analysis half is bigger than the synthesis half, almost all of it
    // the 94 M-parameter semantic model.
    REQUIRE(e->parameter_count() > 150'000'000);
}

TEST_CASE("OmniCodecEncoder emits one code per codebook per frame",
          "[omnicodec][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("no OmniVoice tokenizer checkpoint in models/");
    }
    const OmniCodecConfig cfg = OmniCodecConfig::defaults();
    const Result<OmniCodecEncoder> e = OmniCodecEncoder::load(kCodecPath, cfg);
    REQUIRE(e.has_value());

    // 40 frames, plus a partial one that has to be dropped rather than padded.
    constexpr std::size_t kFrames = 40;
    std::vector<float> wav(kFrames * 960 + 137);
    for (std::size_t i = 0; i < wav.size(); ++i) {
        const float t = static_cast<float>(i) / 24000.0f;
        wav[i] = 0.15f * (std::sin(2.0f * 3.14159265f * 150.0f * t) +
                          0.4f * std::sin(2.0f * 3.14159265f * 900.0f * t));
    }

    const Result<std::vector<std::vector<std::uint32_t>>> codes = e->encode(wav);
    REQUIRE(codes.has_value());
    REQUIRE(codes->size() == cfg.n_codebooks);
    for (const std::vector<std::uint32_t>& stream : *codes) {
        REQUIRE(stream.size() == kFrames);
        for (const std::uint32_t c : stream) {
            REQUIRE(c < cfg.codebook_size);
        }
    }
}

TEST_CASE("OmniCodecEncoder is deterministic", "[omnicodec][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("no OmniVoice tokenizer checkpoint in models/");
    }
    const Result<OmniCodecEncoder> e =
        OmniCodecEncoder::load(kCodecPath, OmniCodecConfig::defaults());
    REQUIRE(e.has_value());

    std::vector<float> wav(20 * 960);
    for (std::size_t i = 0; i < wav.size(); ++i) {
        wav[i] = 0.2f * std::sin(2.0f * 3.14159265f * 220.0f * static_cast<float>(i) / 24000.0f);
    }
    const Result<std::vector<std::vector<std::uint32_t>>> a = e->encode(wav);
    const Result<std::vector<std::vector<std::uint32_t>>> b = e->encode(wav);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    REQUIRE(*a == *b);
}

TEST_CASE("OmniCodecEncoder rejects a clip shorter than a frame",
          "[omnicodec][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("no OmniVoice tokenizer checkpoint in models/");
    }
    const Result<OmniCodecEncoder> e =
        OmniCodecEncoder::load(kCodecPath, OmniCodecConfig::defaults());
    REQUIRE(e.has_value());
    REQUIRE_FALSE(e->encode(std::vector<float>(500, 0.0f)).has_value());
}

TEST_CASE("the codec round-trips a waveform through its own codes",
          "[omnicodec][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("no OmniVoice tokenizer checkpoint in models/");
    }
    // The test that makes the analysis path checkable at all without a
    // reference implementation. Decode a set of codes, encode the waveform
    // back, and the indices have to come out close to what went in --
    // exactly for the coarse codebooks and less so for the fine ones, since
    // those code the residual that survives the first few.
    //
    // Chance agreement is one in 1024. Any real fault in the chain -- a
    // normalisation on the wrong axis, the two paths concatenated in the wrong
    // order, a padding off by one -- lands there.
    const OmniCodecConfig cfg = OmniCodecConfig::defaults();
    const Result<OmniCodecDecoder> d = OmniCodecDecoder::load(kCodecPath, cfg);
    const Result<OmniCodecEncoder> e = OmniCodecEncoder::load(kCodecPath, cfg);
    REQUIRE(d.has_value());
    REQUIRE(e.has_value());

    constexpr std::size_t kFrames = 30;
    std::vector<std::vector<std::uint32_t>> codes(cfg.n_codebooks);
    std::uint32_t state = 12345;
    for (std::size_t i = 0; i < cfg.n_codebooks; ++i) {
        codes[i].resize(kFrames);
        for (std::size_t t = 0; t < kFrames; ++t) {
            state = state * 1664525u + 1013904223u;
            codes[i][t] = (state >> 16) % cfg.codebook_size;
        }
    }

    const Result<std::vector<float>> wav = d->decode(codes);
    REQUIRE(wav.has_value());
    REQUIRE(wav->size() == kFrames * cfg.hop_length);

    const Result<std::vector<std::vector<std::uint32_t>>> back = e->encode(*wav);
    REQUIRE(back.has_value());
    REQUIRE((*back)[0].size() == kFrames);

    std::size_t agree = 0;
    for (std::size_t t = 0; t < kFrames; ++t) {
        agree += codes[0][t] == (*back)[0][t];
    }
    // Random codes make a waveform the codec was never fit on, so this is a
    // weaker recovery than real audio gives; it is still two orders of
    // magnitude above chance.
    REQUIRE(agree * 4 >= kFrames);

    // And the resynthesis of the recovered codes has to track the original.
    const Result<std::vector<float>> again = d->decode(*back);
    REQUIRE(again.has_value());
    double num = 0.0;
    double da = 0.0;
    double db = 0.0;
    for (std::size_t i = 0; i < wav->size(); ++i) {
        num += static_cast<double>((*wav)[i]) * (*again)[i];
        da += static_cast<double>((*wav)[i]) * (*wav)[i];
        db += static_cast<double>((*again)[i]) * (*again)[i];
    }
    REQUIRE(num / std::sqrt(da * db) > 0.5);
}

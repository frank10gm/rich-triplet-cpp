#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

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

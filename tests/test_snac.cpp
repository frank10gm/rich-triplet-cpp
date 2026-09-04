#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "rt/init_rng.hpp"
#include "rt/snac.hpp"
#include "rt/wav.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

namespace {

constexpr const char* kSnacPath = "models/snac_24khz.bin";

[[nodiscard]] Mat identity(std::size_t n) {
    Mat m = Mat::zeros(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        m.at_mut(i, i) = 1.0f;
    }
    return m;
}

/// A quantizer with one-hot codebooks, so a code's contribution is readable
/// straight off the output.
[[nodiscard]] SnacQuantizer toy_quantizer(std::size_t latent_dim,
                                          const std::vector<std::size_t>& strides) {
    SnacQuantizer q;
    q.latent_dim = latent_dim;
    for (std::size_t i = 0; i < strides.size(); ++i) {
        SnacQuantizer::Level level;
        // 4 codes of width 2; code c maps to [c, -c].
        level.codebook = Mat::from_fn(4, 2, [](std::size_t r, std::size_t c) {
            return c == 0 ? static_cast<float>(r) : -static_cast<float>(r);
        });
        // Project [2] -> [latent_dim] by copying into the first two columns.
        level.out_proj_weight = Mat::zeros(latent_dim, 2);
        level.out_proj_weight.at_mut(0, 0) = 1.0f;
        level.out_proj_weight.at_mut(1, 1) = 1.0f;
        level.stride = strides[i];
        q.levels.push_back(std::move(level));
    }
    return q;
}

}  // namespace

// =============================================================================
// Config
// =============================================================================

TEST_CASE("SnacConfig 24 kHz upsamples by 512", "[snac]") {
    const SnacConfig c = SnacConfig::snac_24khz();
    REQUIRE(c.upsample_factor() == 512);
    REQUIRE(c.sampling_rate == 24000);
    REQUIRE(c.latent_dim == 768);
    REQUIRE(c.decoder_dim == 1024);
    REQUIRE(c.vq_strides == std::vector<std::size_t>{4, 2, 1});
    REQUIRE(c.frames_per_group() == 4);
}

TEST_CASE("one code group is 2048 samples, or 85.33 ms", "[snac]") {
    // 7 Orpheus tokens carry 4 frames; 4 * 512 = 2048 samples at 24 kHz. This
    // is what fixes the realtime token rate at ~82/s, so pin it.
    const SnacConfig c = SnacConfig::snac_24khz();
    const std::size_t samples = c.frames_per_group() * c.upsample_factor();
    REQUIRE(samples == 2048);
    const double ms = 1000.0 * static_cast<double>(samples) / static_cast<double>(c.sampling_rate);
    REQUIRE(std::fabs(ms - 85.333) < 0.01);
}

// =============================================================================
// repeat_rows
// =============================================================================

TEST_CASE("repeat_rows repeats rather than tiles", "[snac]") {
    // The distinction that matters: [a; b] with factor 3 must be
    // [a; a; a; b; b; b], not [a; b; a; b; a; b]. Tiling produces a warble
    // that sounds like a plausible codec artefact and is completely wrong.
    const Mat x({1.0f, 2.0f, 10.0f, 20.0f}, 2, 2);
    const Mat out = repeat_rows(x, 3);
    REQUIRE(out.rows == 6);
    REQUIRE(out.cols == 2);
    for (std::size_t k = 0; k < 3; ++k) {
        REQUIRE(approx(out.at(k, 0), 1.0f));
        REQUIRE(approx(out.at(k, 1), 2.0f));
    }
    for (std::size_t k = 3; k < 6; ++k) {
        REQUIRE(approx(out.at(k, 0), 10.0f));
        REQUIRE(approx(out.at(k, 1), 20.0f));
    }
}

TEST_CASE("repeat_rows at factor 1 is the identity", "[snac]") {
    const Mat x({1.0f, 2.0f, 3.0f}, 3, 1);
    const Mat out = repeat_rows(x, 1);
    REQUIRE(out.rows == 3);
    REQUIRE(out.data == x.data);
}

// =============================================================================
// Quantizer
// =============================================================================

TEST_CASE("from_codes sums three time scales", "[snac]") {
    const SnacQuantizer q = toy_quantizer(4, {4, 2, 1});
    // 4 frames: 1 coarse code, 2 mid codes, 4 fine codes.
    const std::vector<std::vector<std::uint32_t>> codes{{1}, {2, 3}, {0, 1, 2, 3}};
    const Result<Mat> z = q.from_codes(codes);
    REQUIRE(z.has_value());
    REQUIRE(z->rows == 4);
    REQUIRE(z->cols == 4);

    // Column 0 is the sum of the three levels' code values at each frame:
    //   frame 0: 1 (coarse) + 2 (mid) + 0 (fine) = 3
    //   frame 1: 1 + 2 + 1 = 4
    //   frame 2: 1 + 3 + 2 = 6
    //   frame 3: 1 + 3 + 3 = 7
    REQUIRE(approx(z->at(0, 0), 3.0f));
    REQUIRE(approx(z->at(1, 0), 4.0f));
    REQUIRE(approx(z->at(2, 0), 6.0f));
    REQUIRE(approx(z->at(3, 0), 7.0f));
    // Column 1 mirrors it, since the toy codebook is [c, -c].
    REQUIRE(approx(z->at(2, 1), -6.0f));
    // Nothing beyond the projected width.
    REQUIRE(approx(z->at(0, 2), 0.0f));
}

TEST_CASE("from_codes rejects code counts that do not cover the frames", "[snac]") {
    const SnacQuantizer q = toy_quantizer(4, {4, 2, 1});
    // The coarse level needs 1 code for 4 frames, not 2.
    const Result<Mat> z = q.from_codes({{1, 1}, {2, 3}, {0, 1, 2, 3}});
    REQUIRE(!z.has_value());
    REQUIRE(z.error().find("does not cover") != std::string::npos);
}

TEST_CASE("from_codes rejects an out-of-range code", "[snac]") {
    const SnacQuantizer q = toy_quantizer(4, {4, 2, 1});
    const Result<Mat> z = q.from_codes({{1}, {2, 3}, {0, 1, 2, 99}});
    REQUIRE(!z.has_value());
    REQUIRE(z.error().find("out of range") != std::string::npos);
}

TEST_CASE("from_codes rejects the wrong number of codebooks", "[snac]") {
    const SnacQuantizer q = toy_quantizer(4, {4, 2, 1});
    const Result<Mat> z = q.from_codes({{1}, {0, 1, 2, 3}});
    REQUIRE(!z.has_value());
    REQUIRE(z.error().find("expected 3 codebooks") != std::string::npos);
}

TEST_CASE("from_codes rejects an empty finest level", "[snac]") {
    const SnacQuantizer q = toy_quantizer(4, {4, 2, 1});
    const Result<Mat> z = q.from_codes({{}, {}, {}});
    REQUIRE(!z.has_value());
}

// =============================================================================
// Noise block
// =============================================================================

TEST_CASE("SnacNoiseBlock in Zero mode is the identity", "[snac]") {
    SnacNoiseBlock nb;
    nb.weight = identity(3);
    const Mat x = Mat::from_fn(5, 3, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r) + 0.1f * static_cast<float>(c);
    });
    InitRng rng(1);
    const Mat out = nb.forward(x, SnacNoise::Zero, rng);
    for (std::size_t i = 0; i < x.data.size(); ++i) {
        REQUIRE(approx(out.data[i], x.data[i]));
    }
}

TEST_CASE("SnacNoiseBlock shares one noise draw across all channels", "[snac]") {
    // The culprit this test exists for: PyTorch draws noise of shape
    // [B, 1, T] -- one sample per timestep, broadcast over channels. Drawing
    // per channel instead only makes the output slightly hissier, which is
    // undetectable by ear, so it has to be pinned here.
    //
    // With an identity weight and an all-ones input, out[t, c] = 1 + noise[t],
    // so every channel in a row must agree exactly, and rows must differ.
    SnacNoiseBlock nb;
    constexpr std::size_t channels = 8;
    nb.weight = identity(channels);
    const Mat x = Mat::ones(6, channels);

    InitRng rng(42);
    const Mat out = nb.forward(x, SnacNoise::Seeded, rng);

    for (std::size_t t = 0; t < out.rows; ++t) {
        for (std::size_t c = 1; c < channels; ++c) {
            REQUIRE(approx(out.at(t, c), out.at(t, 0)));
        }
    }
    // And the draws actually vary between timesteps.
    bool any_different = false;
    for (std::size_t t = 1; t < out.rows; ++t) {
        if (!approx(out.at(t, 0), out.at(0, 0))) {
            any_different = true;
        }
    }
    REQUIRE(any_different);
    // Something was actually added.
    REQUIRE(!approx(out.at(0, 0), 1.0f));
}

// =============================================================================
// Residual unit
// =============================================================================

TEST_CASE("SnacResidualUnit preserves length at every dilation", "[snac]") {
    constexpr std::size_t channels = 4;
    for (const std::size_t dilation : {std::size_t{1}, std::size_t{3}, std::size_t{9}}) {
        SnacResidualUnit unit;
        unit.dilation = dilation;
        unit.kernel = 7;
        unit.depthwise = true;
        unit.alpha1.assign(channels, 1.0f);
        unit.alpha2.assign(channels, 1.0f);
        unit.conv1_weight = Mat::zeros(channels, 7);
        unit.conv2_weight = Mat::zeros(channels, channels);

        const Mat x = Mat::ones(32, channels);
        const Mat out = unit.forward(x);
        REQUIRE(out.rows == 32);
        REQUIRE(out.cols == channels);
    }
}

TEST_CASE("SnacResidualUnit adds its input through the skip", "[snac]") {
    // With both convolutions zeroed, the branch contributes nothing and the
    // output must equal the input exactly -- a missing skip connection would
    // show up as zeros.
    constexpr std::size_t channels = 3;
    SnacResidualUnit unit;
    unit.dilation = 1;
    unit.kernel = 7;
    unit.depthwise = true;
    unit.alpha1.assign(channels, 1.0f);
    unit.alpha2.assign(channels, 1.0f);
    unit.conv1_weight = Mat::zeros(channels, 7);
    unit.conv2_weight = Mat::zeros(channels, channels);

    const Mat x = Mat::from_fn(10, channels, [](std::size_t r, std::size_t c) {
        return 0.1f * static_cast<float>(r) - 0.05f * static_cast<float>(c);
    });
    const Mat out = unit.forward(x);
    for (std::size_t i = 0; i < x.data.size(); ++i) {
        REQUIRE(approx(out.data[i], x.data[i]));
    }
}

// =============================================================================
// Decoder block
// =============================================================================

TEST_CASE("SnacDecoderBlock multiplies length by its stride", "[snac]") {
    for (const std::size_t stride : {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        constexpr std::size_t c_in = 8;
        const std::size_t c_out = c_in / 2;

        SnacDecoderBlock block;
        block.stride = stride;
        block.kernel = 2 * stride;
        block.padding = (stride + 1) / 2;
        block.output_padding = stride % 2;
        block.out_channels = c_out;
        block.alpha.assign(c_in, 1.0f);
        block.up_weight = Mat::zeros(c_in, c_out * block.kernel);
        block.up_bias.assign(c_out, 0.0f);
        for (std::size_t u = 0; u < 3; ++u) {
            SnacResidualUnit unit;
            unit.dilation = u == 0 ? 1 : (u == 1 ? 3 : 9);
            unit.kernel = 7;
            unit.depthwise = true;
            unit.alpha1.assign(c_out, 1.0f);
            unit.alpha2.assign(c_out, 1.0f);
            unit.conv1_weight = Mat::zeros(c_out, 7);
            unit.conv2_weight = Mat::zeros(c_out, c_out);
            block.units.push_back(std::move(unit));
        }

        InitRng rng(3);
        const Mat x = Mat::ones(6, c_in);
        const Mat out = block.forward(x, SnacNoise::Zero, rng);
        REQUIRE(out.rows == 6 * stride);
        REQUIRE(out.cols == c_out);
    }
}

// =============================================================================
// The real decoder
// =============================================================================

TEST_CASE("SnacDecoder loads the 24 kHz checkpoint", "[snac][.integration]") {
    if (!std::filesystem::exists(kSnacPath)) {
        SKIP("models/snac_24khz.bin not present");
    }
    const Result<SnacDecoder> d = SnacDecoder::load(kSnacPath, SnacConfig::snac_24khz());
    if (!d.has_value()) {
        FAIL(d.error());
    }

    REQUIRE(d->blocks.size() == 4);
    REQUIRE(d->quantizer.levels.size() == 3);
    REQUIRE(d->in_mix_weight.has_value());

    // Channel widths halve down the stack: 1024 -> 512 -> 256 -> 128 -> 64.
    const std::vector<std::size_t> want_out{512, 256, 128, 64};
    const std::vector<std::size_t> want_stride{8, 8, 4, 2};
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(d->blocks[i].out_channels == want_out[i]);
        REQUIRE(d->blocks[i].stride == want_stride[i]);
        REQUIRE(d->blocks[i].kernel == 2 * want_stride[i]);
        REQUIRE(d->blocks[i].noise.has_value());
        REQUIRE(d->blocks[i].units.size() == 3);
        // A transposed convolution's weight is [Cin, Cout*K].
        REQUIRE(d->blocks[i].up_weight.cols == want_out[i] * d->blocks[i].kernel);
        // Its bias is per output channel, not per input channel.
        REQUIRE(d->blocks[i].up_bias.size() == want_out[i]);
    }

    REQUIRE(d->out_alpha.size() == 64);
    REQUIRE(d->out_weight.rows == 1);
    REQUIRE(d->out_weight.cols == 64 * 7);

    // The whole checkpoint is 19.9 M parameters including the encoder; the
    // decoder plus quantizer is the bulk of it.
    INFO("decoder parameters: " << d->parameter_count());
    REQUIRE(d->parameter_count() > 10'000'000);
    REQUIRE(d->parameter_count() < 20'000'000);
}

TEST_CASE("SnacDecoder produces exactly frames * 512 samples", "[snac][.integration]") {
    if (!std::filesystem::exists(kSnacPath)) {
        SKIP("models/snac_24khz.bin not present");
    }
    const Result<SnacDecoder> d = SnacDecoder::load(kSnacPath, SnacConfig::snac_24khz());
    REQUIRE(d.has_value());

    // Two Orpheus groups: 8 frames.
    const std::vector<std::vector<std::uint32_t>> codes{
        {100, 200},
        {300, 400, 500, 600},
        {700, 800, 900, 1000, 1100, 1200, 1300, 1400},
    };

    const Result<std::vector<float>> audio = d->decode(codes, SnacNoise::Zero, 0);
    if (!audio.has_value()) {
        FAIL(audio.error());
    }
    REQUIRE(audio->size() == 8 * 512);
    REQUIRE(audio->size() == 4096);

    // Tanh must bound the output, which is what makes the int16 conversion
    // safe. Arbitrary codes are not speech, so only the range is asserted.
    const WaveStats s = wave_stats(*audio);
    INFO(s.describe());
    REQUIRE(s.in_range());
    REQUIRE(s.peak <= 1.0f);
}

TEST_CASE("SnacDecoder is deterministic with noise suppressed", "[snac][.integration]") {
    if (!std::filesystem::exists(kSnacPath)) {
        SKIP("models/snac_24khz.bin not present");
    }
    const Result<SnacDecoder> d = SnacDecoder::load(kSnacPath, SnacConfig::snac_24khz());
    REQUIRE(d.has_value());

    const std::vector<std::vector<std::uint32_t>> codes{{5}, {6, 7}, {8, 9, 10, 11}};

    const Result<std::vector<float>> a = d->decode(codes, SnacNoise::Zero, 0);
    const Result<std::vector<float>> b = d->decode(codes, SnacNoise::Zero, 99);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    // The seed is ignored in Zero mode, so these must be bit-identical.
    REQUIRE(a->size() == b->size());
    for (std::size_t i = 0; i < a->size(); ++i) {
        REQUIRE((*a)[i] == (*b)[i]);
    }

    // Seeded noise changes the output, and does so reproducibly per seed.
    const Result<std::vector<float>> c = d->decode(codes, SnacNoise::Seeded, 7);
    const Result<std::vector<float>> e = d->decode(codes, SnacNoise::Seeded, 7);
    REQUIRE(c.has_value());
    REQUIRE(e.has_value());
    for (std::size_t i = 0; i < c->size(); ++i) {
        REQUIRE((*c)[i] == (*e)[i]);
    }
    bool differs = false;
    for (std::size_t i = 0; i < a->size(); ++i) {
        if ((*a)[i] != (*c)[i]) {
            differs = true;
            break;
        }
    }
    REQUIRE(differs);
}

TEST_CASE("SnacDecoder rejects a mismatched config", "[snac][.integration]") {
    if (!std::filesystem::exists(kSnacPath)) {
        SKIP("models/snac_24khz.bin not present");
    }
    // The 44.1 kHz shape against 24 kHz weights: every dimension is wrong, and
    // the loader must say which rather than producing a decoder that runs and
    // outputs noise.
    SnacConfig wrong = SnacConfig::snac_24khz();
    wrong.decoder_dim = 1536;
    const Result<SnacDecoder> d = SnacDecoder::load(kSnacPath, wrong);
    REQUIRE(!d.has_value());
    INFO(d.error());
    REQUIRE(d.error().find("1536") != std::string::npos);
}

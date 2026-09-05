#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <numbers>
#include <vector>

#include "rt/hubert.hpp"
#include "rt/resample.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

namespace {

constexpr const char* kCodecPath = "models/omnivoice-tokenizer-Q8_0.gguf";

/// A vowel-ish waveform: a fundamental plus two harmonics, slowly modulated.
[[nodiscard]] std::vector<float> voice_like(std::size_t n, float rate) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / rate;
        const float env = 0.6f + 0.4f * std::sin(2.0f * std::numbers::pi_v<float> * 3.0f * t);
        out[i] = 0.2f * env *
                 (std::sin(2.0f * std::numbers::pi_v<float> * 140.0f * t) +
                  0.5f * std::sin(2.0f * std::numbers::pi_v<float> * 700.0f * t) +
                  0.25f * std::sin(2.0f * std::numbers::pi_v<float> * 2400.0f * t));
    }
    return out;
}

}  // namespace

// =============================================================================
// Length arithmetic
// =============================================================================

TEST_CASE("the feature extractor reduces 16 kHz by exactly 320", "[hubert]") {
    const HubertConfig c = HubertConfig::omnivoice_semantic();
    REQUIRE(c.downsample_factor() == 320);
    // 5 * 2^6.
    REQUIRE(c.conv_stride.size() == 7);
    REQUIRE(c.conv_kernel.size() == 7);
    REQUIRE(c.conv_dim.size() == 7);
}

TEST_CASE("a padded second of audio makes exactly fifty frames", "[hubert]") {
    // The identity the codec depends on: n frames of 24 kHz audio become
    // n * 640 samples at 16 kHz, are padded by 160 on each side, and come out
    // as exactly 2n frames -- twice the codec's rate, so keeping every other
    // one lands on n.
    const HubertConfig c = HubertConfig::omnivoice_semantic();
    for (const std::size_t frames : {1u, 2u, 25u, 100u, 251u}) {
        const std::size_t samples = frames * 640 + 320;
        REQUIRE(c.feature_frames(samples) == 2 * frames);
    }
}

TEST_CASE("feature_frames follows the convolution chain", "[hubert]") {
    const HubertConfig c = HubertConfig::omnivoice_semantic();
    // Worked by hand: (960-10)/5+1 = 191, then /2 five more times and /2 again.
    REQUIRE(c.feature_frames(960) == 2);
    REQUIRE(c.feature_frames(16000) == 49);
    // Too short for the stack to consume at all.
    REQUIRE(c.feature_frames(0) == 0);
    REQUIRE(c.feature_frames(9) == 0);
    REQUIRE(c.feature_frames(100) == 0);
}

// =============================================================================
// Normalisation
// =============================================================================

TEST_CASE("layer_norm_rows normalises each row", "[hubert]") {
    Mat x = Mat::from_fn(3, 4, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 10 + c);
    });
    const std::vector<float> gamma(4, 1.0f);
    const std::vector<float> beta(4, 0.0f);
    layer_norm_rows(x, gamma, beta, 1e-5f);

    for (std::size_t r = 0; r < x.rows; ++r) {
        float mean = 0.0f;
        for (std::size_t c = 0; c < x.cols; ++c) {
            mean += x.at(r, c);
        }
        REQUIRE(std::fabs(mean / 4.0f) < 1e-5f);
        float var = 0.0f;
        for (std::size_t c = 0; c < x.cols; ++c) {
            var += x.at(r, c) * x.at(r, c);
        }
        // Biased variance, as PyTorch computes it.
        REQUIRE(approx(var / 4.0f, 1.0f));
    }
    // Every row held the same spread, so every row normalises identically.
    for (std::size_t c = 0; c < x.cols; ++c) {
        REQUIRE(approx(x.at(0, c), x.at(2, c)));
    }
}

TEST_CASE("layer_norm_rows applies gamma and beta", "[hubert]") {
    Mat x = Mat::from_fn(1, 4, [](std::size_t, std::size_t c) {
        return static_cast<float>(c);
    });
    const std::vector<float> gamma{2.0f, 2.0f, 2.0f, 2.0f};
    const std::vector<float> beta{1.0f, 1.0f, 1.0f, 1.0f};
    layer_norm_rows(x, gamma, beta, 1e-5f);
    float mean = 0.0f;
    for (std::size_t c = 0; c < x.cols; ++c) {
        mean += x.at(0, c);
    }
    // Scaled by 2 and shifted by 1, so the mean is the shift.
    REQUIRE(approx(mean / 4.0f, 1.0f));
}

TEST_CASE("group_norm_channels normalises down the columns", "[hubert]") {
    // The transpose of LayerNorm: each channel against its own history. A
    // channel that never moves comes out at exactly the bias.
    Mat x = Mat::from_fn(8, 3, [](std::size_t r, std::size_t c) {
        return c == 2 ? 5.0f : static_cast<float>(r) * (c == 0 ? 1.0f : -2.0f);
    });
    const std::vector<float> gamma{1.0f, 1.0f, 1.0f};
    const std::vector<float> beta{0.0f, 0.0f, 0.5f};
    group_norm_channels(x, gamma, beta, 1e-5f);

    for (std::size_t c = 0; c < 2; ++c) {
        float mean = 0.0f;
        float var = 0.0f;
        for (std::size_t r = 0; r < x.rows; ++r) {
            mean += x.at(r, c);
        }
        REQUIRE(std::fabs(mean / 8.0f) < 1e-4f);
        for (std::size_t r = 0; r < x.rows; ++r) {
            var += x.at(r, c) * x.at(r, c);
        }
        REQUIRE(approx(var / 8.0f, 1.0f));
    }
    for (std::size_t r = 0; r < x.rows; ++r) {
        REQUIRE(approx(x.at(r, 2), 0.5f));
    }
}

// =============================================================================
// Activation
// =============================================================================

TEST_CASE("gelu_erf is the exact GELU, not the tanh approximation", "[hubert]") {
    Mat x(std::vector<float>{0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 3.0f}, 1, 7);
    gelu_erf_inplace(x);
    REQUIRE(approx(x.at(0, 0), 0.0f));
    REQUIRE(approx(x.at(0, 1), 0.841344746f));
    REQUIRE(approx(x.at(0, 2), -0.158655254f));
    REQUIRE(approx(x.at(0, 3), 1.95449974f));
    REQUIRE(approx(x.at(0, 4), 0.345731231f));
    REQUIRE(approx(x.at(0, 5), -0.154268769f));
    REQUIRE(approx(x.at(0, 6), 2.99595031f));

    // The tanh form differs by about 1.5e-4 at x = 1 -- invisible in a
    // language model's logits, and not what this checkpoint was trained with.
    const float tanh_at_one =
        0.5f * (1.0f + std::tanh(std::sqrt(2.0f / std::numbers::pi_v<float>) *
                                 (1.0f + 0.044715f)));
    REQUIRE(std::fabs(x.at(0, 1) - tanh_at_one) > 1e-5f);
    REQUIRE(std::fabs(x.at(0, 1) - tanh_at_one) < 1e-3f);
}

// =============================================================================
// The real checkpoint
// =============================================================================

TEST_CASE("HubertModel loads from the tokenizer GGUF", "[hubert][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("no OmniVoice tokenizer checkpoint in models/");
    }
    const Result<GgufFile> gguf = GgufFile::open(kCodecPath);
    REQUIRE(gguf.has_value());
    const Result<HubertModel> m =
        HubertModel::load(*gguf, HubertConfig::omnivoice_semantic(), "semantic_model");
    REQUIRE(m.has_value());

    REQUIRE(m->feature_layers.size() == 7);
    REQUIRE(m->layers.size() == 12);
    // Only the first convolution carries a norm, which is what
    // feat_extract_norm="group" means.
    REQUIRE(m->feature_layers[0].group_norm);
    for (std::size_t i = 1; i < m->feature_layers.size(); ++i) {
        REQUIRE_FALSE(m->feature_layers[i].group_norm);
    }
    // HuBERT base, to the nearest hundred thousand.
    REQUIRE(m->parameter_count() > 94'000'000);
    REQUIRE(m->parameter_count() < 95'000'000);
}

TEST_CASE("HubertModel produces one frame per 320 samples", "[hubert][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("no OmniVoice tokenizer checkpoint in models/");
    }
    const Result<GgufFile> gguf = GgufFile::open(kCodecPath);
    REQUIRE(gguf.has_value());
    const Result<HubertModel> m =
        HubertModel::load(*gguf, HubertConfig::omnivoice_semantic(), "semantic_model");
    REQUIRE(m.has_value());

    // Half a second of 24 kHz audio, prepared the way the codec prepares it.
    constexpr std::size_t kFrames = 12;
    const std::vector<float> wav = voice_like(kFrames * 960, 24000.0f);
    const std::vector<float> at16 = resample(wav, 24000, 16000);
    REQUIRE(at16.size() == kFrames * 640);

    std::vector<float> padded(160, 0.0f);
    padded.insert(padded.end(), at16.begin(), at16.end());
    padded.insert(padded.end(), 160, 0.0f);

    const Result<Mat> h = m->mean_hidden_states(padded);
    REQUIRE(h.has_value());
    REQUIRE(h->rows == 2 * kFrames);
    REQUIRE(h->cols == 768);

    // Post-LayerNorm features sit near unit scale; anything wildly outside
    // that is a norm applied on the wrong axis.
    double sum_sq = 0.0;
    for (const float v : h->data) {
        REQUIRE(std::isfinite(v));
        sum_sq += static_cast<double>(v) * v;
    }
    const double rms = std::sqrt(sum_sq / static_cast<double>(h->data.size()));
    REQUIRE(rms > 0.05);
    REQUIRE(rms < 5.0);
}

TEST_CASE("HubertModel is deterministic", "[hubert][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("no OmniVoice tokenizer checkpoint in models/");
    }
    const Result<GgufFile> gguf = GgufFile::open(kCodecPath);
    REQUIRE(gguf.has_value());
    const Result<HubertModel> m =
        HubertModel::load(*gguf, HubertConfig::omnivoice_semantic(), "semantic_model");
    REQUIRE(m.has_value());

    const std::vector<float> wav = voice_like(4 * 640 + 320, 16000.0f);
    const Result<Mat> a = m->mean_hidden_states(wav);
    const Result<Mat> b = m->mean_hidden_states(wav);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    REQUIRE(a->data.size() == b->data.size());
    for (std::size_t i = 0; i < a->data.size(); ++i) {
        REQUIRE(a->data[i] == b->data[i]);
    }
}

TEST_CASE("HubertModel rejects a clip it cannot consume", "[hubert][.integration]") {
    if (!std::filesystem::exists(kCodecPath)) {
        SKIP("no OmniVoice tokenizer checkpoint in models/");
    }
    const Result<GgufFile> gguf = GgufFile::open(kCodecPath);
    REQUIRE(gguf.has_value());
    const Result<HubertModel> m =
        HubertModel::load(*gguf, HubertConfig::omnivoice_semantic(), "semantic_model");
    REQUIRE(m.has_value());
    REQUIRE_FALSE(m->mean_hidden_states(std::vector<float>(100, 0.0f)).has_value());
}

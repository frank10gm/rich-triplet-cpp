#include <catch2/catch_test_macros.hpp>

#if RT_FEATURE_METAL

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <filesystem>
#include <memory>
#include <vector>

#include "rt/metal_omnivoice.hpp"
#include "rt/omnivoice.hpp"
#include "rt/tokenizer.hpp"

using namespace rt;

namespace {

constexpr const char* kLmPath = "models/omnivoice-base-Q8_0.gguf";

/// A sequence of the shape a diffusion step actually runs: a style and text
/// prefix, then masked frames.
[[nodiscard]] std::vector<OmniToken> sample_sequence(const Config6& cfg, std::size_t frames) {
    std::vector<OmniToken> seq;
    seq.push_back(OmniToken::text(cfg.lang_start));
    for (std::size_t i = 0; i < 6; ++i) {
        seq.push_back(OmniToken::text(1000 + i));
    }
    seq.push_back(OmniToken::text(cfg.lang_end));
    seq.push_back(OmniToken::text(cfg.text_start));
    for (std::size_t i = 0; i < 10; ++i) {
        seq.push_back(OmniToken::text(2000 + i));
    }
    seq.push_back(OmniToken::text(cfg.text_end));

    // A few decided frames, as voice cloning produces, then masked ones.
    for (std::size_t t = 0; t < 4; ++t) {
        OmniToken token;
        token.audio.resize(cfg.num_audio_codebook);
        for (std::size_t c = 0; c < cfg.num_audio_codebook; ++c) {
            token.audio[c] = static_cast<std::uint32_t>((c * 37 + t * 11) % 1024);
        }
        seq.push_back(std::move(token));
    }
    for (std::size_t t = 0; t < frames; ++t) {
        seq.push_back(OmniToken::masked(cfg.num_audio_codebook, cfg.audio_mask_id));
    }
    return seq;
}

}  // namespace

TEST_CASE("MetalOmniContext matches the CPU forward pass", "[metal][omnivoice][.integration]") {
    if (!std::filesystem::exists(kLmPath)) {
        SKIP("models/omnivoice-base-Q8_0.gguf not present");
    }
    // Two copies of the model: `create` consumes the projections of the one it
    // uploads, so the reference has to be a second load.
    const Config6 cfg = Config6::omnivoice();
    const Result<OmniLm> cpu = OmniLm::load(kLmPath, cfg);
    REQUIRE(cpu.has_value());
    Result<OmniLm> gpu_weights = OmniLm::load(kLmPath, cfg);
    REQUIRE(gpu_weights.has_value());

    const std::unique_ptr<MetalOmniContext> ctx =
        MetalOmniContext::create(*gpu_weights, MetalOmniContext::kMaxTokens);
    REQUIRE(ctx != nullptr);
    REQUIRE(ctx->max_tokens() == MetalOmniContext::kMaxTokens);
    REQUIRE(ctx->buffer_bytes() > 900'000'000);

    const std::vector<OmniToken> seq = sample_sequence(cfg, 40);
    const Result<Mat> want = cpu->forward(seq);
    const Result<Mat> got = ctx->forward(seq);
    REQUIRE(want.has_value());
    REQUIRE(got.has_value());
    REQUIRE(got->rows == want->rows);
    REQUIRE(got->cols == want->cols);
    REQUIRE(got->cols == cfg.audio_table_size());

    // The two paths sum in different orders -- BLAS chunks the weight while
    // the GPU accumulates 8x8 tiles -- so they agree to f32 epsilon and not
    // further. At logits of order 10 that is a few units in the last place.
    double max_rel = 0.0;
    for (std::size_t i = 0; i < want->data.size(); ++i) {
        REQUIRE(std::isfinite(got->data[i]));
        const double scale = std::max(1.0, std::fabs(static_cast<double>(want->data[i])));
        max_rel = std::max(max_rel, std::fabs(static_cast<double>(want->data[i]) -
                                              got->data[i]) / scale);
    }
    REQUIRE(max_rel < 1e-3);

    // What the sampler actually reads is the argmax per (position, codebook),
    // and those have to agree exactly or the two paths would diverge after the
    // first unmasking step.
    const std::size_t vocab = cfg.audio_vocab_size;
    for (std::size_t r = 0; r < want->rows; ++r) {
        for (std::size_t c = 0; c < cfg.num_audio_codebook; ++c) {
            std::size_t best_cpu = 0;
            std::size_t best_gpu = 0;
            float v_cpu = -std::numeric_limits<float>::infinity();
            float v_gpu = -std::numeric_limits<float>::infinity();
            for (std::size_t v = 0; v < vocab; ++v) {
                if (want->at(r, c * vocab + v) > v_cpu) {
                    v_cpu = want->at(r, c * vocab + v);
                    best_cpu = v;
                }
                if (got->at(r, c * vocab + v) > v_gpu) {
                    v_gpu = got->at(r, c * vocab + v);
                    best_gpu = v;
                }
            }
            REQUIRE(best_cpu == best_gpu);
        }
    }
}

TEST_CASE("MetalOmniContext is deterministic", "[metal][omnivoice][.integration]") {
    if (!std::filesystem::exists(kLmPath)) {
        SKIP("models/omnivoice-base-Q8_0.gguf not present");
    }
    const Config6 cfg = Config6::omnivoice();
    Result<OmniLm> lm = OmniLm::load(kLmPath, cfg);
    REQUIRE(lm.has_value());
    const std::unique_ptr<MetalOmniContext> ctx =
        MetalOmniContext::create(*lm, MetalOmniContext::kMaxTokens);
    REQUIRE(ctx != nullptr);

    const std::vector<OmniToken> seq = sample_sequence(cfg, 20);
    const Result<Mat> a = ctx->forward(seq);
    const Result<Mat> b = ctx->forward(seq);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    for (std::size_t i = 0; i < a->data.size(); ++i) {
        REQUIRE(a->data[i] == b->data[i]);
    }
}

TEST_CASE("MetalOmniContext handles lengths that are not tile multiples",
          "[metal][omnivoice][.integration]") {
    if (!std::filesystem::exists(kLmPath)) {
        SKIP("models/omnivoice-base-Q8_0.gguf not present");
    }
    // The GEMM tile is 64 wide, so every sequence but a multiple of it runs
    // with padding rows. Those rows are zeroed and must not leak into the real
    // ones -- attention is the only operator that crosses positions, and it is
    // dispatched over the true length rather than the padded one.
    const Config6 cfg = Config6::omnivoice();
    const Result<OmniLm> cpu = OmniLm::load(kLmPath, cfg);
    REQUIRE(cpu.has_value());
    Result<OmniLm> gpu_weights = OmniLm::load(kLmPath, cfg);
    REQUIRE(gpu_weights.has_value());
    const std::unique_ptr<MetalOmniContext> ctx =
        MetalOmniContext::create(*gpu_weights, MetalOmniContext::kMaxTokens);
    REQUIRE(ctx != nullptr);

    for (const std::size_t frames : {std::size_t{1}, std::size_t{5}, std::size_t{44}}) {
        std::vector<OmniToken> seq;
        for (std::size_t t = 0; t < frames; ++t) {
            seq.push_back(OmniToken::masked(cfg.num_audio_codebook, cfg.audio_mask_id));
        }
        const Result<Mat> want = cpu->forward(seq);
        const Result<Mat> got = ctx->forward(seq);
        REQUIRE(want.has_value());
        REQUIRE(got.has_value());
        REQUIRE(got->rows == frames);
        for (std::size_t i = 0; i < want->data.size(); ++i) {
            const double scale = std::max(1.0, std::fabs(static_cast<double>(want->data[i])));
            REQUIRE(std::fabs(static_cast<double>(want->data[i]) - got->data[i]) / scale < 1e-3);
        }
    }
}

TEST_CASE("MetalOmniContext rejects sequences it cannot run",
          "[metal][omnivoice][.integration]") {
    if (!std::filesystem::exists(kLmPath)) {
        SKIP("models/omnivoice-base-Q8_0.gguf not present");
    }
    const Config6 cfg = Config6::omnivoice();
    Result<OmniLm> lm = OmniLm::load(kLmPath, cfg);
    REQUIRE(lm.has_value());
    // Deliberately small, so the limit is reachable without a huge allocation.
    const std::unique_ptr<MetalOmniContext> ctx = MetalOmniContext::create(*lm, 128);
    REQUIRE(ctx != nullptr);

    REQUIRE_FALSE(ctx->forward({}).has_value());
    REQUIRE_FALSE(ctx->forward(sample_sequence(cfg, 400)).has_value());

    // A bad code has to be caught on the host, before anything is dispatched.
    std::vector<OmniToken> bad = sample_sequence(cfg, 4);
    bad.back().audio[0] = 99999;
    REQUIRE_FALSE(ctx->forward(bad).has_value());
}

#endif  // RT_FEATURE_METAL

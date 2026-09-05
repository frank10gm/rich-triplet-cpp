#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "rt/omnivoice.hpp"
#include "rt/transformer5.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

namespace {

constexpr const char* kLmPath = "models/omnivoice-base-Q8_0.gguf";

}  // namespace

// =============================================================================
// Config
// =============================================================================

TEST_CASE("Config6 matches the OmniVoice checkpoint", "[omnivoice]") {
    const Config6 c = Config6::omnivoice();
    // A Qwen3 0.6B backbone.
    REQUIRE(c.num_hidden_layers == 28);
    REQUIRE(c.hidden_size == 1024);
    REQUIRE(c.num_attention_heads == 16);
    REQUIRE(c.num_key_value_heads == 8);
    REQUIRE(c.head_dim == 128);
    REQUIRE(c.intermediate_size == 3072);
    REQUIRE(c.rope_theta == 1000000.0f);

    // Eight codebooks of 1024 codes plus a mask token each.
    REQUIRE(c.num_audio_codebook == 8);
    REQUIRE(c.audio_vocab_size == 1025);
    REQUIRE(c.audio_mask_id == 1024);
    REQUIRE(c.audio_table_size() == 8200);
}

TEST_CASE("audio rows are one contiguous block per codebook", "[omnivoice]") {
    const Config6 c = Config6::omnivoice();
    // Codebook i occupies rows [i*1025, (i+1)*1025).
    REQUIRE(c.audio_row(0, 0) == 0);
    REQUIRE(c.audio_row(0, 1024) == 1024);   // codebook 0's mask token
    REQUIRE(c.audio_row(1, 0) == 1025);
    REQUIRE(c.audio_row(7, 1024) == 8199);   // the last row of the table
    REQUIRE(c.audio_row(7, 1024) + 1 == c.audio_table_size());
}

// =============================================================================
// Tokens
// =============================================================================

TEST_CASE("OmniToken distinguishes text from audio positions", "[omnivoice]") {
    const OmniToken t = OmniToken::text(1234);
    REQUIRE(!t.is_audio());
    REQUIRE(t.text_id == 1234);

    const OmniToken a = OmniToken::masked(8, 1024);
    REQUIRE(a.is_audio());
    REQUIRE(a.audio.size() == 8);
    for (const std::uint32_t v : a.audio) {
        REQUIRE(v == 1024);
    }
}

// =============================================================================
// Schedules
// =============================================================================

TEST_CASE("omni_time_steps spans 0 to 1", "[omnivoice]") {
    for (const float shift : {0.1f, 1.0f, 3.0f}) {
        const std::vector<float> ts = omni_time_steps(32, shift);
        REQUIRE(ts.size() == 33);
        REQUIRE(approx(ts.front(), 0.0f));
        REQUIRE(approx(ts.back(), 1.0f));
        // Monotone, or the per-step share would go negative.
        for (std::size_t i = 1; i < ts.size(); ++i) {
            REQUIRE(ts[i] >= ts[i - 1]);
        }
    }
}

TEST_CASE("t_shift below 1 front-loads the schedule", "[omnivoice]") {
    // The warp is what decides how much is unmasked early. At shift 1 the grid
    // is linear; below 1 the early steps cover less of the interval, so fewer
    // tokens are committed before the model has context.
    const std::vector<float> linear = omni_time_steps(10, 1.0f);
    const std::vector<float> shifted = omni_time_steps(10, 0.1f);
    REQUIRE(approx(linear[5], 0.5f));
    REQUIRE(shifted[5] < linear[5]);
}

TEST_CASE("omni_unmask_schedule accounts for every position", "[omnivoice]") {
    // Nothing may be left masked by rounding: the codec would reject a mask id
    // as an out-of-range code.
    for (const std::size_t total : {std::size_t{8}, std::size_t{800}, std::size_t{4001}}) {
        for (const std::size_t steps : {std::size_t{1}, std::size_t{8}, std::size_t{32}}) {
            const std::vector<std::size_t> s = omni_unmask_schedule(total, steps, 0.1f);
            REQUIRE(s.size() == steps);
            REQUIRE(std::accumulate(s.begin(), s.end(), std::size_t{0}) == total);
        }
    }
}

TEST_CASE("omni_unmask_schedule never exceeds what is left", "[omnivoice]") {
    const std::vector<std::size_t> s = omni_unmask_schedule(100, 32, 0.1f);
    std::size_t seen = 0;
    for (const std::size_t n : s) {
        seen += n;
        REQUIRE(seen <= 100);
    }
    REQUIRE(seen == 100);
}

TEST_CASE("a single step unmasks everything at once", "[omnivoice]") {
    const std::vector<std::size_t> s = omni_unmask_schedule(64, 1, 0.1f);
    REQUIRE(s.size() == 1);
    REQUIRE(s[0] == 64);
}

// =============================================================================
// Prompt construction
// =============================================================================

TEST_CASE("omni_build_unconditional is target frames only", "[omnivoice]") {
    // Classifier-free guidance conditions on nothing: no style block, no text.
    const Config6 cfg = Config6::omnivoice();
    const std::vector<OmniToken> seq = omni_build_unconditional(cfg, 10);
    REQUIRE(seq.size() == 10);
    for (const OmniToken& t : seq) {
        REQUIRE(t.is_audio());
        REQUIRE(t.audio.size() == cfg.num_audio_codebook);
        REQUIRE(t.audio[0] == cfg.audio_mask_id);
    }
}

TEST_CASE("omni_build_conditional lays out style, text, then frames",
          "[omnivoice][.integration]") {
    if (!std::filesystem::exists(kLmPath)) {
        SKIP("models/omnivoice-base-Q8_0.gguf not present");
    }
    const Result<GgufFile> gguf = GgufFile::open(kLmPath);
    REQUIRE(gguf.has_value());
    const Result<HfBpeTokenizer> tok = load_gguf_tokenizer(*gguf);
    if (!tok.has_value()) {
        FAIL(tok.error());
    }

    const Config6 cfg = Config6::omnivoice();
    OmniRequest r;
    r.text = "Ciao.";
    r.language = "Italian";

    const Result<std::vector<OmniToken>> seq = omni_build_conditional(*tok, cfg, r, 10);
    REQUIRE(seq.has_value());

    // The markers are injected as raw ids, never encoded as literal text.
    REQUIRE((*seq)[0].text_id == cfg.lang_start);
    REQUIRE(!(*seq)[0].is_audio());

    // The trailing block is the masked frames.
    REQUIRE(seq->size() > 10);
    for (std::size_t i = seq->size() - 10; i < seq->size(); ++i) {
        REQUIRE((*seq)[i].is_audio());
        REQUIRE((*seq)[i].audio[0] == cfg.audio_mask_id);
    }
    // Everything before them is text.
    for (std::size_t i = 0; i + 10 < seq->size(); ++i) {
        REQUIRE(!(*seq)[i].is_audio());
    }

    // The markers all appear, in order.
    std::vector<std::size_t> ids;
    for (const OmniToken& t : *seq) {
        if (!t.is_audio()) {
            ids.push_back(t.text_id);
        }
    }
    const auto pos = [&ids](std::size_t id) {
        return std::find(ids.begin(), ids.end(), id) - ids.begin();
    };
    REQUIRE(pos(cfg.lang_start) < pos(cfg.lang_end));
    REQUIRE(pos(cfg.lang_end) < pos(cfg.instruct_start));
    REQUIRE(pos(cfg.instruct_start) < pos(cfg.instruct_end));
    REQUIRE(pos(cfg.instruct_end) < pos(cfg.text_start));
    REQUIRE(pos(cfg.text_start) < pos(cfg.text_end));
}

TEST_CASE("omni_build_conditional rejects empty input", "[omnivoice][.integration]") {
    if (!std::filesystem::exists(kLmPath)) {
        SKIP("models/omnivoice-base-Q8_0.gguf not present");
    }
    const Result<GgufFile> gguf = GgufFile::open(kLmPath);
    REQUIRE(gguf.has_value());
    const Result<HfBpeTokenizer> tok = load_gguf_tokenizer(*gguf);
    REQUIRE(tok.has_value());

    const Config6 cfg = Config6::omnivoice();
    OmniRequest r;
    r.text = "";
    REQUIRE(!omni_build_conditional(*tok, cfg, r, 10).has_value());
    r.text = "Ciao.";
    REQUIRE(!omni_build_conditional(*tok, cfg, r, 0).has_value());
}

// =============================================================================
// The real model
// =============================================================================

TEST_CASE("OmniLm loads and embeds", "[omnivoice][.integration]") {
    if (!std::filesystem::exists(kLmPath)) {
        SKIP("models/omnivoice-base-Q8_0.gguf not present");
    }
    const Result<OmniLm> lm = OmniLm::load(kLmPath, Config6::omnivoice());
    if (!lm.has_value()) {
        FAIL(lm.error());
    }
    REQUIRE(lm->layers.size() == 28);
    REQUIRE(lm->text_embed.has_value());
    REQUIRE(lm->audio_embed.has_value());
    REQUIRE(lm->text_embed->rows == lm->config.text_vocab_size);
    REQUIRE(lm->audio_embed->rows == lm->config.audio_table_size());
    REQUIRE(lm->inv_freq_cache.size() == lm->config.head_dim / 2);

    // A masked audio position is the sum of the eight mask rows, which is what
    // lets decoding start from nothing.
    const std::vector<OmniToken> seq{
        OmniToken::text(1000),
        OmniToken::masked(lm->config.num_audio_codebook, lm->config.audio_mask_id),
    };
    const Result<Mat> e = lm->embed(seq);
    REQUIRE(e.has_value());
    REQUIRE(e->rows == 2);
    REQUIRE(e->cols == lm->config.hidden_size);
    bool nonzero = false;
    for (std::size_t c = 0; c < e->cols; ++c) {
        if (e->at(1, c) != 0.0f) {
            nonzero = true;
        }
    }
    REQUIRE(nonzero);
}

TEST_CASE("OmniLm rejects malformed positions", "[omnivoice][.integration]") {
    if (!std::filesystem::exists(kLmPath)) {
        SKIP("models/omnivoice-base-Q8_0.gguf not present");
    }
    const Result<OmniLm> lm = OmniLm::load(kLmPath, Config6::omnivoice());
    REQUIRE(lm.has_value());

    // Too few codebooks for an audio position.
    OmniToken bad;
    bad.audio = {1, 2, 3};
    REQUIRE(!lm->embed({bad}).has_value());

    // A code past the end of a codebook.
    OmniToken oob = OmniToken::masked(lm->config.num_audio_codebook, lm->config.audio_mask_id);
    oob.audio[0] = 5000;
    REQUIRE(!lm->embed({oob}).has_value());

    // A text id past the end of the vocabulary.
    REQUIRE(!lm->embed({OmniToken::text(lm->config.text_vocab_size + 1)}).has_value());

    REQUIRE(!lm->embed({}).has_value());
}

TEST_CASE("OmniLm produces logits for every position", "[omnivoice][.integration]") {
    // Unlike an autoregressive model there is no last-position shortcut: any
    // position might be unmasked this step, so all of them need logits.
    if (!std::filesystem::exists(kLmPath)) {
        SKIP("models/omnivoice-base-Q8_0.gguf not present");
    }
    const Result<OmniLm> lm = OmniLm::load(kLmPath, Config6::omnivoice());
    REQUIRE(lm.has_value());

    std::vector<OmniToken> seq;
    for (std::size_t i = 0; i < 4; ++i) {
        seq.push_back(OmniToken::text(1000 + i));
    }
    for (std::size_t i = 0; i < 4; ++i) {
        seq.push_back(OmniToken::masked(lm->config.num_audio_codebook, lm->config.audio_mask_id));
    }

    const Result<Mat> logits = lm->forward(seq);
    if (!logits.has_value()) {
        FAIL(logits.error());
    }
    REQUIRE(logits->rows == seq.size());
    REQUIRE(logits->cols == lm->config.audio_table_size());
    for (const float v : logits->data) {
        REQUIRE(std::isfinite(v));
    }
}

TEST_CASE("OmniLm is deterministic", "[omnivoice][.integration]") {
    if (!std::filesystem::exists(kLmPath)) {
        SKIP("models/omnivoice-base-Q8_0.gguf not present");
    }
    const Result<OmniLm> lm = OmniLm::load(kLmPath, Config6::omnivoice());
    REQUIRE(lm.has_value());
    const std::vector<OmniToken> seq{
        OmniToken::text(500),
        OmniToken::masked(lm->config.num_audio_codebook, lm->config.audio_mask_id),
    };
    const Result<Mat> a = lm->forward(seq);
    const Result<Mat> b = lm->forward(seq);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    for (std::size_t i = 0; i < a->data.size(); ++i) {
        REQUIRE(a->data[i] == b->data[i]);
    }
}

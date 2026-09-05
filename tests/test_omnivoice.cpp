#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "rt/omnivoice.hpp"
#include "rt/wav.hpp"
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

// =============================================================================
// Text normalisation
// =============================================================================

TEST_CASE("omni_combine_text trims and joins", "[omnivoice]") {
    REQUIRE(omni_combine_text("  hello  ", "") == "hello");
    REQUIRE(omni_combine_text(" world ", "  hello ") == "hello world");
    // An empty reference is not a leading space.
    REQUIRE(omni_combine_text("hello", "   ") == "hello");
}

TEST_CASE("omni_combine_text folds whitespace", "[omnivoice]") {
    // A line break inside a prompt would be spoken as nothing useful, and runs
    // of spaces as a pause that is not in the text.
    REQUIRE(omni_combine_text("a\nb", "") == "ab");
    REQUIRE(omni_combine_text("a\r\n\r\nb", "") == "ab");
    REQUIRE(omni_combine_text("a  \t  b", "") == "a b");
    REQUIRE(omni_combine_text("one   two    three", "") == "one two three");
}

TEST_CASE("omni_combine_text normalises fullwidth parentheses", "[omnivoice]") {
    REQUIRE(omni_combine_text("（x）", "") == "(x)");
}

TEST_CASE("omni_combine_text removes spaces next to CJK", "[omnivoice]") {
    // Between ideographs a space is typesetting, not a word boundary, and
    // reading it as one puts a pause where none belongs.
    REQUIRE(omni_combine_text("你好 世界", "") == "你好世界");
    REQUIRE(omni_combine_text("hello 你好", "") == "hello你好");
    REQUIRE(omni_combine_text("你好 world", "") == "你好world");
    // Latin text keeps its spaces.
    REQUIRE(omni_combine_text("hello world", "") == "hello world");
}

// =============================================================================
// Reference clips
// =============================================================================

TEST_CASE("omni_prepare_reference trims to whole frames", "[omnivoice]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    // Three frames and a bit.
    const std::vector<float> wav(3 * 960 + 137, 0.2f);
    const Result<OmniReference> ref = omni_prepare_reference(wav, 24000, c);
    REQUIRE(ref.has_value());
    REQUIRE(ref->samples.size() == 3 * 960);
    REQUIRE(approx(static_cast<float>(ref->seconds(c)), 0.12f));
}

TEST_CASE("omni_prepare_reference resamples to the codec's rate", "[omnivoice]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    // One second at 48 kHz becomes one second at 24 kHz, which is 25 frames.
    const std::vector<float> wav(48000, 0.3f);
    const Result<OmniReference> ref = omni_prepare_reference(wav, 48000, c);
    REQUIRE(ref.has_value());
    REQUIRE(ref->samples.size() == 25 * 960);
}

TEST_CASE("omni_prepare_reference lifts a quiet clip to the codec's level",
          "[omnivoice]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    // A constant 0.02 has an RMS of 0.02, so it is scaled by five.
    const std::vector<float> wav(5 * 960, 0.02f);
    const Result<OmniReference> ref = omni_prepare_reference(wav, 24000, c);
    REQUIRE(ref.has_value());
    // The reported RMS is the original one, which is what the output is scaled
    // back to.
    REQUIRE(approx(ref->rms, 0.02f));
    for (const float v : ref->samples) {
        REQUIRE(approx(v, 0.1f));
    }
}

TEST_CASE("omni_prepare_reference leaves a loud clip alone", "[omnivoice]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    const std::vector<float> wav(5 * 960, 0.4f);
    const Result<OmniReference> ref = omni_prepare_reference(wav, 24000, c);
    REQUIRE(ref.has_value());
    REQUIRE(approx(ref->rms, 0.4f));
    for (const float v : ref->samples) {
        REQUIRE(approx(v, 0.4f));
    }
}

TEST_CASE("omni_prepare_reference rejects what it cannot use", "[omnivoice]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    REQUIRE_FALSE(omni_prepare_reference({}, 24000, c).has_value());
    REQUIRE_FALSE(omni_prepare_reference(std::vector<float>(100, 0.1f), 24000, c).has_value());
    REQUIRE_FALSE(omni_prepare_reference(std::vector<float>(4800, 0.1f), 0, c).has_value());
}

// =============================================================================
// Length with a reference
// =============================================================================

TEST_CASE("a reference clip calibrates the length estimate", "[omnivoice]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    const char* text = "Domani andro al mercato con mia sorella.";
    const char* ref = "Ciao, mi chiamo Giulia.";

    // A speaker who took 80 frames to say a 44-frame phrase is slow, and the
    // estimate has to follow them rather than the built-in average.
    const std::size_t slow = omni_estimate_frames_from_reference(text, ref, 80, c);
    const std::size_t fast = omni_estimate_frames_from_reference(text, ref, 30, c);
    REQUIRE(slow > fast);

    // No usable reference falls back to the built-in phrase.
    REQUIRE(omni_estimate_frames_from_reference(text, "", 80, c) ==
            omni_estimate_frames(text, c));
    REQUIRE(omni_estimate_frames_from_reference(text, ref, 0, c) ==
            omni_estimate_frames(text, c));
}

// =============================================================================
// Cloning prompts
// =============================================================================

namespace {

/// A tokenizer with just enough vocabulary to encode the test prompts. The
/// prompt layout is what is under test, not the merges.
[[nodiscard]] HfBpeTokenizer toy_tokenizer() {
    const char* json =
        R"({"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"Ġ":3,"o":4,"e":5,)"
        R"("i":6,"n":7,"N":8,"t":9,".":10,"s":11,"h":12,"l":13,"d":14},"merges":[]}})";
    Result<HfBpeTokenizer> t = HfBpeTokenizer::from_json_str(json);
    REQUIRE(t.has_value());
    return std::move(*t);
}

/// `codebooks` streams of `frames` distinct codes.
[[nodiscard]] std::vector<std::vector<std::uint32_t>> toy_codes(std::size_t codebooks,
                                                                std::size_t frames) {
    std::vector<std::vector<std::uint32_t>> out(codebooks);
    for (std::size_t c = 0; c < codebooks; ++c) {
        out[c].resize(frames);
        for (std::size_t t = 0; t < frames; ++t) {
            out[c][t] = static_cast<std::uint32_t>(c * 100 + t);
        }
    }
    return out;
}

}  // namespace

TEST_CASE("a reference clip becomes decided frames before the masked ones", "[omnivoice]") {
    const Config6 cfg = Config6::omnivoice();
    const HfBpeTokenizer tok = toy_tokenizer();

    OmniRequest r;
    r.text = "note";
    r.ref_text = "abc";
    r.ref_codes = toy_codes(cfg.num_audio_codebook, 5);

    const Result<std::vector<OmniToken>> seq = omni_build_conditional(tok, cfg, r, 4);
    REQUIRE(seq.has_value());

    // The denoise marker leads, and only when there is a recording to clean up.
    REQUIRE((*seq)[0].text_id == cfg.denoise);
    REQUIRE((*seq)[1].text_id == cfg.lang_start);

    // Nine audio positions: five carrying the reference, four masked.
    REQUIRE(seq->size() > 9);
    const std::size_t audio_start = seq->size() - 9;
    for (std::size_t i = 0; i < audio_start; ++i) {
        REQUIRE_FALSE((*seq)[i].is_audio());
    }
    for (std::size_t t = 0; t < 5; ++t) {
        const OmniToken& token = (*seq)[audio_start + t];
        REQUIRE(token.is_audio());
        for (std::size_t c = 0; c < cfg.num_audio_codebook; ++c) {
            REQUIRE(token.audio[c] == c * 100 + t);
            REQUIRE(token.audio[c] != cfg.audio_mask_id);
        }
    }
    for (std::size_t t = 5; t < 9; ++t) {
        const OmniToken& token = (*seq)[audio_start + t];
        REQUIRE(token.is_audio());
        for (const std::uint32_t v : token.audio) {
            REQUIRE(v == cfg.audio_mask_id);
        }
    }
}

TEST_CASE("without a reference there is no denoise marker", "[omnivoice]") {
    const Config6 cfg = Config6::omnivoice();
    const HfBpeTokenizer tok = toy_tokenizer();
    OmniRequest r;
    r.text = "note";

    const Result<std::vector<OmniToken>> seq = omni_build_conditional(tok, cfg, r, 3);
    REQUIRE(seq.has_value());
    REQUIRE((*seq)[0].text_id == cfg.lang_start);
    // Exactly the target frames, and every one of them masked.
    REQUIRE((*seq)[seq->size() - 4].is_audio() == false);
}

TEST_CASE("the target frames stay at the end whatever precedes them", "[omnivoice]") {
    // `omni_synthesize` finds them by counting back from the end, so this is
    // the property that makes cloning need no change to the unmasking loop.
    const Config6 cfg = Config6::omnivoice();
    const HfBpeTokenizer tok = toy_tokenizer();

    OmniRequest plain;
    plain.text = "note";
    OmniRequest cloned;
    cloned.text = "note";
    cloned.ref_text = "abc";
    cloned.ref_codes = toy_codes(cfg.num_audio_codebook, 6);

    const Result<std::vector<OmniToken>> a = omni_build_conditional(tok, cfg, plain, 4);
    const Result<std::vector<OmniToken>> b = omni_build_conditional(tok, cfg, cloned, 4);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    for (std::size_t i = 1; i <= 4; ++i) {
        REQUIRE((*a)[a->size() - i].is_audio());
        REQUIRE((*b)[b->size() - i].is_audio());
        REQUIRE((*a)[a->size() - i].audio[0] == cfg.audio_mask_id);
        REQUIRE((*b)[b->size() - i].audio[0] == cfg.audio_mask_id);
    }
}

TEST_CASE("a reference clip is rejected without its transcript", "[omnivoice]") {
    const Config6 cfg = Config6::omnivoice();
    const HfBpeTokenizer tok = toy_tokenizer();
    OmniRequest r;
    r.text = "note";
    r.ref_codes = toy_codes(cfg.num_audio_codebook, 3);
    REQUIRE_FALSE(omni_build_conditional(tok, cfg, r, 4).has_value());
}

TEST_CASE("malformed reference codes are rejected", "[omnivoice]") {
    const Config6 cfg = Config6::omnivoice();
    const HfBpeTokenizer tok = toy_tokenizer();

    OmniRequest ragged;
    ragged.text = "note";
    ragged.ref_text = "abc";
    ragged.ref_codes = toy_codes(cfg.num_audio_codebook, 3);
    ragged.ref_codes[2].pop_back();
    REQUIRE_FALSE(omni_build_conditional(tok, cfg, ragged, 4).has_value());

    OmniRequest few;
    few.text = "note";
    few.ref_text = "abc";
    few.ref_codes = toy_codes(cfg.num_audio_codebook - 1, 3);
    REQUIRE_FALSE(omni_build_conditional(tok, cfg, few, 4).has_value());

    OmniRequest wild;
    wild.text = "note";
    wild.ref_text = "abc";
    wild.ref_codes = toy_codes(cfg.num_audio_codebook, 3);
    // The mask id is not a code, so it cannot appear in a reference.
    wild.ref_codes[0][1] = static_cast<std::uint32_t>(cfg.audio_mask_id);
    REQUIRE_FALSE(omni_build_conditional(tok, cfg, wild, 4).has_value());
}

// =============================================================================
// End to end
// =============================================================================

TEST_CASE("OmniVoice clones a voice end to end", "[omnivoice][.e2e]") {
    // Loads both halves of the model plus the codec's analysis path, so this
    // is tens of seconds rather than one.
    constexpr const char* kCodecPath = "models/omnivoice-tokenizer-Q8_0.gguf";
    if (!std::filesystem::exists(kLmPath) || !std::filesystem::exists(kCodecPath)) {
        SKIP("OmniVoice weights not present");
    }

    const OmniCodecConfig codec_cfg = OmniCodecConfig::defaults();
    const Result<GgufFile> gguf = GgufFile::open(kLmPath);
    REQUIRE(gguf.has_value());
    const Result<HfBpeTokenizer> tok = load_gguf_tokenizer(*gguf);
    REQUIRE(tok.has_value());
    const Result<OmniLm> lm = OmniLm::load(kLmPath, Config6::omnivoice());
    REQUIRE(lm.has_value());
    const Result<OmniCodecDecoder> decoder = OmniCodecDecoder::load(kCodecPath, codec_cfg);
    REQUIRE(decoder.has_value());
    const Result<OmniCodecEncoder> encoder = OmniCodecEncoder::load(kCodecPath, codec_cfg);
    REQUIRE(encoder.has_value());

    // Build a reference clip the way a user would: real audio in, codes out.
    // Synthesising one first would double the runtime, so this decodes a fixed
    // set of codes instead -- the point is the plumbing, not the voice.
    constexpr std::size_t kRefFrames = 25;
    std::vector<std::vector<std::uint32_t>> seed_codes(codec_cfg.n_codebooks);
    std::uint32_t state = 987654321;
    for (std::size_t c = 0; c < codec_cfg.n_codebooks; ++c) {
        seed_codes[c].resize(kRefFrames);
        for (std::size_t t = 0; t < kRefFrames; ++t) {
            state = state * 1664525u + 1013904223u;
            seed_codes[c][t] = (state >> 16) % codec_cfg.codebook_size;
        }
    }
    const Result<std::vector<float>> ref_wav = decoder->decode(seed_codes);
    REQUIRE(ref_wav.has_value());

    const Result<OmniReference> ref =
        omni_prepare_reference(*ref_wav, codec_cfg.sample_rate, codec_cfg);
    REQUIRE(ref.has_value());
    const Result<std::vector<std::vector<std::uint32_t>>> ref_codes =
        encoder->encode(ref->samples);
    REQUIRE(ref_codes.has_value());
    REQUIRE((*ref_codes)[0].size() == kRefFrames);

    OmniRequest request;
    request.text = "Domani andro al mercato.";
    request.language = "Italian";
    request.ref_text = "Ciao, mi chiamo Giulia.";
    request.ref_codes = *ref_codes;
    request.ref_rms = ref->rms;
    request.duration_seconds = 1.0f;
    request.gen.num_step = 4;  // enough to exercise the loop, not to sound good
    request.gen.seed = 99;

    const Result<OmniResult> out = omni_synthesize(*lm, *decoder, *tok, request);
    REQUIRE(out.has_value());
    REQUIRE(out->frames == 25);
    REQUIRE(out->samples.size() == 25 * codec_cfg.hop_length);
    REQUIRE(out->forward_passes == 8);

    // The reference's frames ride along in the conditional prompt, so it is
    // longer than the plain one by exactly their count.
    OmniRequest plain = request;
    plain.ref_codes.clear();
    plain.ref_text.clear();
    plain.ref_rms = 0.0f;
    const Result<OmniResult> bare = omni_synthesize(*lm, *decoder, *tok, plain);
    REQUIRE(bare.has_value());
    REQUIRE(out->prompt_tokens > bare->prompt_tokens);

    const WaveStats stats = wave_stats(out->samples);
    REQUIRE(stats.in_range());
    for (const float v : out->samples) {
        REQUIRE(std::isfinite(v));
    }

    // The reference changes what comes out. It cannot be checked that it
    // changes it in the *right* direction without listening, but a run that
    // ignored the codes entirely would land on the unconditioned waveform.
    REQUIRE(out->samples.size() == bare->samples.size());
    double num = 0.0;
    double da = 0.0;
    double db = 0.0;
    for (std::size_t i = 0; i < out->samples.size(); ++i) {
        num += static_cast<double>(out->samples[i]) * bare->samples[i];
        da += static_cast<double>(out->samples[i]) * out->samples[i];
        db += static_cast<double>(bare->samples[i]) * bare->samples[i];
    }
    REQUIRE(std::fabs(num / std::sqrt(da * db)) < 0.5);
}

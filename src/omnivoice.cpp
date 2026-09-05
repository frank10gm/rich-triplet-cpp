#include "rt/omnivoice.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <span>
#include <utility>

#include "rt/duration.hpp"
#include "rt/init_rng.hpp"
#include "rt/resample.hpp"
#include "rt/utf8.hpp"

namespace rt {

OmniGenConfig OmniGenConfig::defaults() { return OmniGenConfig{}; }

double OmniResult::audio_seconds() const {
    return sample_rate == 0 ? 0.0
                            : static_cast<double>(samples.size()) / static_cast<double>(sample_rate);
}

double OmniResult::realtime_factor() const {
    const double audio = audio_seconds();
    return audio <= 0.0 ? 0.0 : (generate_seconds + decode_seconds) / audio;
}

// =============================================================================
// Schedules
// =============================================================================

std::vector<float> omni_time_steps(std::size_t num_step, float t_shift) {
    std::vector<float> out(num_step + 1);
    for (std::size_t i = 0; i <= num_step; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(num_step);
        // Warp: below 1, t_shift front-loads the schedule.
        const float denom = 1.0f + (t_shift - 1.0f) * t;
        out[i] = denom == 0.0f ? t : (t_shift * t) / denom;
    }
    return out;
}

std::vector<std::size_t> omni_unmask_schedule(std::size_t total, std::size_t num_step,
                                              float t_shift) {
    const std::vector<float> steps = omni_time_steps(num_step, t_shift);
    std::vector<std::size_t> sched;
    sched.reserve(num_step);

    std::size_t remaining = total;
    for (std::size_t i = 0; i < num_step; ++i) {
        std::size_t n;
        if (i + 1 == num_step) {
            // The last step takes everything left, so rounding can never leave
            // a position masked.
            n = remaining;
        } else {
            const float share = steps[i + 1] - steps[i];
            const auto want = static_cast<std::size_t>(
                std::ceil(static_cast<double>(total) * static_cast<double>(share)));
            n = std::min(want, remaining);
        }
        sched.push_back(n);
        remaining -= n;
    }
    return sched;
}

// =============================================================================
// Prompt
// =============================================================================

std::size_t omni_estimate_frames(std::string_view text, const OmniCodecConfig& codec) {
    const double rate = static_cast<double>(codec.sample_rate) /
                        static_cast<double>(codec.hop_length);

    // The reference calibrates against "Nice to meet you." at 25 audio tokens,
    // and its codec runs at 25 Hz -- so the reference phrase is one second of
    // speech, and the threshold below which short text gets boosted is two.
    // Expressing them that way keeps the numbers identical here and correct if
    // the frame rate ever changes.
    const double est = estimate_duration(text, "Nice to meet you.", rate, 2.0 * rate, 3.0);

    // Truncates rather than rounds, matching the reference's `max(1, int(est))`.
    return std::max<std::size_t>(1, static_cast<std::size_t>(est));
}

std::size_t omni_estimate_frames_from_reference(std::string_view text, std::string_view ref_text,
                                                std::size_t ref_frames,
                                                const OmniCodecConfig& codec) {
    if (ref_text.empty() || ref_frames == 0) {
        return omni_estimate_frames(text, codec);
    }
    const double rate = static_cast<double>(codec.sample_rate) /
                        static_cast<double>(codec.hop_length);
    const double est = estimate_duration(text, ref_text, static_cast<double>(ref_frames),
                                         2.0 * rate, 3.0);
    return std::max<std::size_t>(1, static_cast<std::size_t>(est));
}

// =============================================================================
// Reference clips
// =============================================================================

double OmniReference::seconds(const OmniCodecConfig& codec) const {
    return codec.sample_rate == 0
               ? 0.0
               : static_cast<double>(samples.size()) / static_cast<double>(codec.sample_rate);
}

Result<OmniReference> omni_prepare_reference(std::span<const float> samples,
                                             std::size_t sample_rate,
                                             const OmniCodecConfig& codec) {
    if (samples.empty()) {
        return err("omnivoice: the reference clip is empty");
    }
    if (sample_rate == 0) {
        return err("omnivoice: the reference clip has no sample rate");
    }

    OmniReference ref;
    ref.samples = resample(samples, sample_rate, codec.sample_rate);

    // Whole frames only: the encoder would drop the remainder anyway, and
    // doing it here keeps the length the caller sees honest.
    const std::size_t frames = ref.samples.size() / codec.hop_length;
    if (frames == 0) {
        return err("omnivoice: the reference clip is shorter than one frame at " +
                   std::to_string(codec.sample_rate) + " Hz");
    }
    ref.samples.resize(frames * codec.hop_length);

    double sum_sq = 0.0;
    for (const float v : ref.samples) {
        sum_sq += static_cast<double>(v) * v;
    }
    ref.rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(ref.samples.size())));

    constexpr float kTargetRms = 0.1f;
    if (ref.rms > 0.0f && ref.rms < kTargetRms) {
        const float gain = kTargetRms / ref.rms;
        for (float& v : ref.samples) {
            v *= gain;
        }
    }
    return ref;
}

// =============================================================================
// Text
// =============================================================================

namespace {

/// Trim Unicode whitespace from both ends.
[[nodiscard]] std::string trim(std::string_view s) {
    const std::vector<std::pair<std::size_t, char32_t>> cps = utf8::decode_indices(s);
    std::size_t begin = 0;
    while (begin < cps.size() && utf8::is_whitespace(cps[begin].second)) {
        ++begin;
    }
    std::size_t end = cps.size();
    while (end > begin && utf8::is_whitespace(cps[end - 1].second)) {
        --end;
    }
    if (begin >= end) {
        return {};
    }
    const std::size_t from = cps[begin].first;
    const std::size_t to = end < cps.size() ? cps[end].first : s.size();
    return std::string(s.substr(from, to - from));
}

/// The CJK Unified Ideographs block, where a space between characters is
/// typesetting rather than a word boundary.
[[nodiscard]] bool is_cjk(char32_t cp) { return cp >= 0x4E00 && cp <= 0x9FFF; }

}  // namespace

std::string omni_combine_text(std::string_view text, std::string_view ref_text) {
    const std::string trimmed_ref = trim(ref_text);
    const std::string trimmed = trim(text);
    const std::string joined =
        trimmed_ref.empty() ? trimmed : trimmed_ref + " " + trimmed;

    // Walk once, applying every rule: line breaks vanish, fullwidth
    // parentheses become ASCII, runs of spaces and tabs collapse to one.
    std::vector<char32_t> out;
    for (const char32_t cp : utf8::decode(joined)) {
        if (cp == U'\r' || cp == U'\n') {
            continue;
        }
        if (cp == 0xFF08) {
            out.push_back(U'(');
            continue;
        }
        if (cp == 0xFF09) {
            out.push_back(U')');
            continue;
        }
        if (cp == U' ' || cp == U'\t') {
            if (!out.empty() && out.back() == U' ') {
                continue;
            }
            out.push_back(U' ');
            continue;
        }
        out.push_back(cp);
    }

    // Then drop the spaces that sit against a CJK character on either side.
    std::string result;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] == U' ') {
            const bool cjk_before = i > 0 && is_cjk(out[i - 1]);
            const bool cjk_after = i + 1 < out.size() && is_cjk(out[i + 1]);
            if (cjk_before || cjk_after) {
                continue;
            }
        }
        utf8::encode_into(result, out[i]);
    }
    return result;
}

namespace {

/// Append the text tokens of `s`, if any.
void append_text(std::vector<OmniToken>& out, const HfBpeTokenizer& tok, std::string_view s) {
    if (s.empty()) {
        return;
    }
    for (const std::uint32_t id : tok.encode(s)) {
        out.push_back(OmniToken::text(id));
    }
}

}  // namespace

Result<std::vector<OmniToken>> omni_build_conditional(const HfBpeTokenizer& tok,
                                                      const Config6& cfg,
                                                      const OmniRequest& request,
                                                      std::size_t frames) {
    if (request.text.empty()) {
        return err("omnivoice: prompt text is empty");
    }
    if (frames == 0) {
        return err("omnivoice: asked for zero frames");
    }

    const std::size_t ref_frames = request.ref_frames();
    for (const std::vector<std::uint32_t>& stream : request.ref_codes) {
        if (stream.size() != ref_frames) {
            return err("omnivoice: the reference codebooks disagree on length");
        }
    }
    if (!request.ref_codes.empty() && request.ref_codes.size() != cfg.num_audio_codebook) {
        return err("omnivoice: the reference has " +
                   std::to_string(request.ref_codes.size()) + " codebooks, expected " +
                   std::to_string(cfg.num_audio_codebook));
    }
    if (ref_frames > 0 && request.ref_text.empty()) {
        return err(
            "omnivoice: a reference clip needs its transcript, or the model cannot tell which "
            "part of the text it has already heard");
    }

    std::vector<OmniToken> seq;

    // A reference clip is a recording, so the model is told to clean it up
    // rather than reproduce its room.
    if (ref_frames > 0) {
        seq.push_back(OmniToken::text(cfg.denoise));
    }

    // Style block. The markers are injected as raw ids -- encoding the literal
    // "<|lang_start|>" would split it into ordinary subword pieces.
    seq.push_back(OmniToken::text(cfg.lang_start));
    append_text(seq, tok, request.language.empty() ? "None" : request.language);
    seq.push_back(OmniToken::text(cfg.lang_end));

    seq.push_back(OmniToken::text(cfg.instruct_start));
    append_text(seq, tok, request.instruct.empty() ? "None" : request.instruct);
    seq.push_back(OmniToken::text(cfg.instruct_end));

    // Text block: the reference transcript and the target text as one run, so
    // the model reads them as continuous speech.
    seq.push_back(OmniToken::text(cfg.text_start));
    append_text(seq, tok, omni_combine_text(request.text, request.ref_text));
    seq.push_back(OmniToken::text(cfg.text_end));

    // Reference frames, already decided. These are ordinary audio positions
    // carrying real codes rather than the mask, which is the whole mechanism:
    // the target frames attend to them like any other position.
    for (std::size_t t = 0; t < ref_frames; ++t) {
        OmniToken token;
        token.audio.resize(cfg.num_audio_codebook);
        for (std::size_t c = 0; c < cfg.num_audio_codebook; ++c) {
            if (request.ref_codes[c][t] >= cfg.audio_mask_id) {
                return err("omnivoice: reference code " +
                           std::to_string(request.ref_codes[c][t]) + " is out of range");
            }
            token.audio[c] = request.ref_codes[c][t];
        }
        seq.push_back(std::move(token));
    }

    // Target frames, all masked.
    for (std::size_t i = 0; i < frames; ++i) {
        seq.push_back(OmniToken::masked(cfg.num_audio_codebook, cfg.audio_mask_id));
    }
    return seq;
}

std::vector<OmniToken> omni_build_unconditional(const Config6& cfg, std::size_t frames) {
    // No style, no text: the unconditional branch sees only the frames it is
    // trying to fill.
    std::vector<OmniToken> seq;
    seq.reserve(frames);
    for (std::size_t i = 0; i < frames; ++i) {
        seq.push_back(OmniToken::masked(cfg.num_audio_codebook, cfg.audio_mask_id));
    }
    return seq;
}


// =============================================================================
// Long text
// =============================================================================

namespace {

/// Punctuation a sentence may end on. Fullwidth forms included, since a CJK
/// clause ends on the wide comma rather than the ASCII one.
[[nodiscard]] bool is_split_punctuation(char32_t cp) {
    switch (cp) {
        case U'.': case U',': case U';': case U':': case U'!': case U'?':
        case 0x3002:  // 。
        case 0xFF0C:  // ，
        case 0xFF1B:  // ；
        case 0xFF1A:  // ：
        case 0xFF01:  // ！
        case 0xFF1F:  // ？
            return true;
        default:
            return false;
    }
}

/// Quotes and brackets that close a sentence *after* its full stop, and so
/// belong to the piece that just ended rather than starting the next one.
[[nodiscard]] bool is_closing_mark(char32_t cp) {
    switch (cp) {
        case U'"': case U'\'':
        case 0x201C: case 0x201D:  // “ ”
        case 0x2018: case 0x2019:  // ‘ ’
        case 0xFF09:              // ）
        case U']': case U'>':
        case 0x300B:              // 》
        case 0x300D:              // 」
        case 0x3011:              // 】
            return true;
        default:
            return false;
    }
}

/// Words whose trailing full stop is not the end of a sentence.
[[nodiscard]] bool is_abbreviation(std::string_view word) {
    static const std::vector<std::string_view> kAbbreviations{
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.",
        "Pres.", "Gov.", "Capt.", "Gen.", "Sen.", "Rep.", "Col.", "Maj.", "Lt.",
        "Cmdr.", "Sgt.", "Cpl.", "Co.", "Corp.", "Inc.", "Ltd.", "Est.", "Dept.",
        "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.", "Jan.", "Feb.", "Mar.",
        "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "i.e.", "e.g.",
        "vs.", "Vs.", "Etc.", "approx.", "fig.", "def.",
    };
    return std::find(kAbbreviations.begin(), kAbbreviations.end(), word) !=
           kAbbreviations.end();
}

/// The last whitespace-separated word of a run of code points.
[[nodiscard]] std::string last_word(const std::vector<char32_t>& cps) {
    std::size_t end = cps.size();
    while (end > 0 && utf8::is_whitespace(cps[end - 1])) {
        --end;
    }
    std::size_t begin = end;
    while (begin > 0 && !utf8::is_whitespace(cps[begin - 1])) {
        --begin;
    }
    std::string out;
    for (std::size_t i = begin; i < end; ++i) {
        utf8::encode_into(out, cps[i]);
    }
    return out;
}

[[nodiscard]] std::string to_string(const std::vector<char32_t>& cps) {
    std::string out;
    for (const char32_t cp : cps) {
        utf8::encode_into(out, cp);
    }
    return out;
}

}  // namespace

std::vector<std::string> omni_chunk_text(std::string_view text, std::size_t chunk_chars,
                                         std::size_t min_chunk_chars) {
    const std::vector<char32_t> cps = utf8::decode(text);
    if (cps.empty()) {
        return {};
    }
    if (chunk_chars == 0) {
        return {std::string(text)};
    }

    // 1. Break into sentences at punctuation, keeping the mark with what it
    //    ends.
    std::vector<std::vector<char32_t>> sentences;
    std::vector<char32_t> current;
    for (const char32_t cp : cps) {
        if (current.empty() && !sentences.empty() &&
            (is_split_punctuation(cp) || is_closing_mark(cp))) {
            // A mark that opens a new sentence closes the previous one.
            sentences.back().push_back(cp);
            continue;
        }
        current.push_back(cp);
        if (is_split_punctuation(cp)) {
            if (cp == U'.' && is_abbreviation(last_word(current))) {
                continue;
            }
            sentences.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) {
        sentences.push_back(std::move(current));
    }

    // 2. Merge greedily up to the target size.
    std::vector<std::vector<char32_t>> merged;
    std::vector<char32_t> chunk;
    for (std::vector<char32_t>& sentence : sentences) {
        if (chunk.size() + sentence.size() <= chunk_chars) {
            chunk.insert(chunk.end(), sentence.begin(), sentence.end());
        } else {
            if (!chunk.empty()) {
                merged.push_back(std::move(chunk));
            }
            chunk = std::move(sentence);
        }
    }
    if (!chunk.empty()) {
        merged.push_back(std::move(chunk));
    }

    // 3. Fold away anything too short to be worth its own generation.
    std::vector<std::vector<char32_t>> final_chunks;
    const bool first_is_short = !merged.empty() && merged[0].size() < min_chunk_chars;
    for (std::size_t i = 0; i < merged.size(); ++i) {
        if (i == 1 && first_is_short) {
            // A short opener keeps the piece after it rather than the reverse.
            final_chunks.back().insert(final_chunks.back().end(), merged[i].begin(),
                                       merged[i].end());
        } else if (merged[i].size() >= min_chunk_chars || final_chunks.empty()) {
            final_chunks.push_back(std::move(merged[i]));
        } else {
            final_chunks.back().insert(final_chunks.back().end(), merged[i].begin(),
                                       merged[i].end());
        }
    }

    std::vector<std::string> out;
    for (const std::vector<char32_t>& c : final_chunks) {
        std::string s = trim(to_string(c));
        if (!s.empty()) {
            out.push_back(std::move(s));
        }
    }
    return out;
}

std::vector<float> omni_cross_fade(const std::vector<std::vector<float>>& chunks,
                                   std::size_t sample_rate, float silence_seconds) {
    if (chunks.empty()) {
        return {};
    }
    if (chunks.size() == 1) {
        return chunks.front();
    }

    // A third of the gap fades out, a third is silence, a third fades in.
    const auto total = static_cast<std::size_t>(silence_seconds *
                                                static_cast<float>(sample_rate));
    const std::size_t fade = total / 3;

    std::vector<float> out = chunks.front();
    for (std::size_t i = 1; i < chunks.size(); ++i) {
        const std::size_t fade_out = std::min(fade, out.size());
        for (std::size_t j = 0; j < fade_out; ++j) {
            const float w = 1.0f - static_cast<float>(j) /
                                       static_cast<float>(std::max<std::size_t>(fade_out - 1, 1));
            out[out.size() - fade_out + j] *= w;
        }

        out.insert(out.end(), fade, 0.0f);

        const std::vector<float>& next = chunks[i];
        const std::size_t fade_in = std::min(fade, next.size());
        const std::size_t before = out.size();
        out.insert(out.end(), next.begin(), next.end());
        for (std::size_t j = 0; j < fade_in; ++j) {
            const float w = static_cast<float>(j) /
                            static_cast<float>(std::max<std::size_t>(fade_in - 1, 1));
            out[before + j] *= w;
        }
    }
    return out;
}

// =============================================================================
// Sampling
// =============================================================================

namespace {

/// Log-softmax a single codebook's slice, in place.
void log_softmax(std::span<float> row) {
    float max_v = -std::numeric_limits<float>::infinity();
    for (const float v : row) {
        max_v = std::max(max_v, v);
    }
    float sum = 0.0f;
    for (const float v : row) {
        sum += std::exp(v - max_v);
    }
    const float log_sum = max_v + std::log(sum);
    for (float& v : row) {
        v -= log_sum;
    }
}

/// -log(-log(U)) -- a standard Gumbel draw.
[[nodiscard]] float gumbel(InitRng& rng) {
    const float u = std::clamp(rng.next_f32(), 1e-9f, 1.0f - 1e-7f);
    return -std::log(-std::log(u));
}

}  // namespace

namespace {

/// One piece of text, start to finish: estimate its length, run the unmasking
/// loop, and hand back the codes. Chunked generation calls this once per piece
/// and every caller calls it at least once.
struct OmniChunkResult {
    std::vector<std::vector<std::uint32_t>> codes;
    std::size_t prompt_tokens = 0;
    std::size_t forward_passes = 0;
};

[[nodiscard]] Result<OmniChunkResult> omni_generate_chunk(const OmniLm& lm,
                                                          const OmniCodecDecoder& codec,
                                                          const HfBpeTokenizer& tok,
                                                          const OmniRequest& request,
                                                          const OmniForward& fwd) {
    const Config6& cfg = lm.config;
    const std::size_t codebooks = cfg.num_audio_codebook;
    const std::size_t vocab = cfg.audio_vocab_size;

    const std::size_t frames =
        request.duration_seconds > 0.0f
            ? static_cast<std::size_t>(std::ceil(
                  static_cast<double>(request.duration_seconds) *
                  static_cast<double>(codec.config.sample_rate) /
                  static_cast<double>(codec.config.hop_length)))
            : omni_estimate_frames_from_reference(omni_combine_text(request.text, {}),
                                                 request.ref_text, request.ref_frames(),
                                                 codec.config);
    if (frames == 0) {
        return err("omnivoice: computed zero frames");
    }

    RT_TRY(cond, omni_build_conditional(tok, cfg, request, frames));
    const std::vector<OmniToken> uncond_base = omni_build_unconditional(cfg, frames);
    const bool use_guidance = request.gen.guidance_scale != 0.0f;

    // Where the target frames start in the conditional sequence.
    const std::size_t cond_offset = cond.size() - frames;

    // The working state: every (codebook, frame) pair, initially masked.
    std::vector<std::uint32_t> tokens(codebooks * frames,
                                      static_cast<std::uint32_t>(cfg.audio_mask_id));
    const auto at = [codebooks, frames](std::size_t c, std::size_t t) {
        (void)codebooks;
        return c * frames + t;
    };

    const std::vector<std::size_t> schedule =
        omni_unmask_schedule(codebooks * frames, request.gen.num_step, request.gen.t_shift);

    InitRng rng(request.gen.seed);
    std::vector<OmniToken> cond_seq = cond;
    std::vector<OmniToken> uncond_seq = uncond_base;

    std::vector<float> log_probs(codebooks * frames * vocab);
    std::vector<float> scores(codebooks * frames);
    std::vector<std::uint32_t> predicted(codebooks * frames);
    std::vector<std::size_t> order(codebooks * frames);

    std::size_t passes = 0;

    for (std::size_t step = 0; step < schedule.size(); ++step) {
        const std::size_t k = schedule[step];
        if (k == 0) {
            continue;
        }

        RT_TRY(c_logits, fwd.forward(cond_seq));
        ++passes;
        Mat u_logits;
        if (use_guidance) {
            RT_TRY(u, fwd.forward(uncond_seq));
            u_logits = std::move(u);
            ++passes;
        }

        // Combine, per (codebook, frame).
        for (std::size_t t = 0; t < frames; ++t) {
            for (std::size_t c = 0; c < codebooks; ++c) {
                const std::size_t base = (c * frames + t) * vocab;
                std::span<float> row(log_probs.data() + base, vocab);
                for (std::size_t v = 0; v < vocab; ++v) {
                    row[v] = c_logits.at(cond_offset + t, c * vocab + v);
                }
                log_softmax(row);

                if (use_guidance) {
                    std::vector<float> u_row(vocab);
                    for (std::size_t v = 0; v < vocab; ++v) {
                        u_row[v] = u_logits.at(t, c * vocab + v);
                    }
                    log_softmax(u_row);
                    // c + s * (c - u), renormalized.
                    for (std::size_t v = 0; v < vocab; ++v) {
                        row[v] += request.gen.guidance_scale * (row[v] - u_row[v]);
                    }
                    log_softmax(row);
                }

                // The mask id is never a legitimate prediction.
                row[cfg.audio_mask_id] = -std::numeric_limits<float>::infinity();

                std::size_t best = 0;
                float best_v = -std::numeric_limits<float>::infinity();
                for (std::size_t v = 0; v < vocab; ++v) {
                    if (row[v] > best_v) {
                        best_v = row[v];
                        best = v;
                    }
                }
                if (request.gen.class_temperature > 0.0f) {
                    // Gumbel over the token choice itself.
                    best_v = -std::numeric_limits<float>::infinity();
                    for (std::size_t v = 0; v < vocab; ++v) {
                        if (v == cfg.audio_mask_id) {
                            continue;
                        }
                        const float g = row[v] + request.gen.class_temperature * gumbel(rng);
                        if (g > best_v) {
                            best_v = g;
                            best = v;
                        }
                    }
                    best_v = row[best];
                }

                predicted[at(c, t)] = static_cast<std::uint32_t>(best);

                // Confidence, less a penalty that grows with the codebook
                // index so the coarse ones are decided first.
                float s = row[best] -
                          static_cast<float>(c) * request.gen.layer_penalty_factor;
                if (request.gen.position_temperature > 0.0f) {
                    s += request.gen.position_temperature * gumbel(rng);
                }
                // Anything already decided is out of the running.
                if (tokens[at(c, t)] != cfg.audio_mask_id) {
                    s = -std::numeric_limits<float>::infinity();
                }
                scores[at(c, t)] = s;
            }
        }

        // Unmask the k best.
        std::iota(order.begin(), order.end(), std::size_t{0});
        const std::size_t take = std::min(k, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<long>(take), order.end(),
                          [&scores](std::size_t a, std::size_t b) {
                              return scores[a] > scores[b];
                          });
        for (std::size_t i = 0; i < take; ++i) {
            const std::size_t idx = order[i];
            if (scores[idx] == -std::numeric_limits<float>::infinity()) {
                break;  // nothing left that is still masked
            }
            tokens[idx] = predicted[idx];
        }

        // Feed the decisions back into both sequences.
        for (std::size_t t = 0; t < frames; ++t) {
            for (std::size_t c = 0; c < codebooks; ++c) {
                cond_seq[cond_offset + t].audio[c] = tokens[at(c, t)];
                uncond_seq[t].audio[c] = tokens[at(c, t)];
            }
        }

        if (request.debug) {
            std::size_t remaining = 0;
            for (const std::uint32_t v : tokens) {
                if (v == cfg.audio_mask_id) {
                    ++remaining;
                }
            }
            std::fprintf(stderr, "[ omnivoice ] step %2zu/%zu unmasked %zu, %zu still masked\n",
                         step + 1, schedule.size(), take, remaining);
        }
    }

    // Anything still masked would be an out-of-range code downstream; the
    // schedule guarantees this cannot happen, so treat it as a bug rather than
    // clamping quietly.
    for (const std::uint32_t v : tokens) {
        if (v == cfg.audio_mask_id) {
            return err("omnivoice: the unmasking schedule left positions undecided");
        }
    }

    std::vector<std::vector<std::uint32_t>> codes(codebooks);
    for (std::size_t c = 0; c < codebooks; ++c) {
        codes[c].resize(frames);
        for (std::size_t t = 0; t < frames; ++t) {
            codes[c][t] = tokens[at(c, t)];
        }
    }

    OmniChunkResult out;
    out.codes = std::move(codes);
    out.prompt_tokens = cond.size();
    out.forward_passes = passes;
    return out;
}

}  // namespace

Result<OmniResult> omni_synthesize(const OmniLm& lm, const OmniCodecDecoder& codec,
                                   const HfBpeTokenizer& tok, const OmniRequest& request,
                                   const OmniForward* accel) {
    // The model still owns the config and the prompt layout; only the forward
    // pass moves.
    const OmniForward& fwd = accel != nullptr ? *accel : static_cast<const OmniForward&>(lm);
    const OmniCodecConfig& codec_cfg = codec.config;
    const double rate = static_cast<double>(codec_cfg.sample_rate) /
                        static_cast<double>(codec_cfg.hop_length);

    const auto start = std::chrono::steady_clock::now();

    // How long is this, and is it too long to say in one breath? An explicit
    // --duration is an instruction about the whole output, so it turns
    // splitting off rather than being divided among the pieces.
    const std::size_t estimate =
        omni_estimate_frames_from_reference(omni_combine_text(request.text, {}),
                                            request.ref_text, request.ref_frames(), codec_cfg);
    const auto threshold =
        static_cast<std::size_t>(request.gen.chunk_threshold_seconds * rate);
    const bool split = request.duration_seconds <= 0.0f && threshold > 0 &&
                       request.gen.chunk_seconds > 0.0f && estimate > threshold;

    std::vector<std::string> pieces;
    if (split) {
        // Characters per chunk, from this text's own measured density rather
        // than an average one: a line of digits is worth far more audio per
        // character than a line of Latin letters.
        const std::size_t chars = utf8::decode(request.text).size();
        const double per_chunk = static_cast<double>(request.gen.chunk_seconds) * rate *
                                 static_cast<double>(chars) / static_cast<double>(estimate);
        pieces = omni_chunk_text(request.text,
                                 std::max<std::size_t>(1, static_cast<std::size_t>(per_chunk)));
    }
    if (pieces.size() < 2) {
        pieces = {request.text};
    }

    std::vector<std::vector<float>> waves;
    waves.reserve(pieces.size());
    OmniResult result;
    result.sample_rate = codec_cfg.sample_rate;
    result.chunks = pieces.size();

    // What holds the voice across a split. With a reference clip every piece
    // uses it. Without one, the first piece invents a speaker and every piece
    // after it takes that piece as its reference -- so the seam is a breath
    // rather than a new person.
    std::vector<std::vector<std::uint32_t>> anchor_codes;
    std::string anchor_text;

    double decode_seconds = 0.0;
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        OmniRequest piece = request;
        piece.text = pieces[i];
        if (pieces.size() > 1) {
            // Each piece is measured on its own; the whole-text estimate was
            // only ever used to decide how to cut it.
            piece.duration_seconds = 0.0f;
            if (request.ref_codes.empty() && i > 0) {
                piece.ref_codes = anchor_codes;
                piece.ref_text = anchor_text;
            }
            if (request.debug) {
                std::fprintf(stderr, "[ omnivoice ] chunk %zu/%zu: \"%s\"\n", i + 1,
                             pieces.size(), pieces[i].c_str());
            }
        }

        RT_TRY(chunk, omni_generate_chunk(lm, codec, tok, piece, fwd));
        if (i == 0 && request.ref_codes.empty()) {
            anchor_codes = chunk.codes;
            anchor_text = pieces[0];
        }

        const auto decode_start = std::chrono::steady_clock::now();
        RT_TRY(samples, codec.decode(chunk.codes));
        decode_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start)
                .count();

        // Put the reference's own loudness back. Without this a quiet recording
        // clones into a voice that is the right voice at the wrong level,
        // because the clip was brought up to 0.1 RMS before it was encoded.
        constexpr float kTargetRms = 0.1f;
        if (request.ref_rms > 0.0f && request.ref_rms < kTargetRms) {
            const float gain = request.ref_rms / kTargetRms;
            for (float& v : samples) {
                v *= gain;
            }
        }

        result.frames += chunk.codes.empty() ? 0 : chunk.codes.front().size();
        result.prompt_tokens += chunk.prompt_tokens;
        result.forward_passes += chunk.forward_passes;
        waves.push_back(std::move(samples));
    }

    result.samples = omni_cross_fade(waves, codec_cfg.sample_rate,
                                     request.gen.chunk_gap_seconds);
    const auto end = std::chrono::steady_clock::now();
    result.decode_seconds = decode_seconds;
    result.generate_seconds =
        std::chrono::duration<double>(end - start).count() - decode_seconds;
    return result;
}

}  // namespace rt

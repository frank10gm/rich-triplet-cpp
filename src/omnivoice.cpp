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

    std::vector<OmniToken> seq;

    // Style block. The markers are injected as raw ids -- encoding the literal
    // "<|lang_start|>" would split it into ordinary subword pieces.
    seq.push_back(OmniToken::text(cfg.lang_start));
    append_text(seq, tok, request.language.empty() ? "None" : request.language);
    seq.push_back(OmniToken::text(cfg.lang_end));

    seq.push_back(OmniToken::text(cfg.instruct_start));
    append_text(seq, tok, request.instruct.empty() ? "None" : request.instruct);
    seq.push_back(OmniToken::text(cfg.instruct_end));

    // Text block.
    seq.push_back(OmniToken::text(cfg.text_start));
    append_text(seq, tok, request.text);
    seq.push_back(OmniToken::text(cfg.text_end));

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

Result<OmniResult> omni_synthesize(const OmniLm& lm, const OmniCodecDecoder& codec,
                                   const HfBpeTokenizer& tok, const OmniRequest& request) {
    const Config6& cfg = lm.config;
    const std::size_t codebooks = cfg.num_audio_codebook;
    const std::size_t vocab = cfg.audio_vocab_size;

    const std::size_t frames =
        request.duration_seconds > 0.0f
            ? static_cast<std::size_t>(std::ceil(
                  static_cast<double>(request.duration_seconds) *
                  static_cast<double>(codec.config.sample_rate) /
                  static_cast<double>(codec.config.hop_length)))
            : omni_estimate_frames(request.text, codec.config);
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
    const auto start = std::chrono::steady_clock::now();

    for (std::size_t step = 0; step < schedule.size(); ++step) {
        const std::size_t k = schedule[step];
        if (k == 0) {
            continue;
        }

        RT_TRY(c_logits, lm.forward(cond_seq));
        ++passes;
        Mat u_logits;
        if (use_guidance) {
            RT_TRY(u, lm.forward(uncond_seq));
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

    const auto mid = std::chrono::steady_clock::now();

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

    RT_TRY(samples, codec.decode(codes));
    const auto end = std::chrono::steady_clock::now();

    OmniResult result;
    result.samples = std::move(samples);
    result.sample_rate = codec.config.sample_rate;
    result.frames = frames;
    result.prompt_tokens = cond.size();
    result.forward_passes = passes;
    result.generate_seconds = std::chrono::duration<double>(mid - start).count();
    result.decode_seconds = std::chrono::duration<double>(end - mid).count();
    return result;
}

}  // namespace rt

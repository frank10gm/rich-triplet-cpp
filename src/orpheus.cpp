#include "rt/orpheus.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <utility>

namespace rt {

// =============================================================================
// Config
// =============================================================================

OrpheusConfig OrpheusConfig::defaults() { return OrpheusConfig{}; }

const std::vector<std::string>& orpheus_voices() {
    static const std::vector<std::string> voices{"tara", "leah", "jess", "leo",
                                                 "dan",  "mia",  "zac",  "zoe"};
    return voices;
}

bool orpheus_voice_known(std::string_view voice) {
    const std::vector<std::string>& voices = orpheus_voices();
    return std::find(voices.begin(), voices.end(), voice) != voices.end();
}

SamplingParams orpheus_default_sampling(std::uint64_t seed) {
    SamplingParams p;
    p.temperature = 0.6f;
    p.top_p = 0.8f;
    p.top_k = 0;
    p.repetition_penalty = 1.3f;
    p.seed = seed;
    return p;
}

// =============================================================================
// Code stream
// =============================================================================

bool OrpheusCodeStream::push(std::size_t token_id) {
    if (token_id < cfg_.audio_token_base) {
        ++rejected_;
        return false;
    }

    const std::size_t slot = flat_.size() % cfg_.codes_per_frame;
    const std::size_t offset = cfg_.audio_token_base + slot * cfg_.codebook_size;
    if (token_id < offset) {
        // The token belongs to an earlier slot than the one expected.
        ++rejected_;
        return false;
    }

    const std::size_t code = token_id - offset;
    if (code >= cfg_.codebook_size) {
        // Belongs to a later slot. Not advancing the counter is what lets the
        // stream resynchronise.
        ++rejected_;
        return false;
    }

    flat_.push_back(static_cast<std::uint32_t>(code));
    return true;
}

std::size_t OrpheusCodeStream::frames() const {
    // Level 2 runs at the finest rate and takes four codes per group.
    return complete_groups() * 4;
}

std::vector<std::vector<std::uint32_t>> OrpheusCodeStream::snac_codes() const {
    const std::size_t groups = complete_groups();
    if (groups == 0) {
        return {};
    }

    std::vector<std::uint32_t> level0;
    std::vector<std::uint32_t> level1;
    std::vector<std::uint32_t> level2;
    level0.reserve(groups);
    level1.reserve(groups * 2);
    level2.reserve(groups * 4);

    for (std::size_t g = 0; g < groups; ++g) {
        const std::size_t base = g * cfg_.codes_per_frame;
        // The 1:2:4 interleave. Slots 1 and 4 go to the middle scale; slots
        // 2, 3, 5 and 6 to the finest, in that order.
        level0.push_back(flat_[base + 0]);
        level1.push_back(flat_[base + 1]);
        level1.push_back(flat_[base + 4]);
        level2.push_back(flat_[base + 2]);
        level2.push_back(flat_[base + 3]);
        level2.push_back(flat_[base + 5]);
        level2.push_back(flat_[base + 6]);
    }

    return {std::move(level0), std::move(level1), std::move(level2)};
}

// =============================================================================
// Prompt
// =============================================================================

Result<std::vector<std::size_t>> orpheus_build_prompt(const HfBpeTokenizer& tok,
                                                      const OrpheusConfig& cfg,
                                                      std::string_view voice,
                                                      std::string_view text) {
    if (text.empty()) {
        return err("orpheus: prompt text is empty");
    }

    const std::string body = std::string(voice) + ": " + std::string(text);
    const std::vector<std::uint32_t> encoded = tok.encode(body);
    if (encoded.empty()) {
        return err("orpheus: prompt encoded to no tokens");
    }

    std::vector<std::size_t> ids;
    ids.reserve(encoded.size() + 3 + cfg.prompt_end.size());
    if (cfg.leading_bos) {
        ids.push_back(cfg.bos_token);
    }
    ids.push_back(cfg.prompt_start);
    ids.push_back(cfg.bos_token);
    for (const std::uint32_t id : encoded) {
        ids.push_back(static_cast<std::size_t>(id));
    }
    ids.insert(ids.end(), cfg.prompt_end.begin(), cfg.prompt_end.end());
    return ids;
}

// =============================================================================
// Result helpers
// =============================================================================

double OrpheusResult::audio_seconds() const {
    return sample_rate == 0 ? 0.0
                            : static_cast<double>(samples.size()) / static_cast<double>(sample_rate);
}

double OrpheusResult::realtime_factor() const {
    const double audio = audio_seconds();
    return audio <= 0.0 ? 0.0 : (generate_seconds + decode_seconds) / audio;
}

// =============================================================================
// Synthesis
// =============================================================================

Result<OrpheusResult> orpheus_synthesize(const LlamaModel& model, const SnacDecoder& snac,
                                         const HfBpeTokenizer& tok,
                                         const OrpheusRequest& request,
                                         const OrpheusConfig& cfg) {
    RT_TRY(prompt, orpheus_build_prompt(tok, cfg, request.voice, request.text));

    if (!orpheus_voice_known(request.voice)) {
        std::fprintf(stderr,
                     "[ orpheus ] Warning: '%s' is not a trained voice; output will not sound "
                     "like a consistent speaker\n",
                     request.voice.c_str());
    }

    SamplingParams params = request.sampling;
    // Stops are handled entirely in the callback below, which can recognise
    // any number of them; the sampler's single eos slot would only cover one.
    params.eos_token_id.reset();
    if (request.mask_to_audio) {
        params.allowed_min = cfg.audio_token_base;
        params.allowed_max = cfg.audio_token_limit();
        // The stop marker has to stay reachable or generation runs to max_new.
        params.allowed_extra = cfg.stop_tokens;
    }

    if (request.debug) {
        std::fprintf(stderr, "[ orpheus ] prompt (%zu tokens):", prompt.size());
        for (const std::size_t id : prompt) {
            std::fprintf(stderr, " %zu", id);
        }
        std::fprintf(stderr, "\n");
    }

    OrpheusCodeStream stream(cfg);
    bool hit_stop = false;

    const auto gen_start = std::chrono::steady_clock::now();
    const std::size_t produced =
        model.generate(prompt, request.max_new, params, request.debug,
                       [&](std::size_t id) -> bool {
                           if (std::find(cfg.stop_tokens.begin(), cfg.stop_tokens.end(), id) !=
                               cfg.stop_tokens.end()) {
                               hit_stop = true;
                               return false;
                           }
                           const std::size_t want_slot = stream.accepted() % cfg.codes_per_frame;
                           const bool kept = stream.push(id);
                           if (request.debug) {
                               // Which codebook slot the token's id actually
                               // falls in, against the one the stream wanted.
                               // A healthy stream walks 0,1,2,3,4,5,6 and
                               // repeats; anything else is the frame structure
                               // breaking down, which is invisible in the audio.
                               const long band =
                                   cfg.is_audio_token(id)
                                       ? static_cast<long>((id - cfg.audio_token_base) /
                                                           cfg.codebook_size)
                                       : -1;
                               std::fprintf(stderr, "[ orpheus ] id %6zu band %2ld want %zu %s\n",
                                            id, band, want_slot, kept ? "keep" : "drop");
                           }
                           return true;
                       });
    const auto gen_end = std::chrono::steady_clock::now();

    OrpheusResult result;
    result.sample_rate = snac.config.sampling_rate;
    result.tokens_generated = produced;
    result.codes_accepted = stream.accepted();
    result.codes_rejected = stream.rejected();
    result.groups = stream.complete_groups();
    result.generate_seconds = std::chrono::duration<double>(gen_end - gen_start).count();

    if (request.debug) {
        std::fprintf(stderr,
                     "[ orpheus ] %zu tokens, %zu codes kept, %zu rejected, %zu groups, stop=%s\n",
                     produced, stream.accepted(), stream.rejected(), stream.complete_groups(),
                     hit_stop ? "yes" : "no");
    }

    const std::vector<std::vector<std::uint32_t>> codes = stream.snac_codes();
    if (codes.empty()) {
        return err("orpheus: the model produced no complete 7-code group (" +
                   std::to_string(produced) + " tokens, " + std::to_string(stream.accepted()) +
                   " codes kept, " + std::to_string(stream.rejected()) + " rejected)");
    }

    const auto dec_start = std::chrono::steady_clock::now();
    RT_TRY(samples, snac.decode(codes, request.noise, request.sampling.seed));
    const auto dec_end = std::chrono::steady_clock::now();

    result.samples = std::move(samples);
    result.decode_seconds = std::chrono::duration<double>(dec_end - dec_start).count();
    return result;
}

}  // namespace rt

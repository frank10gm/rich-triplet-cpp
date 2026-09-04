#pragma once

// =============================================================================
// Orpheus TTS -- text to speech over a Llama backbone and a SNAC codec
// =============================================================================
//
// Ties `transformer5.hpp` (the Llama 3.2 backbone) to `snac.hpp` (the codec
// decoder). The backbone generates audio tokens; this file works out which of
// its 156 940 vocabulary entries are audio codes, which SNAC codebook each one
// belongs to, and where a frame begins.
//
// ## The token protocol
//
// Orpheus extends Llama 3.2's 128 256-entry vocabulary with markers and audio
// codes. `<custom_token_0>` is id 128256, so `<custom_token_N>` is
// `128256 + N`, and the reference decoder computes a code as
// `N - 10 - (slot * 4096)`. Substituting gives:
//
//     code = id - 128266 - (slot * 4096)
//
// Audio ids therefore run from 128266 (`<custom_token_10>`) to 156937, seven
// codebook slots of 4096 each.
//
// A prompt is framed as:
//
//     [128000, 128259, 128000] + encode("{voice}: {text}")
//                              + [128009, 128260, 128261, 128257]
//
// and generation stops at 128258 (`<custom_token_2>`).
//
// Both 128000 (`<|begin_of_text|>`) tokens are there for a reason worth
// spelling out, because neither appears in the reference's source.
//
// The reference builds `[128259] + tokenizer(text) + [128009, ...]`, and its
// HuggingFace Llama tokenizer has `add_bos_token` set -- so `tokenizer(text)`
// already begins with 128000. That accounts for the inner one. It then
// *decodes the whole id sequence back to a string* and hands the string to
// vLLM, which tokenizes it again with the same setting and prepends a second
// 128000 at the very front. So the sequence the model is actually served, and
// therefore the one it behaves best on, carries both.
//
// Note also that 128009 (`<|eot_id|>`) appears *inside* the prompt as a
// separator. It is not a stop token here: treating it as one ends generation
// as soon as the model echoes the separator, which it readily does.
//
// ## Slot tracking and resynchronisation
//
// The seven codes of a frame are emitted in order, and slot `i` is offset by
// `i * 4096`. A token is only a valid code for the slot the stream currently
// expects if `code` lands inside `[0, 4096)`:
//
//   * a token really belonging to a later slot computes `code >= 4096`
//   * one belonging to an earlier slot computes `code < 0`
//
// So the range check *is* the slot check, and rejecting a token without
// advancing the slot counter lets the stream resynchronise after the model
// emits something unexpected.
//
// This differs from the reference in one place, deliberately. The reference
// accepts a code only when it is strictly positive, which throws away every
// legitimate code 0 -- and because a rejected token does not advance the slot,
// discarding a valid code 0 shifts every subsequent code by one slot and
// corrupts the rest of the utterance. Checking `0 <= code < 4096` keeps the
// resynchronisation behaviour and drops the off-by-one.
//
// ## Frame layout
//
// Seven codes cover four frames at the finest codebook rate, distributed
// 1:2:4 across the three time scales:
//
//     level 0 (stride 4):  t0
//     level 1 (stride 2):  t1, t4
//     level 2 (stride 1):  t2, t3, t5, t6
//
// which is 2048 samples, or 85.33 ms at 24 kHz. Realtime therefore needs about
// 82 tokens a second.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "rt/result.hpp"
#include "rt/sampling.hpp"
#include "rt/snac.hpp"
#include "rt/tokenizer.hpp"
#include "rt/transformer5.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

struct OrpheusConfig {
    /// Id of the code-0 token for slot 0, i.e. `<custom_token_10>`.
    std::size_t audio_token_base = 128266;
    /// Codes per frame group.
    std::size_t codes_per_frame = 7;
    /// Entries in each SNAC codebook.
    std::size_t codebook_size = 4096;

    /// Prepended to the encoded prompt: `<custom_token_3>`.
    std::size_t prompt_start = 128259;
    /// Llama's `<|begin_of_text|>`. Emitted twice: once before the start
    /// marker and once after it. See the file header for why.
    std::size_t bos_token = 128000;
    /// Whether to emit the leading BOS that vLLM's re-tokenization adds.
    bool leading_bos = true;
    /// Appended: `<|eot_id|>`, `<custom_token_4>`, `<custom_token_5>`,
    /// `<custom_token_1>`.
    std::vector<std::size_t> prompt_end{128009, 128260, 128261, 128257};
    /// Ends generation. `<custom_token_2>` is the end-of-audio marker.
    ///
    /// Deliberately does not include 128009: that token is a separator inside
    /// the prompt, and the model emits it freely without meaning to stop.
    std::vector<std::size_t> stop_tokens{128258};

    [[nodiscard]] static OrpheusConfig defaults();

    /// One past the last audio token id.
    [[nodiscard]] std::size_t audio_token_limit() const {
        return audio_token_base + codes_per_frame * codebook_size;
    }

    /// True when `id` falls in the audio range for any slot.
    [[nodiscard]] bool is_audio_token(std::size_t id) const {
        return id >= audio_token_base && id < audio_token_limit();
    }
};

/// Voices the fine-tuned model was trained on, in the order the model card
/// lists them. `tara` is the reference's default.
[[nodiscard]] const std::vector<std::string>& orpheus_voices();

/// True when `voice` is one of the trained voices. An unknown voice still
/// synthesises -- it is only a prompt prefix -- but it will not sound like a
/// consistent speaker, so callers should warn.
[[nodiscard]] bool orpheus_voice_known(std::string_view voice);

// =============================================================================
// Code stream
// =============================================================================

/// Turns the sampled token stream into SNAC codes, tracking slot position and
/// resynchronising on unexpected tokens.
class OrpheusCodeStream {
   public:
    explicit OrpheusCodeStream(OrpheusConfig cfg) : cfg_(std::move(cfg)) {}

    /// Offer a sampled id.
    ///
    /// Returns true when it was a valid code for the expected slot and was
    /// kept. A false return leaves the slot counter untouched, which is what
    /// lets the stream recover.
    bool push(std::size_t token_id);

    /// Codes accepted so far.
    [[nodiscard]] std::size_t accepted() const { return flat_.size(); }

    /// Codes rejected so far, for reporting -- a high count means the backbone
    /// is drifting out of the audio vocabulary.
    [[nodiscard]] std::size_t rejected() const { return rejected_; }

    /// Complete 7-code groups.
    [[nodiscard]] std::size_t complete_groups() const {
        return flat_.size() / cfg_.codes_per_frame;
    }

    /// Frames at the finest codebook rate, i.e. 4 per complete group.
    [[nodiscard]] std::size_t frames() const;

    /// The three SNAC code levels, truncated to complete groups.
    ///
    /// Empty when no group has completed. Level 0 gets one code per group,
    /// level 1 two, level 2 four.
    [[nodiscard]] std::vector<std::vector<std::uint32_t>> snac_codes() const;

   private:
    OrpheusConfig cfg_;
    /// Accepted codes in emission order.
    std::vector<std::uint32_t> flat_;
    std::size_t rejected_ = 0;
};

// =============================================================================
// Synthesis
// =============================================================================

struct OrpheusRequest {
    std::string text;
    std::string voice = "tara";
    /// 1200 tokens is about 14.6 s of audio.
    std::size_t max_new = 1200;
    /// The reference uses temperature 0.6, top-p 0.8 and repetition penalty
    /// 1.3. Note that this project's penalty applies over a trailing 64-token
    /// window of generated ids, while the reference applies it over the whole
    /// context -- at 7 codes a frame, 64 tokens is about 9 frames.
    SamplingParams sampling;
    SnacNoise noise = SnacNoise::Seeded;
    /// Restrict sampling to the audio range after the prompt.
    ///
    /// The model is fine-tuned to emit only audio tokens here, so this changes
    /// nothing when it behaves. It stops a drifting model from emitting text
    /// that would otherwise be discarded as rejected codes.
    bool mask_to_audio = true;
    bool debug = false;
};

struct OrpheusResult {
    std::vector<float> samples;
    std::size_t sample_rate = 24000;
    std::size_t tokens_generated = 0;
    std::size_t codes_accepted = 0;
    std::size_t codes_rejected = 0;
    std::size_t groups = 0;
    double generate_seconds = 0.0;
    double decode_seconds = 0.0;

    /// Audio duration in seconds.
    [[nodiscard]] double audio_seconds() const;
    /// Wall time over audio duration. Below 1.0 is faster than realtime.
    [[nodiscard]] double realtime_factor() const;
};

/// Build the framed prompt token sequence for a voice and text.
///
/// The markers are injected as raw ids: encoding the literal text
/// "<custom_token_3>" would split it into ordinary subword pieces.
[[nodiscard]] Result<std::vector<std::size_t>> orpheus_build_prompt(
    const HfBpeTokenizer& tok, const OrpheusConfig& cfg, std::string_view voice,
    std::string_view text);

/// Generate audio tokens and decode them to samples.
[[nodiscard]] Result<OrpheusResult> orpheus_synthesize(const LlamaModel& model,
                                                       const SnacDecoder& snac,
                                                       const HfBpeTokenizer& tok,
                                                       const OrpheusRequest& request,
                                                       const OrpheusConfig& cfg);

/// Sensible sampling defaults, matching the reference engine.
[[nodiscard]] SamplingParams orpheus_default_sampling(std::uint64_t seed);

}  // namespace rt

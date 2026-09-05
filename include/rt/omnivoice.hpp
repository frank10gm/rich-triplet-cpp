#pragma once

// =============================================================================
// OmniVoice -- masked diffusion text to speech
// =============================================================================
//
// Ties `transformer6.hpp` (the bidirectional Qwen3 backbone) to
// `omnivoice_codec.hpp` (the acoustic decoder).
//
// ## The sequence
//
// A conditional pass is laid out as:
//
//   <|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>
//   <|text_start|>{text}<|text_end|>
//   [ target frames, all masked ]
//
// The unconditional pass for classifier-free guidance is the **target frames
// alone** -- no style, no text. Both run through the model each step, and their
// log-probabilities are combined.
//
// ## Voice cloning
//
// A reference clip does not add a mode. It adds a prefix:
//
//   <|denoise|><|lang_start|>...<|instruct_end|>
//   <|text_start|>{ref_text} {text}<|text_end|>
//   [ reference frames, decided ][ target frames, all masked ]
//
// so the model is continuing a recording it can see rather than imitating one
// it cannot, and every masked position attends to the reference through the
// same bidirectional attention it uses for everything else. The unconditional
// pass is unchanged -- still the masked frames alone -- which is what makes the
// guidance push *towards* the reference voice.
//
// ## The loop
//
// Every audio position starts masked. Each step runs both passes, scores every
// (codebook, position) pair by confidence, and unmasks the `k` best, where `k`
// comes from a timestep schedule. After `num_step` steps everything is decided.
//
//   log_probs = log_softmax(c + guidance * (c - u))     both already log_softmax
//   scores    = max(log_probs) - codebook_index * layer_penalty
//   unmask the top k of scores, ignoring positions already decided
//
// The layer penalty is what makes decoding coarse to fine: codebook 0 is
// unpenalised, so it is decided first and the later codebooks condition on it.
//
// ## Cost
//
// Nothing is cached, because unmasking a position changes every position that
// attends to it. So this is `num_step * 2` full-sequence forward passes -- 64
// at the defaults. That is more arithmetic than an autoregressive model of the
// same size would do, but it is all batched matmul rather than memory-bound
// GEMV, which is the shape BLAS is good at.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rt/omnivoice_codec.hpp"
#include "rt/result.hpp"
#include "rt/sampling.hpp"
#include "rt/tokenizer.hpp"
#include "rt/transformer6.hpp"

namespace rt {

// =============================================================================
// Generation config
// =============================================================================

struct OmniGenConfig {
    /// Unmasking steps.
    ///
    /// The reference ships 32. This is 12, which is three times faster and
    /// was judged indistinguishable by ear on Italian -- a listening call,
    /// since every number this project prints looks the same across the range.
    /// `--steps 32` restores the reference's setting.
    std::size_t num_step = 12;
    /// Classifier-free guidance strength; 0 disables the unconditional pass
    /// entirely and halves the work.
    float guidance_scale = 2.0f;
    /// Gumbel noise added to the *position* scores, so the choice of which
    /// positions to unmask is stochastic even though the tokens are argmax.
    float position_temperature = 5.0f;
    /// Gumbel noise on the token choice itself. 0 means argmax.
    float class_temperature = 0.0f;
    /// Penalty per codebook index, pushing the coarse codebooks to be decided
    /// first.
    float layer_penalty_factor = 5.0f;
    /// Warps the timestep schedule; below 1 unmasks more early.
    float t_shift = 0.1f;
    std::uint64_t seed = 0;

    // -- long text -----------------------------------------------------------
    //
    // The model was not trained to hold a voice across a monologue, and past
    // roughly half a minute a single pass stops sounding like speech at all --
    // the frames are there and the model runs out of things to put in them.
    // So long text is split, generated piece by piece, and joined.

    /// Split when the estimate exceeds this many seconds. Zero never splits.
    float chunk_threshold_seconds = 30.0f;
    /// Roughly how much audio one chunk should carry.
    float chunk_seconds = 15.0f;
    /// Silence between chunks, a third of it spent fading each side.
    float chunk_gap_seconds = 0.3f;

    [[nodiscard]] static OmniGenConfig defaults();
};

// =============================================================================
// Schedules
// =============================================================================

/// The warped timestep grid: `num_step + 1` values from 0 to 1.
///
/// `t_shift * t / (1 + (t_shift - 1) * t)` bends a linear grid so that with
/// `t_shift < 1` the early steps cover more of the interval, unmasking more
/// tokens up front.
[[nodiscard]] std::vector<float> omni_time_steps(std::size_t num_step, float t_shift);

/// How many (codebook, position) pairs to unmask at each step.
///
/// Sums to exactly `total` -- the last step takes whatever is left, so nothing
/// can be dropped by rounding.
[[nodiscard]] std::vector<std::size_t> omni_unmask_schedule(std::size_t total,
                                                            std::size_t num_step, float t_shift);

// =============================================================================
// Requests
// =============================================================================

struct OmniRequest {
    std::string text;
    /// Language hint, e.g. "Italian". Empty becomes "None".
    std::string language;
    /// Free-text voice description, e.g. "a calm young woman". Empty becomes
    /// "None".
    std::string instruct;
    /// Audio seconds to generate. Zero asks for the length heuristic.
    float duration_seconds = 0.0f;

    // -- voice cloning -------------------------------------------------------
    //
    // A reference clip is not a separate mode. Its codes are prepended to the
    // target frames as *already decided* positions and its transcript to the
    // text, so the model is asked to continue a recording it can see rather
    // than to imitate one it cannot. Everything else about the loop is the
    // same.

    /// What the reference clip says. Required alongside `ref_codes`: the model
    /// has to know which part of the text it has already heard.
    std::string ref_text;
    /// The reference clip encoded, `[codebook][frame]`. Empty means no clone.
    std::vector<std::vector<std::uint32_t>> ref_codes;
    /// The reference clip's loudness before it was levelled for the encoder.
    /// Zero means unknown, which leaves the output gain alone.
    float ref_rms = 0.0f;

    OmniGenConfig gen;
    bool debug = false;

    /// Frames of reference audio, or zero.
    [[nodiscard]] std::size_t ref_frames() const {
        return ref_codes.empty() ? 0 : ref_codes.front().size();
    }
};

struct OmniResult {
    std::vector<float> samples;
    std::size_t sample_rate = 24000;
    std::size_t frames = 0;
    std::size_t prompt_tokens = 0;
    std::size_t forward_passes = 0;
    /// Pieces the text was split into; 1 when it was short enough to run whole.
    std::size_t chunks = 1;
    double generate_seconds = 0.0;
    double decode_seconds = 0.0;

    [[nodiscard]] double audio_seconds() const;
    [[nodiscard]] double realtime_factor() const;
};

// =============================================================================
// Pipeline
// =============================================================================

/// Estimate how many frames a piece of text needs.
///
/// The reference's rule-based estimator, ported in `duration.hpp`: a phonetic
/// weight per character, scaled against "Nice to meet you." at 25 frames.
/// `--duration` still overrides it, because getting the length wrong is
/// audible -- too few frames truncates mid-word, too many and the model runs
/// out of things to say and collapses into silence.
[[nodiscard]] std::size_t omni_estimate_frames(std::string_view text,
                                               const OmniCodecConfig& codec);

/// The same estimate, calibrated on a reference clip instead of the built-in
/// phrase.
///
/// Strictly better when there is one: it measures this speaker's rate rather
/// than an average one, so a slow voice gets the frames it needs. Falls back to
/// the built-in reference when `ref_text` is empty or `ref_frames` is zero.
[[nodiscard]] std::size_t omni_estimate_frames_from_reference(std::string_view text,
                                                              std::string_view ref_text,
                                                              std::size_t ref_frames,
                                                              const OmniCodecConfig& codec);

/// The reference's text normalisation, applied to the prompt with or without a
/// reference clip.
///
/// Trims, joins the reference transcript in front, drops line breaks, folds
/// runs of spaces and tabs, swaps fullwidth parentheses for ASCII ones, and
/// removes whitespace next to a CJK character -- where a space is a typesetting
/// artefact rather than a word boundary and would be spoken as a pause.
[[nodiscard]] std::string omni_combine_text(std::string_view text, std::string_view ref_text);

/// A reference clip, ready for the codec.
struct OmniReference {
    /// Mono, at the codec's sample rate, and a whole number of frames long.
    std::vector<float> samples;
    /// The clip's loudness *before* levelling, which the synthesised audio is
    /// scaled back to at the end.
    float rms = 0.0f;
    /// Seconds of audio, for the caller to warn about.
    [[nodiscard]] double seconds(const OmniCodecConfig& codec) const;
};

/// Prepare a reference recording: resample to the codec's rate, level it, and
/// trim it to a whole number of frames.
///
/// The levelling matters more than it looks. The codec was fit on speech at a
/// particular loudness, and a quiet recording encodes into a part of the
/// codebook space that carries a quiet voice rather than that voice quietly.
/// So a clip under 0.1 RMS is brought up to it, and the original loudness is
/// restored on the way out.
[[nodiscard]] Result<OmniReference> omni_prepare_reference(std::span<const float> samples,
                                                           std::size_t sample_rate,
                                                           const OmniCodecConfig& codec);

// =============================================================================
// Long text
// =============================================================================

/// Split text into pieces of about `chunk_chars` characters, breaking at
/// punctuation.
///
/// Breaking mid-sentence would put a seam where the prosody is still rising,
/// so splits only happen after `.,;:!?` and their fullwidth counterparts. A
/// full stop that ends a known abbreviation -- "Dr.", "e.g.", "No." -- is not a
/// break, since splitting there would strand a title from its name.
///
/// Sentences are then merged greedily up to `chunk_chars`, so a chunk overruns
/// only when a single sentence already does. Pieces shorter than
/// `min_chunk_chars` are folded into a neighbour: a two-character chunk would
/// be given its own speaker and sound like one.
[[nodiscard]] std::vector<std::string> omni_chunk_text(std::string_view text,
                                                       std::size_t chunk_chars,
                                                       std::size_t min_chunk_chars = 3);

/// Join chunk waveforms with a fade-out, a silence, and a fade-in.
///
/// Not an overlap-add: the pieces are separate utterances, not a continuous
/// signal cut in two, so crossfading them onto each other would sound like two
/// people talking over one another. The gap is a breath, and the fades keep
/// its edges from clicking.
[[nodiscard]] std::vector<float> omni_cross_fade(const std::vector<std::vector<float>>& chunks,
                                                 std::size_t sample_rate,
                                                 float silence_seconds = 0.3f);

/// Build the conditional sequence: style markers, text, reference codes if any,
/// then masked frames.
[[nodiscard]] Result<std::vector<OmniToken>> omni_build_conditional(
    const HfBpeTokenizer& tok, const Config6& cfg, const OmniRequest& request,
    std::size_t frames);

/// Build the unconditional sequence: the masked frames alone.
[[nodiscard]] std::vector<OmniToken> omni_build_unconditional(const Config6& cfg,
                                                              std::size_t frames);

/// Run the unmasking loop and decode the result.
///
/// `accel`, when given, runs the forward passes instead of `lm` -- the model is
/// still needed for its config and its prompt layout. Passing a backend whose
/// weights came from a different model is the caller's problem.
[[nodiscard]] Result<OmniResult> omni_synthesize(const OmniLm& lm, const OmniCodecDecoder& codec,
                                                 const HfBpeTokenizer& tok,
                                                 const OmniRequest& request,
                                                 const OmniForward* accel = nullptr);

}  // namespace rt

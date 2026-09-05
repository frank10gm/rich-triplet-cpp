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
    /// Unmasking steps. The reference default.
    std::size_t num_step = 32;
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
    OmniGenConfig gen;
    bool debug = false;
};

struct OmniResult {
    std::vector<float> samples;
    std::size_t sample_rate = 24000;
    std::size_t frames = 0;
    std::size_t prompt_tokens = 0;
    std::size_t forward_passes = 0;
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

/// Build the conditional sequence: style markers, text, then masked frames.
[[nodiscard]] Result<std::vector<OmniToken>> omni_build_conditional(
    const HfBpeTokenizer& tok, const Config6& cfg, const OmniRequest& request,
    std::size_t frames);

/// Build the unconditional sequence: the masked frames alone.
[[nodiscard]] std::vector<OmniToken> omni_build_unconditional(const Config6& cfg,
                                                              std::size_t frames);

/// Run the unmasking loop and decode the result.
[[nodiscard]] Result<OmniResult> omni_synthesize(const OmniLm& lm, const OmniCodecDecoder& codec,
                                                 const HfBpeTokenizer& tok,
                                                 const OmniRequest& request);

}  // namespace rt

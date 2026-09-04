#pragma once

// =============================================================================
// Sampling parameters and RNGs
// =============================================================================
//
// Shared by every generation loop. Two LCG variants exist because the two
// engines derive their float differently from the same state update, and their
// sampled token streams have to stay reproducible.

#include <cstdint>
#include <optional>
#include <vector>

#include "rt/mat.hpp"

namespace rt {

struct SamplingParams {
    /// Divide logits by this before softmax.
    ///   0.0 -> greedy argmax (no randomness)
    ///   1.0 -> unmodified distribution
    ///   >1  -> flatter (more random);  <1 -> sharper (more greedy)
    float temperature = 1.0f;

    /// Keep only the top-k tokens by probability. 0 disables the filter.
    std::size_t top_k = 0;

    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability reaches top_p. 1.0 disables the filter.
    float top_p = 1.0f;

    /// Divide the logit of any previously-seen token by this factor.
    /// 1.0 = no penalty, 1.1-1.3 = mild, 2.0 = aggressive.
    float repetition_penalty = 1.0f;

    /// Seed for the internal LCG. Same seed + same logits -> same sample.
    std::uint64_t seed = 0;

    /// Stop generation at this token id; unset generates exactly `max_new`.
    /// GPT-2's EOS is 50256; tiktoken (GPT-OSS) uses 100257.
    std::optional<std::size_t> eos_token_id;

    /// Reduce a token's logit in proportion to how often it has appeared:
    /// `logit[tok] -= frequency_penalty * count(tok)`. 0.0 disables it.
    /// Typical values 0.2-1.0; positive values discourage word repetition.
    float frequency_penalty = 0.0f;

    /// Reduce the logit of any token that has appeared at all, regardless of
    /// count: `logit[tok] -= presence_penalty`. 0.0 disables it.
    /// Typical values 0.2-1.0; encourages new vocabulary.
    float presence_penalty = 0.0f;

    /// Restrict sampling to ids in `[allowed_min, allowed_max)`, plus any
    /// listed in `allowed_extra`. Both unset means no restriction.
    ///
    /// Useful when a model's vocabulary is partitioned by purpose -- Orpheus
    /// puts 28 672 audio codes alongside 128 256 text tokens and should only
    /// emit the former once its prompt is consumed. Masking is not a
    /// correctness requirement for a well-behaved model; it stops a drifting
    /// one from wandering out of the range the caller can use.
    std::optional<std::size_t> allowed_min;
    std::optional<std::size_t> allowed_max;
    /// Ids permitted regardless of the range -- stop markers, typically,
    /// which must stay reachable or generation runs to `max_new`.
    std::vector<std::size_t> allowed_extra;

    /// Greedy (deterministic) sampling -- always the argmax.
    [[nodiscard]] static SamplingParams greedy() {
        SamplingParams p;
        p.temperature = 0.0f;
        p.top_p = 1.0f;
        return p;
    }

    /// Typical creative-writing defaults.
    [[nodiscard]] static SamplingParams creative(std::uint64_t seed) {
        SamplingParams p;
        p.temperature = 0.8f;
        p.top_k = 50;
        p.top_p = 0.92f;
        p.repetition_penalty = 1.1f;
        p.seed = seed;
        return p;
    }
};

/// LCG with Knuth's multiplier, taking the float from the top 32 bits.
/// Used by the Gemma 3 sampler.
class LcgRng {
   public:
    explicit LcgRng(std::uint64_t seed) : state_(seed + 1) {}

    [[nodiscard]] float next_f32() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>(static_cast<std::uint32_t>(state_ >> 32)) /
               static_cast<float>(UINT32_MAX);
    }

   private:
    std::uint64_t state_;
};

/// Same state update, but the float comes from 53 bits. Used by the GPT-OSS
/// sampler; the two produce different streams and must not be interchanged.
class LcgRng53 {
   public:
    explicit LcgRng53(std::uint64_t seed) : state_(seed + 1) {}

    [[nodiscard]] std::uint64_t next_u64() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return state_;
    }

    [[nodiscard]] float next_f32() {
        return static_cast<float>(next_u64() >> 11) / static_cast<float>(1ull << 53);
    }

   private:
    std::uint64_t state_;
};

}  // namespace rt

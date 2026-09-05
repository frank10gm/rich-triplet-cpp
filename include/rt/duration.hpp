#pragma once

// =============================================================================
// Rule-based duration estimation
// =============================================================================
//
// A masked diffusion model has to be told how long its output is before it
// generates any of it: every frame exists, masked, from the first step. Get it
// wrong and the failure is not graceful. Too few frames truncates mid-word;
// too many and the model has nothing to say for the remainder and collapses
// into near-silence -- measured on one Italian sentence, asking for 10 s of a
// 4 s phrase gave a peak of 0.001 and 98% silence.
//
// So OmniVoice ships an estimator, and it is not a neural one. It is a lookup
// table: every character gets a phonetic weight relative to one Latin letter,
// the weights are summed, and the sum is scaled against a reference phrase of
// known length. This is a port of `RuleDurationEstimator` from
// `omnivoice/utils/duration.py` (k2-fsa/OmniVoice, Apache 2.0, Xiaomi Corp.).
//
// ## Why a weight per character
//
// Characters are not equally expensive to say. A CJK ideograph is a whole
// syllable, a Devanagari cluster is a consonant plus its vowel, and a combining
// accent is silent -- it modifies the letter it follows and adds no time. A
// digit is the outlier: "2024" is four characters and roughly fourteen Latin
// letters' worth of speech, because it is spoken as words.
//
//     cjk 3.0   ethiopic 3.0   yi 3.0        hangul 2.5     kana 2.2
//     indic 1.8 khmer_myanmar 1.8            thai_lao 1.5   arabic 1.5
//     hebrew 1.5                             latin 1.0      cyrillic 1.0
//     greek 1.0 armenian 1.0                 georgian 1.0   default 1.0
//     punctuation 0.5             space 0.2  digit 3.5      mark 0.0
//
// ## How a character is classified
//
// Unicode general category first, script block second. The order matters:
// a Devanagari vowel sign sits inside the Devanagari block but is a combining
// mark, and classifying it by block would charge 1.8 for something silent.
// So marks, punctuation, symbols, separators and numbers are pulled out by
// category, and only what is left is looked up by block.
//
// ## The short-text boost
//
// Linear extrapolation underestimates short utterances, because the fixed
// costs -- onset, final lengthening, the breath at the end -- do not shrink
// with the text. Below 50 frames (2 s) the estimate is pulled up a power
// curve, `50 * (est/50)^(1/3)`, which leaves 50 alone and lifts 25 to 40.

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rt {

// =============================================================================
// Character classes
// =============================================================================

/// The phonetic class of a character. Each carries one weight.
enum class CharClass : std::uint8_t {
    // Logographic -- one character is a syllable or a word.
    Cjk,
    Ethiopic,
    Yi,
    // Syllabic blocks.
    Hangul,
    Kana,
    // Abugidas -- a consonant carrying its vowel.
    Indic,
    KhmerMyanmar,
    ThaiLao,
    // Abjads -- consonant-heavy, vowels often unwritten.
    Arabic,
    Hebrew,
    // Segmental alphabets, the 1.0 baseline.
    Latin,
    Cyrillic,
    Greek,
    Armenian,
    Georgian,
    /// Anything else with no entry of its own, also at 1.0.
    Default,
    // Category-driven classes, which win over the block lookup.
    Punctuation,
    Space,
    Digit,
    Mark,
};

/// The speaking weight of a class, relative to one Latin letter.
[[nodiscard]] float class_weight(CharClass c);

/// Classify one code point.
[[nodiscard]] CharClass char_class(char32_t cp);

/// The speaking weight of one code point.
[[nodiscard]] inline float char_weight(char32_t cp) { return class_weight(char_class(cp)); }

/// Sum the weights of every code point in a UTF-8 string.
[[nodiscard]] double text_weight(std::string_view text);

// =============================================================================
// Estimation
// =============================================================================

/// Estimate how long `target` takes to say, given a reference phrase of known
/// length.
///
/// `ref_duration` sets the unit: pass seconds and the result is seconds, pass
/// codec frames and the result is frames. Returns 0 when the reference is
/// unusable (empty, zero-weight, or non-positive duration).
///
/// `low_threshold` is where the short-text boost stops applying, in the same
/// unit; a non-positive value disables the boost. `boost_strength` is the
/// reciprocal of the curve's exponent -- 1 is linear, 3 is the default cube
/// root.
[[nodiscard]] double estimate_duration(std::string_view target, std::string_view ref,
                                       double ref_duration, double low_threshold = 50.0,
                                       double boost_strength = 3.0);

}  // namespace rt

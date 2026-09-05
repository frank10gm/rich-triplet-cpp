#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <string>

#include "rt/duration.hpp"
#include "rt/omnivoice.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

namespace {

/// The phrase the reference calibrates against, at 25 frames -- one second.
constexpr const char* kRef = "Nice to meet you.";
constexpr double kRefFrames = 25.0;

}  // namespace

// =============================================================================
// Weights
// =============================================================================

TEST_CASE("every class carries its reference weight", "[duration]") {
    REQUIRE(approx(class_weight(CharClass::Cjk), 3.0f));
    REQUIRE(approx(class_weight(CharClass::Ethiopic), 3.0f));
    REQUIRE(approx(class_weight(CharClass::Yi), 3.0f));
    REQUIRE(approx(class_weight(CharClass::Hangul), 2.5f));
    REQUIRE(approx(class_weight(CharClass::Kana), 2.2f));
    REQUIRE(approx(class_weight(CharClass::Indic), 1.8f));
    REQUIRE(approx(class_weight(CharClass::KhmerMyanmar), 1.8f));
    REQUIRE(approx(class_weight(CharClass::ThaiLao), 1.5f));
    REQUIRE(approx(class_weight(CharClass::Arabic), 1.5f));
    REQUIRE(approx(class_weight(CharClass::Hebrew), 1.5f));
    REQUIRE(approx(class_weight(CharClass::Latin), 1.0f));
    REQUIRE(approx(class_weight(CharClass::Cyrillic), 1.0f));
    REQUIRE(approx(class_weight(CharClass::Greek), 1.0f));
    REQUIRE(approx(class_weight(CharClass::Armenian), 1.0f));
    REQUIRE(approx(class_weight(CharClass::Georgian), 1.0f));
    REQUIRE(approx(class_weight(CharClass::Default), 1.0f));
    REQUIRE(approx(class_weight(CharClass::Punctuation), 0.5f));
    REQUIRE(approx(class_weight(CharClass::Space), 0.2f));
    REQUIRE(approx(class_weight(CharClass::Digit), 3.5f));
    REQUIRE(approx(class_weight(CharClass::Mark), 0.0f));
}

// =============================================================================
// Classification
// =============================================================================

TEST_CASE("script blocks map to their class", "[duration]") {
    REQUIRE(char_class(U'A') == CharClass::Latin);
    REQUIRE(char_class(U'z') == CharClass::Latin);
    REQUIRE(char_class(0x00E8) == CharClass::Latin);  // e-grave, precomposed
    REQUIRE(char_class(0x1EF9) == CharClass::Latin);  // Vietnamese y-tilde
    REQUIRE(char_class(0x03B1) == CharClass::Greek);
    REQUIRE(char_class(0x0400) == CharClass::Cyrillic);
    REQUIRE(char_class(0x0561) == CharClass::Armenian);
    REQUIRE(char_class(0x10A0) == CharClass::Georgian);
    REQUIRE(char_class(0x05D0) == CharClass::Hebrew);
    REQUIRE(char_class(0x0627) == CharClass::Arabic);
    REQUIRE(char_class(0x0915) == CharClass::Indic);        // Devanagari ka
    REQUIRE(char_class(0x0F00) == CharClass::Indic);        // Tibetan
    REQUIRE(char_class(0x0E01) == CharClass::ThaiLao);
    REQUIRE(char_class(0x1780) == CharClass::KhmerMyanmar);
    REQUIRE(char_class(0x1000) == CharClass::KhmerMyanmar);  // Myanmar
    REQUIRE(char_class(0x1200) == CharClass::Ethiopic);
    REQUIRE(char_class(0xA000) == CharClass::Yi);
    REQUIRE(char_class(0x3042) == CharClass::Kana);          // Hiragana
    REQUIRE(char_class(0x30A2) == CharClass::Kana);          // Katakana
    REQUIRE(char_class(0xD55C) == CharClass::Hangul);        // Hangul syllable
    REQUIRE(char_class(0x4E2D) == CharClass::Cjk);
}

TEST_CASE("block lookup takes the first end at or above the code point", "[duration]") {
    // The boundaries are what a bisect over end points gets wrong when it is
    // off by one, and every block below is somebody's alphabet.
    REQUIRE(char_class(0x02AF) == CharClass::Latin);     // last of the Latin run
    REQUIRE(char_class(0x02B0) == CharClass::Greek);     // first past it
    REQUIRE(char_class(0x9FFF) == CharClass::Cjk);       // last CJK ideograph
    REQUIRE(char_class(0xA000) == CharClass::Yi);        // first Yi syllable
    REQUIRE(char_class(0xD7AF) == CharClass::Hangul);    // last Hangul syllable
}

TEST_CASE("the category pass wins over the block", "[duration]") {
    // A Devanagari vowel sign lives inside the Devanagari block but is a
    // combining mark: silent, and charging it 1.8 would inflate every Hindi
    // estimate.
    REQUIRE(char_class(0x093F) == CharClass::Mark);  // Devanagari vowel sign I
    REQUIRE(char_class(0x0301) == CharClass::Mark);  // combining acute
    REQUIRE(char_class(0x17D2) == CharClass::Mark);  // Khmer sign coeng
    REQUIRE(char_class(0x064E) == CharClass::Mark);  // Arabic fatha

    // Punctuation inside a CJK run is still a pause, not an ideograph.
    REQUIRE(char_class(0xFF0C) == CharClass::Punctuation);  // fullwidth comma
    REQUIRE(char_class(0x3001) == CharClass::Punctuation);  // ideographic comma
    REQUIRE(char_class(U'.') == CharClass::Punctuation);
    REQUIRE(char_class(0x20AC) == CharClass::Punctuation);   // euro sign (Sc)
    REQUIRE(char_class(0x1F600) == CharClass::Punctuation);  // emoji (So)
}

TEST_CASE("digits weigh 3.5 in every script", "[duration]") {
    REQUIRE(char_class(U'0') == CharClass::Digit);
    REQUIRE(char_class(0x0BE6) == CharClass::Digit);   // Tamil zero
    REQUIRE(char_class(0x00B2) == CharClass::Digit);   // superscript two
    REQUIRE(char_class(0xFF10) == CharClass::Digit);   // fullwidth zero
    REQUIRE(char_class(0x1FBF5) == CharClass::Digit);  // segmented digit five
}

TEST_CASE("separators are spaces, control characters are not", "[duration]") {
    REQUIRE(char_class(U' ') == CharClass::Space);
    REQUIRE(char_class(0x00A0) == CharClass::Space);  // no-break space
    REQUIRE(char_class(0x3000) == CharClass::Space);  // ideographic space
    // A newline is category Cc, which no branch claims, so it falls through to
    // the first block and counts as a Latin letter. The reference does this
    // too, and text arriving here is a single line anyway.
    REQUIRE(char_class(U'\n') == CharClass::Latin);
}

TEST_CASE("Arabic tatweel is silent", "[duration]") {
    // U+0640 is a letter by category -- it stretches the join between two
    // letters and is not pronounced.
    REQUIRE(char_class(0x0640) == CharClass::Mark);
}

TEST_CASE("the supplementary planes split at U+20000", "[duration]") {
    REQUIRE(char_class(0xFFEF) == CharClass::Latin);    // last block entry
    REQUIRE(char_class(0xFFF0) == CharClass::Default);  // past every block
    REQUIRE(char_class(0x20000) == CharClass::Default); // the bound is strict
    REQUIRE(char_class(0x21000) == CharClass::Cjk);     // CJK Extension B
}

// =============================================================================
// Text weight
// =============================================================================

TEST_CASE("text_weight sums the reference phrase to 14.1", "[duration]") {
    // 13 Latin letters, 3 spaces at 0.2, one full stop at 0.5.
    REQUIRE(approx(static_cast<float>(text_weight(kRef)), 14.1f));
    REQUIRE(approx(static_cast<float>(text_weight("")), 0.0f));
}

TEST_CASE("digits weigh what they take to say", "[duration]") {
    // "2024" is four characters and about fourteen letters of speech, because
    // it is spoken as words. Counting characters would ask for a third of the
    // audio it needs.
    REQUIRE(approx(static_cast<float>(text_weight("2024")), 14.0f));
    REQUIRE(approx(static_cast<float>(text_weight("abcd")), 4.0f));
}

TEST_CASE("text_weight is script aware", "[duration]") {
    REQUIRE(approx(static_cast<float>(text_weight("Ciao, mi chiamo Giulia.")), 19.6f));
    // Four ideographs at 3.0 and two fullwidth punctuation marks at 0.5.
    REQUIRE(approx(static_cast<float>(text_weight("你好，世界！")), 13.0f));
    // Hindi: the vowel signs and the virama cost nothing.
    REQUIRE(approx(static_cast<float>(text_weight("नमस्ते दुनिया")), 12.8f));
}

// =============================================================================
// Estimation
// =============================================================================

TEST_CASE("estimate_duration scales against the reference", "[duration]") {
    // Well above the boost threshold, so the relation is linear: 100 Latin
    // letters against 14.1 weight in 25 frames.
    const double est = estimate_duration(std::string(100, 'A'), kRef, kRefFrames);
    REQUIRE(approx(static_cast<float>(est), 177.305f));
    REQUIRE(approx(static_cast<float>(est), static_cast<float>(100.0 / (14.1 / 25.0))));
}

TEST_CASE("short estimates are boosted up a cube root", "[duration]") {
    // The reference phrase measures 25 frames by construction, and the boost
    // lifts it to 50 * (25/50)^(1/3).
    const double est = estimate_duration(kRef, kRef, kRefFrames);
    REQUIRE(approx(static_cast<float>(est), static_cast<float>(50.0 * std::cbrt(0.5))));
    REQUIRE(approx(static_cast<float>(est), 39.685f));

    // Without the boost it is exactly the reference length again.
    REQUIRE(approx(static_cast<float>(estimate_duration(kRef, kRef, kRefFrames, 0.0)), 25.0f));
}

TEST_CASE("the boost applies only below the threshold", "[duration]") {
    const auto linear = [](double n) { return n / (14.1 / 25.0); };

    // 29 Latin letters land just above 50 frames, and pass through untouched.
    const double above = estimate_duration(std::string(29, 'A'), kRef, kRefFrames);
    REQUIRE(above > 50.0);
    REQUIRE(approx(static_cast<float>(above), static_cast<float>(linear(29))));

    // 28 land just below, and get pulled up -- a little here, more further
    // down: 4 letters go from 7.1 frames to 26.
    const double below = estimate_duration(std::string(28, 'A'), kRef, kRefFrames);
    REQUIRE(below < 50.0);
    REQUIRE(below > linear(28));
    REQUIRE(approx(static_cast<float>(estimate_duration("abcd", kRef, kRefFrames)), 26.076f));
}

TEST_CASE("the boost is monotone", "[duration]") {
    // Compression, not saturation: longer text still asks for more frames.
    double prev = 0.0;
    for (std::size_t n = 1; n <= 400; ++n) {
        const double est = estimate_duration(std::string(n, 'A'), kRef, kRefFrames);
        REQUIRE(est > prev);
        prev = est;
    }
}

TEST_CASE("an unusable reference estimates nothing", "[duration]") {
    REQUIRE(estimate_duration("hello", "", kRefFrames) == 0.0);
    REQUIRE(estimate_duration("hello", kRef, 0.0) == 0.0);
    REQUIRE(estimate_duration("hello", kRef, -1.0) == 0.0);
    // A reference of nothing but combining marks weighs zero.
    REQUIRE(estimate_duration("hello", "\u0301\u0301", kRefFrames) == 0.0);
}

// =============================================================================
// Frames
// =============================================================================

TEST_CASE("omni_estimate_frames matches the reference estimator", "[duration]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    // 25 Hz, so these are also the reference implementation's token counts.
    REQUIRE(c.sample_rate / c.hop_length == 25);

    REQUIRE(omni_estimate_frames("Ciao, mi chiamo Giulia.", c) == 44);
    REQUIRE(omni_estimate_frames("Ciao, mi chiamo Giulia. Oggi e una bella giornata a Roma.", c) ==
            84);
    REQUIRE(omni_estimate_frames("Hello world", c) == 35);
    REQUIRE(omni_estimate_frames("你好，世界！", c) == 38);
    REQUIRE(omni_estimate_frames(kRef, c) == 39);
}

TEST_CASE("omni_estimate_frames never asks for zero frames", "[duration]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    // Empty text weighs nothing, and zero frames would be a decode of nothing.
    REQUIRE(omni_estimate_frames("", c) == 1);
    REQUIRE(omni_estimate_frames("\u0301", c) == 1);
}

TEST_CASE("omni_estimate_frames grows with the text", "[duration]") {
    const OmniCodecConfig c = OmniCodecConfig::defaults();
    const std::size_t shorter = omni_estimate_frames("Ciao.", c);
    const std::size_t longer =
        omni_estimate_frames("Ciao, mi chiamo Giulia e oggi e una bella giornata a Roma.", c);
    REQUIRE(shorter > 0);
    REQUIRE(longer > shorter);
    // Digits are spoken as words, so a year is worth more than its characters.
    REQUIRE(omni_estimate_frames("2024", c) > omni_estimate_frames("abcd", c));
}

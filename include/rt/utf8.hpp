#pragma once

// =============================================================================
// UTF-8 helpers
// =============================================================================
//
// The reference implementation iterates strings by Unicode scalar value
// (Rust's `char`), while `std::string` is a byte sequence. These helpers give
// the same view: decode a string into code points, encode a code point back,
// and answer the character-class questions the tokenizers ask.
//
// ## Character-class coverage
//
// `is_whitespace` implements the Unicode White_Space property exactly -- the
// set is small and fixed.
//
// `is_alphabetic` and `is_numeric` cover ASCII exactly and then the major
// letter/digit blocks (Latin-1 and Latin Extended, Greek, Cyrillic, Hebrew,
// Arabic, Devanagari, CJK, Hiragana, Katakana, Hangul, ...). They are *not* a
// complete implementation of the Unicode Alphabetic and Numeric properties, so
// text in a script outside those ranges can pre-tokenize differently than the
// reference. Everything the model paths actually tokenize falls inside them.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rt::utf8 {

/// Number of bytes in the UTF-8 sequence starting with `b0`.
[[nodiscard]] inline std::size_t seq_len(std::uint8_t b0) {
    if (b0 < 0x80) return 1;
    if (b0 < 0xE0) return 2;
    if (b0 < 0xF0) return 3;
    return 4;
}

/// Decode the code point starting at `s[i]`, advancing `i` past it.
/// Malformed bytes decode to U+FFFD and consume one byte.
[[nodiscard]] char32_t decode_at(std::string_view s, std::size_t& i);

/// Decode a whole string into code points.
[[nodiscard]] std::vector<char32_t> decode(std::string_view s);

/// Decode into (byte_offset, code_point) pairs -- the equivalent of
/// `char_indices()`.
[[nodiscard]] std::vector<std::pair<std::size_t, char32_t>> decode_indices(std::string_view s);

/// Append `cp` to `out` as UTF-8.
void encode_into(std::string& out, char32_t cp);

/// A single code point as a UTF-8 string.
[[nodiscard]] std::string encode(char32_t cp);

/// Replace malformed byte sequences with U+FFFD, like `String::from_utf8_lossy`.
[[nodiscard]] std::string from_bytes_lossy(const std::vector<std::uint8_t>& bytes);

/// The Unicode White_Space property (exact).
[[nodiscard]] bool is_whitespace(char32_t cp);

/// Approximates the Unicode Alphabetic property; exact for ASCII.
[[nodiscard]] bool is_alphabetic(char32_t cp);

/// Approximates the Unicode Numeric property; exact for ASCII.
[[nodiscard]] bool is_numeric(char32_t cp);

[[nodiscard]] inline bool is_alphanumeric(char32_t cp) {
    return is_alphabetic(cp) || is_numeric(cp);
}

[[nodiscard]] inline bool is_ascii_digit(char32_t cp) { return cp >= U'0' && cp <= U'9'; }

[[nodiscard]] inline bool is_ascii_graphic(char32_t cp) { return cp >= 0x21 && cp <= 0x7E; }

/// ASCII-only lowercase, leaving every other code point untouched --
/// the same contract as Rust's `to_ascii_lowercase`.
[[nodiscard]] inline char32_t to_ascii_lowercase(char32_t cp) {
    return (cp >= U'A' && cp <= U'Z') ? cp + 32 : cp;
}

/// Replace every occurrence of `from` with `to`.
[[nodiscard]] std::string replace_all(std::string_view s, std::string_view from,
                                      std::string_view to);

}  // namespace rt::utf8

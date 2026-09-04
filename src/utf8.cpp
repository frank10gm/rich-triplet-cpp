#include "rt/utf8.hpp"

#include <algorithm>
#include <array>

namespace rt::utf8 {

char32_t decode_at(std::string_view s, std::size_t& i) {
    if (i >= s.size()) {
        return 0xFFFD;
    }
    const auto b0 = static_cast<std::uint8_t>(s[i]);
    const std::size_t len = seq_len(b0);
    if (i + len > s.size()) {
        ++i;
        return 0xFFFD;
    }

    char32_t cp;
    switch (len) {
        case 1:
            cp = b0;
            break;
        case 2:
            cp = static_cast<char32_t>(b0 & 0x1F);
            break;
        case 3:
            cp = static_cast<char32_t>(b0 & 0x0F);
            break;
        default:
            cp = static_cast<char32_t>(b0 & 0x07);
            break;
    }
    for (std::size_t k = 1; k < len; ++k) {
        const auto b = static_cast<std::uint8_t>(s[i + k]);
        if ((b & 0xC0) != 0x80) {
            ++i;
            return 0xFFFD;
        }
        cp = (cp << 6) | static_cast<char32_t>(b & 0x3F);
    }
    i += len;
    return cp;
}

std::vector<char32_t> decode(std::string_view s) {
    std::vector<char32_t> out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        out.push_back(decode_at(s, i));
    }
    return out;
}

std::vector<std::pair<std::size_t, char32_t>> decode_indices(std::string_view s) {
    std::vector<std::pair<std::size_t, char32_t>> out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const std::size_t start = i;
        out.emplace_back(start, decode_at(s, i));
    }
    return out;
}

void encode_into(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string encode(char32_t cp) {
    std::string out;
    encode_into(out, cp);
    return out;
}

std::string from_bytes_lossy(const std::vector<std::uint8_t>& bytes) {
    std::string out;
    out.reserve(bytes.size());
    std::size_t i = 0;
    while (i < bytes.size()) {
        const std::uint8_t b0 = bytes[i];
        const std::size_t len = seq_len(b0);

        // A well-formed sequence is copied through verbatim; anything else
        // becomes one replacement character and consumes a single byte.
        bool valid = (i + len <= bytes.size());
        if (valid && len > 1) {
            for (std::size_t k = 1; k < len && valid; ++k) {
                valid = (bytes[i + k] & 0xC0) == 0x80;
            }
        }
        if (valid && len == 1 && b0 >= 0x80) {
            valid = false;  // a lone continuation or invalid lead byte
        }

        if (valid) {
            for (std::size_t k = 0; k < len; ++k) {
                out.push_back(static_cast<char>(bytes[i + k]));
            }
            i += len;
        } else {
            encode_into(out, 0xFFFD);
            ++i;
        }
    }
    return out;
}

bool is_whitespace(char32_t cp) {
    // The Unicode White_Space property, in full.
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D:
        case 0x20: case 0x85: case 0xA0: case 0x1680:
        case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return cp >= 0x2000 && cp <= 0x200A;
    }
}

namespace {
struct Range {
    char32_t lo, hi;
};

/// Major Unicode letter blocks. See the header for the coverage caveat.
constexpr std::array<Range, 21> kLetterRanges{{
    {0x00C0, 0x00D6}, {0x00D8, 0x00F6}, {0x00F8, 0x02AF},  // Latin-1 + Latin Extended + IPA
    {0x0370, 0x03FF},                                      // Greek
    {0x0400, 0x04FF}, {0x0500, 0x052F},                    // Cyrillic
    {0x0530, 0x058F},                                      // Armenian
    {0x0590, 0x05FF},                                      // Hebrew
    {0x0600, 0x06FF}, {0x0750, 0x077F},                    // Arabic
    {0x0900, 0x097F},                                      // Devanagari
    {0x0E00, 0x0E7F},                                      // Thai
    {0x10A0, 0x10FF},                                      // Georgian
    {0x1E00, 0x1EFF},                                      // Latin Extended Additional
    {0x3040, 0x309F}, {0x30A0, 0x30FF},                    // Hiragana, Katakana
    {0x3400, 0x4DBF}, {0x4E00, 0x9FFF},                    // CJK
    {0xA000, 0xA48F},                                      // Yi
    {0xAC00, 0xD7AF},                                      // Hangul syllables
    {0xF900, 0xFAFF},                                      // CJK compatibility
}};

/// Non-ASCII digit blocks.
constexpr std::array<Range, 6> kDigitRanges{{
    {0x0660, 0x0669},  // Arabic-Indic
    {0x06F0, 0x06F9},  // Extended Arabic-Indic
    {0x0966, 0x096F},  // Devanagari
    {0x0E50, 0x0E59},  // Thai
    {0xFF10, 0xFF19},  // Fullwidth
    {0x2160, 0x217F},  // Roman numerals (Nl)
}};

template <std::size_t N>
[[nodiscard]] bool in_ranges(const std::array<Range, N>& ranges, char32_t cp) {
    return std::any_of(ranges.begin(), ranges.end(),
                       [cp](const Range& r) { return cp >= r.lo && cp <= r.hi; });
}
}  // namespace

bool is_alphabetic(char32_t cp) {
    if (cp < 0x80) {
        return (cp >= U'A' && cp <= U'Z') || (cp >= U'a' && cp <= U'z');
    }
    if (cp == 0x00AA || cp == 0x00B5 || cp == 0x00BA) {
        return true;  // ordinal indicators and micro sign are Alphabetic
    }
    if (cp >= 0x10000) {
        // Supplementary planes: CJK extensions and most historic scripts.
        return (cp >= 0x20000 && cp <= 0x2FA1F) || (cp >= 0x10000 && cp <= 0x1342F);
    }
    return in_ranges(kLetterRanges, cp);
}

bool is_numeric(char32_t cp) {
    if (cp < 0x80) {
        return cp >= U'0' && cp <= U'9';
    }
    return in_ranges(kDigitRanges, cp);
}

std::string replace_all(std::string_view s, std::string_view from, std::string_view to) {
    if (from.empty()) {
        return std::string(s);
    }
    std::string out;
    out.reserve(s.size());
    std::size_t pos = 0;
    while (true) {
        const std::size_t hit = s.find(from, pos);
        if (hit == std::string_view::npos) {
            out.append(s.substr(pos));
            return out;
        }
        out.append(s.substr(pos, hit - pos));
        out.append(to);
        pos = hit + from.size();
    }
}

}  // namespace rt::utf8

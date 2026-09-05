#include "rt/tokenizer.hpp"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

#include "rt/utf8.hpp"

namespace rt {

/// Parse the `pieces[]` out of a SentencePiece ModelProto. Defined below, but
/// needed by `from_model_bytes` above it.
[[nodiscard]] Result<std::vector<SentencePiece>> proto_read_pieces(
    const std::vector<std::uint8_t>& bytes);

namespace {

/// Read a whole file into a string.
[[nodiscard]] Result<std::string> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return err("cannot read " + path);
    }
    const auto size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::string out(size, '\0');
    f.read(out.data(), static_cast<std::streamsize>(size));
    return out;
}

/// Split a string into lines, dropping the separators.
[[nodiscard]] std::vector<std::string_view> lines_of(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    while (start <= s.size()) {
        const std::size_t nl = s.find('\n', start);
        if (nl == std::string_view::npos) {
            if (start < s.size()) {
                out.push_back(s.substr(start));
            }
            break;
        }
        std::string_view line = s.substr(start, nl - start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        out.push_back(line);
        start = nl + 1;
    }
    return out;
}

[[nodiscard]] std::string_view trim(std::string_view s) {
    const auto is_ws = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    };
    while (!s.empty() && is_ws(s.front())) {
        s.remove_prefix(1);
    }
    while (!s.empty() && is_ws(s.back())) {
        s.remove_suffix(1);
    }
    return s;
}

/// U+2581, the SentencePiece word-boundary marker.
constexpr std::string_view kLowerOneEighthBlock = "\xE2\x96\x81";

/// Represent raw bytes as a string the vocab can hold: each byte becomes the
/// code point of the same value (ISO-8859-1 style), so bytes above 127 occupy
/// two UTF-8 bytes and `decode` can recover them one code point at a time.
[[nodiscard]] std::string bytes_to_token_str(const std::vector<std::uint8_t>& bytes) {
    std::string out;
    for (std::uint8_t b : bytes) {
        utf8::encode_into(out, b);
    }
    return out;
}

}  // namespace

// =============================================================================
// CharTokenizer
// =============================================================================

CharTokenizer CharTokenizer::from_text(std::string_view text) {
    // Collect unique characters, sorted for determinism across runs.
    const std::vector<char32_t> cps = utf8::decode(text);
    const std::set<char32_t> unique(cps.begin(), cps.end());

    CharTokenizer tok;
    for (char32_t ch : unique) {
        tok.char_to_id_.emplace(ch, static_cast<std::uint32_t>(tok.id_to_char_.size()));
        tok.id_to_char_.push_back(ch);
    }
    return tok;
}

void CharTokenizer::print_vocab() const {
    std::printf("Vocabulary (%zu tokens):\n", id_to_char_.size());
    for (std::size_t id = 0; id < id_to_char_.size(); ++id) {
        const char32_t ch = id_to_char_[id];
        std::string display;
        switch (ch) {
            case U'\n': display = "\\n"; break;
            case U'\t': display = "\\t"; break;
            case U' ': display = "SPACE"; break;
            default: display = utf8::encode(ch); break;
        }
        std::printf("  %4zu -> \"%s\"\n", id, display.c_str());
    }
}

std::uint32_t CharTokenizer::unk_id() const {
    // The last id doubles as the unknown token. A real tokenizer would reserve
    // id 0 for it.
    return id_to_char_.empty() ? 0 : static_cast<std::uint32_t>(id_to_char_.size() - 1);
}

std::vector<std::uint32_t> CharTokenizer::encode(std::string_view text) const {
    std::vector<std::uint32_t> out;
    for (char32_t cp : utf8::decode(text)) {
        const auto it = char_to_id_.find(cp);
        out.push_back(it != char_to_id_.end() ? it->second : unk_id());
    }
    return out;
}

std::string CharTokenizer::decode(const std::vector<std::uint32_t>& ids) const {
    std::string out;
    for (std::uint32_t id : ids) {
        // U+FFFD stands in for ids outside the vocabulary.
        const char32_t cp = id < id_to_char_.size() ? id_to_char_[id] : 0xFFFD;
        utf8::encode_into(out, cp);
    }
    return out;
}

// =============================================================================
// GPT-2 byte encoding
// =============================================================================

std::array<char32_t, 256> gpt2_bytes_to_unicode() {
    std::array<char32_t, 256> result{};

    // The bytes that map to themselves: printable ASCII and printable Latin-1.
    std::vector<std::uint8_t> bs;
    for (int b = '!'; b <= '~'; ++b) {
        bs.push_back(static_cast<std::uint8_t>(b));
    }
    for (int b = 0xA1; b <= 0xFF; ++b) {
        bs.push_back(static_cast<std::uint8_t>(b));
    }
    std::vector<char32_t> cs;
    cs.reserve(256);
    for (std::uint8_t b : bs) {
        cs.push_back(b);
    }

    // The remaining 67 bytes take U+0100 onward, in byte order.
    std::vector<bool> in_bs(256, false);
    for (std::uint8_t b : bs) {
        in_bs[b] = true;
    }
    char32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!in_bs[static_cast<std::size_t>(b)]) {
            bs.push_back(static_cast<std::uint8_t>(b));
            cs.push_back(0x100 + n);
            ++n;
        }
    }

    for (std::size_t i = 0; i < bs.size(); ++i) {
        result[bs[i]] = cs[i];
    }
    return result;
}

std::vector<std::uint8_t> gpt2_decode_token_to_bytes(std::string_view s) {
    static const std::array<char32_t, 256> map = gpt2_bytes_to_unicode();
    static const std::map<char32_t, std::uint8_t> inv = [] {
        std::map<char32_t, std::uint8_t> m;
        for (std::size_t b = 0; b < 256; ++b) {
            m.emplace(map[b], static_cast<std::uint8_t>(b));
        }
        return m;
    }();

    std::vector<std::uint8_t> out;
    for (char32_t c : utf8::decode(s)) {
        const auto it = inv.find(c);
        if (it != inv.end()) {
            out.push_back(it->second);
        }
    }
    return out;
}

namespace {

/// Decode a GPT-2 encoded token into the vocab's byte-string representation.
[[nodiscard]] std::string gpt2_decode_token(std::string_view s) {
    return bytes_to_token_str(gpt2_decode_token_to_bytes(s));
}

/// Count every adjacent (left, right) pair of token ids in a sequence.
[[nodiscard]] std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t> count_pairs(
    const std::vector<std::uint32_t>& tokens) {
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t> counts;
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
        ++counts[{tokens[i], tokens[i + 1]}];
    }
    return counts;
}

/// Replace every occurrence of (left_id, right_id) with merged_id, scanning
/// left to right and skipping both members of each matched pair.
[[nodiscard]] std::vector<std::uint32_t> apply_merge(const std::vector<std::uint32_t>& tokens,
                                                     std::uint32_t left_id, std::uint32_t right_id,
                                                     std::uint32_t merged_id) {
    std::vector<std::uint32_t> result;
    result.reserve(tokens.size());
    std::size_t i = 0;
    while (i < tokens.size()) {
        if (i + 1 < tokens.size() && tokens[i] == left_id && tokens[i + 1] == right_id) {
            result.push_back(merged_id);
            i += 2;
        } else {
            result.push_back(tokens[i]);
            ++i;
        }
    }
    return result;
}

/// Parse a `vocab.json` object of `{"token": id, ...}` into id -> token.
///
/// Hand-written rather than pulled from a JSON library, and indexed by code
/// point so multi-byte tokens like U+0120 survive intact. Returns the largest
/// id seen.
[[nodiscard]] Result<std::uint32_t> parse_vocab_json(
    std::string_view json, std::unordered_map<std::uint32_t, std::string>& out) {
    const std::vector<char32_t> chars = utf8::decode(json);
    const std::size_t n = chars.size();
    std::size_t i = 0;
    std::uint32_t max_id = 0;

    while (i < n && chars[i] != U'{') {
        ++i;
    }
    ++i;

    while (true) {
        while (i < n && (chars[i] == U' ' || chars[i] == U',' || chars[i] == U'\n' ||
                         chars[i] == U'\r' || chars[i] == U'\t')) {
            ++i;
        }
        if (i >= n || chars[i] == U'}') {
            break;
        }

        if (chars[i] != U'"') {
            return err("expected '\"' at char pos " + std::to_string(i));
        }
        ++i;
        std::string key;
        while (i < n) {
            if (chars[i] == U'\\' && i + 1 < n) {
                switch (chars[i + 1]) {
                    case U'"': key.push_back('"'); break;
                    case U'\\': key.push_back('\\'); break;
                    case U'n': key.push_back('\n'); break;
                    default:
                        key.push_back('\\');
                        utf8::encode_into(key, chars[i + 1]);
                        break;
                }
                i += 2;
                continue;
            }
            if (chars[i] == U'"') {
                break;
            }
            utf8::encode_into(key, chars[i]);
            ++i;
        }
        ++i;

        while (i < n && (chars[i] == U':' || chars[i] == U' ')) {
            ++i;
        }

        const std::size_t val_start = i;
        while (i < n && utf8::is_ascii_digit(chars[i])) {
            ++i;
        }
        std::string id_str;
        for (std::size_t k = val_start; k < i; ++k) {
            id_str.push_back(static_cast<char>(chars[k]));
        }
        std::uint32_t id = 0;
        const auto res = std::from_chars(id_str.data(), id_str.data() + id_str.size(), id);
        if (res.ec != std::errc{} || id_str.empty()) {
            return err("bad id \"" + id_str + "\" at char pos " + std::to_string(val_start));
        }

        // The key stays GPT-2 encoded; the caller decodes it.
        out.emplace(id, std::move(key));
        max_id = std::max(max_id, id);
    }

    return max_id;
}

}  // namespace

// =============================================================================
// base64
// =============================================================================

namespace {
constexpr std::string_view kBase64Alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}  // namespace

Result<std::vector<std::uint8_t>> base64_decode(std::string_view s) {
    std::array<std::uint8_t, 256> lookup{};
    lookup.fill(255);
    for (std::size_t i = 0; i < kBase64Alphabet.size(); ++i) {
        lookup[static_cast<std::uint8_t>(kBase64Alphabet[i])] = static_cast<std::uint8_t>(i);
    }

    while (!s.empty() && s.back() == '=') {
        s.remove_suffix(1);
    }

    std::vector<std::uint8_t> out;
    out.reserve(s.size() * 3 / 4 + 1);
    std::uint32_t buf = 0;
    std::uint32_t bits = 0;
    for (char c : s) {
        const std::uint8_t val = lookup[static_cast<std::uint8_t>(c)];
        if (val == 255) {
            return err(std::string("invalid base64 char: ") + c);
        }
        buf = (buf << 6) | val;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>(buf >> bits));
            buf &= (1u << bits) - 1;
        }
    }
    return out;
}

std::string base64_encode(const std::vector<std::uint8_t>& data) {
    std::string out;
    for (std::size_t i = 0; i < data.size(); i += 3) {
        const std::size_t remaining = data.size() - i;
        const std::uint32_t b0 = data[i];
        const std::uint32_t b1 = remaining > 1 ? data[i + 1] : 0;
        const std::uint32_t b2 = remaining > 2 ? data[i + 2] : 0;
        const std::uint32_t n = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kBase64Alphabet[(n >> 18) & 63]);
        out.push_back(kBase64Alphabet[(n >> 12) & 63]);
        out.push_back(remaining > 1 ? kBase64Alphabet[(n >> 6) & 63] : '=');
        out.push_back(remaining > 2 ? kBase64Alphabet[n & 63] : '=');
    }
    return out;
}

// =============================================================================
// BpeTokenizer
// =============================================================================

void BpeTokenizer::rebuild_merge_map() {
    // (left_id, right_id) -> merged_id. A pair appearing twice (which valid BPE
    // never produces) keeps the last entry.
    merge_map_.clear();
    for (const auto& [left, right] : merges_) {
        const auto l = token_to_id_.find(left);
        const auto r = token_to_id_.find(right);
        const auto m = token_to_id_.find(left + right);
        if (l != token_to_id_.end() && r != token_to_id_.end() && m != token_to_id_.end()) {
            merge_map_[{l->second, r->second}] = m->second;
        }
    }
}

BpeTokenizer BpeTokenizer::train(std::string_view text, std::size_t num_merges) {
    BpeTokenizer tok;

    // Step 1: the vocabulary starts as all 256 byte values. Working at the byte
    // level means any UTF-8 input is representable.
    for (int b = 0; b < 256; ++b) {
        std::string s;
        if (utf8::is_ascii_graphic(static_cast<char32_t>(b)) || b == ' ') {
            s.push_back(static_cast<char>(b));
        } else {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            s = buf;
        }
        tok.token_to_id_.emplace(s, static_cast<std::uint32_t>(tok.vocab_.size()));
        tok.vocab_.push_back(std::move(s));
    }

    // Step 2: the corpus as a sequence of byte-level token ids.
    std::vector<std::uint32_t> corpus;
    corpus.reserve(text.size());
    for (char c : text) {
        corpus.push_back(static_cast<std::uint8_t>(c));
    }

    std::printf("BPE training: %zu bytes, %zu merges requested\n", corpus.size(), num_merges);

    // Step 3: repeatedly merge the most frequent pair.
    for (std::size_t merge_idx = 0; merge_idx < num_merges; ++merge_idx) {
        const auto pair_counts = count_pairs(corpus);
        if (pair_counts.empty()) {
            std::printf("No more pairs to merge at step %zu\n", merge_idx);
            break;
        }

        // Most frequent pair, breaking ties by pair value for determinism.
        auto best = pair_counts.begin();
        for (auto it = pair_counts.begin(); it != pair_counts.end(); ++it) {
            if (std::pair{it->second, it->first} > std::pair{best->second, best->first}) {
                best = it;
            }
        }
        const auto [left_id, right_id] = best->first;
        const std::string left_str = tok.vocab_[left_id];
        const std::string right_str = tok.vocab_[right_id];
        const std::string merged_str = left_str + right_str;

        const auto new_id = static_cast<std::uint32_t>(tok.vocab_.size());
        tok.vocab_.push_back(merged_str);
        tok.token_to_id_.emplace(merged_str, new_id);
        tok.merges_.emplace_back(left_str, right_str);

        if (merge_idx < 10 || merge_idx % 100 == 0) {
            std::printf("  merge %4zu: \"%s\" + \"%s\" -> \"%s\" (id %u), count=%zu\n", merge_idx,
                        left_str.c_str(), right_str.c_str(), merged_str.c_str(), new_id,
                        best->second);
        }

        corpus = apply_merge(corpus, left_id, right_id, new_id);
    }

    std::printf("BPE done. Vocabulary size: %zu\n", tok.vocab_.size());
    tok.rebuild_merge_map();
    return tok;
}

Result<BpeTokenizer> BpeTokenizer::from_files(const std::string& vocab_path,
                                              const std::string& merges_path) {
    RT_TRY(vocab_str, read_file(vocab_path));
    RT_TRY(merges_str, read_file(merges_path));
    return from_vocab_and_merges_str(vocab_str, merges_str);
}

Result<BpeTokenizer> BpeTokenizer::from_vocab_and_merges_str(std::string_view vocab_json,
                                                             std::string_view merges_txt) {
    std::unordered_map<std::uint32_t, std::string> id_to_token;
    const auto max_id = parse_vocab_json(vocab_json, id_to_token);
    if (!max_id) {
        return err("vocab.json parse error: " + max_id.error());
    }

    BpeTokenizer tok;
    const std::size_t vocab_size = static_cast<std::size_t>(*max_id) + 1;
    tok.vocab_.assign(vocab_size, std::string{});
    tok.token_to_id_.reserve(vocab_size);

    for (const auto& [id, encoded] : id_to_token) {
        // Decode the GPT-2 byte encoding into the vocab's byte-string form.
        std::string decoded = gpt2_decode_token(encoded);
        tok.token_to_id_.emplace(decoded, id);
        tok.vocab_[id] = std::move(decoded);
    }

    // Fill any gaps, which a well-formed vocab.json should not have.
    for (std::size_t i = 0; i < tok.vocab_.size(); ++i) {
        if (tok.vocab_[i].empty()) {
            tok.vocab_[i] = "<unk_" + std::to_string(i) + ">";
        }
    }

    // merges.txt: one "left right" pair per non-comment line.
    for (std::string_view raw : lines_of(merges_txt)) {
        const std::string_view line = trim(raw);
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const std::size_t sp = line.find(' ');
        if (sp == std::string_view::npos) {
            continue;
        }
        const std::string_view left_enc = line.substr(0, sp);
        const std::string_view right_enc = line.substr(sp + 1);
        if (left_enc.empty() || right_enc.empty()) {
            continue;
        }
        tok.merges_.emplace_back(gpt2_decode_token(left_enc), gpt2_decode_token(right_enc));
    }

    tok.rebuild_merge_map();
    return tok;
}

Result<BpeTokenizer> BpeTokenizer::from_tiktoken_file(const std::string& path) {
    RT_TRY(content, read_file(path));
    return from_tiktoken_str(content);
}

Result<BpeTokenizer> BpeTokenizer::from_tiktoken_str(std::string_view content) {
    std::vector<std::pair<std::uint32_t, std::vector<std::uint8_t>>> entries;
    for (std::string_view raw : lines_of(content)) {
        const std::string_view line = trim(raw);
        if (line.empty()) {
            continue;
        }
        const std::size_t sp = line.find(' ');
        const std::string_view b64 = sp == std::string_view::npos ? line : line.substr(0, sp);
        const std::string_view rank_str =
            sp == std::string_view::npos ? std::string_view{} : trim(line.substr(sp + 1));

        std::uint32_t rank = 0;
        const auto res =
            std::from_chars(rank_str.data(), rank_str.data() + rank_str.size(), rank);
        if (res.ec != std::errc{}) {
            return err("bad rank: \"" + std::string(rank_str) + "\"");
        }
        auto token_bytes = base64_decode(b64);
        if (!token_bytes) {
            return err("base64 error on \"" + std::string(b64) + "\": " + token_bytes.error());
        }
        entries.emplace_back(rank, std::move(*token_bytes));
    }

    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // In tiktoken the rank *is* the token id.
    BpeTokenizer tok;
    std::size_t vocab_size = 0;
    for (const auto& [rank, bytes] : entries) {
        vocab_size = std::max(vocab_size, static_cast<std::size_t>(rank) + 1);
    }
    tok.vocab_.assign(vocab_size, std::string{});
    tok.token_to_id_.reserve(vocab_size);

    for (const auto& [rank, bytes] : entries) {
        std::string s = bytes_to_token_str(bytes);
        tok.token_to_id_.emplace(s, rank);
        tok.vocab_[rank] = std::move(s);
    }

    // Recover the merges: a token with rank >= 256 was formed by merging two
    // lower-ranked tokens, so take the split whose two halves have the lowest
    // maximum rank -- that is the pair that existed earliest.
    std::map<std::vector<std::uint8_t>, std::uint32_t> bytes_to_rank;
    for (const auto& [rank, bytes] : entries) {
        bytes_to_rank.emplace(bytes, rank);
    }

    for (const auto& [rank, bytes] : entries) {
        if (rank < 256 || bytes.size() < 2) {
            continue;
        }
        std::optional<std::pair<std::size_t, std::uint32_t>> best;  // (split, max rank)
        for (std::size_t split = 1; split < bytes.size(); ++split) {
            const std::vector<std::uint8_t> left(bytes.begin(),
                                                 bytes.begin() + static_cast<std::ptrdiff_t>(split));
            const std::vector<std::uint8_t> right(
                bytes.begin() + static_cast<std::ptrdiff_t>(split), bytes.end());
            const auto lr = bytes_to_rank.find(left);
            const auto rr = bytes_to_rank.find(right);
            if (lr != bytes_to_rank.end() && rr != bytes_to_rank.end()) {
                const std::uint32_t score = std::max(lr->second, rr->second);
                if (!best || score < best->second) {
                    best = {split, score};
                }
            }
        }
        if (best) {
            const std::size_t split = best->first;
            tok.merges_.emplace_back(
                bytes_to_token_str({bytes.begin(),
                                    bytes.begin() + static_cast<std::ptrdiff_t>(split)}),
                bytes_to_token_str(
                    {bytes.begin() + static_cast<std::ptrdiff_t>(split), bytes.end()}));
        }
    }

    tok.rebuild_merge_map();
    return tok;
}

std::vector<std::uint32_t> BpeTokenizer::encode(std::string_view text) const {
    // Start with one token per byte (id = byte value), then repeatedly apply
    // the highest-priority applicable merge until none remains.
    std::vector<std::uint32_t> tokens;
    tokens.reserve(text.size());
    for (char c : text) {
        tokens.push_back(static_cast<std::uint8_t>(c));
    }

    while (true) {
        std::optional<std::size_t> best_pos;
        std::size_t best_pri = std::numeric_limits<std::size_t>::max();

        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            const std::pair<std::uint32_t, std::uint32_t> pair{tokens[i], tokens[i + 1]};
            const auto merged = merge_map_.find(pair);
            if (merged == merge_map_.end()) {
                continue;
            }
            // Priority is this merge's position in the learned order.
            for (std::size_t p = 0; p < merges_.size(); ++p) {
                const auto& [l, r] = merges_[p];
                const auto li = token_to_id_.find(l);
                const auto ri = token_to_id_.find(r);
                const auto mi = token_to_id_.find(l + r);
                if (li != token_to_id_.end() && li->second == pair.first &&
                    ri != token_to_id_.end() && ri->second == pair.second &&
                    mi != token_to_id_.end() && mi->second == merged->second) {
                    if (p < best_pri) {
                        best_pri = p;
                        best_pos = i;
                    }
                    break;
                }
            }
        }

        if (!best_pos) {
            break;
        }
        const std::size_t pos = *best_pos;
        const std::uint32_t merged_id = merge_map_.at({tokens[pos], tokens[pos + 1]});
        tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(pos) + 1);
        tokens[pos] = merged_id;
    }

    return tokens;
}

std::string BpeTokenizer::decode(const std::vector<std::uint32_t>& ids) const {
    // Concatenate each token's bytes, then interpret the result as UTF-8.
    //
    // train() writes non-printable bytes as "<0xXX>"; the loaders write each
    // byte as the code point of the same value, so a code point <= 0xFF *is*
    // the byte.
    std::vector<std::uint8_t> bytes;
    for (std::uint32_t id : ids) {
        if (id >= vocab_.size()) {
            continue;
        }
        const std::string& token_str = vocab_[id];
        if (token_str.size() == 6 && token_str.starts_with("<0x") && token_str.ends_with('>')) {
            const std::string hex = token_str.substr(3, 2);
            unsigned value = 0;
            if (std::from_chars(hex.data(), hex.data() + hex.size(), value, 16).ec == std::errc{}) {
                bytes.push_back(static_cast<std::uint8_t>(value));
            }
        } else {
            for (char32_t c : utf8::decode(token_str)) {
                if (c <= 0xFF) {
                    bytes.push_back(static_cast<std::uint8_t>(c));
                }
            }
        }
    }
    return utf8::from_bytes_lossy(bytes);
}

// =============================================================================
// SentencePieceTokenizer
// =============================================================================

Result<SentencePieceTokenizer> SentencePieceTokenizer::from_vocab_text(std::string_view text) {
    std::vector<SentencePiece> pieces;
    std::size_t lineno = 0;
    for (std::string_view raw : lines_of(text)) {
        ++lineno;
        const std::string_view line = trim(raw);
        if (line.empty()) {
            continue;
        }
        // Split on the last tab: a piece could in principle contain one.
        const std::size_t tab = line.rfind('\t');
        if (tab == std::string_view::npos) {
            return err("line " + std::to_string(lineno) +
                       ": expected tab-separated <piece>\\t<score>");
        }
        const std::string_view score_str = trim(line.substr(tab + 1));
        float score = 0.0f;
        if (std::from_chars(score_str.data(), score_str.data() + score_str.size(), score).ec !=
            std::errc{}) {
            return err("line " + std::to_string(lineno) + ": cannot parse score \"" +
                       std::string(score_str) + "\"");
        }
        pieces.push_back({std::string(line.substr(0, tab)), score});
    }
    if (pieces.empty()) {
        return err("empty vocabulary");
    }
    return from_pieces(std::move(pieces));
}

Result<SentencePieceTokenizer> SentencePieceTokenizer::from_model_bytes(
    const std::vector<std::uint8_t>& bytes) {
    RT_TRY(pieces, proto_read_pieces(bytes));
    if (pieces.empty()) {
        return err("no pieces found in protobuf");
    }
    return from_pieces(std::move(pieces));
}

Result<SentencePieceTokenizer> SentencePieceTokenizer::from_model_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return err("cannot read " + path);
    }
    const auto size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::uint8_t> bytes(size);
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    return from_model_bytes(bytes);
}

Result<SentencePieceTokenizer> SentencePieceTokenizer::from_tokens_and_scores(
    std::vector<std::string> tokens, const std::vector<float>& scores) {
    if (tokens.empty()) {
        return err("sentencepiece: empty vocabulary");
    }
    if (tokens.size() != scores.size()) {
        return err("sentencepiece: " + std::to_string(tokens.size()) + " tokens but " +
                   std::to_string(scores.size()) + " scores");
    }
    std::vector<SentencePiece> pieces(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        pieces[i].text = std::move(tokens[i]);
        pieces[i].log_prob = scores[i];
    }
    return from_pieces(std::move(pieces));
}

SentencePieceTokenizer SentencePieceTokenizer::from_pieces(std::vector<SentencePiece> pieces) {
    SentencePieceTokenizer tok;
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        tok.piece_to_id_.emplace(pieces[i].text, static_cast<std::uint32_t>(i));
    }
    const auto unk = tok.piece_to_id_.find("<unk>");
    tok.unk_id_ = unk != tok.piece_to_id_.end() ? unk->second : 0;
    tok.trie_ = build_trie(pieces);
    tok.pieces_ = std::move(pieces);
    return tok;
}

std::vector<SentencePieceTokenizer::TrieNode> SentencePieceTokenizer::build_trie(
    const std::vector<SentencePiece>& pieces) {
    // Node 0 is the root; each piece is inserted byte by byte, and its final
    // node records (piece_id, log_prob).
    std::vector<TrieNode> trie(1);
    for (std::size_t id = 0; id < pieces.size(); ++id) {
        const SentencePiece& piece = pieces[id];
        std::size_t node_idx = 0;
        for (char ch : piece.text) {
            const auto b = static_cast<std::uint8_t>(ch);
            const auto it = trie[node_idx].children.find(b);
            if (it != trie[node_idx].children.end()) {
                node_idx = it->second;
            } else {
                const std::size_t new_idx = trie.size();
                trie[node_idx].children.emplace(b, new_idx);
                trie.emplace_back();
                node_idx = new_idx;
            }
        }
        // If two pieces share text, keep the higher log-probability.
        auto& terminal = trie[node_idx].terminal;
        if (!terminal || piece.log_prob > terminal->second) {
            terminal = {static_cast<std::uint32_t>(id), piece.log_prob};
        }
    }
    return trie;
}

std::vector<std::tuple<std::size_t, std::uint32_t, float>> SentencePieceTokenizer::trie_matches(
    std::string_view s, std::size_t start) const {
    std::vector<std::tuple<std::size_t, std::uint32_t, float>> results;
    std::size_t node_idx = 0;
    std::size_t pos = start;
    while (pos < s.size()) {
        const auto b = static_cast<std::uint8_t>(s[pos]);
        const auto it = trie_[node_idx].children.find(b);
        if (it == trie_[node_idx].children.end()) {
            break;
        }
        node_idx = it->second;
        ++pos;
        if (const auto& terminal = trie_[node_idx].terminal) {
            results.emplace_back(pos, terminal->first, terminal->second);
        }
    }
    return results;
}

std::vector<std::uint32_t> SentencePieceTokenizer::encode_bytes(std::string_view s) const {
    const std::size_t n = s.size();
    if (n == 0) {
        return {};
    }

    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    // best[i] is the best log-probability for the prefix s[0..i]; back[i] is
    // the (start, piece_id) that achieved it.
    std::vector<float> best(n + 1, NEG_INF);
    std::vector<std::pair<std::size_t, std::uint32_t>> back(n + 1, {0, unk_id_});
    best[0] = 0.0f;

    // Trie-based Viterbi: O(N * max_piece_len) rather than O(N * V).
    for (std::size_t start = 0; start < n; ++start) {
        if (best[start] == NEG_INF) {
            continue;
        }
        for (const auto& [end, id, log_prob] : trie_matches(s, start)) {
            const float score = best[start] + log_prob;
            if (score > best[end]) {
                best[end] = score;
                back[end] = {start, id};
            }
        }
    }

    // Any position still unreachable falls back to a single-byte unknown.
    for (std::size_t i = 1; i <= n; ++i) {
        if (best[i] == NEG_INF) {
            best[i] = best[i - 1] + (-100.0f);
            back[i] = {i - 1, unk_id_};
        }
    }

    std::vector<std::uint32_t> ids;
    std::size_t pos = n;
    while (pos > 0) {
        const auto [start, id] = back[pos];
        ids.push_back(id);
        pos = start;
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

std::vector<std::uint32_t> SentencePieceTokenizer::encode(std::string_view text) const {
    // Mark the start of the text with U+2581, matching the convention used
    // during SentencePiece training, then map every space to U+2581 too.
    std::string s;
    s.reserve(text.size() + 3);
    s.append(kLowerOneEighthBlock);
    s.append(text);
    return encode_bytes(utf8::replace_all(s, " ", kLowerOneEighthBlock));
}

std::string SentencePieceTokenizer::decode(const std::vector<std::uint32_t>& ids) const {
    std::string out;
    for (std::uint32_t id : ids) {
        out.append(id < pieces_.size() ? pieces_[id].text : "?");
    }
    // U+2581 becomes a space; the leading one came from normalization.
    std::string decoded = utf8::replace_all(out, kLowerOneEighthBlock, " ");
    const std::size_t first = decoded.find_first_not_of(' ');
    return first == std::string::npos ? std::string{} : decoded.substr(first);
}

// =============================================================================
// Minimal protobuf parser -- just enough for SentencePiece ModelProto
// =============================================================================
//
// Each field is a varint tag (field_number << 3 | wire_type) followed by the
// value. Wire types: 0 = varint, 1 = 64-bit, 2 = length-delimited, 5 = 32-bit.
//
// ModelProto carries `repeated SentencePieceProto pieces`, each with a piece
// string (field 1) and a float score (field 2).

Result<std::pair<std::uint64_t, std::size_t>> proto_varint(
    const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint64_t result = 0;
    std::uint32_t shift = 0;
    std::size_t i = offset;
    while (true) {
        if (i >= bytes.size()) {
            return err("truncated varint at offset " + std::to_string(i));
        }
        const auto b = static_cast<std::uint64_t>(bytes[i]);
        result |= (b & 0x7f) << shift;
        ++i;
        if ((b & 0x80) == 0) {
            break;
        }
        shift += 7;
        if (shift >= 64) {
            return err("varint too long");
        }
    }
    return std::pair{result, i - offset};
}

namespace {

[[nodiscard]] Result<SentencePiece> proto_read_one_piece(const std::vector<std::uint8_t>& bytes) {
    std::size_t cursor = 0;
    SentencePiece piece;

    while (cursor < bytes.size()) {
        RT_TRY(tag, proto_varint(bytes, cursor));
        cursor += tag.second;
        const std::uint64_t field_number = tag.first >> 3;
        const std::uint64_t wire_type = tag.first & 0x7;

        switch (wire_type) {
            case 0: {
                RT_TRY(skip, proto_varint(bytes, cursor));
                cursor += skip.second;
                break;
            }
            case 1:
                cursor += 8;
                break;
            case 2: {
                RT_TRY(len, proto_varint(bytes, cursor));
                cursor += len.second;
                const std::size_t end = cursor + static_cast<std::size_t>(len.first);
                if (end > bytes.size()) {
                    return err("truncated length-delimited field");
                }
                if (field_number == 1) {
                    piece.text = utf8::from_bytes_lossy(
                        {bytes.begin() + static_cast<std::ptrdiff_t>(cursor),
                         bytes.begin() + static_cast<std::ptrdiff_t>(end)});
                }
                cursor = end;
                break;
            }
            case 5: {
                if (cursor + 4 > bytes.size()) {
                    return err("truncated 32-bit field");
                }
                if (field_number == 2) {
                    std::uint32_t raw = 0;
                    for (int k = 3; k >= 0; --k) {
                        raw = (raw << 8) | bytes[cursor + static_cast<std::size_t>(k)];
                    }
                    piece.log_prob = std::bit_cast<float>(raw);
                }
                cursor += 4;
                break;
            }
            default:
                return err("unknown wire type " + std::to_string(wire_type) +
                           " in SentencePieceProto");
        }
    }

    return piece;
}

}  // namespace

Result<std::vector<SentencePiece>> proto_read_pieces(const std::vector<std::uint8_t>& bytes) {
    std::size_t cursor = 0;
    std::vector<SentencePiece> pieces;

    while (cursor < bytes.size()) {
        RT_TRY(tag, proto_varint(bytes, cursor));
        cursor += tag.second;
        const std::uint64_t field_number = tag.first >> 3;
        const std::uint64_t wire_type = tag.first & 0x7;

        switch (wire_type) {
            case 0: {
                RT_TRY(skip, proto_varint(bytes, cursor));
                cursor += skip.second;
                break;
            }
            case 1:
                cursor += 8;
                break;
            case 2: {
                RT_TRY(len, proto_varint(bytes, cursor));
                cursor += len.second;
                const std::size_t end = cursor + static_cast<std::size_t>(len.first);
                if (end > bytes.size()) {
                    return err("truncated length-delimited field");
                }
                if (field_number == 1) {
                    RT_TRY(piece,
                           proto_read_one_piece({bytes.begin() +
                                                     static_cast<std::ptrdiff_t>(cursor),
                                                 bytes.begin() +
                                                     static_cast<std::ptrdiff_t>(end)}));
                    pieces.push_back(std::move(piece));
                }
                cursor = end;
                break;
            }
            case 5:
                cursor += 4;
                break;
            default:
                return err("unknown wire type " + std::to_string(wire_type) + " at offset " +
                           std::to_string(cursor));
        }
    }
    return pieces;
}

// =============================================================================
// HfBpeTokenizer
// =============================================================================

namespace {

/// Extract the content between a `{` or `[` at `start` and its match, ignoring
/// braces that appear inside strings.
[[nodiscard]] Result<std::string> extract_block(std::string_view s, std::size_t start, char open,
                                                char close) {
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (std::size_t i = start; i < s.size(); ++i) {
        const char b = s[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (b == '\\' && in_string) {
            escape = true;
            continue;
        }
        if (b == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (b == open) {
            ++depth;
        } else if (b == close) {
            --depth;
            if (depth == 0) {
                return std::string(s.substr(start + 1, i - start - 1));
            }
        }
    }
    return err(std::string("unclosed ") + open + " block");
}

/// Parse a JSON string starting at `s[0] == '"'`, returning (value, consumed).
[[nodiscard]] Result<std::pair<std::string, std::size_t>> parse_json_string(std::string_view s) {
    if (s.empty() || s[0] != '"') {
        return err("expected '\"'");
    }
    std::string out;
    std::size_t i = 1;
    while (i < s.size()) {
        if (s[i] == '\\') {
            ++i;
            if (i >= s.size()) {
                break;
            }
            switch (s[i]) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    if (i + 4 < s.size()) {
                        const std::string_view hex = s.substr(i + 1, 4);
                        unsigned cp = 0;
                        if (std::from_chars(hex.data(), hex.data() + hex.size(), cp, 16).ec ==
                            std::errc{}) {
                            utf8::encode_into(out, static_cast<char32_t>(cp));
                        }
                        i += 4;
                    }
                    break;
                }
                default:
                    out.push_back(s[i]);
                    break;
            }
        } else if (s[i] == '"') {
            ++i;
            break;
        } else {
            // Multi-byte UTF-8 passes through untouched.
            const std::size_t ch_len = utf8::seq_len(static_cast<std::uint8_t>(s[i]));
            const std::size_t end = std::min(i + ch_len, s.size());
            out.append(s.substr(i, end - i));
            i += ch_len;
            continue;
        }
        ++i;
    }
    return std::pair{out, i};
}

[[nodiscard]] bool is_json_space(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

/// Parse the `"vocab"` object into a token -> id map.
[[nodiscard]] Result<std::unordered_map<std::string, std::uint32_t>> parse_hf_vocab(
    std::string_view s) {
    const std::size_t vocab_start = s.find("\"vocab\"");
    if (vocab_start == std::string_view::npos) {
        return err("missing vocab key");
    }
    const std::size_t brace = s.find('{', vocab_start);
    if (brace == std::string_view::npos) {
        return err("vocab not an object");
    }
    RT_TRY(content, extract_block(s, brace, '{', '}'));

    std::unordered_map<std::string, std::uint32_t> map;
    std::size_t pos = 0;
    while (pos < content.size()) {
        while (pos < content.size() && (is_json_space(content[pos]) || content[pos] == ',')) {
            ++pos;
        }
        if (pos >= content.size() || content[pos] == '}') {
            break;
        }
        if (content[pos] != '"') {
            ++pos;
            continue;
        }
        RT_TRY(key, parse_json_string(std::string_view(content).substr(pos)));
        pos += key.second;
        while (pos < content.size() && content[pos] != ':') {
            ++pos;
        }
        ++pos;  // skip the colon
        while (pos < content.size() && is_json_space(content[pos])) {
            ++pos;
        }
        const std::size_t num_start = pos;
        while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9') {
            ++pos;
        }
        if (pos > num_start) {
            std::uint32_t id = 0;
            std::from_chars(content.data() + num_start, content.data() + pos, id);
            map.emplace(std::move(key.first), id);
        }
    }
    return map;
}

/// Parse the `"merges"` array. Both dialects are accepted: `["a", "b"]` pairs
/// and single `"a b"` strings.
[[nodiscard]] Result<std::vector<std::pair<std::string, std::string>>> parse_hf_merges(
    std::string_view s) {
    const std::size_t merges_start = s.find("\"merges\"");
    if (merges_start == std::string_view::npos) {
        return err("missing merges key");
    }
    const std::size_t bracket = s.find('[', merges_start);
    if (bracket == std::string_view::npos) {
        return err("merges not an array");
    }
    RT_TRY(content, extract_block(s, bracket, '[', ']'));

    std::vector<std::pair<std::string, std::string>> merges;
    std::size_t pos = 0;
    while (pos < content.size()) {
        while (pos < content.size() && (is_json_space(content[pos]) || content[pos] == ',')) {
            ++pos;
        }
        if (pos >= content.size() || content[pos] == ']') {
            break;
        }
        if (content[pos] == '[') {
            ++pos;
            while (pos < content.size() && content[pos] != '"') {
                ++pos;
            }
            RT_TRY(left, parse_json_string(std::string_view(content).substr(pos)));
            pos += left.second;
            while (pos < content.size() && content[pos] != '"') {
                ++pos;
            }
            RT_TRY(right, parse_json_string(std::string_view(content).substr(pos)));
            pos += right.second;
            while (pos < content.size() && content[pos] != ']') {
                ++pos;
            }
            ++pos;
            merges.emplace_back(std::move(left.first), std::move(right.first));
        } else if (content[pos] == '"') {
            RT_TRY(pair_str, parse_json_string(std::string_view(content).substr(pos)));
            pos += pair_str.second;
            const std::size_t sp = pair_str.first.find(' ');
            if (sp != std::string::npos) {
                merges.emplace_back(pair_str.first.substr(0, sp), pair_str.first.substr(sp + 1));
            }
        } else {
            ++pos;
        }
    }
    return merges;
}

}  // namespace

Result<HfBpeTokenizer> HfBpeTokenizer::from_json_bytes(const std::vector<std::uint8_t>& bytes) {
    return from_json_str(std::string_view(reinterpret_cast<const char*>(bytes.data()),
                                          bytes.size()));
}

Result<HfBpeTokenizer> HfBpeTokenizer::from_json_file(const std::string& path) {
    RT_TRY(content, read_file(path));
    return from_json_str(content);
}

Result<HfBpeTokenizer> HfBpeTokenizer::from_json_str(std::string_view s) {
    if (s.find("\"model\"") == std::string_view::npos) {
        return err("missing \"model\" key");
    }
    RT_TRY(vocab_map, parse_hf_vocab(s));
    RT_TRY(merges, parse_hf_merges(s));

    HfBpeTokenizer tok;
    tok.id_to_token_.assign(vocab_map.size(), std::string{});
    for (const auto& [token, id] : vocab_map) {
        if (id < tok.id_to_token_.size()) {
            tok.id_to_token_[id] = token;
        }
    }

    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        tok.merge_rank_.emplace(merges[rank], rank);
    }

    const auto unk = vocab_map.find("<unk>");
    tok.unk_id_ = unk != vocab_map.end() ? unk->second : 0;
    // A ByteLevel pre-tokenizer in the file selects the GPT-2 byte encoding.
    tok.byte_level_ = s.find("\"ByteLevel\"") != std::string_view::npos;

    // `end_of_word_suffix` is the discriminator that matters: it is what makes
    // the merge sequence a different algorithm rather than a different regex,
    // and CLIP is the model in this family that declares one.
    if (const std::size_t at = s.find("\"end_of_word_suffix\""); at != std::string_view::npos) {
        const std::size_t open = s.find('"', s.find(':', at) + 1);
        const std::size_t close = open != std::string_view::npos ? s.find('"', open + 1)
                                                                 : std::string_view::npos;
        if (open != std::string_view::npos && close != std::string_view::npos) {
            tok.end_of_word_suffix_ = std::string(s.substr(open + 1, close - open - 1));
        }
    }
    if (!tok.end_of_word_suffix_.empty()) {
        tok.pre_ = PreTokenizer::Clip;
    }
    tok.token_to_id_ = std::move(vocab_map);

    return tok;
}

std::string_view HfBpeTokenizer::token_text(std::uint32_t id) const {
    return id < id_to_token_.size() ? std::string_view(id_to_token_[id]) : std::string_view{};
}

std::optional<std::uint32_t> HfBpeTokenizer::token_id(const std::string& token) const {
    const auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? std::optional{it->second} : std::nullopt;
}

std::vector<std::string> HfBpeTokenizer::gpt2_pretokenize(std::string_view text) {
    return pretokenize(text, PreTokenizer::Gpt2);
}

std::string HfBpeTokenizer::normalize(std::string_view text, PreTokenizer kind) {
    if (kind != PreTokenizer::Clip) {
        return std::string(text);
    }
    // Collapse every whitespace run to one space, then lowercase. NFC
    // composition is part of CLIP's normalizer too and is not done here: it
    // matters only for text that carries combining marks, and getting it wrong
    // costs a token boundary rather than a wrong word.
    std::string out;
    out.reserve(text.size());
    bool in_space = false;
    for (char32_t cp : utf8::decode(text)) {
        if (utf8::is_whitespace(cp)) {
            in_space = true;
            continue;
        }
        if (in_space && !out.empty()) {
            out.push_back(' ');
        }
        in_space = false;
        utf8::encode_into(out, utf8::to_ascii_lowercase(cp));
    }
    return out;
}

std::vector<std::string> HfBpeTokenizer::clip_pretokenize(std::string_view text) {
    // CLIP's regex, directly:
    //   's|'t|'re|'ve|'m|'ll|'d | [\p{L}]+ | [\p{N}] | [^\s\p{L}\p{N}]+
    // with whitespace matching nothing and therefore dropped.
    std::vector<std::string> words;
    const auto chars = utf8::decode_indices(text);
    std::size_t i = 0;

    const auto byte_end_at = [&](std::size_t idx) {
        return idx < chars.size() ? chars[idx].first : text.size();
    };
    const auto push = [&](std::size_t start_byte, std::size_t end_byte) {
        words.emplace_back(text.substr(start_byte, end_byte - start_byte));
    };

    while (i < chars.size()) {
        const auto [byte_start, ch] = chars[i];

        if (utf8::is_whitespace(ch)) {
            ++i;
            continue;
        }

        if (ch == U'\'' && i + 1 < chars.size()) {
            const char32_t n1 = utf8::to_ascii_lowercase(chars[i + 1].second);
            if (n1 == U's' || n1 == U't' || n1 == U'm' || n1 == U'd') {
                push(byte_start, byte_end_at(i + 2));
                i += 2;
                continue;
            }
            if (i + 2 < chars.size()) {
                const char32_t n2 = utf8::to_ascii_lowercase(chars[i + 2].second);
                if ((n1 == U'r' && n2 == U'e') || (n1 == U'v' && n2 == U'e') ||
                    (n1 == U'l' && n2 == U'l')) {
                    push(byte_start, byte_end_at(i + 3));
                    i += 3;
                    continue;
                }
            }
        }

        if (utf8::is_alphabetic(ch)) {
            std::size_t j = i + 1;
            while (j < chars.size() && utf8::is_alphabetic(chars[j].second)) {
                ++j;
            }
            push(byte_start, byte_end_at(j));
            i = j;
            continue;
        }

        // `[\p{N}]` with no repetition: one digit per token, always.
        if (utf8::is_ascii_digit(ch)) {
            push(byte_start, byte_end_at(i + 1));
            ++i;
            continue;
        }

        std::size_t j = i + 1;
        while (j < chars.size() && !utf8::is_whitespace(chars[j].second) &&
               !utf8::is_alphabetic(chars[j].second) &&
               !utf8::is_ascii_digit(chars[j].second)) {
            ++j;
        }
        push(byte_start, byte_end_at(j));
        i = j;
    }

    return words;
}

std::vector<std::string> HfBpeTokenizer::pretokenize(std::string_view text, PreTokenizer kind) {
    if (kind == PreTokenizer::Clip) {
        return clip_pretokenize(text);
    }
    std::vector<std::string> words;
    const auto chars = utf8::decode_indices(text);
    std::size_t i = 0;

    const auto byte_end_at = [&](std::size_t idx) {
        return idx < chars.size() ? chars[idx].first : text.size();
    };
    const auto push = [&](std::size_t start_byte, std::size_t end_byte) {
        words.emplace_back(text.substr(start_byte, end_byte - start_byte));
    };

    while (i < chars.size()) {
        const auto [byte_start, ch] = chars[i];

        // Case 1: contractions ('s, 't, 'd, 'm, 're, 've, 'll).
        if (ch == U'\'' && i + 1 < chars.size()) {
            const char32_t next_lower = utf8::to_ascii_lowercase(chars[i + 1].second);
            if (next_lower == U's' || next_lower == U't' || next_lower == U'd' ||
                next_lower == U'm') {
                push(byte_start, byte_end_at(i + 2));
                i += 2;
                continue;
            }
            if (i + 2 < chars.size()) {
                const char32_t c2 = utf8::to_ascii_lowercase(chars[i + 2].second);
                const bool two_char = (next_lower == U'r' && c2 == U'e') ||
                                      (next_lower == U'v' && c2 == U'e') ||
                                      (next_lower == U'l' && c2 == U'l');
                if (two_char) {
                    push(byte_start, byte_end_at(i + 3));
                    i += 3;
                    continue;
                }
            }
        }

        // Case 2: a letter run, optionally preceded by one non-alphanumeric
        // character (so " is" stays one word).
        if (utf8::is_alphabetic(ch) || ch == U'_') {
            std::size_t j = i + 1;
            while (j < chars.size() &&
                   (utf8::is_alphabetic(chars[j].second) || chars[j].second == U'_')) {
                ++j;
            }
            push(byte_start, byte_end_at(j));
            i = j;
            continue;
        }
        if (!utf8::is_alphanumeric(ch) && ch != U'\r' && ch != U'\n' && i + 1 < chars.size() &&
            utf8::is_alphabetic(chars[i + 1].second)) {
            std::size_t j = i + 1;
            while (j < chars.size() &&
                   (utf8::is_alphabetic(chars[j].second) || chars[j].second == U'_')) {
                ++j;
            }
            push(byte_start, byte_end_at(j));
            i = j;
            continue;
        }

        // Case 3: digits. GPT-2 emits one at a time; Llama 3's regex is
        // `\p{N}{1,3}`, so it takes runs of up to three and "2024" becomes
        // "202" + "4" rather than four separate digits.
        if (utf8::is_ascii_digit(ch)) {
            const std::size_t max_run = kind == PreTokenizer::Llama3 ? 3 : 1;
            std::size_t j = i + 1;
            while (j < chars.size() && j - i < max_run &&
                   utf8::is_ascii_digit(chars[j].second)) {
                ++j;
            }
            push(byte_start, byte_end_at(j));
            i = j;
            continue;
        }

        // Case 4: a run of newlines.
        if (ch == U'\r' || ch == U'\n') {
            std::size_t j = i + 1;
            while (j < chars.size() && (chars[j].second == U'\r' || chars[j].second == U'\n')) {
                ++j;
            }
            push(byte_start, byte_end_at(j));
            i = j;
            continue;
        }

        // Case 5: a run of other whitespace.
        if (utf8::is_whitespace(ch)) {
            std::size_t j = i + 1;
            while (j < chars.size() && utf8::is_whitespace(chars[j].second) &&
                   chars[j].second != U'\r' && chars[j].second != U'\n') {
                ++j;
            }
            push(byte_start, byte_end_at(j));
            i = j;
            continue;
        }

        // Case 6: a space followed by a run of punctuation or symbols.
        if (ch == U' ' && i + 1 < chars.size() && !utf8::is_alphanumeric(chars[i + 1].second) &&
            !utf8::is_whitespace(chars[i + 1].second)) {
            std::size_t j = i + 1;
            while (j < chars.size() && !utf8::is_alphanumeric(chars[j].second) &&
                   !utf8::is_whitespace(chars[j].second)) {
                ++j;
            }
            push(byte_start, byte_end_at(j));
            i = j;
            continue;
        }

        // Case 7: any single remaining character.
        push(byte_start, byte_end_at(i + 1));
        ++i;
    }

    return words;
}

std::vector<std::uint32_t> HfBpeTokenizer::bpe_encode_word(std::string_view word,
                                                           std::string_view end_suffix) const {
    if (word.empty()) {
        return {};
    }

    // Start with one symbol per code point; merged-away slots become empty.
    std::vector<std::string> symbols;
    for (char32_t cp : utf8::decode(word)) {
        symbols.push_back(utf8::encode(cp));
    }
    // The end-of-word marker rides on the *last* symbol rather than becoming a
    // symbol of its own. That is what makes `a</w>` reachable as a single
    // vocabulary entry while `a` stays available mid-word.
    if (!end_suffix.empty() && !symbols.empty()) {
        symbols.back() += std::string(end_suffix);
    }
    std::vector<bool> live(symbols.size(), true);

    while (true) {
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        std::size_t best_i = 0, best_j = 0;

        // Walk the surviving symbols pairwise.
        std::size_t prev = symbols.size();
        for (std::size_t idx = 0; idx < symbols.size(); ++idx) {
            if (!live[idx]) {
                continue;
            }
            if (prev != symbols.size()) {
                const auto it = merge_rank_.find({symbols[prev], symbols[idx]});
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = prev;
                    best_j = idx;
                }
            }
            prev = idx;
        }

        if (best_rank == std::numeric_limits<std::size_t>::max()) {
            break;  // nothing left to merge
        }

        symbols[best_i] += symbols[best_j];
        live[best_j] = false;
    }

    std::vector<std::uint32_t> ids;
    for (std::size_t idx = 0; idx < symbols.size(); ++idx) {
        if (!live[idx]) {
            continue;
        }
        const auto it = token_to_id_.find(symbols[idx]);
        ids.push_back(it != token_to_id_.end() ? it->second : unk_id_);
    }
    return ids;
}

std::vector<std::uint32_t> HfBpeTokenizer::encode_byte_level(std::string_view text) const {
    static const std::array<char32_t, 256> b2u = gpt2_bytes_to_unicode();
    const std::string normalized = normalize(text, pre_);
    std::vector<std::uint32_t> ids;
    for (const std::string& word : pretokenize(normalized, pre_)) {
        // Map each raw byte through the GPT-2 unicode table before merging.
        std::string unicode_word;
        for (char c : word) {
            utf8::encode_into(unicode_word, b2u[static_cast<std::uint8_t>(c)]);
        }
        const std::vector<std::uint32_t> word_ids =
            bpe_encode_word(unicode_word, end_of_word_suffix_);
        ids.insert(ids.end(), word_ids.begin(), word_ids.end());
    }
    return ids;
}

std::string HfBpeTokenizer::decode_byte_level(const std::vector<std::uint32_t>& ids) const {
    static const std::array<char32_t, 256> b2u = gpt2_bytes_to_unicode();
    static const std::map<char32_t, std::uint8_t> u2b = [] {
        std::map<char32_t, std::uint8_t> m;
        for (std::size_t b = 0; b < 256; ++b) {
            m.emplace(b2u[b], static_cast<std::uint8_t>(b));
        }
        return m;
    }();

    std::vector<std::uint8_t> bytes;
    for (std::uint32_t id : ids) {
        if (id >= id_to_token_.size()) {
            continue;
        }
        for (char32_t c : utf8::decode(id_to_token_[id])) {
            const auto it = u2b.find(c);
            if (it != u2b.end()) {
                bytes.push_back(it->second);
            }
            // Special tokens such as <|im_start|> contain characters outside
            // the table; they contribute no bytes.
        }
    }
    return utf8::from_bytes_lossy(bytes);
}

std::vector<std::uint32_t> HfBpeTokenizer::encode(std::string_view text) const {
    if (text.empty()) {
        return {};
    }
    if (byte_level_) {
        return encode_byte_level(text);
    }

    // SentencePiece style: spaces become U+2581, and each word carries it.
    const std::string normalized = utf8::replace_all(text, " ", kLowerOneEighthBlock);

    std::vector<std::string> words;
    std::string current;
    for (char32_t ch : utf8::decode(normalized)) {
        if (ch == 0x2581 && !current.empty()) {
            words.push_back(current);
            current = kLowerOneEighthBlock;
        } else {
            utf8::encode_into(current, ch);
        }
    }
    if (!current.empty()) {
        words.push_back(std::move(current));
    }

    std::vector<std::uint32_t> ids;
    for (const std::string& word : words) {
        const std::vector<std::uint32_t> word_ids = bpe_encode_word(word, end_of_word_suffix_);
        ids.insert(ids.end(), word_ids.begin(), word_ids.end());
    }
    return ids;
}

std::string HfBpeTokenizer::decode(const std::vector<std::uint32_t>& ids) const {
    if (byte_level_) {
        return decode_byte_level(ids);
    }
    std::string out;
    for (std::uint32_t id : ids) {
        if (id < id_to_token_.size()) {
            out.append(id_to_token_[id]);
        }
    }
    return utf8::replace_all(out, kLowerOneEighthBlock, " ");
}


Result<HfBpeTokenizer> HfBpeTokenizer::from_vocab_and_merges(
    std::vector<std::string> tokens, const std::vector<std::string>& merges, bool byte_level,
    PreTokenizer pre) {
    if (tokens.empty()) {
        return err("hf tokenizer: vocabulary is empty");
    }

    HfBpeTokenizer tok;
    tok.byte_level_ = byte_level;
    tok.pre_ = pre;
    tok.id_to_token_ = std::move(tokens);

    tok.token_to_id_.reserve(tok.id_to_token_.size());
    for (std::size_t id = 0; id < tok.id_to_token_.size(); ++id) {
        // First id wins, matching how a JSON vocabulary object would load.
        tok.token_to_id_.emplace(tok.id_to_token_[id], static_cast<std::uint32_t>(id));
    }

    tok.merge_rank_.reserve(merges.size());
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& entry = merges[rank];
        const std::size_t space = entry.find(' ');
        if (space == std::string::npos || space == 0 || space + 1 >= entry.size()) {
            return err("hf tokenizer: merge " + std::to_string(rank) + " ('" + entry +
                       "') is not two space-separated pieces");
        }
        tok.merge_rank_.emplace(
            std::pair<std::string, std::string>{entry.substr(0, space), entry.substr(space + 1)},
            rank);
    }

    // GGUF vocabularies carry no explicit unknown token. Llama 3 never needs
    // one -- byte-level encoding can spell any input -- so point it at id 0 and
    // let a genuinely missing symbol be visible rather than silently dropped.
    tok.unk_id_ = 0;
    const auto unk = tok.token_to_id_.find("<unk>");
    if (unk != tok.token_to_id_.end()) {
        tok.unk_id_ = unk->second;
    }

    return tok;
}

}  // namespace rt

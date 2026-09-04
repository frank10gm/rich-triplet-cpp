#pragma once

// =============================================================================
// Tokenizers -- turning text into numbers (and back)
// =============================================================================
//
// A language model never sees text, only a sequence of integers called tokens.
// The tokenizer defines the mapping:
//
//   "ciao" -> [99, 12, 4, 77]    (encode)
//   [99, 12, 4, 77] -> "ciao"    (decode)
//
// Character codes would work, but the vocabulary would be tiny (~100 entries)
// and the model would have to learn to spell every word from scratch, over very
// long sequences.
//
// Real LLMs use Byte-Pair Encoding: start from individual bytes, then
// repeatedly merge the most frequent adjacent pair into a new token. After
// enough merges "the" is one token and "antidisestablishmentarianism" is five.
// GPT-2 has ~50,000 tokens, GPT-4 ~100,000. A larger vocabulary means shorter
// sequences but a bigger embedding table.
//
// This header provides four implementations behind one interface:
//   CharTokenizer            -- one id per character, for teaching and tests
//   BpeTokenizer             -- byte-level BPE, trainable or loaded from disk
//   SentencePieceTokenizer   -- unigram model with Viterbi decoding
//   HfBpeTokenizer           -- HuggingFace tokenizer.json (Gemma 3, Qwen 3.5)

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rt/result.hpp"

namespace rt {

// =============================================================================
// Shared interface
// =============================================================================

class Tokenizer {
   public:
    virtual ~Tokenizer() = default;

    /// Convert text into a sequence of token ids.
    [[nodiscard]] virtual std::vector<std::uint32_t> encode(std::string_view text) const = 0;

    /// Convert token ids back into text.
    [[nodiscard]] virtual std::string decode(const std::vector<std::uint32_t>& ids) const = 0;

    /// Total number of distinct tokens in the vocabulary.
    [[nodiscard]] virtual std::size_t vocab_size() const = 0;

    /// The id representing "unknown" / out-of-vocabulary input.
    [[nodiscard]] virtual std::uint32_t unk_id() const = 0;
};

// =============================================================================
// 1. Character-level tokenizer
// =============================================================================
//
// Collect every unique character in the training text, sort for determinism,
// and assign ids from 0.
//
//   "ciao mondo" -> [' ', 'a', 'c', 'd', 'i', 'm', 'n', 'o']
//   encode("ciao") -> [2, 4, 1, 7]

class CharTokenizer final : public Tokenizer {
   public:
    /// Build the vocabulary by scanning the provided training text.
    [[nodiscard]] static CharTokenizer from_text(std::string_view text);

    /// Print the full vocabulary (useful for debugging small datasets).
    void print_vocab() const;

    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text) const override;
    [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& ids) const override;
    [[nodiscard]] std::size_t vocab_size() const override { return id_to_char_.size(); }
    /// The last id doubles as the unknown token.
    [[nodiscard]] std::uint32_t unk_id() const override;

   private:
    std::map<char32_t, std::uint32_t> char_to_id_;
    std::vector<char32_t> id_to_char_;
};

// =============================================================================
// 2. Byte-Pair Encoding
// =============================================================================
//
// Training (offline, once per corpus):
//   vocabulary starts as all 256 byte values; then, N times, count every
//   adjacent pair in the corpus, add the most frequent one as a new token, and
//   replace its occurrences. GPT-2 used ~49,744 merges.
//
// Encoding (online): split into bytes, then apply the learned merges in the
// order they were learned, earlier merges first, until none applies.

class BpeTokenizer final : public Tokenizer {
   public:
    /// Learn BPE merges from a corpus. Final vocabulary = 256 + `num_merges`.
    [[nodiscard]] static BpeTokenizer train(std::string_view text, std::size_t num_merges);

    /// Load from a `vocab.json` + `merges.txt` pair -- the standard
    /// HuggingFace format used by GPT-2, GPT-OSS, LLaMA and most open models.
    [[nodiscard]] static Result<BpeTokenizer> from_files(const std::string& vocab_path,
                                                         const std::string& merges_path);

    /// Load from in-memory strings, for tests and embedded vocabularies.
    ///
    /// Tokens in vocab.json use the GPT-2 byte encoding: ' ' is written as
    /// U+0120, '\n' as U+010A, and bytes above 127 land in U+0100..U+016F.
    [[nodiscard]] static Result<BpeTokenizer> from_vocab_and_merges_str(
        std::string_view vocab_json, std::string_view merges_txt);

    /// Load from a tiktoken file (GPT-4 / GPT-OSS format).
    ///
    /// Each line is `<base64_token_bytes> <rank>`, where the rank is both the
    /// token id and the merge priority; there is no separate merges file, so
    /// the merges are recovered by splitting each token into the two
    /// lowest-ranked pieces that compose it.
    [[nodiscard]] static Result<BpeTokenizer> from_tiktoken_file(const std::string& path);
    [[nodiscard]] static Result<BpeTokenizer> from_tiktoken_str(std::string_view content);

    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text) const override;
    [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& ids) const override;
    [[nodiscard]] std::size_t vocab_size() const override { return vocab_.size(); }
    /// Byte-level BPE has no true unknown: every byte sequence is encodable.
    [[nodiscard]] std::uint32_t unk_id() const override { return 0; }

    [[nodiscard]] const std::vector<std::string>& vocab() const { return vocab_; }
    [[nodiscard]] const std::vector<std::pair<std::string, std::string>>& merges() const {
        return merges_;
    }

   private:
    struct PairHash {
        std::size_t operator()(const std::pair<std::uint32_t, std::uint32_t>& p) const {
            return (static_cast<std::size_t>(p.first) << 32) ^ p.second;
        }
    };

    void rebuild_merge_map();

    /// id -> the string representation of that token
    std::vector<std::string> vocab_;
    /// token string -> id
    std::unordered_map<std::string, std::uint32_t> token_to_id_;
    /// Merge rules in the order they were learned.
    std::vector<std::pair<std::string, std::string>> merges_;
    /// (left_id, right_id) -> merged_id, built from `merges_`.
    std::unordered_map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t, PairHash>
        merge_map_;
};

/// The GPT-2 byte->unicode table: byte `b` maps to code point `result[b]`.
///
/// GPT-2 uses a bijection from byte values to printable Unicode so every token
/// string is displayable. Printable ASCII (33-126) and Latin-1 (161-255) map to
/// themselves; the remaining 67 bytes map to U+0100..U+0142. In particular
/// space (32) becomes U+0120.
[[nodiscard]] std::array<char32_t, 256> gpt2_bytes_to_unicode();

/// Decode a GPT-2 encoded token string back to raw bytes.
/// "Ġhello" -> " hello" (the leading U+0120 is a space).
[[nodiscard]] std::vector<std::uint8_t> gpt2_decode_token_to_bytes(std::string_view s);

// =============================================================================
// 3. SentencePiece (unigram)
// =============================================================================

struct SentencePiece {
    /// The piece text; may contain U+2581 marking a word-initial position.
    std::string text;
    /// Log-probability from the unigram language model.
    float log_prob = 0.0f;
};

/// A SentencePiece unigram tokenizer.
///
/// Encoding is a Viterbi forward pass over the piece lattice. A naive scan
/// would test all V pieces at every position -- 130M comparisons for LLaMA-3's
/// 128k vocabulary over a 1024-token context -- so pieces are held in a byte
/// trie, reducing the work to O(N * max_piece_len).
class SentencePieceTokenizer final : public Tokenizer {
   public:
    /// Load from a plain-text vocab: one `<piece>\t<score>` per line, matching
    /// `spm_export_vocab --output_format=tsv`.
    [[nodiscard]] static Result<SentencePieceTokenizer> from_vocab_text(std::string_view text);

    /// Load from the bytes of a SentencePiece `.model` protobuf.
    ///
    /// Only the fields that matter are parsed: `pieces[]`, each with a piece
    /// string and a float score. No protobuf library is needed.
    [[nodiscard]] static Result<SentencePieceTokenizer> from_model_bytes(
        const std::vector<std::uint8_t>& bytes);

    /// Load from a `.model` file on disk.
    [[nodiscard]] static Result<SentencePieceTokenizer> from_model_file(const std::string& path);

    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text) const override;
    [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& ids) const override;
    [[nodiscard]] std::size_t vocab_size() const override { return pieces_.size(); }
    [[nodiscard]] std::uint32_t unk_id() const override { return unk_id_; }

    [[nodiscard]] const std::vector<SentencePiece>& pieces() const { return pieces_; }

    /// Look up a piece id by exact piece text.
    [[nodiscard]] std::optional<std::uint32_t> piece_id(const std::string& text) const {
        const auto it = piece_to_id_.find(text);
        return it != piece_to_id_.end() ? std::optional{it->second} : std::nullopt;
    }

   private:
    /// A single node in the piece trie.
    struct TrieNode {
        /// child byte -> index of the child node
        std::map<std::uint8_t, std::size_t> children;
        /// Set when a piece ends here: (piece_id, log_prob).
        std::optional<std::pair<std::uint32_t, float>> terminal;
    };

    [[nodiscard]] static SentencePieceTokenizer from_pieces(std::vector<SentencePiece> pieces);
    static std::vector<TrieNode> build_trie(const std::vector<SentencePiece>& pieces);

    /// All pieces starting at byte `start`, as (end_pos, piece_id, log_prob).
    [[nodiscard]] std::vector<std::tuple<std::size_t, std::uint32_t, float>> trie_matches(
        std::string_view s, std::size_t start) const;

    [[nodiscard]] std::vector<std::uint32_t> encode_bytes(std::string_view s) const;

    std::vector<SentencePiece> pieces_;
    std::unordered_map<std::string, std::uint32_t> piece_to_id_;
    std::vector<TrieNode> trie_;
    std::uint32_t unk_id_ = 0;
};

// =============================================================================
// 4. HuggingFace tokenizer.json BPE
// =============================================================================
//
// Handles both dialects in use here:
//   - SentencePiece style (Gemma 3): spaces normalize to U+2581, and each word
//     carries that prefix.
//   - GPT-2 byte-level (Qwen 3.5): bytes map through the GPT-2 unicode table
//     and a regex-free pre-tokenizer splits words.
//
// tokenizer.json is the canonical source; tokenizer.model is legacy.

class HfBpeTokenizer final : public Tokenizer {
   public:
    [[nodiscard]] static Result<HfBpeTokenizer> from_json_bytes(
        const std::vector<std::uint8_t>& bytes);
    [[nodiscard]] static Result<HfBpeTokenizer> from_json_file(const std::string& path);
    [[nodiscard]] static Result<HfBpeTokenizer> from_json_str(std::string_view s);

    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text) const override;
    [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& ids) const override;
    [[nodiscard]] std::size_t vocab_size() const override { return id_to_token_.size(); }
    [[nodiscard]] std::uint32_t unk_id() const override { return unk_id_; }

    /// True when the file declared a ByteLevel pre-tokenizer.
    [[nodiscard]] bool byte_level() const { return byte_level_; }

    /// Look up a token string by id; empty when out of range.
    [[nodiscard]] std::string_view token_text(std::uint32_t id) const;

    /// Look up an id by exact token string.
    [[nodiscard]] std::optional<std::uint32_t> token_id(const std::string& token) const;

    /// Split text the way the GPT-2 pre-tokenizer does: contraction suffixes,
    /// letter runs (with an optional leading non-alphanumeric), single digits,
    /// newline runs, other whitespace runs, and punctuation runs.
    [[nodiscard]] static std::vector<std::string> gpt2_pretokenize(std::string_view text);

   private:
    struct StrPairHash {
        std::size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>{}(p.first) * 31 + std::hash<std::string>{}(p.second);
        }
    };

    [[nodiscard]] std::vector<std::uint32_t> encode_byte_level(std::string_view text) const;
    [[nodiscard]] std::string decode_byte_level(const std::vector<std::uint32_t>& ids) const;
    /// BPE-merge one pre-tokenized word into token ids.
    [[nodiscard]] std::vector<std::uint32_t> bpe_encode_word(std::string_view word) const;

    /// id -> token string
    std::vector<std::string> id_to_token_;
    /// token string -> id
    std::unordered_map<std::string, std::uint32_t> token_to_id_;
    /// (left, right) -> merge priority, lower applied first.
    std::unordered_map<std::pair<std::string, std::string>, std::size_t, StrPairHash> merge_rank_;
    std::uint32_t unk_id_ = 0;
    /// GPT-2 byte-level encoding (Qwen, GPT-2) when true; SentencePiece U+2581
    /// encoding (Gemma) when false.
    bool byte_level_ = false;
};

// =============================================================================
// Shared helpers (exposed for testing)
// =============================================================================

/// Decode standard base64. Padding is tolerated but not required.
[[nodiscard]] Result<std::vector<std::uint8_t>> base64_decode(std::string_view s);

/// Encode standard base64 with padding.
[[nodiscard]] std::string base64_encode(const std::vector<std::uint8_t>& data);

/// Read a protobuf varint at `offset`, returning (value, bytes_consumed).
[[nodiscard]] Result<std::pair<std::uint64_t, std::size_t>> proto_varint(
    const std::vector<std::uint8_t>& bytes, std::size_t offset);

}  // namespace rt

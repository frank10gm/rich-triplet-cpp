#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "rt/tokenizer.hpp"
#include "rt/utf8.hpp"

using namespace rt;

namespace {

/// U+2581, the SentencePiece word-boundary marker.
const std::string SP = "\xE2\x96\x81";

/// A tiny SentencePiece vocabulary for unit tests: ASCII pieces, uniform-ish
/// log-probs, and the pieces needed to spell "hello world" and "hi".
SentencePieceTokenizer tiny_sp_vocab() {
    const std::string text = "<unk>\t0\n"
                             "<s>\t0\n"
                             "</s>\t0\n" +
                             SP + "hello\t-1.0\n" + SP + "world\t-2.0\n" + SP + "hi\t-3.0\n" +
                             SP + "h\t-4.0\n"
                                  "e\t-4.0\n"
                                  "l\t-4.0\n"
                                  "o\t-4.0\n"
                                  "w\t-4.0\n"
                                  "r\t-4.0\n"
                                  "d\t-4.0\n"
                                  "i\t-4.0\n";
    auto tok = SentencePieceTokenizer::from_vocab_text(text);
    REQUIRE(tok.has_value());
    return std::move(*tok);
}

/// A minimal vocab.json + merges.txt for ASCII text, in the GPT-2 encoding
/// (space becomes U+0120). Only bytes that need no JSON escaping are included.
std::pair<std::string, std::string> make_tiny_vocab_merges() {
    const auto map = gpt2_bytes_to_unicode();

    std::vector<std::uint8_t> safe_bytes;
    for (int b = 'a'; b <= 'z'; ++b) safe_bytes.push_back(static_cast<std::uint8_t>(b));
    for (int b = 'A'; b <= 'Z'; ++b) safe_bytes.push_back(static_cast<std::uint8_t>(b));
    for (int b = '0'; b <= '9'; ++b) safe_bytes.push_back(static_cast<std::uint8_t>(b));
    safe_bytes.push_back(' ');
    safe_bytes.push_back('\n');

    std::vector<std::string> entries;
    for (std::uint8_t b : safe_bytes) {
        entries.push_back("\"" + utf8::encode(map[b]) + "\":" + std::to_string(b));
    }

    // Two merged tokens: "ab" (id 256) and "ab " (id 257).
    const std::string a_char = utf8::encode(map[static_cast<std::uint8_t>('a')]);
    const std::string b_char = utf8::encode(map[static_cast<std::uint8_t>('b')]);
    const std::string sp_char = utf8::encode(map[static_cast<std::uint8_t>(' ')]);
    entries.push_back("\"" + a_char + b_char + "\":256");
    entries.push_back("\"" + a_char + b_char + sp_char + "\":257");

    std::string vocab_json = "{";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (i) vocab_json += ",";
        vocab_json += entries[i];
    }
    vocab_json += "}";

    const std::string merges_txt = "#version: 0.2\n" + a_char + " " + b_char + "\n" + a_char +
                                   b_char + " " + sp_char + "\n";
    return {vocab_json, merges_txt};
}

std::vector<std::uint8_t> to_bytes(std::string_view s) {
    return {s.begin(), s.end()};
}

// --- protobuf encoding helpers, for the SentencePiece .model tests ---

std::vector<std::uint8_t> encode_varint(std::uint64_t v) {
    std::vector<std::uint8_t> out;
    while (true) {
        auto b = static_cast<std::uint8_t>(v & 0x7f);
        v >>= 7;
        if (v == 0) {
            out.push_back(b);
            break;
        }
        out.push_back(b | 0x80);
    }
    return out;
}

void append(std::vector<std::uint8_t>& dst, const std::vector<std::uint8_t>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

std::vector<std::uint8_t> encode_len_delimited(std::uint64_t field,
                                               const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out = encode_varint((field << 3) | 2);  // wire type 2
    append(out, encode_varint(data.size()));
    append(out, data);
    return out;
}

std::vector<std::uint8_t> encode_f32_field(std::uint64_t field, float v) {
    std::vector<std::uint8_t> out = encode_varint((field << 3) | 5);  // wire type 5
    const auto bits = std::bit_cast<std::uint32_t>(v);
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::uint8_t>(bits >> (8 * i)));
    }
    return out;
}

std::vector<std::uint8_t> make_model_proto(
    const std::vector<std::pair<std::string, float>>& vocab) {
    std::vector<std::uint8_t> out;
    for (const auto& [piece, score] : vocab) {
        // SentencePieceProto { field 1: piece, field 2: score }
        std::vector<std::uint8_t> sp_proto = encode_len_delimited(1, to_bytes(piece));
        append(sp_proto, encode_f32_field(2, score));
        // Wrapped as field 1 (pieces) of ModelProto.
        append(out, encode_len_delimited(1, sp_proto));
    }
    return out;
}

}  // namespace

// -----------------------------------------------------------------------------
// CharTokenizer
// -----------------------------------------------------------------------------

TEST_CASE("char tokenizer round-trips", "[tokenizer]") {
    const std::string text = "ciao mondo, hello world!";
    const CharTokenizer tok = CharTokenizer::from_text(text);
    REQUIRE(tok.decode(tok.encode(text)) == text);
}

TEST_CASE("char tokenizer vocab size counts unique characters", "[tokenizer]") {
    REQUIRE(CharTokenizer::from_text("abc").vocab_size() == 3);
    REQUIRE(CharTokenizer::from_text("aabbcc").vocab_size() == 3);
}

TEST_CASE("char tokenizer handles a bilingual corpus", "[tokenizer]") {
    const CharTokenizer tok = CharTokenizer::from_text("buongiorno, good morning!");
    REQUIRE(tok.decode(tok.encode("buon")) == "buon");
}

TEST_CASE("char tokenizer is deterministic", "[tokenizer]") {
    const std::string text = "un piccolo modello linguistico";
    REQUIRE(CharTokenizer::from_text(text).encode("piccolo") ==
            CharTokenizer::from_text(text).encode("piccolo"));
}

TEST_CASE("char tokenizer handles multi-byte characters", "[tokenizer]") {
    // Each code point is one token, so accents and CJK survive a round-trip.
    const std::string text = "città 東京";
    const CharTokenizer tok = CharTokenizer::from_text(text);
    REQUIRE(tok.decode(tok.encode(text)) == text);
    REQUIRE(tok.encode("東").size() == 1);
}

// -----------------------------------------------------------------------------
// BpeTokenizer -- training
// -----------------------------------------------------------------------------

TEST_CASE("bpe round-trips after training", "[tokenizer]") {
    const BpeTokenizer tok = BpeTokenizer::train(
        "buongiorno buongiorno buongiorno hello hello world world world", 20);
    REQUIRE(tok.decode(tok.encode("buongiorno")) == "buongiorno");
}

TEST_CASE("bpe compresses a repeated word", "[tokenizer]") {
    const std::string word = "buongiorno ";
    std::string corpus;
    for (int i = 0; i < 50; ++i) {
        corpus += word;
    }
    const BpeTokenizer tok = BpeTokenizer::train(corpus, 30);

    const std::size_t token_count = tok.encode(word).size();
    INFO("expected compression: " << token_count << " tokens < " << word.size() << " bytes");
    REQUIRE(token_count < word.size());
}

TEST_CASE("bpe vocabulary grows with merges", "[tokenizer]") {
    const std::string corpus = "abcabc abcabc abcabc";
    REQUIRE(BpeTokenizer::train(corpus, 5).vocab_size() >
            BpeTokenizer::train(corpus, 0).vocab_size());
}

// -----------------------------------------------------------------------------
// BpeTokenizer -- vocab.json + merges.txt
// -----------------------------------------------------------------------------

TEST_CASE("bpe loads vocab.json and merges.txt", "[tokenizer]") {
    const auto [vocab_json, merges_txt] = make_tiny_vocab_merges();
    const auto loaded = BpeTokenizer::from_vocab_and_merges_str(vocab_json, merges_txt);
    REQUIRE(loaded.has_value());
    const BpeTokenizer& tok = *loaded;

    // 256 byte tokens + 2 merged tokens.
    REQUIRE(tok.vocab_size() == 258);

    // "ab " round-trips through the merged token.
    REQUIRE(tok.decode(tok.encode("ab ")) == "ab ");

    // "ab ab ab" is 8 bytes but should merge down.
    const std::size_t n = tok.encode("ab ab ab").size();
    INFO("expected compression: " << n << " tokens for 8 bytes");
    REQUIRE(n < 8);
}

TEST_CASE("bpe reports missing vocab files", "[tokenizer]") {
    REQUIRE_FALSE(BpeTokenizer::from_files("/nope/vocab.json", "/nope/merges.txt").has_value());
}

// -----------------------------------------------------------------------------
// base64
// -----------------------------------------------------------------------------

TEST_CASE("base64 decode", "[tokenizer]") {
    REQUIRE(*base64_decode("aGVsbG8=") == to_bytes("hello"));
    REQUIRE(*base64_decode("aGVsbG8") == to_bytes("hello"));  // padding optional
    REQUIRE(base64_decode("")->empty());
    REQUIRE_FALSE(base64_decode("not base64!").has_value());
}

TEST_CASE("base64 encode round-trips", "[tokenizer]") {
    for (std::string_view s : {"", "a", "ab", "abc", "abcd", "hello world"}) {
        const std::vector<std::uint8_t> bytes = to_bytes(s);
        INFO("round-tripping \"" << s << "\"");
        REQUIRE(*base64_decode(base64_encode(bytes)) == bytes);
    }
}

// -----------------------------------------------------------------------------
// BpeTokenizer -- tiktoken
// -----------------------------------------------------------------------------

TEST_CASE("bpe loads the tiktoken format", "[tokenizer]") {
    // Bytes 0-255 at ranks 0-255, plus one merged token "ab" at rank 256.
    std::string content;
    for (int b = 0; b < 256; ++b) {
        content += base64_encode({static_cast<std::uint8_t>(b)}) + " " + std::to_string(b) + "\n";
    }
    const std::string with_merge = content + base64_encode(to_bytes("ab")) + " 256";

    const auto loaded = BpeTokenizer::from_tiktoken_str(with_merge);
    REQUIRE(loaded.has_value());

    const std::vector<std::uint32_t> ids = loaded->encode("ab");
    INFO("expected 'ab' to merge into one token, got " << ids.size());
    REQUIRE(ids.size() == 1);
    REQUIRE(loaded->decode(ids) == "ab");

    // Without the merge line the vocabulary is exactly the 256 bytes.
    const auto bytes_only = BpeTokenizer::from_tiktoken_str(content);
    REQUIRE(bytes_only.has_value());
    REQUIRE(bytes_only->vocab_size() == 256);
}

// -----------------------------------------------------------------------------
// GPT-2 byte encoding
// -----------------------------------------------------------------------------

TEST_CASE("gpt2 byte encoding", "[tokenizer]") {
    // U+0120 decodes to a space.
    REQUIRE(gpt2_decode_token_to_bytes("\xC4\xA0") == to_bytes(" "));
    // Printable ASCII passes through.
    REQUIRE(gpt2_decode_token_to_bytes("hello") == to_bytes("hello"));

    // Every byte round-trips through the table.
    const auto map = gpt2_bytes_to_unicode();
    for (int b = 0; b < 256; ++b) {
        INFO("byte " << b);
        REQUIRE(gpt2_decode_token_to_bytes(utf8::encode(map[b])) ==
                std::vector<std::uint8_t>{static_cast<std::uint8_t>(b)});
    }

    // The table is a bijection: 256 distinct code points.
    std::set<char32_t> distinct(map.begin(), map.end());
    REQUIRE(distinct.size() == 256);
}

// -----------------------------------------------------------------------------
// SentencePieceTokenizer
// -----------------------------------------------------------------------------

TEST_CASE("sentencepiece vocab size", "[tokenizer]") {
    REQUIRE(tiny_sp_vocab().vocab_size() == 14);
}

TEST_CASE("sentencepiece encodes a known word as one token", "[tokenizer]") {
    const SentencePieceTokenizer tok = tiny_sp_vocab();
    const std::vector<std::uint32_t> ids = tok.encode("hello");
    INFO("expected 1 token for 'hello', got " << ids.size());
    REQUIRE(ids.size() == 1);
    REQUIRE(tok.decode(ids) == "hello");
}

TEST_CASE("sentencepiece encodes two words as two tokens", "[tokenizer]") {
    const SentencePieceTokenizer tok = tiny_sp_vocab();
    const std::vector<std::uint32_t> ids = tok.encode("hello world");
    INFO("expected 2 tokens for 'hello world', got " << ids.size());
    REQUIRE(ids.size() == 2);
    REQUIRE(tok.decode(ids) == "hello world");
}

TEST_CASE("sentencepiece round-trips a short word", "[tokenizer]") {
    const SentencePieceTokenizer tok = tiny_sp_vocab();
    REQUIRE(tok.decode(tok.encode("hi")) == "hi");
}

TEST_CASE("sentencepiece falls back for unknown bytes", "[tokenizer]") {
    // 'z' is not in the vocabulary, so the Viterbi fallback emits <unk> rather
    // than failing.
    const SentencePieceTokenizer tok = tiny_sp_vocab();
    const std::vector<std::uint32_t> ids = tok.encode("hz");
    REQUIRE_FALSE(ids.empty());
    REQUIRE(tok.unk_id() == 0);
}

TEST_CASE("sentencepiece rejects malformed vocab text", "[tokenizer]") {
    REQUIRE_FALSE(SentencePieceTokenizer::from_vocab_text("nospace\n").has_value());
    REQUIRE_FALSE(SentencePieceTokenizer::from_vocab_text("").has_value());
    REQUIRE_FALSE(SentencePieceTokenizer::from_model_file("/nope/model.model").has_value());
}

TEST_CASE("sentencepiece piece lookup", "[tokenizer]") {
    const SentencePieceTokenizer tok = tiny_sp_vocab();
    REQUIRE(tok.piece_id(SP + "hello") == 3u);
    REQUIRE_FALSE(tok.piece_id("nonexistent").has_value());
}

// -----------------------------------------------------------------------------
// SentencePiece .model protobuf
// -----------------------------------------------------------------------------

TEST_CASE("sentencepiece parses a hand-built ModelProto", "[tokenizer]") {
    const std::vector<std::uint8_t> bytes = make_model_proto({{"<unk>", 0.0f},
                                                              {"<s>", 0.0f},
                                                              {"</s>", 0.0f},
                                                              {SP + "hello", -1.0f},
                                                              {SP + "world", -2.0f}});
    const auto tok = SentencePieceTokenizer::from_model_bytes(bytes);
    REQUIRE(tok.has_value());
    REQUIRE(tok->vocab_size() == 5);

    const auto id = tok->piece_id(SP + "hello");
    REQUIRE(id == 3u);
    REQUIRE(std::fabs(tok->pieces()[*id].log_prob - (-1.0f)) < 1e-6f);
}

TEST_CASE("sentencepiece encodes through a ModelProto vocabulary", "[tokenizer]") {
    const std::vector<std::uint8_t> bytes = make_model_proto({{"<unk>", 0.0f},
                                                              {"<s>", 0.0f},
                                                              {"</s>", 0.0f},
                                                              {SP + "hello", -1.0f},
                                                              {SP + "world", -2.0f},
                                                              {SP + "h", -4.0f},
                                                              {"e", -4.0f},
                                                              {"l", -4.0f},
                                                              {"o", -4.0f}});
    const auto tok = SentencePieceTokenizer::from_model_bytes(bytes);
    REQUIRE(tok.has_value());
    REQUIRE(tok->decode(tok->encode("hello world")) == "hello world");
}

TEST_CASE("proto_varint reads multi-byte values", "[tokenizer]") {
    for (std::uint64_t v : {0ull, 1ull, 127ull, 128ull, 300ull, 16384ull, 1ull << 40}) {
        const std::vector<std::uint8_t> encoded = encode_varint(v);
        const auto got = proto_varint(encoded, 0);
        REQUIRE(got.has_value());
        INFO("varint " << v);
        REQUIRE(got->first == v);
        REQUIRE(got->second == encoded.size());
    }
    // A varint whose continuation bit runs off the end is an error.
    REQUIRE_FALSE(proto_varint({0x80}, 0).has_value());
}

// -----------------------------------------------------------------------------
// HfBpeTokenizer
// -----------------------------------------------------------------------------

TEST_CASE("hf bpe loads a SentencePiece-style tokenizer.json", "[tokenizer]") {
    // Gemma style: no ByteLevel, so spaces normalize to U+2581.
    const std::string json = R"({"model":{"type":"BPE","vocab":{)"
                             "\"<unk>\":0,"
                             "\"" + SP + "\":1,"
                             "\"h\":2,\"e\":3,\"l\":4,\"o\":5,"
                             "\"" + SP + "h\":6,"
                             "\"" + SP + "he\":7,"
                             "\"" + SP + "hello\":8"
                             R"(},"merges":[")" + SP + R"( h",")" + SP + R"(h e"]}})";

    const auto tok = HfBpeTokenizer::from_json_str(json);
    REQUIRE(tok.has_value());
    REQUIRE_FALSE(tok->byte_level());
    REQUIRE(tok->vocab_size() == 9);
    REQUIRE(tok->unk_id() == 0);
    REQUIRE(tok->token_id(SP + "hello") == 8u);
    REQUIRE(tok->token_text(2) == "h");

    // Without a leading space there is no U+2581, so no merge applies and each
    // character stays its own token.
    const std::vector<std::uint32_t> bare = tok->encode("hello");
    REQUIRE(bare == std::vector<std::uint32_t>{2, 3, 4, 4, 5});
    REQUIRE(tok->decode(bare) == "hello");

    // A leading space normalizes to U+2581, and the two merges then build "▁he".
    const std::vector<std::uint32_t> spaced = tok->encode(" hello");
    REQUIRE(spaced == std::vector<std::uint32_t>{7, 4, 4, 5});
    REQUIRE(tok->decode(spaced) == " hello");
}

TEST_CASE("hf bpe detects a ByteLevel pre-tokenizer", "[tokenizer]") {
    const auto map = gpt2_bytes_to_unicode();
    const auto enc = [&](std::string_view s) {
        std::string out;
        for (char c : s) {
            out += utf8::encode(map[static_cast<std::uint8_t>(c)]);
        }
        return out;
    };

    // Qwen style: a ByteLevel pre-tokenizer, so bytes go through the GPT-2 table.
    std::string json = R"({"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{)";
    json += "\"" + enc("h") + "\":0,";
    json += "\"" + enc("i") + "\":1,";
    json += "\"" + enc("hi") + "\":2,";
    json += "\"" + enc(" ") + "\":3,";
    json += "\"" + enc("t") + "\":4,";
    json += "\"" + enc("here") + "\":5,";
    json += "\"" + enc("e") + "\":6,";
    json += "\"" + enc("r") + "\":7";
    json += R"(},"merges":[[")" + enc("h") + "\",\"" + enc("i") + "\"]]}}";

    const auto tok = HfBpeTokenizer::from_json_str(json);
    REQUIRE(tok.has_value());
    REQUIRE(tok->byte_level());

    const std::vector<std::uint32_t> ids = tok->encode("hi");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == 2u);
    REQUIRE(tok->decode(ids) == "hi");
}

TEST_CASE("hf bpe rejects malformed json", "[tokenizer]") {
    REQUIRE_FALSE(HfBpeTokenizer::from_json_str("{}").has_value());
    REQUIRE_FALSE(HfBpeTokenizer::from_json_str(R"({"model":{}})").has_value());
    REQUIRE_FALSE(HfBpeTokenizer::from_json_file("/nope/tokenizer.json").has_value());
}

TEST_CASE("gpt2 pretokenizer splits like the reference", "[tokenizer]") {
    using V = std::vector<std::string>;

    // Letter runs absorb one leading non-alphanumeric, so " is" is one word.
    REQUIRE(HfBpeTokenizer::gpt2_pretokenize("What is 2+2?") ==
            V{"What", " is", " ", "2", "+", "2", "?"});

    // Contractions split off as their own words; newline runs group together
    // separately from other whitespace; digits are always single tokens.
    REQUIRE(HfBpeTokenizer::gpt2_pretokenize("don't we're I'll a\n\nb 123") ==
            V{"don", "'t", " we", "'re", " I", "'ll", " a", "\n\n", "b", " ", "1", "2", "3"});

    REQUIRE(HfBpeTokenizer::gpt2_pretokenize("").empty());
}

// =============================================================================
// CLIP pre-tokenization
// =============================================================================
//
// CLIP is a different family from GPT-2 and Llama 3: the text is lowercased and
// its whitespace collapsed before splitting, the split drops whitespace rather
// than attaching it to the following word, and each word carries an explicit
// `</w>` marker instead of a leading space.

TEST_CASE("CLIP normalization lowercases and collapses whitespace", "[tokenizer][clip]") {
    using PT = HfBpeTokenizer::PreTokenizer;
    REQUIRE(HfBpeTokenizer::normalize("A  Photo\tOf\nThings", PT::Clip) == "a photo of things");
    REQUIRE(HfBpeTokenizer::normalize("   leading and trailing   ", PT::Clip) ==
            "leading and trailing");
    // The other pre-tokenizers must be left exactly as they were: GPT-2 carries
    // the space into the next token and depends on it surviving.
    REQUIRE(HfBpeTokenizer::normalize("A  Photo", PT::Gpt2) == "A  Photo");
    REQUIRE(HfBpeTokenizer::normalize("A  Photo", PT::Llama3) == "A  Photo");
}

TEST_CASE("CLIP pre-tokenization drops whitespace rather than attaching it",
          "[tokenizer][clip]") {
    const std::vector<std::string> words =
        HfBpeTokenizer::clip_pretokenize("a photograph of a harbour at dawn");
    REQUIRE(words == std::vector<std::string>{"a", "photograph", "of", "a", "harbour", "at",
                                              "dawn"});
    // GPT-2 keeps the space, which is exactly the difference.
    const std::vector<std::string> gpt2 = HfBpeTokenizer::gpt2_pretokenize("a photograph");
    REQUIRE(gpt2.size() == 2);
    REQUIRE(gpt2[1] == " photograph");
}

TEST_CASE("CLIP pre-tokenization splits digits one at a time", "[tokenizer][clip]") {
    // `[\p{N}]` with no repetition, unlike Llama 3's runs of up to three.
    REQUIRE(HfBpeTokenizer::clip_pretokenize("2024") ==
            std::vector<std::string>{"2", "0", "2", "4"});
}

TEST_CASE("CLIP pre-tokenization keeps contractions together", "[tokenizer][clip]") {
    REQUIRE(HfBpeTokenizer::clip_pretokenize("it's") == std::vector<std::string>{"it", "'s"});
    REQUIRE(HfBpeTokenizer::clip_pretokenize("they're") ==
            std::vector<std::string>{"they", "'re"});
    REQUIRE(HfBpeTokenizer::clip_pretokenize("we'll") == std::vector<std::string>{"we", "'ll"});
    REQUIRE(HfBpeTokenizer::clip_pretokenize("i've") == std::vector<std::string>{"i", "'ve"});
}

TEST_CASE("CLIP pre-tokenization runs punctuation together", "[tokenizer][clip]") {
    REQUIRE(HfBpeTokenizer::clip_pretokenize("wow!!! really?") ==
            std::vector<std::string>{"wow", "!!!", "really", "?"});
}

TEST_CASE("CLIP pre-tokenization handles an empty and whitespace-only string",
          "[tokenizer][clip]") {
    REQUIRE(HfBpeTokenizer::clip_pretokenize("").empty());
    REQUIRE(HfBpeTokenizer::clip_pretokenize("   \t\n ").empty());
}

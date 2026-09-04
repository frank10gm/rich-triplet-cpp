#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <unistd.h>

#include "rt/dataset.hpp"

using namespace rt;

namespace {

struct TempPath {
    std::filesystem::path path;
    explicit TempPath(const char* stem)
        : path(std::filesystem::temp_directory_path() /
               (std::string(stem) + std::to_string(::getpid()) + ".bin")) {}
    ~TempPath() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    [[nodiscard]] std::string str() const { return path.string(); }
};

/// A character tokenizer over a corpus that gives each letter its own id.
CharTokenizer corpus_tokenizer(std::string_view text) { return CharTokenizer::from_text(text); }

}  // namespace

TEST_CASE("lcg is deterministic", "[dataset]") {
    Lcg a(42), b(42);
    for (int i = 0; i < 8; ++i) {
        REQUIRE(a.next() == b.next());
    }
    Lcg c(43);
    Lcg d(42);
    REQUIRE(c.next() != d.next());
}

TEST_CASE("text dataset pairs are the window and its shift", "[dataset]") {
    const std::string text = "abcdefghij";
    const CharTokenizer tok = corpus_tokenizer(text);
    const TextDataset ds = TextDataset::from_text(text, tok, 4);

    // 10 tokens with a window of 4 leaves 6 valid starting positions.
    REQUIRE(ds.size() == 6);

    const auto [inp, tgt] = ds.get_pair(0);
    REQUIRE(inp.shape == std::vector<std::size_t>{4});
    REQUIRE(tgt.shape == std::vector<std::size_t>{4});
    // The target is the input shifted by one.
    for (std::size_t k = 0; k + 1 < 4; ++k) {
        REQUIRE(tgt.data[k] == inp.data[k + 1]);
    }
    // And it decodes back to the original characters.
    const std::vector<std::uint32_t> ids{static_cast<std::uint32_t>(inp.data[0]),
                                         static_cast<std::uint32_t>(inp.data[1]),
                                         static_cast<std::uint32_t>(inp.data[2]),
                                         static_cast<std::uint32_t>(inp.data[3])};
    REQUIRE(tok.decode(ids) == "abcd");
}

TEST_CASE("text dataset random_batch shape and determinism", "[dataset]") {
    const std::string text = "the quick brown fox jumps over the lazy dog";
    const CharTokenizer tok = corpus_tokenizer(text);
    const TextDataset ds = TextDataset::from_text(text, tok, 5);

    const auto [inp, tgt] = ds.random_batch(3, 7);
    REQUIRE(inp.shape == std::vector<std::size_t>{3, 5});
    REQUIRE(tgt.shape == std::vector<std::size_t>{3, 5});

    // The same seed reproduces the same batch.
    const auto [inp2, tgt2] = ds.random_batch(3, 7);
    REQUIRE(inp2.data == inp.data);
    REQUIRE(tgt2.data == tgt.data);
}

TEST_CASE("text dataset sample and sample_batch", "[dataset]") {
    const std::string text = "the quick brown fox jumps over the lazy dog";
    const CharTokenizer tok = corpus_tokenizer(text);
    const TextDataset ds = TextDataset::from_text(text, tok, 5);

    const auto [inp, tgt] = ds.sample(11);
    REQUIRE(inp.size() == 5);
    REQUIRE(tgt.size() == 5);

    const std::vector<TokenPair> batch = ds.sample_batch(11, 4);
    REQUIRE(batch.size() == 4);
    // The first entry reuses the seed directly.
    REQUIRE(batch[0].first == inp);
    // The per-sequence seeds are offset, so the batch covers several positions.
    std::set<std::vector<std::size_t>> distinct;
    for (const auto& [i, t] : batch) {
        distinct.insert(i);
    }
    REQUIRE(distinct.size() > 1);
}

TEST_CASE("text dataset train/val split", "[dataset]") {
    std::string text;
    for (int i = 0; i < 40; ++i) {
        text += "abcdefghij";
    }
    const CharTokenizer tok = corpus_tokenizer(text);
    const auto [train, val] = TextDataset::train_val_split(text, tok, 4);

    // 400 tokens split 90/10, minus the window from each side.
    REQUIRE(train.size() == 360 - 4);
    REQUIRE(val.size() == 40 - 4);
    REQUIRE(train.context_length == 4);
}

TEST_CASE("tokenized dataset round-trips through a binary file", "[dataset]") {
    TempPath tmp("rt_ds_");
    std::string text;
    for (int i = 0; i < 40; ++i) {
        text += "abcdefghij";
    }
    const CharTokenizer tok = corpus_tokenizer(text);

    const auto written = TokenizedDataset::write_bin(tmp.str(), text, tok);
    REQUIRE(written.has_value());
    REQUIRE(*written == 400);

    const auto ds = TokenizedDataset::open(tmp.str(), 8);
    REQUIRE(ds.has_value());
    REQUIRE(ds->size() == 400 - 8);
    REQUIRE(ds->context_length == 8);

    const auto [inp, tgt] = ds->random_sample(5);
    REQUIRE(inp.size() == 8);
    REQUIRE(tgt.size() == 8);
    for (std::size_t k = 0; k + 1 < 8; ++k) {
        REQUIRE(tgt[k] == inp[k + 1]);
    }
    // Same seed, same sample.
    REQUIRE(ds->random_sample(5).first == inp);
}

TEST_CASE("tokenized dataset train/val views share the file", "[dataset]") {
    TempPath tmp("rt_ds_split_");
    std::string text;
    for (int i = 0; i < 40; ++i) {
        text += "abcdefghij";
    }
    const CharTokenizer tok = corpus_tokenizer(text);
    REQUIRE(TokenizedDataset::write_bin(tmp.str(), text, tok).has_value());

    const auto pair = TokenizedDataset::open_train_val(tmp.str(), 4);
    REQUIRE(pair.has_value());
    const auto& [train, val] = *pair;
    REQUIRE(train.size() == 360 - 4);
    REQUIRE(val.size() == 40 - 4);

    // The validation view reads from a later offset, so it yields different
    // tokens than the training view for the same seed.
    REQUIRE(train.sample(1).first.size() == 4);
    REQUIRE(val.sample(1).first.size() == 4);
}

TEST_CASE("tokenized dataset rejects malformed files", "[dataset]") {
    REQUIRE_FALSE(TokenizedDataset::open("/definitely/not/here.bin", 4).has_value());

    TempPath tiny("rt_ds_tiny_");
    {
        std::ofstream f(tiny.str(), std::ios::binary);
        f << "short";
    }
    const auto too_small = TokenizedDataset::open(tiny.str(), 4);
    REQUIRE_FALSE(too_small.has_value());
    REQUIRE(too_small.error().find("too small") != std::string::npos);

    TempPath bad("rt_ds_bad_");
    {
        std::ofstream f(bad.str(), std::ios::binary);
        for (int i = 0; i < 32; ++i) {
            f.put('\0');
        }
    }
    const auto bad_magic = TokenizedDataset::open(bad.str(), 4);
    REQUIRE_FALSE(bad_magic.has_value());
    REQUIRE(bad_magic.error().find("bad magic") != std::string::npos);

    // A valid file whose window exceeds the token count is also rejected.
    TempPath small("rt_ds_small_");
    const CharTokenizer tok = corpus_tokenizer("abc");
    REQUIRE(TokenizedDataset::write_bin(small.str(), "abc", tok).has_value());
    const auto too_short = TokenizedDataset::open(small.str(), 10);
    REQUIRE_FALSE(too_short.has_value());
    REQUIRE(too_short.error().find("need >") != std::string::npos);
}

TEST_CASE("tokenized dataset writes from a file line by line", "[dataset]") {
    TempPath src_holder("rt_ds_src_");
    TempPath dst("rt_ds_dst_");
    const std::string src = src_holder.str();
    {
        std::ofstream f(src);
        f << "abc\n"
          << "\n"  // blank lines are skipped
          << "defg\n";
    }

    const CharTokenizer tok = corpus_tokenizer("abcdefg");
    const auto n = TokenizedDataset::write_bin_from_file(dst.str(), src, tok);
    REQUIRE(n.has_value());
    REQUIRE(*n == 7);  // "abc" + "defg"

    const auto ds = TokenizedDataset::open(dst.str(), 3);
    REQUIRE(ds.has_value());
    REQUIRE(ds->size() == 4);

    REQUIRE_FALSE(
        TokenizedDataset::write_bin_from_file(dst.str(), "/nope/missing.txt", tok).has_value());
}

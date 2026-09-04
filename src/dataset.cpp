#include "rt/dataset.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace rt {

std::vector<TokenPair> DataSource::sample_batch(std::uint64_t rng_seed,
                                                std::size_t batch_size) const {
    std::vector<TokenPair> out;
    out.reserve(batch_size);
    for (std::size_t i = 0; i < batch_size; ++i) {
        // A prime stride keeps the per-sequence seeds well separated.
        out.push_back(sample(rng_seed + static_cast<std::uint64_t>(i) * 7919));
    }
    return out;
}

// =============================================================================
// TextDataset
// =============================================================================

TextDataset TextDataset::from_text(std::string_view text, const Tokenizer& tokenizer,
                                   std::size_t context_length) {
    TextDataset ds;
    ds.tokens_ = tokenizer.encode(text);
    ds.context_length = context_length;
    std::printf("Dataset: %zu chars -> %zu tokens (vocab size %zu)\n", text.size(),
                ds.tokens_.size(), tokenizer.vocab_size());
    return ds;
}

std::pair<TextDataset, TextDataset> TextDataset::train_val_split(std::string_view text,
                                                                 const Tokenizer& tokenizer,
                                                                 std::size_t context_length) {
    const std::vector<std::uint32_t> all = tokenizer.encode(text);
    const auto split = static_cast<std::size_t>(static_cast<double>(all.size()) * 0.9);

    TextDataset train, val;
    train.tokens_.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(split));
    train.context_length = context_length;
    val.tokens_.assign(all.begin() + static_cast<std::ptrdiff_t>(split), all.end());
    val.context_length = context_length;

    std::printf("Train/val split: %zu / %zu tokens\n", train.tokens_.size(), val.tokens_.size());
    return {std::move(train), std::move(val)};
}

std::size_t TextDataset::size() const {
    return tokens_.size() <= context_length ? 0 : tokens_.size() - context_length;
}

std::pair<Tensor, Tensor> TextDataset::get_pair(std::size_t idx) const {
    assert(idx + context_length < tokens_.size() && "index out of bounds");

    std::vector<float> input(context_length), target(context_length);
    for (std::size_t k = 0; k < context_length; ++k) {
        input[k] = static_cast<float>(tokens_[idx + k]);
        target[k] = static_cast<float>(tokens_[idx + k + 1]);
    }
    return {Tensor(std::move(input), {context_length}),
            Tensor(std::move(target), {context_length})};
}

std::pair<Tensor, Tensor> TextDataset::random_batch(std::size_t batch_size,
                                                    std::uint64_t rng_seed) const {
    const std::size_t n = size();
    assert(n >= batch_size && "dataset too small for this batch size");

    Lcg rng(rng_seed);
    std::vector<float> input_data, target_data;
    input_data.reserve(batch_size * context_length);
    target_data.reserve(batch_size * context_length);

    for (std::size_t b = 0; b < batch_size; ++b) {
        const std::size_t idx = static_cast<std::size_t>(rng.next()) % n;
        const auto [inp, tgt] = get_pair(idx);
        input_data.insert(input_data.end(), inp.data.begin(), inp.data.end());
        target_data.insert(target_data.end(), tgt.data.begin(), tgt.data.end());
    }

    const std::vector<std::size_t> shape{batch_size, context_length};
    return {Tensor(std::move(input_data), shape), Tensor(std::move(target_data), shape)};
}

TokenPair TextDataset::sample(std::uint64_t rng_seed) const {
    const auto [inp_t, tgt_t] = random_batch(1, rng_seed);
    std::vector<std::size_t> input(inp_t.data.size()), target(tgt_t.data.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<std::size_t>(inp_t.data[i]);
        target[i] = static_cast<std::size_t>(tgt_t.data[i]);
    }
    return {std::move(input), std::move(target)};
}

// =============================================================================
// TokenizedDataset
// =============================================================================

namespace {
[[nodiscard]] std::uint64_t read_u64_le(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | p[i];
    }
    return v;
}

void put_u64_le(std::ostream& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.put(static_cast<char>((v >> (8 * i)) & 0xff));
    }
}

void put_u32_le(std::ostream& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.put(static_cast<char>((v >> (8 * i)) & 0xff));
    }
}
}  // namespace

Result<TokenizedDataset> TokenizedDataset::open(const std::string& path,
                                                std::size_t context_length) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return err("cannot read " + path);
    }
    const auto size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    auto bytes = std::make_shared<std::vector<std::uint8_t>>(size);
    f.read(reinterpret_cast<char*>(bytes->data()), static_cast<std::streamsize>(size));

    if (bytes->size() < 16) {
        return err(path + ": file too small (" + std::to_string(bytes->size()) + " bytes)");
    }

    const std::uint64_t magic = read_u64_le(bytes->data());
    if (magic != BIN_MAGIC) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%s: bad magic %016llx (expected %016llx)", path.c_str(),
                      static_cast<unsigned long long>(magic),
                      static_cast<unsigned long long>(BIN_MAGIC));
        return err(msg);
    }

    const auto n_tokens = static_cast<std::size_t>(read_u64_le(bytes->data() + 8));
    constexpr std::size_t data_offset = 16;
    const std::size_t expected_bytes = data_offset + n_tokens * 4;
    if (bytes->size() < expected_bytes) {
        return err(path + ": file truncated: need " + std::to_string(expected_bytes) +
                   " bytes, got " + std::to_string(bytes->size()));
    }
    if (n_tokens <= context_length) {
        return err(path + ": only " + std::to_string(n_tokens) + " tokens, need > " +
                   std::to_string(context_length));
    }

    TokenizedDataset ds;
    ds.bytes_ = std::move(bytes);
    ds.data_offset_ = data_offset;
    ds.n_tokens_ = n_tokens;
    ds.context_length = context_length;
    return ds;
}

Result<std::pair<TokenizedDataset, TokenizedDataset>> TokenizedDataset::open_train_val(
    const std::string& path, std::size_t context_length) {
    RT_TRY(full, open(path, context_length));
    const auto split = static_cast<std::size_t>(static_cast<double>(full.n_tokens_) * 0.9);

    std::printf("TokenizedDataset: %zu tokens total -> %zu train / %zu val\n", full.n_tokens_,
                split, full.n_tokens_ - split);

    // Both views share the same bytes; only the offset and length differ.
    TokenizedDataset train;
    train.bytes_ = full.bytes_;
    train.data_offset_ = full.data_offset_;
    train.n_tokens_ = split;
    train.context_length = context_length;

    TokenizedDataset val;
    val.bytes_ = full.bytes_;
    val.data_offset_ = full.data_offset_ + split * 4;
    val.n_tokens_ = full.n_tokens_ - split;
    val.context_length = context_length;

    return std::pair{std::move(train), std::move(val)};
}

std::size_t TokenizedDataset::token_at(std::size_t i) const {
    const std::uint8_t* p = bytes_->data() + data_offset_ + i * 4;
    return static_cast<std::size_t>(p[0]) | (static_cast<std::size_t>(p[1]) << 8) |
           (static_cast<std::size_t>(p[2]) << 16) | (static_cast<std::size_t>(p[3]) << 24);
}

std::size_t TokenizedDataset::size() const {
    return n_tokens_ <= context_length ? 0 : n_tokens_ - context_length;
}

TokenPair TokenizedDataset::random_sample(std::uint64_t rng_seed) const {
    Lcg rng(rng_seed);
    const std::size_t idx = static_cast<std::size_t>(rng.next()) % size();

    std::vector<std::size_t> input(context_length), target(context_length);
    for (std::size_t k = 0; k < context_length; ++k) {
        input[k] = token_at(idx + k);
        target[k] = token_at(idx + k + 1);
    }
    return {std::move(input), std::move(target)};
}

TokenPair TokenizedDataset::sample(std::uint64_t rng_seed) const {
    return random_sample(rng_seed);
}

Result<std::size_t> TokenizedDataset::write_bin_tokens(
    const std::string& path, const std::vector<std::uint32_t>& tokens) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return err("cannot create " + path);
    }
    put_u64_le(f, BIN_MAGIC);
    put_u64_le(f, tokens.size());
    for (std::uint32_t tok : tokens) {
        put_u32_le(f, tok);
    }
    if (!f) {
        return err("write error on " + path);
    }
    std::printf("Wrote %zu tokens to %s\n", tokens.size(), path.c_str());
    return tokens.size();
}

Result<std::size_t> TokenizedDataset::write_bin(const std::string& path, std::string_view text,
                                                const Tokenizer& tokenizer) {
    return write_bin_tokens(path, tokenizer.encode(text));
}

Result<std::size_t> TokenizedDataset::write_bin_from_file(const std::string& dst_path,
                                                          const std::string& src_path,
                                                          const Tokenizer& tokenizer) {
    std::ifstream src(src_path);
    if (!src) {
        return err("cannot open " + src_path);
    }
    std::ofstream out(dst_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return err("cannot create " + dst_path);
    }

    // The token count is not known until the end, so the header is written with
    // a placeholder and patched afterwards.
    put_u64_le(out, BIN_MAGIC);
    put_u64_le(out, 0);

    std::size_t n_tokens = 0;
    std::string line;
    while (std::getline(src, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::uint32_t> toks = tokenizer.encode(line);
        for (std::uint32_t tok : toks) {
            put_u32_le(out, tok);
        }
        n_tokens += toks.size();
    }

    out.seekp(8);
    put_u64_le(out, n_tokens);
    if (!out) {
        return err("write error on " + dst_path);
    }

    std::printf("Wrote %zu tokens to %s\n", n_tokens, dst_path.c_str());
    return n_tokens;
}

}  // namespace rt

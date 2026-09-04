#pragma once

// =============================================================================
// Datasets -- turning a corpus into (input, target) training pairs
// =============================================================================
//
// Both sources yield the same thing: a window of `context_length` tokens as
// input, and the same window shifted one position as the target, so the model
// learns to predict the next token at every position.
//
// `TextDataset` keeps the whole corpus in memory. `TokenizedDataset` reads a
// pre-tokenized binary file, so a corpus far larger than RAM still trains.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rt/result.hpp"
#include "rt/tensor.hpp"
#include "rt/tokenizer.hpp"

namespace rt {

/// Minimal LCG, so no dependency is needed for reproducible sampling.
///
///   state = (a * state + c) mod 2^64
///
/// with Knuth's a = 6364136223846793005 and c = 1442695040888963407.
class Lcg {
   public:
    explicit Lcg(std::uint64_t seed) : state_(seed + 1) {}

    [[nodiscard]] std::uint64_t next() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return state_;
    }

   private:
    std::uint64_t state_;
};

/// One training example: the input token ids and the targets they predict.
using TokenPair = std::pair<std::vector<std::size_t>, std::vector<std::size_t>>;

/// A source of random (input, target) pairs. The training loop takes either
/// implementation.
class DataSource {
   public:
    virtual ~DataSource() = default;

    [[nodiscard]] virtual TokenPair sample(std::uint64_t rng_seed) const = 0;
    [[nodiscard]] virtual std::size_t size() const = 0;

    /// Sample `batch_size` independent pairs. Each sequence gets its own seed
    /// derived from `rng_seed`, so the batch covers different positions in the
    /// corpus rather than repeating one.
    [[nodiscard]] std::vector<TokenPair> sample_batch(std::uint64_t rng_seed,
                                                      std::size_t batch_size) const;
};

// =============================================================================
// TextDataset -- the whole corpus held in memory
// =============================================================================

class TextDataset final : public DataSource {
   public:
    /// How many tokens the model sees at once, also called block size.
    std::size_t context_length = 0;

    /// Tokenize `text` and keep the result.
    [[nodiscard]] static TextDataset from_text(std::string_view text, const Tokenizer& tokenizer,
                                               std::size_t context_length);

    /// Split a corpus 90/10 into training and validation datasets.
    ///
    /// The validation portion is never trained on -- it is how the run measures
    /// generalization to unseen text.
    [[nodiscard]] static std::pair<TextDataset, TextDataset> train_val_split(
        std::string_view text, const Tokenizer& tokenizer, std::size_t context_length);

    /// How many (input, target) pairs exist. A pair needs `context_length + 1`
    /// tokens: the window, plus one more for the final target.
    [[nodiscard]] std::size_t size() const override;

    /// The pair starting at `idx`, as two 1-D tensors of `context_length`
    /// values: `tokens[idx ..]` and `tokens[idx+1 ..]`.
    [[nodiscard]] std::pair<Tensor, Tensor> get_pair(std::size_t idx) const;

    /// A random batch, as two [batch_size, context_length] tensors.
    [[nodiscard]] std::pair<Tensor, Tensor> random_batch(std::size_t batch_size,
                                                         std::uint64_t rng_seed) const;

    [[nodiscard]] TokenPair sample(std::uint64_t rng_seed) const override;

   private:
    /// The corpus, concatenated into one flat token sequence.
    std::vector<std::uint32_t> tokens_;
};

// =============================================================================
// TokenizedDataset -- streaming from a pre-tokenized binary file
// =============================================================================
//
// Holding a 10 GB corpus in memory costs ~10 GB before the model is even built.
// Instead the tokens live on disk as a flat u32 array and are read on demand.
//
// File format:
//   [magic: u64 LE = 0x526963685472697C "RichTri|"]
//   [n_tokens: u64 LE]
//   [token_0: u32 LE] [token_1: u32 LE] ...
//
// Write the file once with `write_bin`, then open it for every training run.

inline constexpr std::uint64_t BIN_MAGIC = 0x526963685472697Cull;  // "RichTri|"

class TokenizedDataset final : public DataSource {
   public:
    std::size_t context_length = 0;

    /// Open a file written by `write_bin`. Fails if it is missing, truncated,
    /// carries the wrong magic, or holds too few tokens for one window.
    [[nodiscard]] static Result<TokenizedDataset> open(const std::string& path,
                                                       std::size_t context_length);

    /// Open and split 90/10 into training and validation views. Both share the
    /// same underlying bytes.
    [[nodiscard]] static Result<std::pair<TokenizedDataset, TokenizedDataset>> open_train_val(
        const std::string& path, std::size_t context_length);

    /// Tokenize `text` and write it to `path`. A one-time operation.
    [[nodiscard]] static Result<std::size_t> write_bin(const std::string& path,
                                                       std::string_view text,
                                                       const Tokenizer& tokenizer);

    /// Tokenize `src_path` line by line into `dst_path`. The low-memory path
    /// for large corpora: peak usage is one line, not the whole corpus.
    [[nodiscard]] static Result<std::size_t> write_bin_from_file(const std::string& dst_path,
                                                                 const std::string& src_path,
                                                                 const Tokenizer& tokenizer);

    [[nodiscard]] std::size_t size() const override;

    /// A random (input, target) pair, drawn with an LCG seeded by `rng_seed`.
    [[nodiscard]] TokenPair random_sample(std::uint64_t rng_seed) const;

    [[nodiscard]] TokenPair sample(std::uint64_t rng_seed) const override;

   private:
    [[nodiscard]] static Result<std::size_t> write_bin_tokens(
        const std::string& path, const std::vector<std::uint32_t>& tokens);

    /// The token at position `i`, relative to this view's start.
    [[nodiscard]] std::size_t token_at(std::size_t i) const;

    /// The file bytes. Shared between the train and validation views.
    std::shared_ptr<std::vector<std::uint8_t>> bytes_;
    /// Byte offset of this view's first token.
    std::size_t data_offset_ = 0;
    /// Tokens in this view, which may be a sub-range of the file.
    std::size_t n_tokens_ = 0;
};

}  // namespace rt

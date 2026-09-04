#pragma once

// =============================================================================
// A GPT built on the scalar engine
// =============================================================================
//
// The teaching model: embeddings, causal self-attention, a pre-norm block, and
// a language-modelling head, all on scalar `Value` nodes. Correct and readable;
// `transformer2` is the same architecture on the tensor engine, and that is the
// one that trains at any useful speed.

#include <cstddef>
#include <vector>

#include "rt/nn.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

struct Config {
    /// Number of distinct tokens.
    std::size_t vocab_size = 0;
    /// Maximum sequence length -- the context window.
    std::size_t context_length = 0;
    /// Embedding dimension. Every layer takes and returns vectors this wide.
    std::size_t d_model = 0;
    /// Number of transformer layers.
    std::size_t n_layers = 0;
    /// Number of attention heads. Must divide `d_model` evenly.
    std::size_t n_heads = 0;

    /// A tiny model, sized for learning and fast CPU training.
    [[nodiscard]] static Config nano(std::size_t vocab_size);

    /// Width of each head's key/query/value vectors: d_model / n_heads.
    [[nodiscard]] std::size_t d_head() const;

    /// Approximate parameter count, for display.
    [[nodiscard]] std::size_t param_count() const;
};

// =============================================================================
// Embedding
// =============================================================================
//
// Each token id maps to a learned d_model vector -- a lookup table that
// training shapes so that related tokens end up near each other.
//
// A transformer sees all positions at once and has no inherent sense of order,
// so a learned vector per position is added on top:
//
//   input[t] = token_embed[token_id[t]] + pos_embed[t]
//
// The original paper used fixed sinusoids; GPT-2 learns them, which is what
// this does. Newer models use rotary embeddings instead.

class Embedding : public Module {
   public:
    /// [vocab_size, d_model]
    std::vector<std::vector<Value>> token_embed;
    /// [context_length, d_model]
    std::vector<std::vector<Value>> pos_embed;
    Config config;

    Embedding(const Config& config, InitRng& rng);

    /// token_ids (length T) -> T vectors of d_model values.
    [[nodiscard]] std::vector<std::vector<Value>> forward(
        const std::vector<std::size_t>& token_ids) const;

    [[nodiscard]] std::vector<Value> parameters() const override;
};

// =============================================================================
// Causal self-attention
// =============================================================================

/// One attention head.
///
/// Projects each token to a query, key and value, scores every query against
/// every key it is allowed to see, softmaxes those scores, and returns the
/// weighted sum of values.
///
/// The causal mask uses -1e9 rather than -infinity: `exp(-1e9)` underflows to
/// zero cleanly, while -infinity would produce NaN in the gradient.
class AttentionHead : public Module {
   public:
    Linear w_q;  // d_model -> d_head
    Linear w_k;
    Linear w_v;
    std::size_t d_head = 0;

    AttentionHead(std::size_t d_model, std::size_t d_head, InitRng& rng);

    /// T vectors of d_model -> T vectors of d_head.
    [[nodiscard]] std::vector<std::vector<Value>> forward(
        const std::vector<std::vector<Value>>& x) const;

    [[nodiscard]] std::vector<Value> parameters() const override;
};

/// Several heads in parallel, concatenated and projected back to d_model.
///
/// Different heads learn different relationships -- one may track syntax,
/// another coreference, another local context.
class MultiHeadAttention : public Module {
   public:
    std::vector<AttentionHead> heads;
    /// Concatenated heads (n_heads * d_head == d_model) -> d_model.
    Linear w_o;
    std::size_t n_heads = 0;
    std::size_t d_model = 0;

    MultiHeadAttention(const Config& config, InitRng& rng);

    [[nodiscard]] std::vector<std::vector<Value>> forward(
        const std::vector<std::vector<Value>>& x) const;

    [[nodiscard]] std::vector<Value> parameters() const override;
};

// =============================================================================
// TransformerBlock
// =============================================================================
//
// One layer, in the modern pre-norm arrangement:
//
//   x = x + MultiHeadAttention(LayerNorm(x))
//   x = x + Mlp(LayerNorm(x))
//
// The additions are residual connections. Without them the gradient has to
// multiply through every layer and vanishes; with them it also flows straight
// down the identity path, which is what made deep networks trainable.

class TransformerBlock : public Module {
   public:
    LayerNorm ln1;
    MultiHeadAttention attn;
    LayerNorm ln2;
    Mlp mlp;

    TransformerBlock(const Config& config, InitRng& rng);

    [[nodiscard]] std::vector<std::vector<Value>> forward(
        const std::vector<std::vector<Value>>& x) const;

    [[nodiscard]] std::vector<Value> parameters() const override;
};

// =============================================================================
// Gpt
// =============================================================================

class Gpt : public Module {
   public:
    // Declaration order is initialization order, and every constructor here
    // draws from the same InitRng. The reference builds the blocks first, so
    // they must be declared first for the weights to come out identical.
    std::vector<TransformerBlock> blocks;
    Embedding embed;
    LayerNorm ln_final;
    Linear lm_head;  // d_model -> vocab_size
    Config config;

    Gpt(Config config, InitRng& rng);

    /// token_ids (length T <= context_length) -> T logit vectors of vocab_size.
    [[nodiscard]] std::vector<std::vector<Value>> forward(
        const std::vector<std::size_t>& token_ids) const;

    /// Mean cross-entropy over the sequence.
    ///
    /// At each position the loss is `-log(p[target])`: near zero when the model
    /// is confident and right, unbounded when it assigns the correct token
    /// almost no probability. A uniform model over V tokens scores log(V) --
    /// about 4.17 for a 65-token character vocabulary.
    [[nodiscard]] Value loss(const std::vector<std::size_t>& token_ids,
                             const std::vector<std::size_t>& targets) const;

    [[nodiscard]] std::vector<Value> parameters() const override;
};

}  // namespace rt

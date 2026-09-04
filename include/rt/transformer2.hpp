#pragma once

// =============================================================================
// GPT-2 on the tensor engine
// =============================================================================
//
// The same architecture as `transformer.hpp`, but every operation is one
// TensorNode over whole matrices rather than thousands of scalar nodes. This
// is the version that actually trains.

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "rt/nn2.hpp"
#include "rt/transformer.hpp"

namespace rt {

// =============================================================================
// Embedding2
// =============================================================================
//
// Gathers a token row and a position row and adds them. Implemented as one
// fused node holding the id sequence, so backward scatters the output gradient
// straight back into the rows that were read.

class Embedding2 : public Module2 {
   public:
    /// [vocab_size, d_model]
    TensorNode token_embed;
    /// [context_length, d_model]
    TensorNode pos_embed;
    std::size_t vocab_size = 0;
    std::size_t context_length = 0;
    std::size_t d_model = 0;

    Embedding2(const Config& config, InitRng& rng);

    /// token_ids (length T) -> [T, d_model].
    [[nodiscard]] TensorNode forward(const std::vector<std::size_t>& token_ids) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
};

// =============================================================================
// AttentionHead2
// =============================================================================
//
//   Q = x @ W_Q.T,  K = x @ W_K.T,  V = x @ W_V.T
//   out = causal_attention(Q, K, V)

class AttentionHead2 : public Module2 {
   public:
    Linear2 w_q;
    Linear2 w_k;
    Linear2 w_v;
    std::size_t d_head = 0;

    AttentionHead2(std::size_t d_model, std::size_t d_head, InitRng& rng);

    /// x [T, d_model] -> [T, d_head].
    [[nodiscard]] TensorNode forward(const TensorNode& x) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
};

// =============================================================================
// MultiHeadAttention2
// =============================================================================
//
// Projects each head separately, concatenates, and runs all heads through one
// batched attention call rather than a per-head loop.

class MultiHeadAttention2 : public Module2 {
   public:
    std::vector<AttentionHead2> heads;
    /// [d_model, d_model] -- n_heads * d_head equals d_model.
    Linear2 w_o;
    std::size_t d_model = 0;

    MultiHeadAttention2(const Config& config, InitRng& rng);

    /// x [T, d_model] -> [T, d_model].
    [[nodiscard]] TensorNode forward(const TensorNode& x) const;

    /// Concatenate [T, d_head] nodes into [T, n*d_head], with a backward that
    /// scatters the gradient back to each head.
    [[nodiscard]] static TensorNode concat_heads(const std::vector<TensorNode>& heads,
                                                 std::size_t t, std::size_t n, std::size_t dh);

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
};

// =============================================================================
// TransformerBlock2
// =============================================================================
//
//   x = x + Attn(LN1(x))
//   x = x + Mlp(LN2(x))

class TransformerBlock2 : public Module2 {
   public:
    LayerNorm2 ln1;
    MultiHeadAttention2 attn;
    LayerNorm2 ln2;
    Mlp2 mlp;

    TransformerBlock2(const Config& config, InitRng& rng);

    [[nodiscard]] TensorNode forward(const TensorNode& x) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
};

// =============================================================================
// KV cache
// =============================================================================
//
// Generating without a cache is O(T^2): each new token recomputes attention
// over every previous token. With a cache, the prompt runs once and each new
// token attends against stored K/V, which is O(T) per step.

/// One attention head's cache, growing to at most `max_len` rows.
class HeadKvCache {
   public:
    Mat k;  // [seq_len, d_head]
    Mat v;

    HeadKvCache(std::size_t d_head, std::size_t max_len);

    /// Append a row, dropping the oldest once the cache is full.
    void append_k(std::span<const float> row);
    void append_v(std::span<const float> row);

   private:
    std::size_t d_head_ = 0;
    std::size_t max_len_ = 0;
};

/// One `HeadKvCache` per attention head in a block.
struct BlockKvCache {
    std::vector<HeadKvCache> heads;

    BlockKvCache(std::size_t n_heads, std::size_t d_head, std::size_t max_len);
};

/// One `BlockKvCache` per transformer block.
struct Gpt2KvCache {
    std::vector<BlockKvCache> blocks;

    explicit Gpt2KvCache(const Config& config);
};

/// Attention for the new token(s) against the cached K/V.
///
/// No causal mask is needed: the cache holds only past positions, so every
/// cached key is valid context.
[[nodiscard]] TensorNode mha_forward_cached(const MultiHeadAttention2& attn,
                                            const TensorNode& x_new, BlockKvCache& cache);

/// One block's cached forward.
[[nodiscard]] TensorNode block_forward_cached(const TransformerBlock2& block, const TensorNode& x,
                                              BlockKvCache& cache);

// =============================================================================
// Gpt2
// =============================================================================

class Gpt2 : public Trainable {
   public:
    // Declaration order is initialization order, and every constructor here
    // draws from the same InitRng. The reference builds the blocks first, so
    // they must be declared first for the weights to come out identical.
    std::vector<TransformerBlock2> blocks;
    Embedding2 embed;
    LayerNorm2 ln_final;
    Linear2 lm_head;  // d_model -> vocab_size
    Config config;

    Gpt2(Config config, InitRng& rng);

    /// token_ids -> logits [T, vocab_size].
    [[nodiscard]] TensorNode forward(const std::vector<std::size_t>& token_ids) const;

    /// Mean cross-entropy for next-token prediction, as a 1x1 scalar node.
    ///
    /// Softmax and cross-entropy are fused: only the loss scalar is needed, so
    /// the [T, V] softmax never becomes a graph node. The gradient is
    /// `(softmax(logits) - one_hot(target)) / T`.
    [[nodiscard]] TensorNode loss(const std::vector<std::size_t>& token_ids,
                                  const std::vector<std::size_t>& targets) const;

    /// Point `lm_head.weight` at `embed.token_embed`.
    ///
    /// The embedding maps a token id to a d_model vector and the head maps a
    /// d_model vector back to a per-token logit: two views of the same
    /// representation. Sharing them removes vocab_size * d_model parameters
    /// (~25M for GPT-2), keeps the two consistent, and is what GPT-2, LLaMA and
    /// Mistral all do.
    void tie_weights();

    /// Generate with the KV cache -- O(T) per step.
    /// `temperature <= 0` is greedy.
    [[nodiscard]] std::vector<std::size_t> generate_cached(
        const std::vector<std::size_t>& token_ids, std::size_t max_new, float temperature) const;

    /// Same, with top-k sampling and a per-token callback for streaming.
    /// `top_k == 0` samples over the full vocabulary.
    void generate_cached_streaming(const std::vector<std::size_t>& token_ids, std::size_t max_new,
                                   float temperature, std::size_t top_k,
                                   const std::function<void(std::size_t)>& callback) const;

    [[nodiscard]] std::vector<TensorNode> parameters() const override;
    [[nodiscard]] TensorNode forward_tokens(
        const std::vector<std::size_t>& token_ids) const override;
    [[nodiscard]] TensorNode loss_tokens(
        const std::vector<std::size_t>& token_ids,
        const std::vector<std::size_t>& targets) const override;
};

/// Greedy or temperature-scaled argmax over a logits row.
[[nodiscard]] std::size_t gpt2_sample_token(const Mat& logits, std::size_t pos, float temperature);

/// Temperature plus top-k sampling over a logits row.
[[nodiscard]] std::size_t gpt2_sample_token_topk(const Mat& logits, std::size_t pos,
                                                 float temperature, std::size_t top_k);

}  // namespace rt

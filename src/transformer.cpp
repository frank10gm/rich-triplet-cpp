#include "rt/transformer.hpp"

#include <cassert>
#include <cmath>

namespace rt {

// =============================================================================
// Config
// =============================================================================

Config Config::nano(std::size_t vocab_size) {
    Config c;
    c.vocab_size = vocab_size;
    c.context_length = 64;
    c.d_model = 64;
    c.n_layers = 2;
    c.n_heads = 2;
    return c;
}

std::size_t Config::d_head() const {
    assert(d_model % n_heads == 0 && "d_model must be divisible by n_heads");
    return d_model / n_heads;
}

std::size_t Config::param_count() const {
    const std::size_t embed = vocab_size * d_model + context_length * d_model;
    const std::size_t attn_per_layer = 4 * d_model * d_model;      // Q, K, V, O
    const std::size_t ffn_per_layer = 2 * d_model * 4 * d_model;   // two linears
    const std::size_t norm_per_layer = 2 * 2 * d_model;            // two LayerNorms
    const std::size_t blocks = n_layers * (attn_per_layer + ffn_per_layer + norm_per_layer);
    const std::size_t output_head = d_model * vocab_size;
    return embed + blocks + output_head;
}

// =============================================================================
// Embedding
// =============================================================================

namespace {
/// A table of `rows` vectors, each `cols` samples from N(0, std_dev).
[[nodiscard]] std::vector<std::vector<Value>> random_table(std::size_t rows, std::size_t cols,
                                                           float std_dev, InitRng& rng) {
    std::vector<std::vector<Value>> table;
    table.reserve(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        std::vector<Value> row;
        row.reserve(cols);
        for (float v : rng.normal_vec(cols, std_dev)) {
            row.emplace_back(v);
        }
        table.push_back(std::move(row));
    }
    return table;
}

/// Flatten a table of vectors into one parameter list.
void append_table(std::vector<Value>& out, const std::vector<std::vector<Value>>& table) {
    for (const auto& row : table) {
        out.insert(out.end(), row.begin(), row.end());
    }
}

/// Element-wise sum of two same-length vectors of nodes.
[[nodiscard]] std::vector<Value> add_vectors(const std::vector<Value>& a,
                                             const std::vector<Value>& b) {
    std::vector<Value> out;
    out.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out.push_back(a[i].add(b[i]));
    }
    return out;
}

/// Apply `f` to every token vector in a sequence.
template <typename F>
[[nodiscard]] std::vector<std::vector<Value>> map_tokens(
    const std::vector<std::vector<Value>>& x, F&& f) {
    std::vector<std::vector<Value>> out;
    out.reserve(x.size());
    for (const auto& xi : x) {
        out.push_back(f(xi));
    }
    return out;
}
}  // namespace

Embedding::Embedding(const Config& config, InitRng& rng)
    : token_embed(random_table(config.vocab_size, config.d_model, 0.02f, rng)),
      pos_embed(random_table(config.context_length, config.d_model, 0.01f, rng)),
      config(config) {}

std::vector<std::vector<Value>> Embedding::forward(
    const std::vector<std::size_t>& token_ids) const {
    std::vector<std::vector<Value>> out;
    out.reserve(token_ids.size());
    for (std::size_t pos = 0; pos < token_ids.size(); ++pos) {
        out.push_back(add_vectors(token_embed[token_ids[pos]], pos_embed[pos]));
    }
    return out;
}

std::vector<Value> Embedding::parameters() const {
    std::vector<Value> p;
    append_table(p, token_embed);
    append_table(p, pos_embed);
    return p;
}

// =============================================================================
// AttentionHead
// =============================================================================

AttentionHead::AttentionHead(std::size_t d_model, std::size_t d_head_, InitRng& rng)
    : w_q(d_model, d_head_, rng),
      w_k(d_model, d_head_, rng),
      w_v(d_model, d_head_, rng),
      d_head(d_head_) {}

std::vector<std::vector<Value>> AttentionHead::forward(
    const std::vector<std::vector<Value>>& x) const {
    const std::size_t t = x.size();
    const float scale = std::sqrt(static_cast<float>(d_head));

    const auto queries = map_tokens(x, [this](const auto& xi) { return w_q.forward(xi); });
    const auto keys = map_tokens(x, [this](const auto& xi) { return w_k.forward(xi); });
    const auto values = map_tokens(x, [this](const auto& xi) { return w_v.forward(xi); });

    // weights[i] is the attention distribution for query i.
    std::vector<std::vector<Value>> weights;
    weights.reserve(t);
    for (std::size_t i = 0; i < t; ++i) {
        std::vector<Value> raw_scores;
        raw_scores.reserve(t);
        for (std::size_t j = 0; j < t; ++j) {
            if (j > i) {
                // Causal mask. -1e9 rather than -infinity: exp() underflows to
                // zero cleanly, where -infinity would give NaN in the gradient.
                raw_scores.emplace_back(-1e9f);
                continue;
            }
            Value dot = queries[i][0].mul(keys[j][0]);
            for (std::size_t d = 1; d < d_head; ++d) {
                dot = dot.add(queries[i][d].mul(keys[j][d]));
            }
            raw_scores.push_back(dot.mul(Value(1.0f / scale)));
        }
        weights.push_back(softmax(raw_scores));
    }

    // out[i][d] = sum_j weights[i][j] * values[j][d]
    std::vector<std::vector<Value>> out;
    out.reserve(t);
    for (std::size_t i = 0; i < t; ++i) {
        std::vector<Value> row;
        row.reserve(d_head);
        for (std::size_t d = 0; d < d_head; ++d) {
            Value acc = weights[i][0].mul(values[0][d]);
            for (std::size_t j = 1; j < t; ++j) {
                acc = acc.add(weights[i][j].mul(values[j][d]));
            }
            row.push_back(acc);
        }
        out.push_back(std::move(row));
    }
    return out;
}

std::vector<Value> AttentionHead::parameters() const {
    std::vector<Value> p = w_q.parameters();
    for (const Linear* l : {&w_k, &w_v}) {
        const std::vector<Value> q = l->parameters();
        p.insert(p.end(), q.begin(), q.end());
    }
    return p;
}

// =============================================================================
// MultiHeadAttention
// =============================================================================

namespace {
/// Build `n_heads` attention heads, drawing from `rng` in order.
///
/// The reference draws the heads before the output projection, so this has to
/// run during member initialization of `heads` -- which C++ orders by
/// declaration, not by the initializer list.
[[nodiscard]] std::vector<AttentionHead> make_heads(const Config& cfg, InitRng& rng) {
    std::vector<AttentionHead> heads;
    heads.reserve(cfg.n_heads);
    for (std::size_t i = 0; i < cfg.n_heads; ++i) {
        heads.emplace_back(cfg.d_model, cfg.d_head(), rng);
    }
    return heads;
}
}  // namespace

MultiHeadAttention::MultiHeadAttention(const Config& config, InitRng& rng)
    : heads(make_heads(config, rng)),
      w_o(config.d_model, config.d_model, rng),
      n_heads(config.n_heads),
      d_model(config.d_model) {}

std::vector<std::vector<Value>> MultiHeadAttention::forward(
    const std::vector<std::vector<Value>>& x) const {
    const std::size_t t = x.size();

    // Heads run one after another here; on a GPU they run in parallel.
    std::vector<std::vector<std::vector<Value>>> head_outputs;
    head_outputs.reserve(heads.size());
    for (const AttentionHead& h : heads) {
        head_outputs.push_back(h.forward(x));
    }

    // Concatenate along the feature dimension, then project.
    std::vector<std::vector<Value>> out;
    out.reserve(t);
    for (std::size_t i = 0; i < t; ++i) {
        std::vector<Value> concatenated;
        concatenated.reserve(d_model);
        for (std::size_t h = 0; h < heads.size(); ++h) {
            concatenated.insert(concatenated.end(), head_outputs[h][i].begin(),
                                head_outputs[h][i].end());
        }
        out.push_back(w_o.forward(concatenated));
    }
    return out;
}

std::vector<Value> MultiHeadAttention::parameters() const {
    std::vector<Value> p;
    for (const AttentionHead& h : heads) {
        const std::vector<Value> hp = h.parameters();
        p.insert(p.end(), hp.begin(), hp.end());
    }
    const std::vector<Value> op = w_o.parameters();
    p.insert(p.end(), op.begin(), op.end());
    return p;
}

// =============================================================================
// TransformerBlock
// =============================================================================

TransformerBlock::TransformerBlock(const Config& config, InitRng& rng)
    : ln1(config.d_model), attn(config, rng), ln2(config.d_model), mlp(config.d_model, rng) {}

std::vector<std::vector<Value>> TransformerBlock::forward(
    const std::vector<std::vector<Value>>& x) const {
    const std::size_t t = x.size();

    // Attention sub-block: pre-norm, attend, add the residual.
    const auto attn_out = attn.forward(map_tokens(x, [this](const auto& xi) {
        return ln1.forward(xi);
    }));
    std::vector<std::vector<Value>> x_after_attn;
    x_after_attn.reserve(t);
    for (std::size_t i = 0; i < t; ++i) {
        x_after_attn.push_back(add_vectors(x[i], attn_out[i]));
    }

    // MLP sub-block: pre-norm, transform, add the residual.
    const auto mlp_out = map_tokens(
        map_tokens(x_after_attn, [this](const auto& xi) { return ln2.forward(xi); }),
        [this](const auto& xi) { return mlp.forward(xi); });

    std::vector<std::vector<Value>> out;
    out.reserve(t);
    for (std::size_t i = 0; i < t; ++i) {
        out.push_back(add_vectors(x_after_attn[i], mlp_out[i]));
    }
    return out;
}

std::vector<Value> TransformerBlock::parameters() const {
    std::vector<Value> p = ln1.parameters();
    for (const Module* m : {static_cast<const Module*>(&attn), static_cast<const Module*>(&ln2),
                            static_cast<const Module*>(&mlp)}) {
        const std::vector<Value> q = m->parameters();
        p.insert(p.end(), q.begin(), q.end());
    }
    return p;
}

// =============================================================================
// Gpt
// =============================================================================

namespace {
/// Build `n_layers` blocks, drawing from `rng` in order.
[[nodiscard]] std::vector<TransformerBlock> make_blocks(const Config& cfg, InitRng& rng) {
    std::vector<TransformerBlock> blocks;
    blocks.reserve(cfg.n_layers);
    for (std::size_t i = 0; i < cfg.n_layers; ++i) {
        blocks.emplace_back(cfg, rng);
    }
    return blocks;
}
}  // namespace

Gpt::Gpt(Config cfg, InitRng& rng)
    : blocks(make_blocks(cfg, rng)),
      embed(cfg, rng),
      ln_final(cfg.d_model),
      lm_head(cfg.d_model, cfg.vocab_size, rng),
      config(std::move(cfg)) {}

std::vector<std::vector<Value>> Gpt::forward(const std::vector<std::size_t>& token_ids) const {
    assert(token_ids.size() <= config.context_length && "sequence exceeds context_length");

    std::vector<std::vector<Value>> x = embed.forward(token_ids);
    for (const TransformerBlock& block : blocks) {
        x = block.forward(x);
    }
    return map_tokens(map_tokens(x, [this](const auto& xi) { return ln_final.forward(xi); }),
                      [this](const auto& xi) { return lm_head.forward(xi); });
}

Value Gpt::loss(const std::vector<std::size_t>& token_ids,
                const std::vector<std::size_t>& targets) const {
    const auto logits = forward(token_ids);
    const std::size_t t = logits.size();
    assert(t == targets.size());

    // Sum of -log(p[correct]) over the sequence.
    Value loss_sum = softmax(logits[0])[targets[0]].ln().neg();
    for (std::size_t i = 1; i < t; ++i) {
        loss_sum = loss_sum.add(softmax(logits[i])[targets[i]].ln().neg());
    }
    return loss_sum.mul(Value(1.0f / static_cast<float>(t)));
}

std::vector<Value> Gpt::parameters() const {
    std::vector<Value> p = embed.parameters();
    for (const TransformerBlock& block : blocks) {
        const std::vector<Value> bp = block.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    for (const Module* m : {static_cast<const Module*>(&ln_final),
                            static_cast<const Module*>(&lm_head)}) {
        const std::vector<Value> q = m->parameters();
        p.insert(p.end(), q.begin(), q.end());
    }
    return p;
}

}  // namespace rt

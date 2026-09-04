#include "rt/transformer2.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>

namespace rt {

// =============================================================================
// Embedding2
// =============================================================================

Embedding2::Embedding2(const Config& config, InitRng& rng)
    : token_embed(TensorNode::leaf(Mat(rng.normal_vec(config.vocab_size * config.d_model, 0.02f),
                                       config.vocab_size, config.d_model))),
      pos_embed(TensorNode::leaf(Mat(
          rng.normal_vec(config.context_length * config.d_model, 0.01f), config.context_length,
          config.d_model))),
      vocab_size(config.vocab_size),
      context_length(config.context_length),
      d_model(config.d_model) {}

TensorNode Embedding2::forward(const std::vector<std::size_t>& token_ids) const {
    const std::size_t t = token_ids.size();
    const std::size_t d = d_model;

    const Mat& te = token_embed.data();
    const Mat& pe = pos_embed.data();

    TensorNode out = TensorNode::leaf(Mat::from_fn(t, d, [&](std::size_t row, std::size_t col) {
        return te.at(token_ids[row], col) + pe.at(row, col);
    }));

    TensorNode te_node = token_embed, pe_node = pos_embed, out_c = out;
    const std::vector<std::size_t> ids = token_ids;
    out.set_backward(
        [te_node, pe_node, out_c, ids, d] {
            const Mat& dout = out_c.grad();

            // Scatter each output row back into the token row it came from...
            Mat dte = te_node.grad();
            for (std::size_t row = 0; row < ids.size(); ++row) {
                for (std::size_t col = 0; col < d; ++col) {
                    dte.at_mut(ids[row], col) += dout.at(row, col);
                }
            }
            te_node.set_grad(std::move(dte));

            // ...and into the matching position row.
            Mat dpe = pe_node.grad();
            for (std::size_t row = 0; row < ids.size(); ++row) {
                for (std::size_t col = 0; col < d; ++col) {
                    dpe.at_mut(row, col) += dout.at(row, col);
                }
            }
            pe_node.set_grad(std::move(dpe));
        },
        {token_embed, pos_embed});
    return out;
}

std::vector<TensorNode> Embedding2::parameters() const { return {token_embed, pos_embed}; }

// =============================================================================
// AttentionHead2
// =============================================================================

AttentionHead2::AttentionHead2(std::size_t d_model, std::size_t d_head_, InitRng& rng)
    : w_q(d_model, d_head_, rng),
      w_k(d_model, d_head_, rng),
      w_v(d_model, d_head_, rng),
      d_head(d_head_) {}

TensorNode AttentionHead2::forward(const TensorNode& x) const {
    return TensorNode::flash_attention(w_q.forward(x), w_k.forward(x), w_v.forward(x), d_head);
}

namespace {
/// Concatenate the parameter lists of several modules.
template <typename... Modules>
[[nodiscard]] std::vector<TensorNode> collect_params(const Modules&... modules) {
    std::vector<TensorNode> p;
    const auto append = [&p](const auto& m) {
        const std::vector<TensorNode> q = m.parameters();
        p.insert(p.end(), q.begin(), q.end());
    };
    (append(modules), ...);
    return p;
}
}  // namespace

std::vector<TensorNode> AttentionHead2::parameters() const {
    return collect_params(w_q, w_k, w_v);
}

// =============================================================================
// MultiHeadAttention2
// =============================================================================

namespace {
/// Build `n_heads` attention heads, drawing from `rng` in order.
///
/// The reference draws the heads before the output projection, so this has to
/// run during member initialization of `heads` -- which C++ orders by
/// declaration, not by the initializer list.
[[nodiscard]] std::vector<AttentionHead2> make_heads(const Config& cfg, InitRng& rng) {
    std::vector<AttentionHead2> heads;
    heads.reserve(cfg.n_heads);
    for (std::size_t i = 0; i < cfg.n_heads; ++i) {
        heads.emplace_back(cfg.d_model, cfg.d_head(), rng);
    }
    return heads;
}
}  // namespace

MultiHeadAttention2::MultiHeadAttention2(const Config& config, InitRng& rng)
    : heads(make_heads(config, rng)),
      w_o(config.d_model, config.d_model, rng),
      d_model(config.d_model) {}

TensorNode MultiHeadAttention2::concat_heads(const std::vector<TensorNode>& heads, std::size_t t,
                                             std::size_t n, std::size_t dh) {
    TensorNode concat = TensorNode::leaf(
        Mat::from_fn(t, n * dh, [&](std::size_t row, std::size_t col) {
            return heads[col / dh].data().at(row, col % dh);
        }));

    const std::vector<TensorNode> heads_c = heads;
    TensorNode concat_c = concat;
    concat.set_backward(
        [heads_c, concat_c, t, dh] {
            const Mat& dconcat = concat_c.grad();
            for (std::size_t h = 0; h < heads_c.size(); ++h) {
                Mat dh_grad = heads_c[h].grad();
                for (std::size_t row = 0; row < t; ++row) {
                    for (std::size_t c = 0; c < dh; ++c) {
                        dh_grad.at_mut(row, c) += dconcat.at(row, h * dh + c);
                    }
                }
                heads_c[h].set_grad(std::move(dh_grad));
            }
        },
        heads);
    return concat;
}

TensorNode MultiHeadAttention2::forward(const TensorNode& x) const {
    const std::size_t t = x.data().rows;
    const std::size_t n = heads.size();
    const std::size_t dh = d_model / n;

    // Project each head, then concatenate so all heads go through one batched
    // attention call rather than a sequential loop.
    std::vector<TensorNode> qs, ks, vs;
    qs.reserve(n);
    ks.reserve(n);
    vs.reserve(n);
    for (const AttentionHead2& h : heads) {
        qs.push_back(h.w_q.forward(x));
        ks.push_back(h.w_k.forward(x));
        vs.push_back(h.w_v.forward(x));
    }

    // n_q == n_kv == n_heads, so the group size is 1.
    const TensorNode attn_out = TensorNode::batched_gqa_attention(
        concat_heads(qs, t, n, dh), concat_heads(ks, t, n, dh), concat_heads(vs, t, n, dh), n, n,
        dh);

    return w_o.forward(attn_out);
}

std::vector<TensorNode> MultiHeadAttention2::parameters() const {
    std::vector<TensorNode> p;
    for (const AttentionHead2& h : heads) {
        const std::vector<TensorNode> hp = h.parameters();
        p.insert(p.end(), hp.begin(), hp.end());
    }
    const std::vector<TensorNode> op = w_o.parameters();
    p.insert(p.end(), op.begin(), op.end());
    return p;
}

// =============================================================================
// TransformerBlock2
// =============================================================================

TransformerBlock2::TransformerBlock2(const Config& config, InitRng& rng)
    : ln1(config.d_model), attn(config, rng), ln2(config.d_model), mlp(config.d_model, rng) {}

TensorNode TransformerBlock2::forward(const TensorNode& x) const {
    const TensorNode x2 = x.add(attn.forward(ln1.forward(x)));
    return x2.add(mlp.forward(ln2.forward(x2)));
}

std::vector<TensorNode> TransformerBlock2::parameters() const {
    return collect_params(ln1, attn, ln2, mlp);
}

// =============================================================================
// KV cache
// =============================================================================

HeadKvCache::HeadKvCache(std::size_t d_head, std::size_t max_len)
    : k(Mat::zeros(0, d_head)), v(Mat::zeros(0, d_head)), d_head_(d_head), max_len_(max_len) {}

namespace {
/// Append `row` to `m`, dropping the oldest row once `max_len` is reached.
void append_row(Mat& m, std::span<const float> row, std::size_t d_head, std::size_t max_len) {
    assert(row.size() == d_head);
    const std::size_t start = m.rows >= max_len ? d_head : 0;

    std::vector<float> new_data;
    new_data.reserve(max_len * d_head);
    new_data.insert(new_data.end(), m.data.begin() + static_cast<std::ptrdiff_t>(start),
                    m.data.end());
    new_data.insert(new_data.end(), row.begin(), row.end());

    const std::size_t new_rows = new_data.size() / d_head;
    m = Mat(std::move(new_data), new_rows, d_head);
}
}  // namespace

void HeadKvCache::append_k(std::span<const float> row) {
    append_row(k, row, d_head_, max_len_);
}

void HeadKvCache::append_v(std::span<const float> row) {
    append_row(v, row, d_head_, max_len_);
}

BlockKvCache::BlockKvCache(std::size_t n_heads, std::size_t d_head, std::size_t max_len) {
    heads.reserve(n_heads);
    for (std::size_t i = 0; i < n_heads; ++i) {
        heads.emplace_back(d_head, max_len);
    }
}

Gpt2KvCache::Gpt2KvCache(const Config& config) {
    blocks.reserve(config.n_layers);
    for (std::size_t i = 0; i < config.n_layers; ++i) {
        blocks.emplace_back(config.n_heads, config.d_head(), config.context_length);
    }
}

TensorNode mha_forward_cached(const MultiHeadAttention2& attn, const TensorNode& x_new,
                              BlockKvCache& cache) {
    const std::size_t d = attn.d_model;
    const std::size_t n_heads = attn.heads.size();
    const std::size_t dh = d / n_heads;
    const std::size_t t_new = x_new.data().rows;  // 1 while decoding, T during prefill

    std::vector<TensorNode> head_outs;
    head_outs.reserve(n_heads);

    for (std::size_t h = 0; h < n_heads; ++h) {
        const AttentionHead2& head = attn.heads[h];
        const TensorNode q_new = head.w_q.forward(x_new);
        const TensorNode k_new = head.w_k.forward(x_new);
        const TensorNode v_new = head.w_v.forward(x_new);

        for (std::size_t r = 0; r < t_new; ++r) {
            cache.heads[h].append_k(k_new.data().row(r));
            cache.heads[h].append_v(v_new.data().row(r));
        }

        // No causal mask: the cache holds only past tokens, so every cached
        // key is valid context for this query.
        const Mat& k_full = cache.heads[h].k;
        const Mat& v_full = cache.heads[h].v;
        const std::size_t t_total = k_full.rows;
        const float scale = std::sqrt(static_cast<float>(dh));

        const Mat scores = Mat::from_fn(t_new, t_total, [&](std::size_t tq, std::size_t tk) {
            float dot = 0.0f;
            for (std::size_t j = 0; j < dh; ++j) {
                dot += q_new.data().at(tq, j) * k_full.at(tk, j);
            }
            return dot / scale;
        });

        Mat weights = Mat::zeros(t_new, t_total);
        for (std::size_t r = 0; r < t_new; ++r) {
            float mx = -std::numeric_limits<float>::infinity();
            for (std::size_t c = 0; c < t_total; ++c) {
                mx = std::max(mx, scores.at(r, c));
            }
            float sum = 0.0f;
            for (std::size_t c = 0; c < t_total; ++c) {
                weights.at_mut(r, c) = std::exp(scores.at(r, c) - mx);
                sum += weights.at(r, c);
            }
            for (std::size_t c = 0; c < t_total; ++c) {
                weights.at_mut(r, c) /= sum;
            }
        }

        head_outs.push_back(TensorNode::leaf(
            Mat::from_fn(t_new, dh, [&](std::size_t tq, std::size_t j) {
                float acc = 0.0f;
                for (std::size_t tk = 0; tk < t_total; ++tk) {
                    acc += weights.at(tq, tk) * v_full.at(tk, j);
                }
                return acc;
            })));
    }

    const TensorNode concat = TensorNode::leaf(
        Mat::from_fn(t_new, d, [&](std::size_t row, std::size_t col) {
            return head_outs[col / dh].data().at(row, col % dh);
        }));
    return attn.w_o.forward(concat);
}

TensorNode block_forward_cached(const TransformerBlock2& block, const TensorNode& x,
                                BlockKvCache& cache) {
    const TensorNode x2 = x.add(mha_forward_cached(block.attn, block.ln1.forward(x), cache));
    return x2.add(block.mlp.forward(block.ln2.forward(x2)));
}

// =============================================================================
// Gpt2
// =============================================================================

namespace {
/// Build `n_layers` blocks, drawing from `rng` in order.
[[nodiscard]] std::vector<TransformerBlock2> make_blocks(const Config& cfg, InitRng& rng) {
    std::vector<TransformerBlock2> blocks;
    blocks.reserve(cfg.n_layers);
    for (std::size_t i = 0; i < cfg.n_layers; ++i) {
        blocks.emplace_back(cfg, rng);
    }
    return blocks;
}
}  // namespace

Gpt2::Gpt2(Config cfg, InitRng& rng)
    : blocks(make_blocks(cfg, rng)),
      embed(cfg, rng),
      ln_final(cfg.d_model),
      lm_head(cfg.d_model, cfg.vocab_size, rng),
      config(std::move(cfg)) {}

TensorNode Gpt2::forward(const std::vector<std::size_t>& token_ids) const {
    assert(token_ids.size() <= config.context_length && "sequence exceeds context_length");

    TensorNode x = embed.forward(token_ids);
    for (const TransformerBlock2& block : blocks) {
        x = block.forward(x);
    }
    return lm_head.forward(ln_final.forward(x));
}

TensorNode Gpt2::loss(const std::vector<std::size_t>& token_ids,
                      const std::vector<std::size_t>& targets) const {
    const TensorNode logits_node = forward(token_ids);
    const Mat& logits = logits_node.data();
    const std::size_t t = logits.rows;
    const std::size_t v = logits.cols;
    assert(t == targets.size());

    Mat probs = Mat::zeros(t, v);
    float loss_val = 0.0f;
    for (std::size_t r = 0; r < t; ++r) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < v; ++c) {
            row_max = std::max(row_max, logits.at(r, c));
        }
        float sum_exp = 0.0f;
        for (std::size_t c = 0; c < v; ++c) {
            probs.at_mut(r, c) = std::exp(logits.at(r, c) - row_max);
            sum_exp += probs.at(r, c);
        }
        for (std::size_t c = 0; c < v; ++c) {
            probs.at_mut(r, c) /= sum_exp;
        }
        loss_val += -std::log(probs.at(r, targets[r]));
    }
    loss_val /= static_cast<float>(t);

    TensorNode loss = TensorNode::leaf(Mat({loss_val}, 1, 1));
    TensorNode logits_c = logits_node;
    const std::vector<std::size_t> targets_v = targets;

    loss.set_backward(
        [logits_c, probs = std::move(probs), targets_v, t] {
            // d loss / d logits[r, j] = (p[r,j] - one_hot(j == target[r])) / T
            Mat dlogits = probs;
            for (std::size_t r = 0; r < t; ++r) {
                dlogits.at_mut(r, targets_v[r]) -= 1.0f;
            }
            logits_c.set_grad(logits_c.grad().add(dlogits.scale(1.0f / static_cast<float>(t))));
        },
        {logits_node});
    return loss;
}

void Gpt2::tie_weights() {
    assert(lm_head.weight.data().rows == embed.token_embed.data().rows &&
           lm_head.weight.data().cols == embed.token_embed.data().cols &&
           "tie_weights: lm_head.weight and embed.token_embed have different shapes");
    lm_head.weight = embed.token_embed;
}

namespace {
/// Prefill the cache with the prompt and return the resulting logits.
Mat prefill_cached(const Gpt2& model, const std::vector<std::size_t>& token_ids,
                   Gpt2KvCache& cache) {
    TensorNode x = model.embed.forward(token_ids);
    for (std::size_t bi = 0; bi < model.blocks.size(); ++bi) {
        x = block_forward_cached(model.blocks[bi], x, cache.blocks[bi]);
    }
    return model.lm_head.forward(model.ln_final.forward(x)).data();
}

/// Run one token through every block and return its logits.
Mat decode_cached(const Gpt2& model, std::size_t token, Gpt2KvCache& cache) {
    TensorNode x = model.embed.forward({token});
    for (std::size_t bi = 0; bi < model.blocks.size(); ++bi) {
        x = block_forward_cached(model.blocks[bi], x, cache.blocks[bi]);
    }
    return model.lm_head.forward(model.ln_final.forward(x)).data();
}
}  // namespace

std::vector<std::size_t> Gpt2::generate_cached(const std::vector<std::size_t>& token_ids,
                                               std::size_t max_new, float temperature) const {
    Gpt2KvCache cache(config);

    const Mat first_logits = prefill_cached(*this, token_ids, cache);
    std::vector<std::size_t> generated;
    generated.reserve(max_new);

    std::size_t prev_tok = gpt2_sample_token(first_logits, token_ids.size() - 1, temperature);
    generated.push_back(prev_tok);

    for (std::size_t i = 1; i < max_new; ++i) {
        prev_tok = gpt2_sample_token(decode_cached(*this, prev_tok, cache), 0, temperature);
        generated.push_back(prev_tok);
    }
    return generated;
}

void Gpt2::generate_cached_streaming(const std::vector<std::size_t>& token_ids,
                                     std::size_t max_new, float temperature, std::size_t top_k,
                                     const std::function<void(std::size_t)>& callback) const {
    Gpt2KvCache cache(config);

    const Mat first_logits = prefill_cached(*this, token_ids, cache);
    std::size_t prev_tok =
        gpt2_sample_token_topk(first_logits, token_ids.size() - 1, temperature, top_k);
    callback(prev_tok);

    for (std::size_t i = 1; i < max_new; ++i) {
        prev_tok = gpt2_sample_token_topk(decode_cached(*this, prev_tok, cache), 0, temperature,
                                          top_k);
        callback(prev_tok);
    }
}

std::vector<TensorNode> Gpt2::parameters() const {
    std::vector<TensorNode> p = embed.parameters();
    for (const TransformerBlock2& block : blocks) {
        const std::vector<TensorNode> bp = block.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    const std::vector<TensorNode> tail = collect_params(ln_final, lm_head);
    p.insert(p.end(), tail.begin(), tail.end());
    return p;
}

TensorNode Gpt2::forward_tokens(const std::vector<std::size_t>& token_ids) const {
    return forward(token_ids);
}

TensorNode Gpt2::loss_tokens(const std::vector<std::size_t>& token_ids,
                             const std::vector<std::size_t>& targets) const {
    return loss(token_ids, targets);
}

// =============================================================================
// Samplers
// =============================================================================

namespace {
/// Argmax over a logits row.
[[nodiscard]] std::size_t argmax_row(const Mat& logits, std::size_t pos) {
    std::size_t best = 0;
    for (std::size_t c = 1; c < logits.cols; ++c) {
        if (logits.at(pos, best) < logits.at(pos, c)) {
            best = c;
        }
    }
    return best;
}

/// Temperature-scaled softmax over a logits row.
[[nodiscard]] std::vector<float> temperature_probs(const Mat& logits, std::size_t pos,
                                                   float temperature) {
    const std::size_t v = logits.cols;
    float row_max = -std::numeric_limits<float>::infinity();
    for (std::size_t c = 0; c < v; ++c) {
        row_max = std::max(row_max, logits.at(pos, c));
    }
    std::vector<float> probs(v);
    float sum = 0.0f;
    for (std::size_t c = 0; c < v; ++c) {
        probs[c] = std::exp((logits.at(pos, c) - row_max) / temperature);
        sum += probs[c];
    }
    for (float& p : probs) {
        p /= sum;
    }
    return probs;
}
}  // namespace

std::size_t gpt2_sample_token(const Mat& logits, std::size_t pos, float temperature) {
    if (temperature <= 0.0f) {
        return argmax_row(logits, pos);
    }
    // Temperature reshapes the distribution, then the most likely token wins:
    // this sampler is deterministic by design.
    const std::vector<float> probs = temperature_probs(logits, pos, temperature);
    return static_cast<std::size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

std::size_t gpt2_sample_token_topk(const Mat& logits, std::size_t pos, float temperature,
                                   std::size_t top_k) {
    const std::size_t v = logits.cols;
    if (temperature <= 0.0f) {
        return argmax_row(logits, pos);
    }

    std::vector<float> probs = temperature_probs(logits, pos, temperature);

    // Keep only the k most likely tokens, by thresholding at the k-th value.
    const std::size_t k = top_k == 0 ? v : std::min(top_k, v);
    std::vector<float> sorted = probs;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const float threshold = sorted[k - 1];

    std::vector<float> filtered(v);
    float fsum = 0.0f;
    for (std::size_t c = 0; c < v; ++c) {
        filtered[c] = probs[c] >= threshold ? probs[c] : 0.0f;
        fsum += filtered[c];
    }
    if (fsum > 0.0f) {
        for (float& p : filtered) {
            p /= fsum;
        }
    }

    // A process-wide counter drives the LCG, so successive calls draw
    // different samples without the caller managing RNG state.
    static std::atomic<std::uint64_t> counter{54321};
    const std::uint64_t seed = counter.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
    const float rand_val = static_cast<float>(s >> 33) / static_cast<float>(UINT32_MAX);

    float cumsum = 0.0f;
    for (std::size_t i = 0; i < v; ++i) {
        cumsum += filtered[i];
        if (rand_val <= cumsum) {
            return i;
        }
    }
    return static_cast<std::size_t>(std::max_element(filtered.begin(), filtered.end()) -
                                    filtered.begin());
}

}  // namespace rt

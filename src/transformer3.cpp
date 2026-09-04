#include "rt/transformer3.hpp"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>

namespace rt {

// =============================================================================
// Config3
// =============================================================================

namespace {
/// The fields both GPT-OSS sizes share.
Config3 gpt_oss_common() {
    Config3 c;
    c.vocab_size = 201088;
    c.num_key_value_heads = 8;
    c.experts_per_token = 4;
    c.max_position_embeddings = 131072;
    c.rope_theta = 150000.0f;
    c.rms_norm_eps = 1e-5f;
    c.swiglu_limit = 7.0f;
    c.sliding_window = 128;
    return c;
}
}  // namespace

Config3 Config3::gpt_oss_20b() {
    Config3 c = gpt_oss_common();
    c.hidden_size = 2880;
    c.num_hidden_layers = 24;
    c.num_attention_heads = 64;
    c.intermediate_size = 2880;
    c.num_local_experts = 32;
    return c;
}

Config3 Config3::gpt_oss_120b() {
    Config3 c = gpt_oss_common();
    c.hidden_size = 7168;
    c.num_hidden_layers = 36;
    c.num_attention_heads = 128;
    c.intermediate_size = 7168;
    c.num_local_experts = 128;
    return c;
}

std::size_t Config3::d_head() const {
    assert(hidden_size % num_attention_heads == 0 &&
           "hidden_size must be divisible by num_attention_heads");
    return hidden_size / num_attention_heads;
}

// =============================================================================
// KV cache
// =============================================================================

LayerKvCache::LayerKvCache(std::size_t n_kv_heads, std::size_t d_head, std::size_t max_seq_len)
    : k(Mat::zeros(max_seq_len, n_kv_heads * d_head)),
      v(Mat::zeros(max_seq_len, n_kv_heads * d_head)) {}

void LayerKvCache::append(const Mat& new_k, const Mat& new_v) {
    const std::size_t n_new = new_k.rows;
    const std::size_t d = new_k.cols;
    assert(d == k.cols);
    for (std::size_t r = 0; r < n_new; ++r) {
        for (std::size_t c = 0; c < d; ++c) {
            k.at_mut(seq_len + r, c) = new_k.at(r, c);
            v.at_mut(seq_len + r, c) = new_v.at(r, c);
        }
    }
    seq_len += n_new;
}

namespace {
/// The trailing `rows` rows of `m`, starting at `start`.
[[nodiscard]] Mat rows_from(const Mat& m, std::size_t start, std::size_t rows) {
    return Mat::from_fn(rows, m.cols,
                        [&](std::size_t r, std::size_t c) { return m.at(start + r, c); });
}
}  // namespace

Mat LayerKvCache::k_filled() const { return rows_from(k, 0, seq_len); }
Mat LayerKvCache::v_filled() const { return rows_from(v, 0, seq_len); }

Mat LayerKvCache::k_last(std::size_t window) const {
    const std::size_t start = seq_len > window ? seq_len - window : 0;
    return rows_from(k, start, seq_len - start);
}

Mat LayerKvCache::v_last(std::size_t window) const {
    const std::size_t start = seq_len > window ? seq_len - window : 0;
    return rows_from(v, start, seq_len - start);
}

KvCache::KvCache(const Config3& config) {
    layers.reserve(config.num_hidden_layers);
    for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers.emplace_back(config.num_key_value_heads, config.d_head(),
                            config.max_position_embeddings);
    }
}

void KvCache::clear() {
    for (LayerKvCache& layer : layers) {
        layer.seq_len = 0;
    }
}

// =============================================================================
// GptOssAttention
// =============================================================================

GptOssAttention::GptOssAttention(const Config3& config, InitRng& rng,
                                 std::optional<std::size_t> window)
    : q_proj(config.hidden_size, config.num_attention_heads * config.d_head(), rng),
      k_proj(config.hidden_size, config.num_key_value_heads * config.d_head(), rng),
      v_proj(config.hidden_size, config.num_key_value_heads * config.d_head(), rng),
      o_proj(config.num_attention_heads * config.d_head(), config.hidden_size, rng),
      n_q_heads(config.num_attention_heads),
      n_kv_heads(config.num_key_value_heads),
      d_head(config.d_head()),
      rope_theta(config.rope_theta),
      // Anything past the standard 4096 needs YaRN to stay coherent.
      use_yarn(config.max_position_embeddings > 4096),
      max_ctx(config.max_position_embeddings),
      sliding_window(window) {}

TensorNode GptOssAttention::apply_rope_to_all_heads(const TensorNode& x, std::size_t n_heads,
                                                    std::size_t t, std::size_t dh,
                                                    std::size_t seq_offset) const {
    const Mat& x_data = x.data();

    const float scale = static_cast<float>(max_ctx) / static_cast<float>(original_ctx);
    // YaRN's attention-temperature correction, and the interpolation factor for
    // the slow frequencies.
    const float mscale = use_yarn ? 0.1f * std::log(scale) + 1.0f : 1.0f;
    const float inv_scale = use_yarn ? 1.0f / scale : 1.0f;
    constexpr float beta_fast = 32.0f;
    constexpr float beta_slow = 1.0f;

    return TensorNode::leaf(
        Mat::from_fn(t, n_heads * dh, [&](std::size_t row, std::size_t col) {
            const std::size_t h = col / dh;
            const std::size_t dim = col % dh;
            const std::size_t pair = dim / 2;
            const bool is_odd = dim % 2 == 1;
            const auto pos = static_cast<float>(seq_offset + row);

            const float omega =
                1.0f / std::pow(rope_theta, 2.0f * static_cast<float>(pair) / static_cast<float>(dh));

            float effective_pos = pos;
            if (use_yarn) {
                // How many cycles this frequency completes over the trained
                // context decides how much its position is interpolated.
                const float cycles = static_cast<float>(original_ctx) * omega /
                                     (2.0f * std::numbers::pi_v<float>);
                float ramp;
                if (cycles < beta_slow) {
                    ramp = 0.0f;
                } else if (cycles > beta_fast) {
                    ramp = 1.0f;
                } else {
                    ramp = (cycles - beta_slow) / (beta_fast - beta_slow);
                }
                effective_pos = (1.0f - ramp) * pos * inv_scale + ramp * pos;
            }

            const float angle = effective_pos * omega * mscale;
            const float cos_a = std::cos(angle);
            const float sin_a = std::sin(angle);

            // Interleaved pairing: dimension 2i rotates with 2i + 1.
            const std::size_t base_col = h * dh + (dim & ~std::size_t{1});
            if (!is_odd) {
                return x_data.at(row, base_col) * cos_a - x_data.at(row, base_col + 1) * sin_a;
            }
            return x_data.at(row, base_col + 1) * cos_a + x_data.at(row, base_col) * sin_a;
        }));
}

namespace {
/// Softmax each row of `scores` in place, over its first `cols` entries.
void softmax_rows(Mat& w, const Mat& scores, std::size_t rows, std::size_t cols) {
    for (std::size_t r = 0; r < rows; ++r) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < cols; ++c) {
            row_max = std::max(row_max, scores.at(r, c));
        }
        float row_sum = 0.0f;
        for (std::size_t c = 0; c < cols; ++c) {
            w.at_mut(r, c) = std::exp(scores.at(r, c) - row_max);
            row_sum += w.at(r, c);
        }
        for (std::size_t c = 0; c < cols; ++c) {
            w.at_mut(r, c) /= row_sum;
        }
    }
}

/// Extract head `h`'s [rows, d_head] block from a [rows, H*d_head] matrix.
[[nodiscard]] Mat head_block(const Mat& m, std::size_t h, std::size_t d_head, std::size_t rows) {
    return Mat::from_fn(rows, d_head,
                        [&](std::size_t r, std::size_t c) { return m.at(r, h * d_head + c); });
}

/// Write a [rows, d_head] block back into head `h` of `dst`.
void write_head_block(Mat& dst, const Mat& src, std::size_t h, std::size_t d_head) {
    for (std::size_t r = 0; r < src.rows; ++r) {
        for (std::size_t c = 0; c < d_head; ++c) {
            dst.at_mut(r, h * d_head + c) = src.at(r, c);
        }
    }
}
}  // namespace

TensorNode GptOssAttention::forward(const TensorNode& x) const {
    const std::size_t t = x.data().rows;

    const TensorNode q_rope =
        apply_rope_to_all_heads(q_proj.forward(x), n_q_heads, t, d_head);
    const TensorNode k_rope =
        apply_rope_to_all_heads(k_proj.forward(x), n_kv_heads, t, d_head);
    const TensorNode v = v_proj.forward(x);

    if (!sliding_window) {
        return o_proj.forward(
            TensorNode::batched_gqa_attention(q_rope, k_rope, v, n_q_heads, n_kv_heads, d_head));
    }

    // Windowed: a token at i attends to j only when j <= i and i - j < window.
    const Mat& q_data = q_rope.data();
    const Mat& k_data = k_rope.data();
    const Mat& v_data = v.data();
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    const std::size_t group_size = n_q_heads / n_kv_heads;
    const std::size_t window = *sliding_window;

    Mat out_data = Mat::zeros(t, n_q_heads * d_head);
    for (std::size_t qh = 0; qh < n_q_heads; ++qh) {
        const std::size_t kvh = qh / group_size;
        const Mat q_h = head_block(q_data, qh, d_head, t);
        const Mat k_h = head_block(k_data, kvh, d_head, t);
        const Mat v_h = head_block(v_data, kvh, d_head, t);

        Mat scores = q_h.matmul(k_h.transpose()).scale(scale);
        for (std::size_t i = 0; i < t; ++i) {
            for (std::size_t j = 0; j < t; ++j) {
                if (j > i || (i - j) >= window) {
                    scores.at_mut(i, j) = -1e9f;
                }
            }
        }

        Mat w = Mat::zeros(t, t);
        softmax_rows(w, scores, t, t);
        write_head_block(out_data, w.matmul(v_h), qh, d_head);
    }

    return o_proj.forward(TensorNode::leaf(std::move(out_data)));
}

TensorNode GptOssAttention::forward_cached(const TensorNode& x, LayerKvCache& cache) const {
    const std::size_t n_new = x.data().rows;
    const std::size_t seq_offset = cache.seq_len;

    const TensorNode q_rope =
        apply_rope_to_all_heads(q_proj.forward(x), n_q_heads, n_new, d_head, seq_offset);
    const TensorNode k_rope =
        apply_rope_to_all_heads(k_proj.forward(x), n_kv_heads, n_new, d_head, seq_offset);
    const TensorNode v = v_proj.forward(x);

    cache.append(k_rope.data(), v.data());

    // Local layers see only the trailing window of the cache.
    const Mat k_full = sliding_window ? cache.k_last(*sliding_window) : cache.k_filled();
    const Mat v_full = sliding_window ? cache.v_last(*sliding_window) : cache.v_filled();

    // No causal mask is needed: the cache holds only past tokens.
    const Mat& q_data = q_rope.data();
    const std::size_t t_q = q_data.rows;
    const std::size_t t_kv = k_full.rows;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    const std::size_t group_size = n_q_heads / n_kv_heads;

    Mat out_data = Mat::zeros(t_q, n_q_heads * d_head);
    for (std::size_t qh = 0; qh < n_q_heads; ++qh) {
        const std::size_t kvh = qh / group_size;
        const Mat q_h = head_block(q_data, qh, d_head, t_q);
        const Mat k_h = head_block(k_full, kvh, d_head, t_kv);
        const Mat v_h = head_block(v_full, kvh, d_head, t_kv);

        const Mat scores = q_h.matmul(k_h.transpose()).scale(scale);
        Mat w = Mat::zeros(t_q, t_kv);
        softmax_rows(w, scores, t_q, t_kv);
        write_head_block(out_data, w.matmul(v_h), qh, d_head);
    }

    return o_proj.forward(TensorNode::leaf(std::move(out_data)));
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

std::vector<TensorNode> GptOssAttention::parameters() const {
    return collect_params(q_proj, k_proj, v_proj, o_proj);
}

// =============================================================================
// MoELayer
// =============================================================================

namespace {
/// Build the experts, drawing from `rng` in order. The reference draws the
/// router first, so this runs after it -- which C++ orders by declaration.
[[nodiscard]] std::vector<SwiGluMlp2> make_experts(const Config3& cfg, InitRng& rng) {
    std::vector<SwiGluMlp2> experts;
    experts.reserve(cfg.num_local_experts);
    for (std::size_t i = 0; i < cfg.num_local_experts; ++i) {
        experts.emplace_back(cfg.hidden_size, cfg.intermediate_size, cfg.swiglu_limit, rng);
    }
    return experts;
}
}  // namespace

MoELayer::MoELayer(const Config3& config, InitRng& rng)
    : router(config.hidden_size, config.num_local_experts, rng),
      experts(make_experts(config, rng)),
      num_experts(config.num_local_experts),
      experts_per_token(config.experts_per_token) {}

TensorNode MoELayer::forward(const TensorNode& x) const {
    const Mat& x_data = x.data();
    const std::size_t t = x_data.rows;
    const std::size_t d = x_data.cols;
    const std::size_t k = experts_per_token;

    const Mat rlogits = router.forward(x).data();  // [T, num_experts]
    Mat out_data = Mat::zeros(t, d);

    for (std::size_t row = 0; row < t; ++row) {
        // Softmax the router scores for this token.
        float row_max = -std::numeric_limits<float>::infinity();
        for (std::size_t e = 0; e < num_experts; ++e) {
            row_max = std::max(row_max, rlogits.at(row, e));
        }
        std::vector<float> probs(num_experts);
        float sum_exp = 0.0f;
        for (std::size_t e = 0; e < num_experts; ++e) {
            probs[e] = std::exp(rlogits.at(row, e) - row_max);
            sum_exp += probs[e];
        }
        for (float& p : probs) {
            p /= sum_exp;
        }

        std::vector<std::pair<std::size_t, float>> indexed(num_experts);
        for (std::size_t e = 0; e < num_experts; ++e) {
            indexed[e] = {e, probs[e]};
        }
        std::sort(indexed.begin(), indexed.end(),
                  [](const auto& a, const auto& b) { return b.second < a.second; });

        // The selected experts' weights are renormalized to sum to 1.
        float weight_sum = 0.0f;
        for (std::size_t i = 0; i < k; ++i) {
            weight_sum += indexed[i].second;
        }

        const TensorNode token_node = TensorNode::leaf(
            Mat::from_fn(1, d, [&](std::size_t, std::size_t c) { return x_data.at(row, c); }));

        for (std::size_t i = 0; i < k; ++i) {
            const auto [expert_idx, weight] = indexed[i];
            const Mat expert_out = experts[expert_idx].forward(token_node).data();
            const float normalized_weight = weight / weight_sum;
            for (std::size_t c = 0; c < d; ++c) {
                out_data.at_mut(row, c) += normalized_weight * expert_out.at(0, c);
            }
        }
    }

    return TensorNode::leaf(std::move(out_data));
}

std::vector<TensorNode> MoELayer::parameters() const {
    std::vector<TensorNode> p = router.parameters();
    for (const SwiGluMlp2& expert : experts) {
        const std::vector<TensorNode> ep = expert.parameters();
        p.insert(p.end(), ep.begin(), ep.end());
    }
    return p;
}

// =============================================================================
// GptOssBlock
// =============================================================================

GptOssBlock::GptOssBlock(const Config3& config, std::size_t layer_idx, InitRng& rng)
    : input_layernorm(config.hidden_size),
      // Even layers attend to the whole context, odd layers to a local window.
      self_attn(config, rng, layer_idx % 2 == 0 ? std::nullopt : config.sliding_window),
      post_attention_layernorm(config.hidden_size),
      mlp(config, rng) {}

TensorNode GptOssBlock::forward(const TensorNode& x) const {
    const TensorNode x2 = x.add(self_attn.forward(input_layernorm.forward(x)));
    return x2.add(mlp.forward(post_attention_layernorm.forward(x2)));
}

TensorNode GptOssBlock::forward_cached(const TensorNode& x, LayerKvCache& cache) const {
    const TensorNode x2 = x.add(self_attn.forward_cached(input_layernorm.forward(x), cache));
    return x2.add(mlp.forward(post_attention_layernorm.forward(x2)));
}

std::vector<TensorNode> GptOssBlock::parameters() const {
    return collect_params(input_layernorm, self_attn, post_attention_layernorm, mlp);
}

// =============================================================================
// GptOssModel
// =============================================================================

namespace {
[[nodiscard]] std::vector<GptOssBlock> make_layers(const Config3& cfg, InitRng& rng) {
    std::vector<GptOssBlock> layers;
    layers.reserve(cfg.num_hidden_layers);
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers.emplace_back(cfg, i, rng);
    }
    return layers;
}
}  // namespace

GptOssModel::GptOssModel(Config3 cfg, InitRng& rng)
    : embed_tokens(TensorNode::leaf(Mat(rng.normal_vec(cfg.vocab_size * cfg.hidden_size, 0.02f),
                                        cfg.vocab_size, cfg.hidden_size))),
      layers(make_layers(cfg, rng)),
      norm(cfg.hidden_size),
      lm_head(cfg.hidden_size, cfg.vocab_size, rng),
      config(std::move(cfg)) {}

namespace {
/// Gather embedding rows for `token_ids`. Position comes from RoPE, so no
/// positional embedding is added here.
[[nodiscard]] Mat gather_embeddings(const Mat& te, const std::vector<std::size_t>& token_ids,
                                    std::size_t d) {
    return Mat::from_fn(token_ids.size(), d, [&](std::size_t row, std::size_t col) {
        return te.at(token_ids[row], col);
    });
}
}  // namespace

TensorNode GptOssModel::forward(const std::vector<std::size_t>& token_ids) const {
    TensorNode x =
        TensorNode::leaf(gather_embeddings(embed_tokens.data(), token_ids, config.hidden_size));
    for (const GptOssBlock& layer : layers) {
        x = layer.forward(x);
    }
    return lm_head.forward(norm.forward(x));
}

std::vector<TensorNode> GptOssModel::forward_batch(
    const std::vector<std::vector<std::size_t>>& sequences) const {
    std::vector<TensorNode> out;
    out.reserve(sequences.size());
    for (const auto& seq : sequences) {
        out.push_back(forward(seq));
    }
    return out;
}

TensorNode GptOssModel::loss(const std::vector<std::size_t>& token_ids,
                             const std::vector<std::size_t>& targets) const {
    const TensorNode logits_node = forward(token_ids);
    const Mat& logits = logits_node.data();
    const std::size_t t = logits.rows;
    const std::size_t v = logits.cols;
    assert(t == targets.size() && "loss: token_ids len != targets len");

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
        loss_val -= std::log(probs.at(r, targets[r]));
    }
    loss_val /= static_cast<float>(t);

    TensorNode loss_node = TensorNode::leaf(Mat({loss_val}, 1, 1));
    TensorNode logits_c = logits_node;
    const std::vector<std::size_t> targets_v = targets;

    // d(loss)/d(logits[r,c]) = (p[r,c] - one_hot(c == target[r])) / T
    loss_node.set_backward(
        [logits_c, probs = std::move(probs), targets_v, t, v] {
            Mat dlogits = logits_c.grad();
            for (std::size_t r = 0; r < t; ++r) {
                for (std::size_t c = 0; c < v; ++c) {
                    const float indicator = c == targets_v[r] ? 1.0f : 0.0f;
                    dlogits.at_mut(r, c) += (probs.at(r, c) - indicator) / static_cast<float>(t);
                }
            }
            logits_c.set_grad(std::move(dlogits));
            logits_c.call_backward_fn();
        },
        {logits_node});
    return loss_node;
}

std::size_t GptOssModel::predict_next(const std::vector<std::size_t>& token_ids) const {
    const Mat logits = forward(token_ids).data();
    const std::size_t last = logits.rows - 1;
    std::size_t best = 0;
    for (std::size_t c = 1; c < logits.cols; ++c) {
        if (logits.at(last, best) < logits.at(last, c)) {
            best = c;
        }
    }
    return best;
}

// =============================================================================
// Sampling
// =============================================================================

std::size_t sample_token_full(const Mat& logits, std::size_t pos, const SamplingParams& params,
                              const std::vector<std::size_t>& seen_ids, LcgRng53& rng) {
    const std::size_t v = logits.cols;

    // Frequency counts drive all three penalties.
    std::vector<std::uint32_t> freq(v, 0);
    for (std::size_t tok : seen_ids) {
        if (tok < v) {
            ++freq[tok];
        }
    }

    std::vector<float> row(v);
    for (std::size_t c = 0; c < v; ++c) {
        row[c] = logits.at(pos, c);
    }

    // Dividing shrinks the logit toward zero whether it was positive or
    // negative, which is what makes this work on raw logits.
    if (params.repetition_penalty != 1.0f) {
        for (std::size_t tok = 0; tok < v; ++tok) {
            if (freq[tok] > 0) {
                row[tok] /= params.repetition_penalty;
            }
        }
    }
    if (params.frequency_penalty != 0.0f) {
        for (std::size_t tok = 0; tok < v; ++tok) {
            if (freq[tok] > 0) {
                row[tok] -= params.frequency_penalty * static_cast<float>(freq[tok]);
            }
        }
    }
    if (params.presence_penalty != 0.0f) {
        for (std::size_t tok = 0; tok < v; ++tok) {
            if (freq[tok] > 0) {
                row[tok] -= params.presence_penalty;
            }
        }
    }

    if (params.temperature <= 0.0f) {
        return static_cast<std::size_t>(std::max_element(row.begin(), row.end()) - row.begin());
    }

    for (float& x : row) {
        x /= params.temperature;
    }

    std::vector<std::size_t> candidates(v);
    for (std::size_t i = 0; i < v; ++i) {
        candidates[i] = i;
    }
    if (params.top_k > 0 && params.top_k < v) {
        std::sort(candidates.begin(), candidates.end(),
                  [&row](std::size_t a, std::size_t b) { return row[b] < row[a]; });
        candidates.resize(params.top_k);
    }

    float row_max = -std::numeric_limits<float>::infinity();
    for (std::size_t i : candidates) {
        row_max = std::max(row_max, row[i]);
    }
    std::vector<std::pair<std::size_t, float>> probs;
    probs.reserve(candidates.size());
    float sum = 0.0f;
    for (std::size_t i : candidates) {
        const float e = std::exp(row[i] - row_max);
        probs.emplace_back(i, e);
        sum += e;
    }
    for (auto& [tok, p] : probs) {
        p /= sum;
    }

    // Nucleus: keep the shortest prefix whose cumulative mass reaches top_p.
    if (params.top_p < 1.0f) {
        std::sort(probs.begin(), probs.end(),
                  [](const auto& a, const auto& b) { return b.second < a.second; });
        float cumsum = 0.0f;
        std::size_t cutoff = probs.size();
        for (std::size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i].second;
            if (cumsum >= params.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        probs.resize(cutoff);
        float total = 0.0f;
        for (const auto& [tok, p] : probs) {
            total += p;
        }
        for (auto& [tok, p] : probs) {
            p /= total;
        }
    }

    const float u = rng.next_f32();
    float cumsum = 0.0f;
    for (const auto& [tok, p] : probs) {
        cumsum += p;
        if (u < cumsum) {
            return tok;
        }
    }
    return probs.empty() ? 0 : probs.back().first;
}

namespace {
/// The legacy temperature-only sampler: greedy at or below zero, otherwise a
/// plain temperature draw with a fixed seed.
[[nodiscard]] std::size_t sample_token_simple(const Mat& logits, std::size_t pos,
                                              float temperature) {
    SamplingParams params;
    if (temperature <= 0.0f) {
        params = SamplingParams::greedy();
    } else {
        params.temperature = temperature;
        params.top_k = 0;
        params.top_p = 1.0f;
        params.repetition_penalty = 1.0f;
    }
    LcgRng53 rng(params.seed);
    return sample_token_full(logits, pos, params, {}, rng);
}
}  // namespace

// =============================================================================
// Cached generation
// =============================================================================

namespace {
/// Run `token_ids` through every layer against `cache`, returning the logits.
[[nodiscard]] Mat run_cached(const GptOssModel& model, const std::vector<std::size_t>& token_ids,
                             KvCache& cache) {
    TensorNode x = TensorNode::leaf(
        gather_embeddings(model.embed_tokens.data(), token_ids, model.config.hidden_size));
    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        x = model.layers[i].forward_cached(x, cache.layers[i]);
    }
    return model.lm_head.forward(model.norm.forward(x)).data();
}
}  // namespace

std::vector<std::size_t> GptOssModel::generate_cached(const std::vector<std::size_t>& token_ids,
                                                      std::size_t max_new,
                                                      float temperature) const {
    KvCache cache(config);

    const Mat first_logits = run_cached(*this, token_ids, cache);
    std::vector<std::size_t> generated;
    generated.reserve(max_new);

    std::size_t prev_tok = sample_token_simple(first_logits, token_ids.size() - 1, temperature);
    generated.push_back(prev_tok);

    for (std::size_t i = 1; i < max_new; ++i) {
        prev_tok = sample_token_simple(run_cached(*this, {prev_tok}, cache), 0, temperature);
        generated.push_back(prev_tok);
    }
    return generated;
}

void GptOssModel::generate_cached_streaming(
    const std::vector<std::size_t>& token_ids, std::size_t max_new, float temperature,
    const std::function<void(std::size_t)>& callback) const {
    KvCache cache(config);

    const Mat first_logits = run_cached(*this, token_ids, cache);
    std::size_t prev_tok = sample_token_simple(first_logits, token_ids.size() - 1, temperature);
    callback(prev_tok);

    for (std::size_t i = 1; i < max_new; ++i) {
        prev_tok = sample_token_simple(run_cached(*this, {prev_tok}, cache), 0, temperature);
        callback(prev_tok);
    }
}

std::vector<std::size_t> GptOssModel::generate_with_params(
    const std::vector<std::size_t>& token_ids, std::size_t max_new,
    const SamplingParams& params) const {
    KvCache cache(config);
    LcgRng53 rng(params.seed);

    // Penalties look only at generated tokens, so the prompt is not penalized.
    std::vector<std::size_t> generated;
    generated.reserve(max_new);

    Mat logits = run_cached(*this, token_ids, cache);
    std::size_t pos = token_ids.size() - 1;

    for (std::size_t i = 0; i < max_new; ++i) {
        const std::size_t tok = sample_token_full(logits, pos, params, generated, rng);
        generated.push_back(tok);
        if (params.eos_token_id && tok == *params.eos_token_id) {
            break;
        }
        logits = run_cached(*this, {tok}, cache);
        pos = 0;
    }
    return generated;
}

void GptOssModel::tie_weights() {
    assert(lm_head.weight.data().rows == embed_tokens.data().rows &&
           lm_head.weight.data().cols == embed_tokens.data().cols &&
           "tie_weights: lm_head.weight and embed_tokens have different shapes");
    lm_head.weight = embed_tokens;
}

Q4QuantStats GptOssModel::quantize_all_linear_weights() const {
    Q4QuantStats stats;

    for (const TensorNode& param : parameters()) {
        const std::size_t rows = param.data().rows;
        const std::size_t cols = param.data().cols;
        // Norm gammas and scalars are 1-D; quantizing them buys nothing.
        if (rows <= 1 || cols <= 1) {
            continue;
        }

        const Q4Mat q4 = Q4Mat::quantize(param.data());
        param.set_data(q4.dequantize());

        ++stats.n_tensors;
        stats.f32_bytes += rows * cols * 4;
        stats.q4_bytes += q4.size_bytes();
    }

    if (stats.q4_bytes > 0) {
        stats.compression_ratio =
            static_cast<float>(stats.f32_bytes) / static_cast<float>(stats.q4_bytes);
    }
    return stats;
}

std::vector<TensorNode> GptOssModel::parameters() const {
    std::vector<TensorNode> p{embed_tokens};
    for (const GptOssBlock& layer : layers) {
        const std::vector<TensorNode> lp = layer.parameters();
        p.insert(p.end(), lp.begin(), lp.end());
    }
    const std::vector<TensorNode> tail = collect_params(norm, lm_head);
    p.insert(p.end(), tail.begin(), tail.end());
    return p;
}

TensorNode GptOssModel::forward_tokens(const std::vector<std::size_t>& token_ids) const {
    return forward(token_ids);
}

TensorNode GptOssModel::loss_tokens(const std::vector<std::size_t>& token_ids,
                                    const std::vector<std::size_t>& targets) const {
    return loss(token_ids, targets);
}

// =============================================================================
// safetensors
// =============================================================================

namespace {

/// Iterate the top-level `"key": {...}` pairs of a JSON object, returning
/// (name, value_json) for each. Hand-written so the project stays
/// dependency-free.
[[nodiscard]] std::vector<std::pair<std::string, std::string>> iter_top_level_pairs(
    std::string_view json) {
    std::vector<std::pair<std::string, std::string>> pairs;
    const std::size_t n = json.size();
    std::size_t i = 0;

    while (i < n) {
        while (i < n && json[i] != '"') {
            ++i;
        }
        if (i >= n) {
            break;
        }
        ++i;  // opening quote
        const std::size_t key_start = i;
        while (i < n && json[i] != '"') {
            ++i;
        }
        std::string key(json.substr(key_start, i - key_start));
        ++i;  // closing quote

        while (i < n && (json[i] == ':' || json[i] == ' ' || json[i] == '\n' || json[i] == '\r')) {
            ++i;
        }
        if (i >= n || json[i] != '{') {
            continue;
        }

        // Match the braces to find the end of this tensor's object.
        const std::size_t val_start = i;
        int depth = 0;
        while (i < n) {
            if (json[i] == '{') {
                ++depth;
            } else if (json[i] == '}') {
                if (--depth == 0) {
                    ++i;
                    break;
                }
            }
            ++i;
        }
        pairs.emplace_back(std::move(key), std::string(json.substr(val_start, i - val_start)));
    }
    return pairs;
}

/// The string value of `"key":"value"`.
[[nodiscard]] std::optional<std::string> extract_quoted_value(std::string_view json,
                                                              std::string_view key) {
    const std::string pattern = "\"" + std::string(key) + "\":";
    const std::size_t pos = json.find(pattern);
    if (pos == std::string_view::npos) {
        return std::nullopt;
    }
    std::string_view after = json.substr(pos + pattern.size());
    while (!after.empty() && (after.front() == ' ' || after.front() == '\n')) {
        after.remove_prefix(1);
    }
    if (after.empty() || after.front() != '"') {
        return std::nullopt;
    }
    after.remove_prefix(1);
    const std::size_t end = after.find('"');
    if (end == std::string_view::npos) {
        return std::nullopt;
    }
    return std::string(after.substr(0, end));
}

/// The integer array value of `"key":[1,2,3]`.
[[nodiscard]] std::optional<std::vector<std::int64_t>> extract_int_array(std::string_view json,
                                                                        std::string_view key) {
    const std::string pattern = "\"" + std::string(key) + "\":";
    const std::size_t pos = json.find(pattern);
    if (pos == std::string_view::npos) {
        return std::nullopt;
    }
    std::string_view after = json.substr(pos + pattern.size());
    while (!after.empty() && (after.front() == ' ' || after.front() == '\n')) {
        after.remove_prefix(1);
    }
    if (after.empty() || after.front() != '[') {
        return std::nullopt;
    }
    const std::size_t end = after.find(']');
    if (end == std::string_view::npos) {
        return std::nullopt;
    }

    std::vector<std::int64_t> out;
    std::string_view inner = after.substr(1, end - 1);
    std::size_t start = 0;
    while (start <= inner.size()) {
        const std::size_t comma = inner.find(',', start);
        std::string_view item =
            inner.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                                : comma - start);
        while (!item.empty() && item.front() == ' ') {
            item.remove_prefix(1);
        }
        while (!item.empty() && item.back() == ' ') {
            item.remove_suffix(1);
        }
        std::int64_t value = 0;
        if (std::from_chars(item.data(), item.data() + item.size(), value).ec != std::errc{}) {
            return std::nullopt;
        }
        out.push_back(value);
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

[[nodiscard]] std::uint64_t read_u64_le(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | p[i];
    }
    return v;
}

/// Decode a raw byte range into f32 (and, for BF16, the raw bits too).
[[nodiscard]] bool decode_tensor_bytes(const std::string& dtype, const std::uint8_t* raw,
                                       std::size_t byte_len, bool skip_bf16_to_f32,
                                       std::vector<float>& data,
                                       std::optional<std::vector<std::uint16_t>>& bf16_data) {
    if (dtype == "F32") {
        data.resize(byte_len / 4);
        std::memcpy(data.data(), raw, byte_len);
        bf16_data.reset();
        return true;
    }
    if (dtype == "BF16") {
        std::vector<std::uint16_t> bits(byte_len / 2);
        std::memcpy(bits.data(), raw, byte_len);
        // The f32 form is twice the size; callers that read the bits directly
        // skip building it.
        if (skip_bf16_to_f32) {
            data.clear();
        } else {
            data.resize(bits.size());
            for (std::size_t i = 0; i < bits.size(); ++i) {
                data[i] = bf16_to_f32(bits[i]);
            }
        }
        bf16_data = std::move(bits);
        return true;
    }
    if (dtype == "F16") {
        const std::size_t n = byte_len / 2;
        data.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto bits =
                static_cast<std::uint16_t>(raw[2 * i] | (raw[2 * i + 1] << 8));
            data[i] = f16_to_f32(bits);
        }
        bf16_data.reset();
        return true;
    }
    return false;
}

[[nodiscard]] Result<std::vector<SafeTensor>> parse_safetensors_inner(
    const std::vector<std::uint8_t>& bytes, bool skip_bf16_to_f32) {
    if (bytes.size() < 8) {
        return err("safetensors: file too small");
    }
    const auto header_len = static_cast<std::size_t>(read_u64_le(bytes.data()));
    if (bytes.size() < 8 + header_len) {
        return err("safetensors: header_len " + std::to_string(header_len) +
                   " exceeds file size");
    }
    const std::string_view header_json(reinterpret_cast<const char*>(bytes.data() + 8),
                                       header_len);
    const std::uint8_t* data_section = bytes.data() + 8 + header_len;
    const std::size_t data_len = bytes.size() - 8 - header_len;

    std::vector<SafeTensor> tensors;
    for (auto& [name, value_json] : iter_top_level_pairs(header_json)) {
        if (name == "__metadata__") {
            continue;
        }
        const auto dtype = extract_quoted_value(value_json, "dtype");
        const auto shape = extract_int_array(value_json, "shape");
        const auto offsets = extract_int_array(value_json, "data_offsets");
        if (!dtype || !shape || !offsets || offsets->size() != 2) {
            continue;
        }
        const auto byte_start = static_cast<std::size_t>((*offsets)[0]);
        const auto byte_end = static_cast<std::size_t>((*offsets)[1]);
        if (byte_end > data_len) {
            continue;
        }

        SafeTensor t;
        if (!decode_tensor_bytes(*dtype, data_section + byte_start, byte_end - byte_start,
                                 skip_bf16_to_f32, t.data, t.bf16_data)) {
            continue;  // an unsupported dtype is skipped, not an error
        }
        t.name = std::move(name);
        t.shape.reserve(shape->size());
        for (std::int64_t d : *shape) {
            t.shape.push_back(static_cast<std::size_t>(d));
        }
        tensors.push_back(std::move(t));
    }
    return tensors;
}

}  // namespace

Result<std::vector<SafeTensor>> parse_safetensors(const std::vector<std::uint8_t>& bytes) {
    return parse_safetensors_inner(bytes, false);
}

Result<std::vector<SafeTensor>> parse_safetensors_skip_bf16_f32(
    const std::vector<std::uint8_t>& bytes) {
    return parse_safetensors_inner(bytes, true);
}

Result<std::pair<std::size_t, std::vector<SafeTensorEntry>>> parse_safetensors_header(
    const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return err("safetensors: cannot open " + path);
    }

    std::uint8_t len_buf[8];
    f.read(reinterpret_cast<char*>(len_buf), 8);
    if (!f) {
        return err("safetensors: cannot read header length");
    }
    const auto header_len = static_cast<std::size_t>(read_u64_le(len_buf));
    const std::size_t data_start = 8 + header_len;

    std::string header(header_len, '\0');
    f.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!f) {
        return err("safetensors: cannot read header");
    }

    std::vector<SafeTensorEntry> entries;
    for (auto& [name, value_json] : iter_top_level_pairs(header)) {
        if (name == "__metadata__") {
            continue;
        }
        const auto dtype = extract_quoted_value(value_json, "dtype");
        const auto shape = extract_int_array(value_json, "shape");
        const auto offsets = extract_int_array(value_json, "data_offsets");
        if (!dtype || !shape || !offsets || offsets->size() != 2) {
            continue;
        }
        SafeTensorEntry e;
        e.name = std::move(name);
        e.dtype = *dtype;
        e.shape.reserve(shape->size());
        for (std::int64_t d : *shape) {
            e.shape.push_back(static_cast<std::size_t>(d));
        }
        e.byte_start = static_cast<std::size_t>((*offsets)[0]);
        e.byte_end = static_cast<std::size_t>((*offsets)[1]);
        entries.push_back(std::move(e));
    }
    return std::pair{data_start, std::move(entries)};
}

Result<SafeTensor> read_safetensor_from_file(const std::string& path, std::size_t data_start,
                                             const SafeTensorEntry& entry,
                                             bool skip_bf16_to_f32) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return err("safetensors: cannot open " + path);
    }
    const std::size_t byte_len = entry.byte_end - entry.byte_start;
    f.seekg(static_cast<std::streamoff>(data_start + entry.byte_start));

    std::vector<std::uint8_t> raw(byte_len);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_len));
    if (!f) {
        return err("safetensors: read error for " + entry.name);
    }

    SafeTensor t;
    if (!decode_tensor_bytes(entry.dtype, raw.data(), byte_len, skip_bf16_to_f32, t.data,
                             t.bf16_data)) {
        return err("safetensors: unsupported dtype " + entry.dtype + " for " + entry.name);
    }
    t.name = entry.name;
    t.shape = entry.shape;
    return t;
}

// =============================================================================
// Applying weights to the model
// =============================================================================

namespace {

/// Set a node's data from a flat f32 slice, if the element count matches.
[[nodiscard]] bool set_node(const TensorNode& node, const std::vector<float>& data,
                            std::size_t rows, std::size_t cols) {
    if (data.size() != rows * cols) {
        return false;
    }
    node.set_data(Mat(data, rows, cols));
    return true;
}

/// Parse a leading `<index>.` prefix, returning the index and the remainder.
[[nodiscard]] std::optional<std::pair<std::size_t, std::string_view>> split_index(
    std::string_view s) {
    const std::size_t dot = s.find('.');
    if (dot == std::string_view::npos) {
        return std::nullopt;
    }
    std::size_t idx = 0;
    if (std::from_chars(s.data(), s.data() + dot, idx).ec != std::errc{}) {
        return std::nullopt;
    }
    return std::pair{idx, s.substr(dot + 1)};
}

/// Apply one tensor to the matching field, returning whether it was recognized.
[[nodiscard]] bool apply_tensor(GptOssModel& model, const SafeTensor& t) {
    const std::string_view name = t.name;

    if (name == "model.embed_tokens.weight") {
        return set_node(model.embed_tokens, t.data, t.shape[0], t.shape[1]);
    }
    if (name == "model.norm.weight") {
        return set_node(model.norm.gamma, t.data, 1, t.shape[0]);
    }
    if (name == "lm_head.weight") {
        return set_node(model.lm_head.weight, t.data, t.shape[0], t.shape[1]);
    }

    constexpr std::string_view kLayerPrefix = "model.layers.";
    if (!name.starts_with(kLayerPrefix)) {
        return false;
    }
    const auto layer_split = split_index(name.substr(kLayerPrefix.size()));
    if (!layer_split) {
        return false;
    }
    const auto [layer_idx, layer_name] = *layer_split;
    if (layer_idx >= model.layers.size()) {
        return false;
    }
    GptOssBlock& layer = model.layers[layer_idx];

    if (layer_name == "input_layernorm.weight") {
        return set_node(layer.input_layernorm.gamma, t.data, 1, t.shape[0]);
    }
    if (layer_name == "post_attention_layernorm.weight") {
        return set_node(layer.post_attention_layernorm.gamma, t.data, 1, t.shape[0]);
    }
    if (layer_name == "self_attn.q_proj.weight") {
        return set_node(layer.self_attn.q_proj.weight, t.data, t.shape[0], t.shape[1]);
    }
    if (layer_name == "self_attn.k_proj.weight") {
        return set_node(layer.self_attn.k_proj.weight, t.data, t.shape[0], t.shape[1]);
    }
    if (layer_name == "self_attn.v_proj.weight") {
        return set_node(layer.self_attn.v_proj.weight, t.data, t.shape[0], t.shape[1]);
    }
    if (layer_name == "self_attn.o_proj.weight") {
        return set_node(layer.self_attn.o_proj.weight, t.data, t.shape[0], t.shape[1]);
    }
    if (layer_name == "mlp.router.weight") {
        return set_node(layer.mlp.router.weight, t.data, t.shape[0], t.shape[1]);
    }

    constexpr std::string_view kExpertPrefix = "mlp.experts.";
    if (!layer_name.starts_with(kExpertPrefix)) {
        return false;
    }
    const auto expert_split = split_index(layer_name.substr(kExpertPrefix.size()));
    if (!expert_split) {
        return false;
    }
    const auto [expert_idx, expert_name] = *expert_split;
    if (expert_idx >= layer.mlp.experts.size()) {
        return false;
    }
    SwiGluMlp2& expert = layer.mlp.experts[expert_idx];

    if (expert_name == "gate_proj.weight") {
        return set_node(expert.gate_proj.weight, t.data, t.shape[0], t.shape[1]);
    }
    if (expert_name == "up_proj.weight") {
        return set_node(expert.up_proj.weight, t.data, t.shape[0], t.shape[1]);
    }
    if (expert_name == "down_proj.weight") {
        return set_node(expert.down_proj.weight, t.data, t.shape[0], t.shape[1]);
    }
    return false;
}

}  // namespace

void load_into_model(GptOssModel& model, const std::vector<SafeTensor>& tensors) {
    for (const SafeTensor& tensor : tensors) {
        // An unrecognized name is either from a shard layout we do not handle
        // or an extra tensor such as "model.rotary_emb.inv_freq"; skip it.
        (void)apply_tensor(model, tensor);
    }
}

Result<void> GptOssModel::load_weights_from_dir(const std::string& dir) {
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    if (ec) {
        return err("cannot read dir " + dir + ": " + ec.message());
    }

    std::size_t loaded_shards = 0;
    std::size_t loaded_tensors = 0;

    for (const auto& entry : it) {
        if (entry.path().extension() != ".safetensors") {
            continue;
        }
        const std::string path = entry.path().string();

        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            return err("cannot read " + path);
        }
        const auto size = static_cast<std::size_t>(f.tellg());
        f.seekg(0);
        std::vector<std::uint8_t> bytes(size);
        f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));

        const auto tensors = parse_safetensors(bytes);
        if (!tensors) {
            return err("parse error in " + path + ": " + tensors.error());
        }

        loaded_tensors += tensors->size();
        load_into_model(*this, *tensors);
        ++loaded_shards;
    }

    if (loaded_shards == 0) {
        return err("no .safetensors files found in " + dir);
    }

    std::printf("Loaded %zu tensors from %zu shards in %s\n", loaded_tensors, loaded_shards,
                dir.c_str());
    return {};
}

}  // namespace rt

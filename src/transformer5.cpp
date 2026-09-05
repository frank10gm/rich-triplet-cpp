#include "rt/transformer5.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <utility>

#include "rt/transformer4.hpp"

namespace rt {

// =============================================================================
// Config5
// =============================================================================

Config5 Config5::orpheus_3b() {
    Config5 c;
    c.vocab_size = 156940;  // 128 256 base + 28 672 audio + 12 markers
    c.hidden_size = 3072;
    c.num_hidden_layers = 28;
    c.num_attention_heads = 24;
    c.num_key_value_heads = 8;
    c.intermediate_size = 8192;
    c.head_dim = 128;
    c.rope_theta = 500000.0f;
    c.rms_norm_eps = 1e-5f;
    c.max_position_embeddings = 131072;
    c.eos_token_id = 128009;
    // Left empty here; the loader fills it from `rope_freqs.weight`.
    return c;
}

std::vector<float> Config5::inv_freq() const {
    const std::size_t half = head_dim / 2;
    std::vector<float> out(half);
    for (std::size_t i = 0; i < half; ++i) {
        const float exponent = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
        float f = 1.0f / std::pow(rope_theta, exponent);
        if (i < rope_freq_divisors.size()) {
            // Divisors, not multipliers: `rope_freqs.weight` runs from 1.0 up
            // to 32.0, and dividing by 32 is what stretches a frequency band.
            const float d = rope_freq_divisors[i];
            if (d > 0.0f) {
                f /= d;
            }
        }
        out[i] = f;
    }
    return out;
}

// =============================================================================
// KV cache
// =============================================================================

void LlamaLayerKvCache::append(const Mat& new_k, const Mat& new_v) {
    assert(new_k.cols == k.cols && new_v.cols == v.cols && "llama kv cache: width mismatch");
    assert(seq_len + new_k.rows <= k.rows && "llama kv cache: overflow");
    for (std::size_t r = 0; r < new_k.rows; ++r) {
        const std::size_t dst = seq_len + r;
        std::copy(new_k.row(r).begin(), new_k.row(r).end(), k.row_mut(dst).begin());
        std::copy(new_v.row(r).begin(), new_v.row(r).end(), v.row_mut(dst).begin());
    }
    seq_len += new_k.rows;
}

LlamaKvCache::LlamaKvCache(const Config5& config, std::size_t max_tokens) {
    layers.reserve(config.num_hidden_layers);
    for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers.emplace_back(config.num_key_value_heads, config.head_dim, max_tokens);
    }
}

void LlamaKvCache::clear() {
    for (LlamaLayerKvCache& layer : layers) {
        layer.seq_len = 0;
    }
}

void LlamaKvCache::free() {
    for (LlamaLayerKvCache& layer : layers) {
        layer.k = Mat::zeros(0, 0);
        layer.v = Mat::zeros(0, 0);
        layer.seq_len = 0;
    }
}

// =============================================================================
// RoPE
// =============================================================================

Mat llama_rope(const Mat& x, std::size_t n_heads, std::size_t head_dim, std::size_t offset,
               const std::vector<float>& inv_freq, RopePairing pairing) {
    const std::size_t half = head_dim / 2;
    assert(inv_freq.size() >= half && "llama_rope: not enough inverse frequencies");
    assert(x.cols == n_heads * head_dim && "llama_rope: width is not n_heads * head_dim");

    Mat out = x;
    for (std::size_t row = 0; row < x.rows; ++row) {
        const float pos = static_cast<float>(offset + row);
        for (std::size_t i = 0; i < half; ++i) {
            const float angle = pos * inv_freq[i];
            const float cos_a = std::cos(angle);
            const float sin_a = std::sin(angle);
            for (std::size_t h = 0; h < n_heads; ++h) {
                // Interleaved pairs (2i, 2i+1) for GGUF's permuted llama
                // weights; half-split (i, i + head_dim/2) for HuggingFace's.
                const std::size_t c0 = pairing == RopePairing::Interleaved
                                           ? h * head_dim + 2 * i
                                           : h * head_dim + i;
                const std::size_t c1 =
                    pairing == RopePairing::Interleaved ? c0 + 1 : c0 + half;
                const float x0 = x.at(row, c0);
                const float x1 = x.at(row, c1);
                out.at_mut(row, c0) = x0 * cos_a - x1 * sin_a;
                out.at_mut(row, c1) = x0 * sin_a + x1 * cos_a;
            }
        }
    }
    return out;
}

// =============================================================================
// LlamaAttention
// =============================================================================

LlamaAttention::LlamaAttention(const Config5& cfg)
    : q_proj(Linear2::new_no_bias_zeros(cfg.hidden_size,
                                        cfg.num_attention_heads * cfg.head_dim)),
      k_proj(Linear2::new_no_bias_zeros(cfg.hidden_size,
                                        cfg.num_key_value_heads * cfg.head_dim)),
      v_proj(Linear2::new_no_bias_zeros(cfg.hidden_size,
                                        cfg.num_key_value_heads * cfg.head_dim)),
      o_proj(Linear2::new_no_bias_zeros(cfg.num_attention_heads * cfg.head_dim,
                                        cfg.hidden_size)),
      n_q_heads(cfg.num_attention_heads),
      n_kv_heads(cfg.num_key_value_heads),
      head_dim(cfg.head_dim),
      rope_pairing(cfg.rope_pairing),
      // Plain 1/sqrt(head_dim) -- Gemma's query_pre_attn_scalar has no analogue.
      attn_scale(1.0f / std::sqrt(static_cast<float>(cfg.head_dim))) {}

LlamaAttention LlamaAttention::new_for_inference(const Config5& cfg) {
    return LlamaAttention(cfg);
}

Mat LlamaAttention::forward_cached(const Mat& x, LlamaLayerKvCache& cache) const {
    assert(inv_freq != nullptr && "llama attention: inverse frequencies not bound");
    const std::size_t seq_offset = cache.seq_len;

    const TensorNode xn = TensorNode::leaf(x);
    // No per-head Q/K RMSNorm: RoPE goes straight onto the projections.
    const Mat q = llama_rope(q_proj.forward(xn).data(), n_q_heads, head_dim, seq_offset, *inv_freq,
                             rope_pairing);
    const Mat k = llama_rope(k_proj.forward(xn).data(), n_kv_heads, head_dim, seq_offset, *inv_freq,
                             rope_pairing);
    const Mat v = v_proj.forward(xn).data();

    cache.append(k, v);

    // Every layer is global: attend over the whole cache. The cache holds only
    // past tokens, so prefill still needs the causal mask that
    // `gqa_attention_cached` applies relative to `k_start`.
    Mat attn = gqa_attention_cached(q, cache.k, cache.v, 0, cache.seq_len, n_q_heads, n_kv_heads,
                                    head_dim, attn_scale);

    return o_proj.forward(TensorNode::leaf(std::move(attn))).data();
}

// =============================================================================
// LlamaMlp
// =============================================================================

LlamaMlp::LlamaMlp(const Config5& cfg)
    : gate_proj(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size)),
      up_proj(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size)),
      down_proj(Linear2::new_no_bias_zeros(cfg.intermediate_size, cfg.hidden_size)) {}

LlamaMlp LlamaMlp::new_for_inference(const Config5& cfg) { return LlamaMlp(cfg); }

Mat LlamaMlp::forward(const Mat& x) const {
    const TensorNode xn = TensorNode::leaf(x);
    // SiLU, not Gemma's gelu_pytorch_tanh.
    const TensorNode gate = gate_proj.forward(xn).silu();
    return down_proj.forward(gate.mul_elem_node(up_proj.forward(xn))).data();
}

// =============================================================================
// LlamaBlock
// =============================================================================

LlamaBlock::LlamaBlock(const Config5& cfg)
    : input_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      self_attn(LlamaAttention::new_for_inference(cfg)),
      post_attention_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      mlp(LlamaMlp::new_for_inference(cfg)) {}

LlamaBlock LlamaBlock::new_for_inference(const Config5& cfg) { return LlamaBlock(cfg); }

Mat LlamaBlock::forward_cached(const Mat& x, LlamaLayerKvCache& cache) const {
    // Plain RMSNorm, not Gemma's (1 + gamma) variant.
    const Mat normed = input_layernorm.forward(TensorNode::leaf(x)).data();
    Mat h = self_attn.forward_cached(normed, cache);
    h.add_assign(x);

    const Mat normed2 = post_attention_layernorm.forward(TensorNode::leaf(h)).data();
    Mat ff = mlp.forward(normed2);
    ff.add_assign(h);
    return ff;
}

// =============================================================================
// LlamaModel
// =============================================================================

LlamaModel::LlamaModel(Config5 cfg)
    : norm(cfg.hidden_size, cfg.rms_norm_eps),
      lm_head(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.vocab_size)),
      config(std::move(cfg)) {
    layers.reserve(config.num_hidden_layers);
    for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers.push_back(LlamaBlock::new_for_inference(config));
    }
    rebind_inv_freq();
}

LlamaModel LlamaModel::new_for_inference(Config5 cfg) { return LlamaModel(std::move(cfg)); }

void LlamaModel::rebind_inv_freq() {
    inv_freq_cache = config.inv_freq();
    for (LlamaBlock& layer : layers) {
        layer.self_attn.inv_freq = &inv_freq_cache;
        layer.self_attn.rope_pairing = config.rope_pairing;
    }
}

Mat LlamaModel::embed_rows(const std::vector<std::size_t>& token_ids) const {
    const std::size_t t = token_ids.size();
    const std::size_t h = config.hidden_size;
    // No sqrt(hidden_size) scaling -- that is Gemma's.
    assert(embed_bf16.has_value() && "llama: embedding table not loaded");
    const std::vector<std::uint16_t>& bits = *embed_bf16->data;
    return Mat::from_fn(t, h, [&](std::size_t row, std::size_t col) {
        return bf16_to_f32(bits[token_ids[row] * h + col]);
    });
}

Mat LlamaModel::forward_cached(const std::vector<std::size_t>& token_ids,
                               LlamaKvCache& cache) const {
    Mat x = embed_rows(token_ids);
    for (std::size_t i = 0; i < layers.size(); ++i) {
        x = layers[i].forward_cached(x, cache.layers[i]);
    }
    // Only the final row's logits matter, so the lm_head runs as a GEMV rather
    // than a T x 156940 gemm.
    const Mat last =
        Mat::from_fn(1, x.cols, [&](std::size_t, std::size_t c) { return x.at(x.rows - 1, c); });
    return lm_head.forward(norm.forward(TensorNode::leaf(last))).data();
}

std::size_t LlamaModel::weight_bytes() const {
    std::size_t n = 0;
    if (embed_bf16) {
        n += embed_bf16->size_bytes();
    }
    const auto linear_bytes = [](const Linear2& l) -> std::size_t {
        if (l.q4k_weight) {
            return l.q4k_weight->size_bytes();
        }
        if (l.bf16_weight) {
            return l.bf16_weight->size_bytes();
        }
        return l.weight.data().numel() * sizeof(float);
    };
    n += linear_bytes(lm_head);
    for (const LlamaBlock& layer : layers) {
        n += linear_bytes(layer.self_attn.q_proj) + linear_bytes(layer.self_attn.k_proj) +
             linear_bytes(layer.self_attn.v_proj) + linear_bytes(layer.self_attn.o_proj) +
             linear_bytes(layer.mlp.gate_proj) + linear_bytes(layer.mlp.up_proj) +
             linear_bytes(layer.mlp.down_proj);
    }
    return n;
}

void LlamaModel::quantize_lm_head() { lm_head.quantize_bf16_to_q4k(); }

std::size_t LlamaModel::projection_count() const { return 7 * layers.size() + 1; }

std::size_t LlamaModel::bf16_projection_count() const {
    std::size_t n = lm_head.bf16_weight ? 1 : 0;
    for (const LlamaBlock& layer : layers) {
        for (const Linear2* l : {&layer.self_attn.q_proj, &layer.self_attn.k_proj,
                                 &layer.self_attn.v_proj, &layer.self_attn.o_proj,
                                 &layer.mlp.gate_proj, &layer.mlp.up_proj,
                                 &layer.mlp.down_proj}) {
            if (l->bf16_weight) {
                ++n;
            }
        }
    }
    return n;
}

std::size_t LlamaModel::quantize_projections_to_q4k() {
    // `quantize_bf16_to_q4k` is a no-op without a BF16 weight, so anything
    // already stored as Q4_K passes through untouched.
    std::size_t converted = 0;
    const auto convert = [&converted](Linear2& l) {
        if (l.bf16_weight) {
            l.quantize_bf16_to_q4k();
            ++converted;
        }
    };
    for (LlamaBlock& layer : layers) {
        convert(layer.self_attn.q_proj);
        convert(layer.self_attn.k_proj);
        convert(layer.self_attn.v_proj);
        convert(layer.self_attn.o_proj);
        convert(layer.mlp.gate_proj);
        convert(layer.mlp.up_proj);
        convert(layer.mlp.down_proj);
    }
    convert(lm_head);
    return converted;
}

// =============================================================================
// Sampling
// =============================================================================

std::size_t sample_token_large_vocab(const Mat& logits, std::size_t row,
                                     const SamplingParams& params,
                                     const std::vector<std::size_t>& seen, LcgRng& rng) {
    const std::size_t v = logits.cols;
    std::vector<float> scores(v);
    for (std::size_t c = 0; c < v; ++c) {
        scores[c] = logits.at(row, c);
    }

    // Vocabulary mask. -infinity survives the temperature division and makes
    // exp() underflow to exactly zero, so masked ids cannot be drawn and
    // cannot be the greedy argmax either.
    if (params.allowed_min || params.allowed_max) {
        const std::size_t lo = params.allowed_min.value_or(0);
        const std::size_t hi = params.allowed_max.value_or(v);
        for (std::size_t c = 0; c < v; ++c) {
            if (c < lo || c >= hi) {
                scores[c] = -std::numeric_limits<float>::infinity();
            }
        }
        for (const std::size_t id : params.allowed_extra) {
            if (id < v) {
                scores[id] = logits.at(row, id);
            }
        }
    }

    // Repetition penalty over the trailing window of generated tokens.
    if (params.repetition_penalty != 1.0f && !seen.empty()) {
        constexpr std::size_t kWindow = 64;
        const std::size_t start = seen.size() > kWindow ? seen.size() - kWindow : 0;
        for (std::size_t i = start; i < seen.size(); ++i) {
            const std::size_t id = seen[i];
            if (id >= v) {
                continue;
            }
            scores[id] = scores[id] > 0.0f ? scores[id] / params.repetition_penalty
                                           : scores[id] * params.repetition_penalty;
        }
    }

    // Greedy: no need to build a distribution at all.
    if (params.temperature <= 0.0f) {
        std::size_t best = 0;
        float best_v = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < v; ++c) {
            if (scores[c] > best_v) {
                best_v = scores[c];
                best = c;
            }
        }
        return best;
    }

    for (float& s : scores) {
        s /= params.temperature;
    }

    float max_s = -std::numeric_limits<float>::infinity();
    for (const float s : scores) {
        max_s = std::max(max_s, s);
    }
    std::vector<float> probs(v);
    float sum_exp = 0.0f;
    for (std::size_t c = 0; c < v; ++c) {
        probs[c] = std::exp(scores[c] - max_s);
        sum_exp += probs[c];
    }
    if (sum_exp <= 0.0f) {
        return 0;
    }
    for (float& p : probs) {
        p /= sum_exp;
    }

    // Candidate set. Top-k selects with nth_element -- linear, rather than
    // sorting 156 940 entries per token.
    std::vector<std::size_t> idx;
    const std::size_t k = params.top_k > 0 ? std::min(params.top_k, v) : v;
    if (k < v) {
        idx.resize(v);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::nth_element(idx.begin(), idx.begin() + static_cast<long>(k), idx.end(),
                         [&probs](std::size_t a, std::size_t b) { return probs[a] > probs[b]; });
        idx.resize(k);
    } else if (params.top_p < 1.0f) {
        // Nucleus sampling has to look at the mass in descending order, but it
        // only ever needs the head of the distribution. Anything below
        // 1/(4v) cannot enter a nucleus of any useful size, and dropping it
        // first keeps the sort small.
        const float floor_p = 0.25f / static_cast<float>(v);
        idx.reserve(1024);
        for (std::size_t c = 0; c < v; ++c) {
            if (probs[c] > floor_p) {
                idx.push_back(c);
            }
        }
        if (idx.empty()) {
            idx.resize(v);
            std::iota(idx.begin(), idx.end(), std::size_t{0});
        }
    }

    if (!idx.empty()) {
        // Only the survivors get sorted.
        std::sort(idx.begin(), idx.end(),
                  [&probs](std::size_t a, std::size_t b) { return probs[a] > probs[b]; });

        if (params.top_p < 1.0f) {
            float cumulative = 0.0f;
            std::size_t keep = 0;
            for (; keep < idx.size(); ++keep) {
                cumulative += probs[idx[keep]];
                if (cumulative >= params.top_p) {
                    ++keep;  // include the token that crossed the threshold
                    break;
                }
            }
            idx.resize(std::max<std::size_t>(keep, 1));
        }

        float mass = 0.0f;
        for (const std::size_t c : idx) {
            mass += probs[c];
        }
        if (mass <= 0.0f) {
            return idx.front();
        }
        // Inverse CDF over the candidates.
        const float draw = rng.next_f32() * mass;
        float running = 0.0f;
        for (const std::size_t c : idx) {
            running += probs[c];
            if (running >= draw) {
                return c;
            }
        }
        return idx.back();
    }

    // No filtering: draw straight from the full distribution.
    const float draw = rng.next_f32();
    float running = 0.0f;
    for (std::size_t c = 0; c < v; ++c) {
        running += probs[c];
        if (running >= draw) {
            return c;
        }
    }
    return v - 1;
}

// =============================================================================
// Generation
// =============================================================================

std::size_t LlamaModel::generate(const std::vector<std::size_t>& prompt, std::size_t max_new,
                                 const SamplingParams& params, bool debug,
                                 const std::function<bool(std::size_t)>& on_token) const {
    if (prompt.empty()) {
        return 0;
    }

    LlamaKvCache cache(config, prompt.size() + max_new + 1);
    LcgRng rng(params.seed);

    Mat logits = forward_cached(prompt, cache);

    std::vector<std::size_t> generated;
    generated.reserve(max_new);

    for (std::size_t step = 0; step < max_new; ++step) {
        const std::size_t next = sample_token_large_vocab(logits, 0, params, generated, rng);

        if (debug) {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (std::size_t c = 0; c < logits.cols; ++c) {
                max_logit = std::max(max_logit, logits.at(0, c));
            }
            std::fprintf(stderr, "[ llama ] step %zu -> id %zu (logit %.3f, max %.3f)\n", step,
                         next, static_cast<double>(logits.at(0, next)),
                         static_cast<double>(max_logit));
        }

        generated.push_back(next);
        if (!on_token(next)) {
            return generated.size();
        }
        if (params.eos_token_id && next == *params.eos_token_id) {
            return generated.size();
        }
        if (step + 1 == max_new) {
            break;
        }
        logits = forward_cached({next}, cache);
    }

    return generated.size();
}

}  // namespace rt

#include "rt/transformer_qwen35.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace rt {

// =============================================================================
// ConfigQwen35
// =============================================================================

namespace {
/// The fields every Qwen 3.5 size shares.
ConfigQwen35 qwen35_common() {
    ConfigQwen35 c;
    c.vocab_size = 248320;
    c.num_hidden_layers = 32;
    c.num_attention_heads = 16;
    c.num_key_value_heads = 4;
    c.head_dim = 256;
    c.linear_num_key_heads = 16;
    c.linear_num_value_heads = 32;
    c.linear_key_head_dim = 128;
    c.linear_value_head_dim = 128;
    c.linear_conv_kernel_dim = 4;
    c.rms_norm_eps = 1e-6f;
    c.rope_theta = 10000000.0f;
    c.partial_rotary_factor = 0.25f;
    c.full_attention_interval = 4;
    c.max_position_embeddings = 262144;
    c.eos_token_id = 248044;
    c.tie_word_embeddings = true;
    return c;
}
}  // namespace

ConfigQwen35 ConfigQwen35::qwen35_4b() {
    ConfigQwen35 c = qwen35_common();
    c.hidden_size = 2560;
    c.intermediate_size = 9216;
    return c;
}

ConfigQwen35 ConfigQwen35::qwen35_9b() {
    ConfigQwen35 c = qwen35_common();
    c.hidden_size = 4096;
    c.intermediate_size = 12288;
    c.tie_word_embeddings = false;
    return c;
}

ConfigQwen35 ConfigQwen35::qwen35_0_8b() {
    ConfigQwen35 c = qwen35_common();
    c.hidden_size = 1024;
    c.num_hidden_layers = 24;
    c.num_attention_heads = 8;
    c.num_key_value_heads = 2;
    c.linear_num_value_heads = 16;
    c.intermediate_size = 3584;
    return c;
}

// =============================================================================
// Caches
// =============================================================================

DeltaNetState::DeltaNetState(const ConfigQwen35& cfg)
    : state(cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim, 0.0f),
      conv_state(cfg.deltanet_qkv_dim() * (cfg.linear_conv_kernel_dim - 1), 0.0f),
      num_v_heads(cfg.linear_num_value_heads),
      key_head_dim(cfg.linear_key_head_dim),
      value_head_dim(cfg.linear_value_head_dim),
      conv_dim(cfg.deltanet_qkv_dim()),
      conv_kernel(cfg.linear_conv_kernel_dim) {}

void FullAttnKvCache::append(const Mat& new_k, const Mat& new_v) {
    const std::size_t n_new = new_k.rows;
    const std::size_t d = new_k.cols;
    for (std::size_t r = 0; r < n_new; ++r) {
        for (std::size_t c = 0; c < d; ++c) {
            k.at_mut(seq_len + r, c) = new_k.at(r, c);
            v.at_mut(seq_len + r, c) = new_v.at(r, c);
        }
    }
    seq_len += n_new;
}

Qwen35Cache::Qwen35Cache(const ConfigQwen35& cfg, std::size_t max_tokens) {
    const std::size_t max = std::min(max_tokens, cfg.max_position_embeddings);
    layers.reserve(cfg.num_hidden_layers);
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        if (cfg.is_full_attention_layer(i)) {
            layers.emplace_back(FullAttnKvCache(cfg.num_key_value_heads, cfg.head_dim, max));
        } else {
            layers.emplace_back(DeltaNetState(cfg));
        }
    }
}

// =============================================================================
// Scalar helpers
// =============================================================================

float qwen_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float qwen_softplus(float x) {
    // Above ~20 the function is indistinguishable from the identity, and
    // exp(x) would overflow.
    return x > 20.0f ? x : std::log(1.0f + std::exp(x));
}

float qwen_silu(float x) { return x * qwen_sigmoid(x); }

void l2_normalize_heads(std::span<float> x, std::size_t n_heads, std::size_t dim) {
    constexpr float eps = 1e-12f;
    for (std::size_t h = 0; h < n_heads; ++h) {
        float* slice = x.data() + h * dim;
        float sum_sq = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            sum_sq += slice[i] * slice[i];
        }
        const float norm = std::max(std::sqrt(sum_sq), eps);
        for (std::size_t i = 0; i < dim; ++i) {
            slice[i] /= norm;
        }
    }
}

void apply_per_head_norm_raw(std::span<float> x, const RmsNorm2& norm, std::size_t n_heads,
                             std::size_t dim) {
    const Mat& gamma = norm.gamma.data();
    const float eps = norm.eps;
    for (std::size_t h = 0; h < n_heads; ++h) {
        float* slice = x.data() + h * dim;
        float sum_sq = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            sum_sq += slice[i] * slice[i];
        }
        const float rms = std::sqrt(sum_sq / static_cast<float>(dim) + eps);
        for (std::size_t j = 0; j < dim; ++j) {
            slice[j] = (slice[j] / rms) * (1.0f + gamma.at(0, j));
        }
    }
}

void apply_partial_rope(std::span<float> x, std::size_t n_heads, std::size_t head_dim,
                        std::size_t rope_dim, float theta, std::size_t pos) {
    const std::size_t half = rope_dim / 2;
    for (std::size_t h = 0; h < n_heads; ++h) {
        const std::size_t base = h * head_dim;
        for (std::size_t i = 0; i < half; ++i) {
            const float freq =
                1.0f / std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(rope_dim));
            const float angle = static_cast<float>(pos) * freq;
            const float cos_a = std::cos(angle);
            const float sin_a = std::sin(angle);

            // NeoX half-split within the rotated prefix: pair i with i + half.
            const std::size_t idx0 = base + i;
            const std::size_t idx1 = base + i + half;
            const float x0 = x[idx0];
            const float x1 = x[idx1];
            x[idx0] = x0 * cos_a - x1 * sin_a;
            x[idx1] = x1 * cos_a + x0 * sin_a;
        }
    }
}

std::vector<float> qwen_gqa_attention_cached(std::span<const float> q, const Mat& k_cache,
                                             const Mat& v_cache, std::size_t k_start,
                                             std::size_t k_end, std::size_t nq, std::size_t nkv,
                                             std::size_t d, float scale) {
    const std::size_t ctx_len = k_end - k_start;
    std::vector<float> output(nq * d, 0.0f);
    if (ctx_len == 0) {
        return output;
    }

    const std::size_t heads_per_kv = nq / nkv;

    for (std::size_t qh = 0; qh < nq; ++qh) {
        const std::size_t kvh = qh / heads_per_kv;
        const float* q_slice = q.data() + qh * d;

        std::vector<float> scores(ctx_len);
        for (std::size_t t = 0; t < ctx_len; ++t) {
            float dot = 0.0f;
            for (std::size_t j = 0; j < d; ++j) {
                dot += q_slice[j] * k_cache.at(k_start + t, kvh * d + j);
            }
            scores[t] = dot * scale;
        }

        float max_s = -std::numeric_limits<float>::infinity();
        for (float s : scores) {
            max_s = std::max(max_s, s);
        }
        float sum = 0.0f;
        for (float& s : scores) {
            s = std::exp(s - max_s);
            sum += s;
        }
        for (float& s : scores) {
            s /= sum;
        }

        float* out_slice = output.data() + qh * d;
        for (std::size_t t = 0; t < ctx_len; ++t) {
            const float w = scores[t];
            if (w > 0.0f) {
                for (std::size_t j = 0; j < d; ++j) {
                    out_slice[j] += w * v_cache.at(k_start + t, kvh * d + j);
                }
            }
        }
    }

    return output;
}

// =============================================================================
// Qwen35DeltaNet
// =============================================================================

Qwen35DeltaNet::Qwen35DeltaNet(const ConfigQwen35& cfg)
    : in_proj_qkv(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.deltanet_qkv_dim())),
      in_proj_z(Linear2::new_no_bias_zeros(
          cfg.hidden_size, cfg.linear_num_value_heads * cfg.linear_value_head_dim)),
      in_proj_a(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.linear_num_value_heads)),
      in_proj_b(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.linear_num_value_heads)),
      out_proj(Linear2::new_no_bias_zeros(
          cfg.linear_num_value_heads * cfg.linear_value_head_dim, cfg.hidden_size)),
      conv1d_weight(cfg.deltanet_qkv_dim() * cfg.linear_conv_kernel_dim, 0.0f),
      a_log(cfg.linear_num_value_heads, 0.0f),
      dt_bias(cfg.linear_num_value_heads, 0.0f),
      norm_weight(cfg.linear_value_head_dim, 0.0f),
      num_k_heads(cfg.linear_num_key_heads),
      num_v_heads(cfg.linear_num_value_heads),
      key_head_dim(cfg.linear_key_head_dim),
      value_head_dim(cfg.linear_value_head_dim),
      conv_kernel(cfg.linear_conv_kernel_dim),
      qkv_dim(cfg.deltanet_qkv_dim()),
      value_dim(cfg.linear_num_value_heads * cfg.linear_value_head_dim) {}

Qwen35DeltaNet Qwen35DeltaNet::new_for_inference(const ConfigQwen35& cfg) {
    return Qwen35DeltaNet(cfg);
}

std::vector<float> Qwen35DeltaNet::apply_conv1d(std::span<const float> input,
                                                DeltaNetState& state) const {
    const std::size_t dim = qkv_dim;
    const std::size_t ks = conv_kernel;
    const std::size_t hist = ks - 1;
    std::vector<float> output(dim);

    for (std::size_t c = 0; c < dim; ++c) {
        // The kernel spans the stored history plus the incoming sample.
        const std::size_t w_base = c * ks;
        const std::size_t s_base = c * hist;
        float val = 0.0f;
        for (std::size_t t = 0; t < hist; ++t) {
            val += conv1d_weight[w_base + t] * state.conv_state[s_base + t];
        }
        val += conv1d_weight[w_base + hist] * input[c];

        output[c] = qwen_silu(val);

        // Shift the history left and append this sample.
        for (std::size_t t = 0; t + 1 < hist; ++t) {
            state.conv_state[s_base + t] = state.conv_state[s_base + t + 1];
        }
        state.conv_state[s_base + hist - 1] = input[c];
    }

    return output;
}

namespace {

/// How the delta-rule update associates its multiplications.
///
/// The reference's decode and prefill paths spell the same update differently:
/// decode forms `delta = (v - kv_mem) * beta` and then multiplies by `k`, while
/// prefill writes `k * (v - kv_mem) * beta`, which groups as `(k * (v - kv)) *
/// beta`. Float multiplication is not associative, so the two round
/// differently and the paths genuinely diverge in the last ulp. Both are
/// reproduced rather than unified.
enum class DeltaRuleAssoc {
    DeltaFirst,  // k * ((v - kv_mem) * beta)   -- decode
    KeyFirst,    // (k * (v - kv_mem)) * beta   -- prefill
};

/// One DeltaNet token step, shared by decode and prefill.
///
/// Reads the projected rows for a single token, advances `state`, and writes
/// the gated output into `dst` (length nv * vd).
void deltanet_step(const Qwen35DeltaNet& dn, std::span<const float> qkv_row,
                   std::span<const float> z_row, std::span<const float> a_row,
                   std::span<const float> b_row, DeltaNetState& state, std::span<float> dst,
                   DeltaRuleAssoc assoc) {
    const std::size_t nk = dn.num_k_heads;
    const std::size_t nv = dn.num_v_heads;
    const std::size_t kd = dn.key_head_dim;
    const std::size_t vd = dn.value_head_dim;
    const std::size_t v_per_k = nv / nk;
    const std::size_t key_dim = nk * kd;
    const std::size_t value_dim = nv * vd;

    // Causal conv1d + SiLU, then split [Q_all | K_all | V_all].
    const std::vector<float> qkv_conv = dn.apply_conv1d(qkv_row, state);
    std::vector<float> q_flat(qkv_conv.begin(), qkv_conv.begin() + static_cast<std::ptrdiff_t>(key_dim));
    std::vector<float> k_flat(qkv_conv.begin() + static_cast<std::ptrdiff_t>(key_dim),
                              qkv_conv.begin() + static_cast<std::ptrdiff_t>(key_dim * 2));
    std::vector<float> v_flat(
        qkv_conv.begin() + static_cast<std::ptrdiff_t>(key_dim * 2),
        qkv_conv.begin() + static_cast<std::ptrdiff_t>(key_dim * 2 + value_dim));

    // Gates: beta writes, g decays.
    std::vector<float> beta(nv), g_decay(nv);
    for (std::size_t h = 0; h < nv; ++h) {
        beta[h] = qwen_sigmoid(b_row[h]);
        g_decay[h] = -std::exp(dn.a_log[h]) * qwen_softplus(a_row[h] + dn.dt_bias[h]);
    }

    l2_normalize_heads(q_flat, nk, kd);
    l2_normalize_heads(k_flat, nk, kd);

    // Repeat-interleave the key heads up to the value-head count.
    std::vector<float> q_exp(nv * kd), k_exp(nv * kd);
    for (std::size_t g = 0; g < nk; ++g) {
        for (std::size_t vi = 0; vi < v_per_k; ++vi) {
            const std::size_t dst_off = (g * v_per_k + vi) * kd;
            std::copy_n(q_flat.begin() + static_cast<std::ptrdiff_t>(g * kd), kd,
                        q_exp.begin() + static_cast<std::ptrdiff_t>(dst_off));
            std::copy_n(k_flat.begin() + static_cast<std::ptrdiff_t>(g * kd), kd,
                        k_exp.begin() + static_cast<std::ptrdiff_t>(dst_off));
        }
    }

    const float q_scale = 1.0f / std::sqrt(static_cast<float>(kd));
    for (float& value : q_exp) {
        value *= q_scale;
    }

    // Per-head delta-rule state update.
    std::vector<float> output(nv * vd, 0.0f);
    std::vector<float> kv_mem(vd);
    for (std::size_t h = 0; h < nv; ++h) {
        const std::span<float> s = state.head_state(h);  // [kd, vd] row-major
        const float* k_h = k_exp.data() + h * kd;
        const float* v_h = v_flat.data() + h * vd;
        const float* q_h = q_exp.data() + h * kd;
        const float decay = std::exp(g_decay[h]);  // g < 0, so decay < 1
        const float beta_h = beta[h];

        for (float& value : s) {
            value *= decay;
        }

        // kv_mem = S^T k
        std::fill(kv_mem.begin(), kv_mem.end(), 0.0f);
        for (std::size_t i = 0; i < kd; ++i) {
            const float k_i = k_h[i];
            if (k_i != 0.0f) {
                for (std::size_t j = 0; j < vd; ++j) {
                    kv_mem[j] += s[i * vd + j] * k_i;
                }
            }
        }

        // S += outer(k, (v - kv_mem) * beta)
        for (std::size_t i = 0; i < kd; ++i) {
            const float k_i = k_h[i];
            if (k_i != 0.0f) {
                if (assoc == DeltaRuleAssoc::DeltaFirst) {
                    for (std::size_t j = 0; j < vd; ++j) {
                        s[i * vd + j] += k_i * ((v_h[j] - kv_mem[j]) * beta_h);
                    }
                } else {
                    for (std::size_t j = 0; j < vd; ++j) {
                        s[i * vd + j] += k_i * (v_h[j] - kv_mem[j]) * beta_h;
                    }
                }
            }
        }

        // out = S^T q
        float* out_h = output.data() + h * vd;
        for (std::size_t i = 0; i < kd; ++i) {
            const float q_i = q_h[i];
            if (q_i != 0.0f) {
                for (std::size_t j = 0; j < vd; ++j) {
                    out_h[j] += s[i * vd + j] * q_i;
                }
            }
        }
    }

    // Gated RMSNorm per head: rms_norm(out) * norm_weight * silu(z).
    constexpr float eps = 1e-6f;
    for (std::size_t h = 0; h < nv; ++h) {
        const float* out_h = output.data() + h * vd;
        const float* z_h = z_row.data() + h * vd;
        float sum_sq = 0.0f;
        for (std::size_t j = 0; j < vd; ++j) {
            sum_sq += out_h[j] * out_h[j];
        }
        const float rms = std::sqrt(sum_sq / static_cast<float>(vd) + eps);
        for (std::size_t j = 0; j < vd; ++j) {
            dst[h * vd + j] = (out_h[j] / rms) * dn.norm_weight[j] * qwen_silu(z_h[j]);
        }
    }
}

}  // namespace

TensorNode Qwen35DeltaNet::forward_cached(const TensorNode& x, DeltaNetState& state) const {
    const TensorNode qkv_tn = in_proj_qkv.forward(x);  // [1, qkv_dim]
    const TensorNode z_tn = in_proj_z.forward(x);      // [1, value_dim]
    const TensorNode a_tn = in_proj_a.forward(x);      // [1, num_v_heads]
    const TensorNode b_tn = in_proj_b.forward(x);      // [1, num_v_heads]

    std::vector<float> gated(value_dim);
    deltanet_step(*this, qkv_tn.data().data, z_tn.data().data, a_tn.data().data,
                  b_tn.data().data, state, gated, DeltaRuleAssoc::DeltaFirst);

    return out_proj.forward(TensorNode::leaf(Mat(std::move(gated), 1, value_dim)));
}

TensorNode Qwen35DeltaNet::forward_prefill(const TensorNode& x, DeltaNetState& state) const {
    const std::size_t t = x.data().rows;
    const std::size_t nv = num_v_heads;

    // Projections batch as GEMM; the conv1d and recurrence stay sequential.
    const TensorNode qkv_all = in_proj_qkv.forward(x);  // [T, qkv_dim]
    const TensorNode z_all = in_proj_z.forward(x);      // [T, value_dim]
    const TensorNode a_all = in_proj_a.forward(x);      // [T, nv]
    const TensorNode b_all = in_proj_b.forward(x);      // [T, nv]

    std::vector<float> all_gated(t * value_dim);
    for (std::size_t tok = 0; tok < t; ++tok) {
        deltanet_step(*this,
                      std::span(qkv_all.data().data).subspan(tok * qkv_dim, qkv_dim),
                      std::span(z_all.data().data).subspan(tok * value_dim, value_dim),
                      std::span(a_all.data().data).subspan(tok * nv, nv),
                      std::span(b_all.data().data).subspan(tok * nv, nv), state,
                      std::span(all_gated).subspan(tok * value_dim, value_dim),
                      DeltaRuleAssoc::KeyFirst);
    }

    return out_proj.forward(TensorNode::leaf(Mat(std::move(all_gated), t, value_dim)));
}

// =============================================================================
// Qwen35FullAttention
// =============================================================================

Qwen35FullAttention::Qwen35FullAttention(const ConfigQwen35& cfg)
    : q_proj(Linear2::new_no_bias_zeros(cfg.hidden_size,
                                        cfg.num_attention_heads * cfg.head_dim * 2)),
      k_proj(Linear2::new_no_bias_zeros(cfg.hidden_size,
                                        cfg.num_key_value_heads * cfg.head_dim)),
      v_proj(Linear2::new_no_bias_zeros(cfg.hidden_size,
                                        cfg.num_key_value_heads * cfg.head_dim)),
      o_proj(Linear2::new_no_bias_zeros(cfg.num_attention_heads * cfg.head_dim,
                                        cfg.hidden_size)),
      q_norm(cfg.head_dim, cfg.rms_norm_eps),
      k_norm(cfg.head_dim, cfg.rms_norm_eps),
      n_q_heads(cfg.num_attention_heads),
      n_kv_heads(cfg.num_key_value_heads),
      head_dim(cfg.head_dim),
      rope_theta(cfg.rope_theta),
      rope_dim(cfg.rope_dim()) {}

Qwen35FullAttention Qwen35FullAttention::new_for_inference(const ConfigQwen35& cfg) {
    return Qwen35FullAttention(cfg);
}

namespace {
/// Split `q_proj`'s output for one token into the query and gate halves.
/// Each head lays out `d` query values followed by `d` gate values.
void split_q_and_gate(std::span<const float> qg_row, std::size_t nq, std::size_t d,
                      std::span<float> q_out, std::span<float> gate_out) {
    for (std::size_t h = 0; h < nq; ++h) {
        const std::size_t src_base = h * d * 2;
        for (std::size_t j = 0; j < d; ++j) {
            q_out[h * d + j] = qg_row[src_base + j];
            gate_out[h * d + j] = qg_row[src_base + d + j];
        }
    }
}
}  // namespace

TensorNode Qwen35FullAttention::forward_cached(const TensorNode& x,
                                               FullAttnKvCache& cache) const {
    const std::size_t d = head_dim;
    const std::size_t nq = n_q_heads;
    const std::size_t nkv = n_kv_heads;
    const std::size_t seq_offset = cache.seq_len;

    const TensorNode qg_tn = q_proj.forward(x);  // [1, nq*d*2]
    const TensorNode k_tn = k_proj.forward(x);   // [1, nkv*d]
    const TensorNode v_tn = v_proj.forward(x);   // [1, nkv*d]

    std::vector<float> q_raw(nq * d), gate_raw(nq * d);
    split_q_and_gate(qg_tn.data().data, nq, d, q_raw, gate_raw);
    std::vector<float> k_raw = k_tn.data().data;

    // Per-head RMSNorm ((1 + gamma) variant), then partial RoPE.
    apply_per_head_norm_raw(q_raw, q_norm, nq, d);
    apply_per_head_norm_raw(k_raw, k_norm, nkv, d);
    apply_partial_rope(q_raw, nq, d, rope_dim, rope_theta, seq_offset);
    apply_partial_rope(k_raw, nkv, d, rope_dim, rope_theta, seq_offset);

    cache.append(Mat(std::move(k_raw), 1, nkv * d), Mat(v_tn.data().data, 1, nkv * d));

    const std::vector<float> attn_out =
        qwen_gqa_attention_cached(q_raw, cache.k, cache.v, 0, cache.seq_len, nq, nkv, d,
                                  1.0f / std::sqrt(static_cast<float>(d)));

    // Sigmoid output gate.
    std::vector<float> gated(nq * d);
    for (std::size_t i = 0; i < nq * d; ++i) {
        gated[i] = attn_out[i] * qwen_sigmoid(gate_raw[i]);
    }

    return o_proj.forward(TensorNode::leaf(Mat(std::move(gated), 1, nq * d)));
}

TensorNode Qwen35FullAttention::forward_prefill(const TensorNode& x,
                                                FullAttnKvCache& cache) const {
    const std::size_t d = head_dim;
    const std::size_t nq = n_q_heads;
    const std::size_t nkv = n_kv_heads;
    const std::size_t groups = nq / nkv;
    const std::size_t t = x.data().rows;
    const std::size_t start_pos = cache.seq_len;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    const TensorNode qg_tn = q_proj.forward(x);  // [T, nq*d*2]
    const TensorNode k_tn = k_proj.forward(x);   // [T, nkv*d]
    const TensorNode v_tn = v_proj.forward(x);   // [T, nkv*d]

    std::vector<float> q_all(t * nq * d), gate_all(t * nq * d), k_all(t * nkv * d);
    const std::vector<float>& v_all = v_tn.data().data;

    for (std::size_t tok = 0; tok < t; ++tok) {
        const std::size_t q_off = tok * nq * d;
        const std::size_t k_off = tok * nkv * d;

        split_q_and_gate(std::span(qg_tn.data().data).subspan(tok * nq * d * 2, nq * d * 2), nq, d,
                         std::span(q_all).subspan(q_off, nq * d),
                         std::span(gate_all).subspan(q_off, nq * d));
        std::copy_n(k_tn.data().data.begin() + static_cast<std::ptrdiff_t>(k_off), nkv * d,
                    k_all.begin() + static_cast<std::ptrdiff_t>(k_off));

        apply_per_head_norm_raw(std::span(q_all).subspan(q_off, nq * d), q_norm, nq, d);
        apply_per_head_norm_raw(std::span(k_all).subspan(k_off, nkv * d), k_norm, nkv, d);

        const std::size_t pos = start_pos + tok;
        apply_partial_rope(std::span(q_all).subspan(q_off, nq * d), nq, d, rope_dim, rope_theta,
                           pos);
        apply_partial_rope(std::span(k_all).subspan(k_off, nkv * d), nkv, d, rope_dim, rope_theta,
                           pos);
    }

    for (std::size_t tok = 0; tok < t; ++tok) {
        const std::size_t k_off = tok * nkv * d;
        cache.append(Mat(std::vector<float>(k_all.begin() + static_cast<std::ptrdiff_t>(k_off),
                                            k_all.begin() +
                                                static_cast<std::ptrdiff_t>(k_off + nkv * d)),
                         1, nkv * d),
                     Mat(std::vector<float>(v_all.begin() + static_cast<std::ptrdiff_t>(k_off),
                                            v_all.begin() +
                                                static_cast<std::ptrdiff_t>(k_off + nkv * d)),
                         1, nkv * d));
    }

    // Causal attention: query `qi` sees cache positions 0 ..= start_pos + qi.
    std::vector<float> attn_out(t * nq * d, 0.0f);
    for (std::size_t h = 0; h < nq; ++h) {
        const std::size_t kv_h = h / groups;
        for (std::size_t qi = 0; qi < t; ++qi) {
            const std::size_t cache_end = start_pos + qi + 1;
            std::vector<float> scores(cache_end);

            const std::size_t q_off = qi * nq * d + h * d;
            for (std::size_t ki = 0; ki < cache_end; ++ki) {
                const float* k_row = cache.k.data.data() + ki * nkv * d + kv_h * d;
                float dot = 0.0f;
                for (std::size_t j = 0; j < d; ++j) {
                    dot += q_all[q_off + j] * k_row[j];
                }
                scores[ki] = dot * scale;
            }

            float max_s = -std::numeric_limits<float>::infinity();
            for (float s : scores) {
                max_s = std::max(max_s, s);
            }
            float sum_exp = 0.0f;
            for (float& s : scores) {
                s = std::exp(s - max_s);
                sum_exp += s;
            }
            for (float& s : scores) {
                s /= sum_exp;
            }

            const std::size_t out_off = qi * nq * d + h * d;
            for (std::size_t ki = 0; ki < cache_end; ++ki) {
                const float* v_row = cache.v.data.data() + ki * nkv * d + kv_h * d;
                const float w = scores[ki];
                for (std::size_t j = 0; j < d; ++j) {
                    attn_out[out_off + j] += w * v_row[j];
                }
            }
        }
    }

    std::vector<float> gated(t * nq * d);
    for (std::size_t i = 0; i < gated.size(); ++i) {
        gated[i] = attn_out[i] * qwen_sigmoid(gate_all[i]);
    }

    return o_proj.forward(TensorNode::leaf(Mat(std::move(gated), t, nq * d)));
}

// =============================================================================
// Qwen35Mlp
// =============================================================================

Qwen35Mlp::Qwen35Mlp(const ConfigQwen35& cfg)
    : gate_proj(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size)),
      up_proj(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size)),
      down_proj(Linear2::new_no_bias_zeros(cfg.intermediate_size, cfg.hidden_size)) {}

Qwen35Mlp Qwen35Mlp::new_for_inference(const ConfigQwen35& cfg) { return Qwen35Mlp(cfg); }

TensorNode Qwen35Mlp::forward(const TensorNode& x) const {
    return down_proj.forward(gate_proj.forward(x).silu().mul_elem_node(up_proj.forward(x)));
}

// =============================================================================
// Qwen35Block
// =============================================================================

Qwen35Block::Qwen35Block(const ConfigQwen35& cfg, std::size_t layer_idx)
    : input_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      token_mixer(cfg.is_full_attention_layer(layer_idx)
                      ? TokenMixer(Qwen35FullAttention::new_for_inference(cfg))
                      : TokenMixer(Qwen35DeltaNet::new_for_inference(cfg))),
      post_attention_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      mlp(Qwen35Mlp::new_for_inference(cfg)) {}

Qwen35Block Qwen35Block::new_for_inference(const ConfigQwen35& cfg, std::size_t layer_idx) {
    return Qwen35Block(cfg, layer_idx);
}

namespace {
/// Dispatch a block's token mixer against its cache, requiring the two to be
/// the same kind. `step` picks decode or prefill.
template <typename Step>
TensorNode mix_tokens(const TokenMixer& mixer, LayerCache& cache, const TensorNode& normed,
                      Step step) {
    if (const auto* dn = std::get_if<Qwen35DeltaNet>(&mixer)) {
        auto* state = std::get_if<DeltaNetState>(&cache);
        assert(state != nullptr && "layer/cache type mismatch");
        return step(*dn, *state);
    }
    const auto& fa = std::get<Qwen35FullAttention>(mixer);
    auto* kv = std::get_if<FullAttnKvCache>(&cache);
    assert(kv != nullptr && "layer/cache type mismatch");
    return step(fa, *kv);
}
}  // namespace

TensorNode Qwen35Block::forward_cached(const TensorNode& x, LayerCache& cache) const {
    // Pre-norm (the (1 + gamma) variant) -> token mixer -> residual.
    const TensorNode normed = input_layernorm.forward_gemma3(x);
    const TensorNode attn = mix_tokens(
        token_mixer, cache, normed,
        [&normed](const auto& mixer, auto& state) { return mixer.forward_cached(normed, state); });
    const TensorNode x2 = x.add(attn);

    // Pre-norm -> MLP -> residual.
    return x2.add(mlp.forward(post_attention_layernorm.forward_gemma3(x2)));
}

TensorNode Qwen35Block::forward_prefill(const TensorNode& x, LayerCache& cache) const {
    const TensorNode normed = input_layernorm.forward_gemma3(x);
    const TensorNode attn = mix_tokens(
        token_mixer, cache, normed,
        [&normed](const auto& mixer, auto& state) { return mixer.forward_prefill(normed, state); });
    const TensorNode x2 = x.add(attn);

    return x2.add(mlp.forward(post_attention_layernorm.forward_gemma3(x2)));
}

// =============================================================================
// Qwen35Model
// =============================================================================

Qwen35Model::Qwen35Model(ConfigQwen35 cfg)
    : norm(cfg.hidden_size, cfg.rms_norm_eps),
      lm_head(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.vocab_size)),
      config(std::move(cfg)) {
    layers.reserve(config.num_hidden_layers);
    for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers.push_back(Qwen35Block::new_for_inference(config, i));
    }
    lm_head.weight = embed_tokens;
}

Qwen35Model Qwen35Model::new_for_inference(ConfigQwen35 cfg) {
    return Qwen35Model(std::move(cfg));
}

Mat Qwen35Model::embed_rows(const std::vector<std::size_t>& token_ids) const {
    const std::size_t t = token_ids.size();
    const std::size_t h = config.hidden_size;

    // Qwen 3.5 uses the raw embedding rows -- no sqrt(hidden) scaling.
    if (embed_bf16) {
        const std::vector<std::uint16_t>& bits = *embed_bf16->data;
        return Mat::from_fn(t, h, [&](std::size_t row, std::size_t col) {
            return bf16_to_f32(bits[token_ids[row] * h + col]);
        });
    }
    const Mat& te = embed_tokens.data();
    return Mat::from_fn(
        t, h, [&](std::size_t row, std::size_t col) { return te.at(token_ids[row], col); });
}

namespace {
/// Logits for the last row of `hidden`, through the final norm and lm_head.
Mat last_row_logits(const RmsNorm2& norm, const Linear2& lm_head, const Mat& hidden) {
    const Mat last = Mat::from_fn(
        1, hidden.cols, [&](std::size_t, std::size_t c) { return hidden.at(hidden.rows - 1, c); });
    return lm_head.forward(norm.forward_gemma3(TensorNode::leaf(last))).data();
}
}  // namespace

Mat Qwen35Model::prefill(const std::vector<std::size_t>& token_ids, Qwen35Cache& cache) const {
    TensorNode x = TensorNode::leaf(embed_rows(token_ids));
    for (std::size_t i = 0; i < layers.size(); ++i) {
        x = layers[i].forward_prefill(x, cache.layers[i]);
    }
    return last_row_logits(norm, lm_head, x.data());
}

Mat Qwen35Model::decode_step(std::size_t token_id, Qwen35Cache& cache) const {
    TensorNode x = TensorNode::leaf(embed_rows({token_id}));
    for (std::size_t i = 0; i < layers.size(); ++i) {
        x = layers[i].forward_cached(x, cache.layers[i]);
    }
    return last_row_logits(norm, lm_head, x.data());
}

}  // namespace rt

#include "rt/transformer4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

#if RT_FEATURE_BLAS
#include <Accelerate/Accelerate.h>
#endif

#if defined(__APPLE__)
#include <malloc/malloc.h>
#include <mach/mach.h>
#endif

namespace rt {

// =============================================================================
// Config4
// =============================================================================

Config4 Config4::gemma3_1b() {
    Config4 c;
    c.vocab_size = 262144;
    c.hidden_size = 1152;
    c.num_hidden_layers = 26;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 1;
    c.intermediate_size = 6912;
    c.head_dim = 256;
    c.sliding_window = 512;
    c.rope_theta_local = 10000.0f;
    c.rope_theta_global = 1000000.0f;
    c.rope_freq_scale_local = 1.0f;
    c.rope_freq_scale_global = 1.0f;
    c.rms_norm_eps = 1e-6f;
    c.query_pre_attn_scalar = 256.0f;
    c.eos_token_id = 1;  // <eos> in the Gemma tokenizer
    c.max_position_embeddings = 32768;
    return c;
}

Config4 Config4::gemma3_4b() {
    Config4 c;
    c.vocab_size = 262208;
    c.hidden_size = 2560;
    c.num_hidden_layers = 34;
    c.num_attention_heads = 8;
    c.num_key_value_heads = 4;
    c.intermediate_size = 10240;
    c.head_dim = 256;
    c.sliding_window = 1024;
    c.rope_theta_local = 10000.0f;
    c.rope_theta_global = 1000000.0f;
    c.rope_freq_scale_local = 1.0f;
    c.rope_freq_scale_global = 0.125f;  // 1 / rope_scaling.factor
    c.rms_norm_eps = 1e-6f;
    c.query_pre_attn_scalar = 256.0f;
    c.eos_token_id = 1;  // 106 (<end_of_turn>) is checked separately
    c.max_position_embeddings = 32768;
    return c;
}

// =============================================================================
// KV cache
// =============================================================================

void Gemma3LayerKvCache::append(const Mat& new_k, const Mat& new_v) {
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

Mat Gemma3LayerKvCache::k_filled() const {
    return Mat::from_fn(seq_len, k.cols, [this](std::size_t r, std::size_t c) { return k.at(r, c); });
}

Mat Gemma3LayerKvCache::v_filled() const {
    return Mat::from_fn(seq_len, v.cols, [this](std::size_t r, std::size_t c) { return v.at(r, c); });
}

Mat Gemma3LayerKvCache::k_last(std::size_t window) const {
    const std::size_t start = seq_len > window ? seq_len - window : 0;
    return Mat::from_fn(seq_len - start, k.cols,
                        [&](std::size_t r, std::size_t c) { return k.at(start + r, c); });
}

Mat Gemma3LayerKvCache::v_last(std::size_t window) const {
    const std::size_t start = seq_len > window ? seq_len - window : 0;
    return Mat::from_fn(seq_len - start, v.cols,
                        [&](std::size_t r, std::size_t c) { return v.at(start + r, c); });
}

Gemma3KvCache::Gemma3KvCache(const Config4& config, std::size_t max_tokens) {
    const std::size_t max = std::min(max_tokens, config.max_position_embeddings);
    layers.reserve(config.num_hidden_layers);
    for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers.emplace_back(config.num_key_value_heads, config.head_dim, max);
    }
}

void Gemma3KvCache::clear() {
    for (auto& layer : layers) {
        layer.seq_len = 0;
    }
}

void Gemma3KvCache::free() {
    for (auto& layer : layers) {
        layer.k = Mat::zeros(0, 0);
        layer.v = Mat::zeros(0, 0);
        layer.seq_len = 0;
    }
}

// =============================================================================
// Helpers
// =============================================================================

TensorNode apply_per_head_norm(const TensorNode& x, const RmsNorm2& norm, std::size_t t,
                               std::size_t n_heads, std::size_t head_dim) {
    const Mat& x_data = x.data();
    Mat out = Mat::zeros(t, n_heads * head_dim);

    for (std::size_t h = 0; h < n_heads; ++h) {
        // Normalize this head's slice on its own.
        const Mat head_data = Mat::from_fn(t, head_dim, [&](std::size_t row, std::size_t col) {
            return x_data.at(row, h * head_dim + col);
        });
        const Mat normed = norm.forward_gemma3(TensorNode::leaf(head_data)).data();
        for (std::size_t row = 0; row < t; ++row) {
            for (std::size_t col = 0; col < head_dim; ++col) {
                out.at_mut(row, h * head_dim + col) = normed.at(row, col);
            }
        }
    }

    TensorNode result = TensorNode::leaf(std::move(out));

    // Backward runs the Gemma 3 RMSNorm gradient per head, recomputing the RMS
    // from the stored input rather than keeping per-head activations.
    TensorNode x_c = x, result_c = result, gamma_c = norm.gamma;
    const float eps = norm.eps;
    result.set_backward(
        [x_c, result_c, gamma_c, eps, t, n_heads, head_dim] {
            const Mat& dout = result_c.grad();
            const Mat& x_data = x_c.data();
            Mat dx = x_c.grad();

            for (std::size_t h = 0; h < n_heads; ++h) {
                for (std::size_t row = 0; row < t; ++row) {
                    float sq_sum = 0.0f;
                    for (std::size_t col = 0; col < head_dim; ++col) {
                        const float v = x_data.at(row, h * head_dim + col);
                        sq_sum += v * v;
                    }
                    const float inv_rms =
                        1.0f / std::sqrt(sq_sum / static_cast<float>(head_dim) + eps);

                    // The effective scale is (1 + gamma), the Gemma 3 form.
                    float dot_dy_gamma_x = 0.0f;
                    for (std::size_t col = 0; col < head_dim; ++col) {
                        dot_dy_gamma_x += dout.at(row, h * head_dim + col) *
                                          (1.0f + gamma_c.data().at(0, col)) *
                                          x_data.at(row, h * head_dim + col);
                    }

                    for (std::size_t col = 0; col < head_dim; ++col) {
                        const float g = 1.0f + gamma_c.data().at(0, col);
                        const float xi = x_data.at(row, h * head_dim + col);
                        const float dy = dout.at(row, h * head_dim + col);
                        const float term1 = dy * g * inv_rms;
                        const float term2 = xi * inv_rms * inv_rms * inv_rms * dot_dy_gamma_x /
                                            static_cast<float>(head_dim);
                        dx.at_mut(row, h * head_dim + col) += term1 - term2;
                    }
                }
            }
            x_c.set_grad(std::move(dx));
            x_c.call_backward_fn();
        },
        {x});

    return result;
}

namespace {

/// Rotate every head in place using the NeoX half-split pairing, with row `row`
/// at absolute position `pos_of(row)`.
template <typename PosOf>
Mat rope_forward_heads(const Mat& x_data, std::size_t n_heads, std::size_t rows,
                       std::size_t head_dim, float theta, float freq_scale, PosOf pos_of) {
    Mat out = x_data;
    const std::size_t half = head_dim / 2;
    for (std::size_t h = 0; h < n_heads; ++h) {
        for (std::size_t row = 0; row < rows; ++row) {
            const float pos = static_cast<float>(pos_of(row));
            for (std::size_t i = 0; i < half; ++i) {
                const float angle =
                    (pos * freq_scale) /
                    std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
                const float cos_a = std::cos(angle);
                const float sin_a = std::sin(angle);
                const std::size_t c0 = h * head_dim + i;
                const std::size_t c1 = c0 + half;
                const float x0 = x_data.at(row, c0);
                const float x1 = x_data.at(row, c1);
                out.at_mut(row, c0) = x0 * cos_a - x1 * sin_a;
                out.at_mut(row, c1) = x0 * sin_a + x1 * cos_a;
            }
        }
    }
    return out;
}

/// The inverse rotation, accumulated into `dx`.
template <typename PosOf>
void rope_backward_heads(const Mat& dout, Mat& dx, std::size_t n_heads, std::size_t rows,
                         std::size_t head_dim, float theta, float freq_scale, PosOf pos_of) {
    const std::size_t half = head_dim / 2;
    for (std::size_t h = 0; h < n_heads; ++h) {
        for (std::size_t row = 0; row < rows; ++row) {
            const float pos = static_cast<float>(pos_of(row));
            for (std::size_t i = 0; i < half; ++i) {
                const float angle =
                    (pos * freq_scale) /
                    std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
                const float cos_a = std::cos(angle);
                const float sin_a = std::sin(angle);
                const std::size_t c0 = h * head_dim + i;
                const std::size_t c1 = c0 + half;
                const float dy0 = dout.at(row, c0);
                const float dy1 = dout.at(row, c1);
                // Rotation is orthogonal, so backward is the transpose.
                dx.at_mut(row, c0) += dy0 * cos_a + dy1 * sin_a;
                dx.at_mut(row, c1) += -dy0 * sin_a + dy1 * cos_a;
            }
        }
    }
}

}  // namespace

TensorNode apply_rope_to_all_heads(const TensorNode& x, std::size_t n_heads, std::size_t t,
                                   std::size_t head_dim, float theta, float freq_scale) {
    const auto pos_of = [](std::size_t row) { return row; };
    TensorNode result = TensorNode::leaf(
        rope_forward_heads(x.data(), n_heads, t, head_dim, theta, freq_scale, pos_of));

    TensorNode x_c = x, result_c = result;
    result.set_backward(
        [x_c, result_c, n_heads, t, head_dim, theta, freq_scale] {
            Mat dx = x_c.grad();
            rope_backward_heads(result_c.grad(), dx, n_heads, t, head_dim, theta, freq_scale,
                                [](std::size_t row) { return row; });
            x_c.set_grad(std::move(dx));
            x_c.call_backward_fn();
        },
        {x});
    return result;
}

TensorNode apply_rope_at_offset(const TensorNode& x, std::size_t n_heads, std::size_t n_new,
                                std::size_t head_dim, float theta, std::size_t offset,
                                float freq_scale) {
    const auto pos_of = [offset](std::size_t row) { return offset + row; };
    TensorNode result = TensorNode::leaf(
        rope_forward_heads(x.data(), n_heads, n_new, head_dim, theta, freq_scale, pos_of));

    TensorNode x_c = x, result_c = result;
    result.set_backward(
        [x_c, result_c, n_heads, n_new, head_dim, theta, offset, freq_scale] {
            Mat dx = x_c.grad();
            rope_backward_heads(result_c.grad(), dx, n_heads, n_new, head_dim, theta, freq_scale,
                                [offset](std::size_t row) { return offset + row; });
            x_c.set_grad(std::move(dx));
            x_c.call_backward_fn();
        },
        {x});
    return result;
}

TensorNode scale_tensor(const TensorNode& x, float factor) {
    if (std::fabs(factor - 1.0f) < 1e-9f) {
        return x;
    }
    TensorNode result = TensorNode::leaf(x.data().scale(factor));
    TensorNode x_c = x, result_c = result;
    result.set_backward(
        [x_c, result_c, factor] {
            x_c.set_grad(x_c.grad().add(result_c.grad().scale(factor)));
            x_c.call_backward_fn();
        },
        {x});
    return result;
}

TensorNode gqa_attention_full(const TensorNode& q, const TensorNode& k, const TensorNode& v,
                              std::size_t n_q_heads, std::size_t n_kv_heads, std::size_t d_head,
                              float scale) {
    // `batched_gqa_attention` bakes in 1/sqrt(d_head); Gemma 3 wants
    // 1/sqrt(query_pre_attn_scalar). Pre-scaling Q by the ratio gets there
    // without duplicating the kernel.
    const float default_scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    return TensorNode::batched_gqa_attention(scale_tensor(q, scale / default_scale), k, v,
                                             n_q_heads, n_kv_heads, d_head);
}

TensorNode gqa_attention_windowed(const TensorNode& q, const TensorNode& k, const TensorNode& v,
                                  std::size_t n_q_heads, std::size_t n_kv_heads,
                                  std::size_t d_head, std::size_t window, float scale) {
    const std::size_t t = q.data().rows;
    const std::size_t group = n_q_heads / n_kv_heads;
    const Mat& q_data = q.data();
    const Mat& k_data = k.data();
    const Mat& v_data = v.data();

    Mat out_data = Mat::zeros(t, n_q_heads * d_head);

    for (std::size_t h = 0; h < n_q_heads; ++h) {
        const std::size_t kv_head = h / group;
        const Mat q_mat = Mat::from_fn(t, d_head, [&](std::size_t r, std::size_t c) {
            return q_data.at(r, h * d_head + c);
        });
        const Mat k_mat = Mat::from_fn(t, d_head, [&](std::size_t r, std::size_t c) {
            return k_data.at(r, kv_head * d_head + c);
        });
        const Mat v_mat = Mat::from_fn(t, d_head, [&](std::size_t r, std::size_t c) {
            return v_data.at(r, kv_head * d_head + c);
        });

        const Mat scores = q_mat.matmul(k_mat.transpose()).scale(scale);

        // Causal *and* inside the window: key c is visible to query r when
        // c <= r and c >= r + 1 - window.
        Mat masked = Mat::from_fn(t, t, [&](std::size_t r, std::size_t c) {
            const bool causal_ok = c <= r;
            const bool window_ok = r < window || c + window >= r + 1;
            return (causal_ok && window_ok) ? scores.at(r, c)
                                            : -std::numeric_limits<float>::infinity();
        });

        for (std::size_t r = 0; r < t; ++r) {
            float row_max = -std::numeric_limits<float>::infinity();
            for (std::size_t c = 0; c < t; ++c) {
                row_max = std::max(row_max, masked.at(r, c));
            }
            float sum_exp = 0.0f;
            for (std::size_t c = 0; c < t; ++c) {
                const float value = masked.at(r, c) == -std::numeric_limits<float>::infinity()
                                        ? 0.0f
                                        : std::exp(masked.at(r, c) - row_max);
                masked.at_mut(r, c) = value;
                sum_exp += value;
            }
            if (sum_exp > 0.0f) {
                for (std::size_t c = 0; c < t; ++c) {
                    masked.at_mut(r, c) /= sum_exp;
                }
            }
        }

        const Mat head_out = masked.matmul(v_mat);
        for (std::size_t r = 0; r < t; ++r) {
            for (std::size_t c = 0; c < d_head; ++c) {
                out_data.at_mut(r, h * d_head + c) = head_out.at(r, c);
            }
        }
    }

    // Inference-only: no backward is wired for the windowed path.
    return TensorNode::leaf(std::move(out_data));
}

Mat gqa_attention_cached(const Mat& q_data, const Mat& k_cache, const Mat& v_cache,
                         std::size_t k_start, std::size_t k_end, std::size_t n_q_heads,
                         std::size_t n_kv_heads, std::size_t d_head, float scale) {
    const std::size_t t_q = q_data.rows;
    const std::size_t t_kv = k_end - k_start;
    const std::size_t kv_stride = k_cache.cols;  // n_kv_heads * d_head
    const std::size_t group = n_q_heads / n_kv_heads;
    Mat out = Mat::zeros(t_q, n_q_heads * d_head);

    // --- Decode path (a single query token) ---
    // No per-head temporaries at all.
    if (t_q == 1) {
        std::vector<float> scores(t_kv);

        for (std::size_t qh = 0; qh < n_q_heads; ++qh) {
            const std::size_t kvh = qh / group;
            const std::size_t q_off = qh * d_head;    // into q_data row 0
            const std::size_t kv_off = kvh * d_head;  // into each cache row
            const std::size_t out_off = qh * d_head;  // into out row 0

            // scores[c] = dot(q_h, k_h[c]) * scale
            for (std::size_t ci = 0; ci < t_kv; ++ci) {
                const std::size_t k_base = (k_start + ci) * kv_stride + kv_off;
#if RT_FEATURE_BLAS
                scores[ci] = cblas_sdot(static_cast<int>(d_head), q_data.data.data() + q_off, 1,
                                        k_cache.data.data() + k_base, 1) *
                             scale;
#else
                float dot = 0.0f;
                for (std::size_t di = 0; di < d_head; ++di) {
                    dot += q_data.data[q_off + di] * k_cache.data[k_base + di];
                }
                scores[ci] = dot * scale;
#endif
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
            if (sum_exp > 0.0f) {
                for (float& s : scores) {
                    s /= sum_exp;
                }
            }

            // out_h = sum_c scores[c] * v_h[c]
            for (std::size_t ci = 0; ci < t_kv; ++ci) {
                const std::size_t v_base = (k_start + ci) * kv_stride + kv_off;
#if RT_FEATURE_BLAS
                cblas_saxpy(static_cast<int>(d_head), scores[ci], v_cache.data.data() + v_base, 1,
                            out.data.data() + out_off, 1);
#else
                for (std::size_t di = 0; di < d_head; ++di) {
                    out.data[out_off + di] += scores[ci] * v_cache.data[v_base + di];
                }
#endif
            }
        }
        return out;
    }

    // --- Prefill path (multiple query tokens) ---
    // Per-head copies buy contiguous matrices for sgemm.
    for (std::size_t qh = 0; qh < n_q_heads; ++qh) {
        const std::size_t kvh = qh / group;
        const std::size_t kv_off = kvh * d_head;

        const Mat q_h = Mat::from_fn(t_q, d_head, [&](std::size_t r, std::size_t c) {
            return q_data.at(r, qh * d_head + c);
        });
        const Mat k_h = Mat::from_fn(t_kv, d_head, [&](std::size_t r, std::size_t c) {
            return k_cache.data[(k_start + r) * kv_stride + kv_off + c];
        });
        const Mat v_h = Mat::from_fn(t_kv, d_head, [&](std::size_t r, std::size_t c) {
            return v_cache.data[(k_start + r) * kv_stride + kv_off + c];
        });

        const Mat raw_scores = q_h.matmul(k_h.transpose()).scale(scale);

        // Causal masking relative to k_start. Window index c holds absolute
        // position k_start + c, and query r sits at absolute position r during
        // prefill, so key c is visible when k_start + c <= r.
        //
        // When r < k_start the query precedes every windowed key, and the row
        // stays all zero -- which happens when the prompt is longer than the
        // window.
        Mat w = Mat::zeros(t_q, t_kv);
        for (std::size_t r = 0; r < t_q; ++r) {
            if (r < k_start) {
                continue;
            }
            const std::size_t max_kv = std::min(r - k_start, t_kv - 1);
            float row_max = -std::numeric_limits<float>::infinity();
            for (std::size_t c = 0; c <= max_kv; ++c) {
                row_max = std::max(row_max, raw_scores.at(r, c));
            }
            float row_sum = 0.0f;
            for (std::size_t c = 0; c < t_kv; ++c) {
                const float e = c <= max_kv ? std::exp(raw_scores.at(r, c) - row_max) : 0.0f;
                w.at_mut(r, c) = e;
                row_sum += e;
            }
            if (row_sum > 0.0f) {
                for (std::size_t c = 0; c < t_kv; ++c) {
                    w.at_mut(r, c) /= row_sum;
                }
            }
        }

        const Mat out_h = w.matmul(v_h);
        for (std::size_t r = 0; r < t_q; ++r) {
            for (std::size_t c = 0; c < d_head; ++c) {
                out.at_mut(r, qh * d_head + c) = out_h.at(r, c);
            }
        }
    }

    return out;
}

// =============================================================================
// Gemma3Attention
// =============================================================================

namespace {
/// The per-layer attention constants, which depend only on the config and
/// whether this layer is global.
struct LayerAttnParams {
    float attn_scale;
    std::optional<std::size_t> sliding_window;
    float rope_theta;
    float rope_freq_scale;
};

[[nodiscard]] LayerAttnParams layer_attn_params(const Config4& cfg, std::size_t layer_idx) {
    const bool is_global = cfg.is_global_layer(layer_idx);
    return {1.0f / std::sqrt(cfg.query_pre_attn_scalar),
            is_global ? std::nullopt : cfg.sliding_window,
            is_global ? cfg.rope_theta_global : cfg.rope_theta_local,
            is_global ? cfg.rope_freq_scale_global : cfg.rope_freq_scale_local};
}
}  // namespace

Gemma3Attention::Gemma3Attention(const Config4& cfg, std::size_t layer_idx, InitRng& rng)
    : q_proj(Linear2::new_no_bias(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, rng)),
      k_proj(Linear2::new_no_bias(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, rng)),
      v_proj(Linear2::new_no_bias(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, rng)),
      o_proj(Linear2::new_no_bias(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, rng)),
      q_norm(cfg.head_dim, cfg.rms_norm_eps),
      k_norm(cfg.head_dim, cfg.rms_norm_eps),
      n_q_heads(cfg.num_attention_heads),
      n_kv_heads(cfg.num_key_value_heads),
      head_dim(cfg.head_dim) {
    const LayerAttnParams p = layer_attn_params(cfg, layer_idx);
    attn_scale = p.attn_scale;
    sliding_window = p.sliding_window;
    rope_theta = p.rope_theta;
    rope_freq_scale = p.rope_freq_scale;
}

Gemma3Attention::Gemma3Attention(const Config4& cfg, std::size_t layer_idx, InferenceInit)
    : q_proj(Linear2::new_no_bias_zeros(cfg.hidden_size,
                                        cfg.num_attention_heads * cfg.head_dim)),
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
      head_dim(cfg.head_dim) {
    const LayerAttnParams p = layer_attn_params(cfg, layer_idx);
    attn_scale = p.attn_scale;
    sliding_window = p.sliding_window;
    rope_theta = p.rope_theta;
    rope_freq_scale = p.rope_freq_scale;
}

Gemma3Attention Gemma3Attention::new_for_inference(const Config4& cfg, std::size_t layer_idx) {
    return Gemma3Attention(cfg, layer_idx, InferenceInit{});
}

TensorNode Gemma3Attention::forward(const TensorNode& x) const {
    const std::size_t t = x.data().rows;

    const TensorNode q_raw = q_proj.forward(x);  // [T, nq*d]
    const TensorNode k_raw = k_proj.forward(x);  // [T, nkv*d]
    const TensorNode v = v_proj.forward(x);      // [T, nkv*d]

    // Per-head RMSNorm on Q and K, then RoPE (NeoX half-split pairing).
    const TensorNode q = apply_rope_to_all_heads(
        apply_per_head_norm(q_raw, q_norm, t, n_q_heads, head_dim), n_q_heads, t, head_dim,
        rope_theta, rope_freq_scale);
    const TensorNode k = apply_rope_to_all_heads(
        apply_per_head_norm(k_raw, k_norm, t, n_kv_heads, head_dim), n_kv_heads, t, head_dim,
        rope_theta, rope_freq_scale);

    const TensorNode attn_out =
        sliding_window ? gqa_attention_windowed(q, k, v, n_q_heads, n_kv_heads, head_dim,
                                                *sliding_window, attn_scale)
                       : gqa_attention_full(q, k, v, n_q_heads, n_kv_heads, head_dim, attn_scale);

    return o_proj.forward(attn_out);
}

TensorNode Gemma3Attention::forward_cached(const TensorNode& x,
                                           Gemma3LayerKvCache& cache) const {
    const std::size_t n_new = x.data().rows;
    const std::size_t seq_offset = cache.seq_len;

    const TensorNode q_raw = q_proj.forward(x);
    const TensorNode k_raw = k_proj.forward(x);
    const TensorNode v = v_proj.forward(x);

    const TensorNode q = apply_rope_at_offset(
        apply_per_head_norm(q_raw, q_norm, n_new, n_q_heads, head_dim), n_q_heads, n_new, head_dim,
        rope_theta, seq_offset, rope_freq_scale);
    const TensorNode k = apply_rope_at_offset(
        apply_per_head_norm(k_raw, k_norm, n_new, n_kv_heads, head_dim), n_kv_heads, n_new,
        head_dim, rope_theta, seq_offset, rope_freq_scale);

    cache.append(k.data(), v.data());

    // Attend over cache rows [k_start, k_end) -- the whole cache on global
    // layers, the trailing window on local ones.
    const std::size_t k_end = cache.seq_len;
    const std::size_t k_start =
        sliding_window ? (k_end > *sliding_window ? k_end - *sliding_window : 0) : 0;

    Mat attn_out = gqa_attention_cached(q.data(), cache.k, cache.v, k_start, k_end, n_q_heads,
                                        n_kv_heads, head_dim, attn_scale);

    return o_proj.forward(TensorNode::leaf(std::move(attn_out)));
}

std::vector<TensorNode> Gemma3Attention::parameters() const {
    return {q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight, q_norm.gamma, k_norm.gamma};
}

// =============================================================================
// Gemma3Mlp
// =============================================================================

Gemma3Mlp::Gemma3Mlp(const Config4& cfg, InitRng& rng)
    : gate_proj(Linear2::new_no_bias(cfg.hidden_size, cfg.intermediate_size, rng)),
      up_proj(Linear2::new_no_bias(cfg.hidden_size, cfg.intermediate_size, rng)),
      down_proj(Linear2::new_no_bias(cfg.intermediate_size, cfg.hidden_size, rng)) {}

Gemma3Mlp::Gemma3Mlp(const Config4& cfg, InferenceInit)
    : gate_proj(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size)),
      up_proj(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size)),
      down_proj(Linear2::new_no_bias_zeros(cfg.intermediate_size, cfg.hidden_size)) {}

Gemma3Mlp Gemma3Mlp::new_for_inference(const Config4& cfg) {
    return Gemma3Mlp(cfg, InferenceInit{});
}

TensorNode Gemma3Mlp::forward(const TensorNode& x) const {
    // Gemma 3 gates with gelu_pytorch_tanh, not SiLU.
    const TensorNode gate = gate_proj.forward(x).gelu_tanh();
    return down_proj.forward(gate.mul_elem_node(up_proj.forward(x)));
}

std::vector<TensorNode> Gemma3Mlp::parameters() const {
    return {gate_proj.weight, up_proj.weight, down_proj.weight};
}

// =============================================================================
// Gemma3Block
// =============================================================================

Gemma3Block::Gemma3Block(const Config4& cfg, std::size_t layer_idx, InitRng& rng)
    : input_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      self_attn(cfg, layer_idx, rng),
      post_attention_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      pre_feedforward_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      post_feedforward_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      mlp(cfg, rng) {}

Gemma3Block::Gemma3Block(const Config4& cfg, std::size_t layer_idx, InferenceInit)
    : input_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      self_attn(cfg, layer_idx, InferenceInit{}),
      post_attention_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      pre_feedforward_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      post_feedforward_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      mlp(cfg, InferenceInit{}) {}

Gemma3Block Gemma3Block::new_for_inference(const Config4& cfg, std::size_t layer_idx) {
    return Gemma3Block(cfg, layer_idx, InferenceInit{});
}

TensorNode Gemma3Block::forward(const TensorNode& x) const {
    const TensorNode attn =
        post_attention_layernorm.forward_gemma3(self_attn.forward(input_layernorm.forward_gemma3(x)));
    const TensorNode x2 = x.add(attn);

    const TensorNode mlp_out = post_feedforward_layernorm.forward_gemma3(
        mlp.forward(pre_feedforward_layernorm.forward_gemma3(x2)));
    return x2.add(mlp_out);
}

TensorNode Gemma3Block::forward_cached(const TensorNode& x, Gemma3LayerKvCache& cache) const {
    const TensorNode attn = post_attention_layernorm.forward_gemma3(
        self_attn.forward_cached(input_layernorm.forward_gemma3(x), cache));
    const TensorNode x2 = x.add(attn);

    const TensorNode mlp_out = post_feedforward_layernorm.forward_gemma3(
        mlp.forward(pre_feedforward_layernorm.forward_gemma3(x2)));
    return x2.add(mlp_out);
}

std::vector<TensorNode> Gemma3Block::parameters() const {
    std::vector<TensorNode> p{input_layernorm.gamma};
    const std::vector<TensorNode> attn = self_attn.parameters();
    p.insert(p.end(), attn.begin(), attn.end());
    p.push_back(post_attention_layernorm.gamma);
    p.push_back(pre_feedforward_layernorm.gamma);
    p.push_back(post_feedforward_layernorm.gamma);
    const std::vector<TensorNode> mlp_p = mlp.parameters();
    p.insert(p.end(), mlp_p.begin(), mlp_p.end());
    return p;
}

// =============================================================================
// Gemma3Model
// =============================================================================

Gemma3Model::Gemma3Model(Config4 cfg, InitRng& rng)
    : embed_tokens(TensorNode::leaf(Mat(rng.normal_vec(cfg.vocab_size * cfg.hidden_size, 0.02f),
                                        cfg.vocab_size, cfg.hidden_size))),
      norm(cfg.hidden_size, cfg.rms_norm_eps),
      lm_head(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.vocab_size)),
      config(std::move(cfg)) {
    layers.reserve(config.num_hidden_layers);
    for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers.emplace_back(config, i, rng);
    }
    // lm_head is weight-tied to embed_tokens: sharing the node shares storage.
    lm_head.weight = embed_tokens;
    lm_head.bias = TensorNode::leaf(Mat::zeros(1, config.vocab_size));
}

Gemma3Model::Gemma3Model(Config4 cfg, InferenceInit)
    : norm(cfg.hidden_size, cfg.rms_norm_eps),
      lm_head(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.vocab_size)),
      config(std::move(cfg)) {
    layers.reserve(config.num_hidden_layers);
    for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers.push_back(Gemma3Block::new_for_inference(config, i));
    }
    // The embedding stays a 0x0 placeholder; the real table arrives as BF16.
    // lm_head has no bias in Gemma 3, so it stays zero-sized too.
    lm_head.weight = embed_tokens;
}

Gemma3Model Gemma3Model::new_for_inference(Config4 cfg) {
    return Gemma3Model(std::move(cfg), InferenceInit{});
}

void Gemma3Model::clear_cpu_weights() {
    for (Gemma3Block& layer : layers) {
        layer.self_attn.q_proj.clear_weight_data();
        layer.self_attn.k_proj.clear_weight_data();
        layer.self_attn.v_proj.clear_weight_data();
        layer.self_attn.o_proj.clear_weight_data();
        layer.mlp.gate_proj.clear_weight_data();
        layer.mlp.up_proj.clear_weight_data();
        layer.mlp.down_proj.clear_weight_data();
        // Norm gammas are ~10 KB each -- not worth clearing.
    }
    // The embedding table is the big one: 262144 x 2560 x 2 bytes ~= 1.3 GB.
    if (embed_bf16) {
        mark_pages_reusable(*embed_bf16->data);
    }
    embed_bf16.reset();
    embed_tokens.set_data(Mat::zeros(0, 0));
    lm_head.clear_weight_data();
    std::fprintf(stderr, "[ Metal ] Freed CPU-side weight data (~3 GB)\n");
}

void Gemma3Model::quantize_all_weights() {
    for (Gemma3Block& layer : layers) {
        layer.self_attn.q_proj.quantize_bf16_and_free();
        layer.self_attn.k_proj.quantize_bf16_and_free();
        layer.self_attn.v_proj.quantize_bf16_and_free();
        layer.self_attn.o_proj.quantize_bf16_and_free();
        layer.mlp.gate_proj.quantize_bf16_and_free();
        layer.mlp.up_proj.quantize_bf16_and_free();
        layer.mlp.down_proj.quantize_bf16_and_free();
    }
}

Mat Gemma3Model::embed_rows(const std::vector<std::size_t>& token_ids) const {
    const std::size_t t = token_ids.size();
    const std::size_t h = config.hidden_size;
    // Gemma 3 multiplies the looked-up embeddings by sqrt(hidden_size).
    const float scale = std::sqrt(static_cast<float>(h));

    if (embed_bf16) {
        const std::vector<std::uint16_t>& bits = *embed_bf16->data;
        return Mat::from_fn(t, h, [&](std::size_t row, std::size_t col) {
            return bf16_to_f32(bits[token_ids[row] * h + col]) * scale;
        });
    }
    const Mat& te = embed_tokens.data();
    return Mat::from_fn(
        t, h, [&](std::size_t row, std::size_t col) { return te.at(token_ids[row], col) * scale; });
}

TensorNode Gemma3Model::forward(const std::vector<std::size_t>& token_ids) const {
    const std::size_t h = config.hidden_size;
    const float scale = std::sqrt(static_cast<float>(h));

    TensorNode x = TensorNode::leaf(embed_rows(token_ids));

    // Backward scatters the gradient back into the embedding rows that were
    // looked up.
    TensorNode embed_node = embed_tokens, x_c = x;
    const std::vector<std::size_t> ids = token_ids;
    x.set_backward(
        [embed_node, x_c, ids, h, scale] {
            const Mat& dout = x_c.grad();
            Mat dte = embed_node.grad();
            for (std::size_t row = 0; row < ids.size(); ++row) {
                for (std::size_t col = 0; col < h; ++col) {
                    dte.at_mut(ids[row], col) += dout.at(row, col) * scale;
                }
            }
            embed_node.set_grad(std::move(dte));
        },
        {embed_tokens});

    for (const Gemma3Block& layer : layers) {
        x = layer.forward(x);
    }

    return lm_head.forward(norm.forward_gemma3(x));
}

Mat Gemma3Model::forward_cached(const std::vector<std::size_t>& token_ids,
                                Gemma3KvCache& cache) const {
    TensorNode x = TensorNode::leaf(embed_rows(token_ids));

    for (std::size_t i = 0; i < layers.size(); ++i) {
        x = layers[i].forward_cached(x, cache.layers[i]);
    }

    // Only the last row's logits matter, so the lm_head GEMV runs on one row.
    const Mat& hidden = x.data();
    const Mat last_row = Mat::from_fn(
        1, hidden.cols, [&](std::size_t, std::size_t c) { return hidden.at(hidden.rows - 1, c); });

    return lm_head.forward(norm.forward_gemma3(TensorNode::leaf(last_row))).data();
}

std::vector<TensorNode> Gemma3Model::parameters() const {
    std::vector<TensorNode> p{embed_tokens};
    for (const Gemma3Block& layer : layers) {
        const std::vector<TensorNode> lp = layer.parameters();
        p.insert(p.end(), lp.begin(), lp.end());
    }
    p.push_back(norm.gamma);
    // lm_head is weight-tied, so this is the same node as embed_tokens and the
    // list holds it twice. Kept as-is to match the reference parameter order.
    p.push_back(lm_head.weight);
    return p;
}

TensorNode Gemma3Model::forward_tokens(const std::vector<std::size_t>& token_ids) const {
    return forward(token_ids);
}

TensorNode Gemma3Model::loss_tokens(const std::vector<std::size_t>& token_ids,
                                    const std::vector<std::size_t>& targets) const {
    const TensorNode logits_node = forward(token_ids);
    const Mat& logits = logits_node.data();
    const std::size_t t = logits.rows;
    const std::size_t v = logits.cols;
    assert(t == targets.size());

    // Softmax and cross-entropy in one pass; the stored probabilities are what
    // backward needs.
    Mat probs = Mat::zeros(t, v);
    float loss_val = 0.0f;
    for (std::size_t r = 0; r < t; ++r) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < v; ++c) {
            row_max = std::max(row_max, logits.at(r, c));
        }
        float sum_exp = 0.0f;
        for (std::size_t c = 0; c < v; ++c) {
            const float e = std::exp(logits.at(r, c) - row_max);
            probs.at_mut(r, c) = e;
            sum_exp += e;
        }
        for (std::size_t c = 0; c < v; ++c) {
            probs.at_mut(r, c) /= sum_exp;
        }
        loss_val -= std::max(std::log(probs.at(r, targets[r])), -100.0f);
    }
    loss_val /= static_cast<float>(t);

    TensorNode loss = TensorNode::leaf(Mat({loss_val}, 1, 1));
    TensorNode logits_c = logits_node;
    const std::vector<std::size_t> targets_v = targets;

    // d(logits) = (softmax - one_hot) / T, the fused softmax + cross-entropy rule.
    loss.set_backward(
        [logits_c, probs = std::move(probs), targets_v, t, v] {
            Mat dlogits = logits_c.grad();
            for (std::size_t r = 0; r < t; ++r) {
                for (std::size_t c = 0; c < v; ++c) {
                    const float ind = c == targets_v[r] ? 1.0f : 0.0f;
                    dlogits.at_mut(r, c) += (probs.at(r, c) - ind) / static_cast<float>(t);
                }
            }
            logits_c.set_grad(std::move(dlogits));
            logits_c.call_backward_fn();
        },
        {logits_node});

    return loss;
}

// =============================================================================
// Sampling
// =============================================================================

std::size_t sample_token(const Mat& logits, std::size_t row, const SamplingParams& params,
                         const std::vector<std::size_t>& seen, LcgRng& rng) {
    const std::size_t v = logits.cols;
    std::vector<float> scores(v);
    for (std::size_t c = 0; c < v; ++c) {
        scores[c] = logits.at(row, c);
    }

    // Repetition penalty, over the last 64 generated tokens.
    if (params.repetition_penalty != 1.0f) {
        const std::size_t start = seen.size() > 64 ? seen.size() - 64 : 0;
        for (std::size_t i = start; i < seen.size(); ++i) {
            const std::size_t tok = seen[i];
            if (tok < v) {
                if (scores[tok] >= 0.0f) {
                    scores[tok] /= params.repetition_penalty;
                } else {
                    scores[tok] *= params.repetition_penalty;
                }
            }
        }
    }

    if (params.temperature > 0.0f && params.temperature != 1.0f) {
        for (float& s : scores) {
            s /= params.temperature;
        }
    }

    float max_s = -std::numeric_limits<float>::infinity();
    for (float s : scores) {
        max_s = std::max(max_s, s);
    }
    std::vector<float> probs(v);
    float sum = 0.0f;
    for (std::size_t c = 0; c < v; ++c) {
        probs[c] = std::exp(scores[c] - max_s);
        sum += probs[c];
    }
    for (float& p : probs) {
        p /= sum;
    }

    const auto renormalize = [&probs] {
        float s = 0.0f;
        for (float p : probs) {
            s += p;
        }
        if (s > 0.0f) {
            for (float& p : probs) {
                p /= s;
            }
        }
    };

    if (params.top_k > 0 && params.top_k < v) {
        std::vector<std::pair<std::size_t, float>> indexed(v);
        for (std::size_t c = 0; c < v; ++c) {
            indexed[c] = {c, probs[c]};
        }
        std::sort(indexed.begin(), indexed.end(),
                  [](const auto& a, const auto& b) { return b.second < a.second; });
        for (std::size_t i = params.top_k; i < v; ++i) {
            probs[indexed[i].first] = 0.0f;
        }
        renormalize();
    }

    // Nucleus sampling, applied after top-k so it works on the already
    // truncated distribution.
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        std::vector<std::pair<std::size_t, float>> indexed;
        for (std::size_t c = 0; c < v; ++c) {
            if (probs[c] > 0.0f) {
                indexed.emplace_back(c, probs[c]);
            }
        }
        std::sort(indexed.begin(), indexed.end(),
                  [](const auto& a, const auto& b) { return b.second < a.second; });
        float cumulative = 0.0f;
        std::size_t cutoff = indexed.size();
        for (std::size_t i = 0; i < indexed.size(); ++i) {
            cumulative += indexed[i].second;
            if (cumulative >= params.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        for (std::size_t i = cutoff; i < indexed.size(); ++i) {
            probs[indexed[i].first] = 0.0f;
        }
        renormalize();
    }

    if (params.temperature == 0.0f) {
        // total_cmp ordering: NaN-safe argmax.
        std::size_t best = 0;
        for (std::size_t c = 1; c < v; ++c) {
            if (probs[best] < probs[c] || std::isnan(probs[best])) {
                best = c;
            }
        }
        return best;
    }

    const float r = rng.next_f32();
    float cumulative = 0.0f;
    for (std::size_t i = 0; i < v; ++i) {
        cumulative += probs[i];
        if (r <= cumulative) {
            return i;
        }
    }
    return v - 1;
}

// =============================================================================
// Memory utilities
// =============================================================================

std::vector<std::uint16_t> f32s_to_bf16_and_drop(std::vector<float> f32s) {
    std::vector<std::uint16_t> out(f32s.size());
    for (std::size_t i = 0; i < f32s.size(); ++i) {
        out[i] = f32_to_bf16(f32s[i]);
    }
    // Release the f32 storage now rather than at the caller's scope exit.
    std::vector<float>().swap(f32s);
    out.shrink_to_fit();
    return out;
}

void release_memory_to_os() {
#if defined(__APPLE__)
    malloc_zone_pressure_relief(nullptr, 0);
#endif
}

void print_rss(const char* label) {
#if defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS) {
        std::fprintf(stderr, "[ RSS ] %-32s %.2f GB\n", label,
                     static_cast<double>(info.resident_size) / (1024.0 * 1024.0 * 1024.0));
    }
#else
    (void)label;
#endif
}

}  // namespace rt

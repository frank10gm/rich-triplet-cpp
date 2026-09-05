#include "rt/transformer6.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

#include "rt/transformer4.hpp"

namespace rt {

// =============================================================================
// Config6
// =============================================================================

Config6 Config6::omnivoice() { return Config6{}; }

std::vector<float> Config6::inv_freq() const {
    const std::size_t half = head_dim / 2;
    std::vector<float> out(half);
    for (std::size_t i = 0; i < half; ++i) {
        const float exponent = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
        out[i] = 1.0f / std::pow(rope_theta, exponent);
    }
    return out;
}

// =============================================================================
// Attention helpers
// =============================================================================

Mat apply_head_norm(const Mat& x, const RmsNorm2& norm, std::size_t n_heads,
                    std::size_t head_dim) {
    assert(x.cols == n_heads * head_dim && "apply_head_norm: width mismatch");
    const Mat& gamma = norm.gamma.data();
    assert(gamma.cols == head_dim && "apply_head_norm: gamma is not head_dim wide");

    Mat out = Mat::zeros(x.rows, x.cols);
    for (std::size_t r = 0; r < x.rows; ++r) {
        const float* row = x.row(r).data();
        float* orow = out.row_mut(r).data();
        for (std::size_t h = 0; h < n_heads; ++h) {
            const std::size_t base = h * head_dim;
            // Each head normalizes over its own slice, independently.
            float sum_sq = 0.0f;
            for (std::size_t i = 0; i < head_dim; ++i) {
                sum_sq += row[base + i] * row[base + i];
            }
            const float inv =
                1.0f / std::sqrt(sum_sq / static_cast<float>(head_dim) + norm.eps);
            for (std::size_t i = 0; i < head_dim; ++i) {
                orow[base + i] = row[base + i] * inv * gamma.data[i];
            }
        }
    }
    return out;
}

Mat bidirectional_gqa_attention(const Mat& q, const Mat& k, const Mat& v, std::size_t n_q_heads,
                                std::size_t n_kv_heads, std::size_t head_dim, float scale) {
    const std::size_t t = q.rows;
    const std::size_t group = n_q_heads / n_kv_heads;
    Mat out = Mat::zeros(t, n_q_heads * head_dim);

    // Per-head contiguous copies so each head's scores are one sgemm. With no
    // causal mask the score matrix is dense, which is what makes this the
    // BLAS-friendly shape despite being O(T^2).
    for (std::size_t qh = 0; qh < n_q_heads; ++qh) {
        const std::size_t kvh = qh / group;
        const Mat q_h = Mat::from_fn(t, head_dim, [&](std::size_t r, std::size_t c) {
            return q.at(r, qh * head_dim + c);
        });
        const Mat k_h = Mat::from_fn(t, head_dim, [&](std::size_t r, std::size_t c) {
            return k.at(r, kvh * head_dim + c);
        });
        const Mat v_h = Mat::from_fn(t, head_dim, [&](std::size_t r, std::size_t c) {
            return v.at(r, kvh * head_dim + c);
        });

        Mat scores = q_h.matmul_bt(k_h);
        for (std::size_t r = 0; r < t; ++r) {
            float* row = scores.row_mut(r).data();
            float max_s = -std::numeric_limits<float>::infinity();
            for (std::size_t c = 0; c < t; ++c) {
                row[c] *= scale;
                max_s = std::max(max_s, row[c]);
            }
            float sum_exp = 0.0f;
            for (std::size_t c = 0; c < t; ++c) {
                row[c] = std::exp(row[c] - max_s);
                sum_exp += row[c];
            }
            if (sum_exp > 0.0f) {
                for (std::size_t c = 0; c < t; ++c) {
                    row[c] /= sum_exp;
                }
            }
        }

        const Mat head_out = scores.matmul(v_h);
        for (std::size_t r = 0; r < t; ++r) {
            for (std::size_t c = 0; c < head_dim; ++c) {
                out.at_mut(r, qh * head_dim + c) = head_out.at(r, c);
            }
        }
    }
    return out;
}

// =============================================================================
// OmniAttention
// =============================================================================

OmniAttention::OmniAttention(const Config6& cfg)
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
      head_dim(cfg.head_dim),
      attn_scale(1.0f / std::sqrt(static_cast<float>(cfg.head_dim))),
      rope_pairing(cfg.rope_pairing) {}

Mat OmniAttention::forward(const Mat& x, const std::vector<float>& inv_freq) const {
    assert(inv_freq.size() >= head_dim / 2 && "omnivoice lm: not enough inverse frequencies");
    const TensorNode xn = TensorNode::leaf(x);

    // Qwen3 normalizes each head of Q and K before rotating, like Gemma 3.
    const Mat q_normed = apply_head_norm(q_proj.forward(xn).data(), q_norm, n_q_heads, head_dim);
    const Mat k_normed = apply_head_norm(k_proj.forward(xn).data(), k_norm, n_kv_heads, head_dim);

    // Positions are absolute from 0: there is no cache, so the whole sequence
    // is rotated every pass.
    const Mat q = llama_rope(q_normed, n_q_heads, head_dim, 0, inv_freq, rope_pairing);
    const Mat k = llama_rope(k_normed, n_kv_heads, head_dim, 0, inv_freq, rope_pairing);
    const Mat v = v_proj.forward(xn).data();

    Mat attn =
        bidirectional_gqa_attention(q, k, v, n_q_heads, n_kv_heads, head_dim, attn_scale);
    return o_proj.forward(TensorNode::leaf(std::move(attn))).data();
}

// =============================================================================
// OmniMlp / OmniBlock
// =============================================================================

OmniMlp::OmniMlp(const Config6& cfg)
    : gate_proj(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size)),
      up_proj(Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size)),
      down_proj(Linear2::new_no_bias_zeros(cfg.intermediate_size, cfg.hidden_size)) {}

Mat OmniMlp::forward(const Mat& x) const {
    const TensorNode xn = TensorNode::leaf(x);
    const TensorNode gate = gate_proj.forward(xn).silu();
    return down_proj.forward(gate.mul_elem_node(up_proj.forward(xn))).data();
}

OmniBlock::OmniBlock(const Config6& cfg)
    : input_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      self_attn(cfg),
      post_attention_layernorm(cfg.hidden_size, cfg.rms_norm_eps),
      mlp(cfg) {}

Mat OmniBlock::forward(const Mat& x, const std::vector<float>& inv_freq) const {
    const Mat normed = input_layernorm.forward(TensorNode::leaf(x)).data();
    Mat h = self_attn.forward(normed, inv_freq);
    h.add_assign(x);

    const Mat normed2 = post_attention_layernorm.forward(TensorNode::leaf(h)).data();
    Mat ff = mlp.forward(normed2);
    ff.add_assign(h);
    return ff;
}

// =============================================================================
// OmniLm
// =============================================================================

OmniLm::OmniLm(Config6 cfg)
    : config(std::move(cfg)),
      norm(config.hidden_size, config.rms_norm_eps),
      audio_head(Linear2::new_no_bias_zeros(config.hidden_size, config.audio_table_size())) {
    layers.reserve(config.num_hidden_layers);
    for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers.emplace_back(config);
    }
    refresh_inv_freq();
}

void OmniLm::refresh_inv_freq() {
    inv_freq_cache = config.inv_freq();
    for (OmniBlock& layer : layers) {
        layer.self_attn.rope_pairing = config.rope_pairing;
    }
}

Result<Mat> OmniLm::embed(const std::vector<OmniToken>& tokens) const {
    if (tokens.empty()) {
        return err("omnivoice lm: empty sequence");
    }
    if (!text_embed || !audio_embed) {
        return err("omnivoice lm: embeddings not loaded");
    }

    const std::size_t h = config.hidden_size;
    const std::vector<std::uint16_t>& text_bits = *text_embed->data;
    const std::vector<std::uint16_t>& audio_bits = *audio_embed->data;

    Mat out = Mat::zeros(tokens.size(), h);
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        float* row = out.row_mut(t).data();
        const OmniToken& tok = tokens[t];

        if (!tok.is_audio()) {
            if (tok.text_id >= text_embed->rows) {
                return err("omnivoice lm: text id " + std::to_string(tok.text_id) +
                           " is outside the vocabulary");
            }
            for (std::size_t c = 0; c < h; ++c) {
                row[c] = bf16_to_f32(text_bits[tok.text_id * h + c]);
            }
            continue;
        }

        if (tok.audio.size() != config.num_audio_codebook) {
            return err("omnivoice lm: audio position " + std::to_string(t) + " carries " +
                       std::to_string(tok.audio.size()) + " codes, expected " +
                       std::to_string(config.num_audio_codebook));
        }
        // Sum across codebooks. A fully masked position is the sum of the eight
        // mask rows, which is a perfectly well-defined embedding -- that is what
        // lets decoding start from nothing.
        for (std::size_t i = 0; i < config.num_audio_codebook; ++i) {
            const std::uint32_t code = tok.audio[i];
            if (code >= config.audio_vocab_size) {
                return err("omnivoice lm: audio code " + std::to_string(code) + " at codebook " +
                           std::to_string(i) + " is outside the codebook");
            }
            const std::size_t base = config.audio_row(i, code) * h;
            for (std::size_t c = 0; c < h; ++c) {
                row[c] += bf16_to_f32(audio_bits[base + c]);
            }
        }
    }
    return out;
}

Result<Mat> OmniLm::forward(const std::vector<OmniToken>& tokens) const {
    RT_TRY(x, embed(tokens));

    Mat h = std::move(x);
    for (const OmniBlock& layer : layers) {
        h = layer.forward(h, inv_freq_cache);
    }

    // Unlike an autoregressive model there is no "last position" shortcut:
    // every position's logits matter, because any of them might be unmasked
    // this step.
    const Mat normed = norm.forward(TensorNode::leaf(h)).data();
    return audio_head.forward(TensorNode::leaf(normed)).data();
}

std::size_t OmniLm::weight_bytes() const {
    const auto linear_bytes = [](const Linear2& l) -> std::size_t {
        if (l.q4k_weight) {
            return l.q4k_weight->size_bytes();
        }
        if (l.bf16_weight) {
            return l.bf16_weight->size_bytes();
        }
        return l.weight.data().numel() * sizeof(float);
    };
    std::size_t n = linear_bytes(audio_head);
    if (text_embed) {
        n += text_embed->size_bytes();
    }
    if (audio_embed) {
        n += audio_embed->size_bytes();
    }
    for (const OmniBlock& layer : layers) {
        n += linear_bytes(layer.self_attn.q_proj) + linear_bytes(layer.self_attn.k_proj) +
             linear_bytes(layer.self_attn.v_proj) + linear_bytes(layer.self_attn.o_proj) +
             linear_bytes(layer.mlp.gate_proj) + linear_bytes(layer.mlp.up_proj) +
             linear_bytes(layer.mlp.down_proj);
    }
    return n;
}

std::size_t OmniLm::quantize_projections_to_q4k() {
    std::size_t converted = 0;
    const auto convert = [&converted](Linear2& l) {
        if (l.bf16_weight) {
            l.quantize_bf16_to_q4k();
            ++converted;
        }
    };
    for (OmniBlock& layer : layers) {
        convert(layer.self_attn.q_proj);
        convert(layer.self_attn.k_proj);
        convert(layer.self_attn.v_proj);
        convert(layer.self_attn.o_proj);
        convert(layer.mlp.gate_proj);
        convert(layer.mlp.up_proj);
        convert(layer.mlp.down_proj);
    }
    convert(audio_head);
    return converted;
}

}  // namespace rt

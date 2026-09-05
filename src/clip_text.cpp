#include "rt/clip_text.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "rt/conv2d.hpp"
#include "rt/mat.hpp"
#include "rt/qlinear.hpp"
#include "rt/result.hpp"
#include "rt/transformer3.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

ClipTextConfig ClipTextConfig::large() {
    ClipTextConfig c;
    c.vocab_size = 49408;
    c.hidden_size = 768;
    c.intermediate_size = 3072;
    c.n_layers = 12;
    c.n_heads = 12;
    c.max_position_embeddings = 77;
    c.eot_token_id = 49407;
    c.layer_norm_eps = 1e-5f;
    return c;
}

// =============================================================================
// Norm
// =============================================================================

Mat clip_layer_norm(const Mat& x, std::span<const float> weight, std::span<const float> bias,
                    float eps) {
    assert(weight.size() == x.cols && "clip_layer_norm: weight length != hidden_size");
    Mat out = Mat::zeros(x.rows, x.cols);
    const auto n = static_cast<double>(x.cols);
    for (std::size_t r = 0; r < x.rows; ++r) {
        const float* src = x.row(r).data();
        double sum = 0.0;
        for (std::size_t c = 0; c < x.cols; ++c) {
            sum += src[c];
        }
        const double mean = sum / n;
        double var = 0.0;
        for (std::size_t c = 0; c < x.cols; ++c) {
            const double d = static_cast<double>(src[c]) - mean;
            var += d * d;
        }
        const auto inv = static_cast<float>(1.0 / std::sqrt(var / n + static_cast<double>(eps)));
        const auto mean_f = static_cast<float>(mean);
        float* dst = out.row_mut(r).data();
        for (std::size_t c = 0; c < x.cols; ++c) {
            dst[c] = (src[c] - mean_f) * inv * weight[c] + (bias.empty() ? 0.0f : bias[c]);
        }
    }
    return out;
}

// =============================================================================
// Layers
// =============================================================================

Mat ClipAttention::forward(const Mat& x) const {
    const std::size_t t = x.rows;
    const std::size_t inner = n_heads * head_dim;

    const Mat q = this->q.forward(x);
    const Mat k = this->k.forward(x);
    const Mat v = this->v.forward(x);
    assert(q.cols == inner && "clip attention: projection width != n_heads * head_dim");

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    Mat context = Mat::zeros(t, inner);
    std::vector<float> scores(t);
    for (std::size_t h = 0; h < n_heads; ++h) {
        const std::size_t off = h * head_dim;
        for (std::size_t i = 0; i < t; ++i) {
            const float* qi = q.row(i).data() + off;
            float max_score = -std::numeric_limits<float>::infinity();
            // Causal: position `i` reads keys 0..i. CLIP's text tower is
            // masked even though nothing downstream generates text.
            for (std::size_t j = 0; j <= i; ++j) {
                const float* kj = k.row(j).data() + off;
                float acc = 0.0f;
                for (std::size_t c = 0; c < head_dim; ++c) {
                    acc += qi[c] * kj[c];
                }
                scores[j] = acc * scale;
                max_score = std::max(max_score, scores[j]);
            }
            float denom = 0.0f;
            for (std::size_t j = 0; j <= i; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                denom += scores[j];
            }
            const float inv = 1.0f / denom;
            float* dst = context.row_mut(i).data() + off;
            for (std::size_t j = 0; j <= i; ++j) {
                const float weight = scores[j] * inv;
                const float* vj = v.row(j).data() + off;
                for (std::size_t c = 0; c < head_dim; ++c) {
                    dst[c] += weight * vj[c];
                }
            }
        }
    }
    return out.forward(context);
}

Mat ClipMlp::forward(const Mat& x) const {
    Mat h = fc1.forward(x);
    quick_gelu_inplace(h);
    return fc2.forward(h);
}

Mat ClipLayer::forward(const Mat& x, float eps) const {
    Mat h = attn.forward(clip_layer_norm(x, norm1_weight, norm1_bias, eps));
    for (std::size_t i = 0; i < h.data.size(); ++i) {
        h.data[i] += x.data[i];
    }
    Mat m = mlp.forward(clip_layer_norm(h, norm2_weight, norm2_bias, eps));
    for (std::size_t i = 0; i < m.data.size(); ++i) {
        m.data[i] += h.data[i];
    }
    return m;
}

// =============================================================================
// Loading
// =============================================================================

namespace {

using TensorMap = std::unordered_map<std::string, SafeTensor>;

/// Checkpoints ship under two prefixes: HuggingFace's `text_model.` and the
/// flat ComfyUI export that drops it. Try both before failing.
[[nodiscard]] SafeTensor* find_tensor(TensorMap& map, const std::string& suffix) {
    if (auto it = map.find("text_model." + suffix); it != map.end()) {
        return &it->second;
    }
    if (auto it = map.find(suffix); it != map.end()) {
        return &it->second;
    }
    return nullptr;
}

[[nodiscard]] Result<Mat> take_mat(TensorMap& map, const std::string& suffix, std::size_t rows,
                                   std::size_t cols) {
    SafeTensor* t = find_tensor(map, suffix);
    if (t == nullptr) {
        return err("clip: missing tensor " + suffix);
    }
    if (t->data.size() != rows * cols) {
        return err("clip: " + suffix + " has " + std::to_string(t->data.size()) +
                   " elements, expected " + std::to_string(rows * cols));
    }
    return Mat(std::move(t->data), rows, cols);
}

[[nodiscard]] Result<std::vector<float>> take_vec(TensorMap& map, const std::string& suffix,
                                                  std::size_t len) {
    SafeTensor* t = find_tensor(map, suffix);
    if (t == nullptr) {
        return err("clip: missing tensor " + suffix);
    }
    if (t->data.size() != len) {
        return err("clip: " + suffix + " has " + std::to_string(t->data.size()) +
                   " elements, expected " + std::to_string(len));
    }
    return std::move(t->data);
}

[[nodiscard]] Result<QLinear> take_linear(TensorMap& map, const std::string& prefix,
                                          std::size_t out_features, std::size_t in_features) {
    RT_TRY(w, take_mat(map, prefix + ".weight", out_features, in_features));
    RT_TRY(b, take_vec(map, prefix + ".bias", out_features));
    return QLinear::from_f32(std::move(w), std::move(b));
}

}  // namespace

Result<ClipTextEncoder> ClipTextEncoder::load(const std::string& path, ClipTextConfig cfg) {
    RT_TRY(header, parse_safetensors_header(path));
    const std::size_t data_offset = header.first;

    TensorMap map;
    for (const SafeTensorEntry& e : header.second) {
        // The vision tower shares a file in some exports and is three times the
        // size of what is wanted here.
        if (e.name.contains("vision_model")) {
            continue;
        }
        RT_TRY(t, read_safetensor_from_file(path, data_offset, e, false));
        map.emplace(e.name, std::move(t));
    }
    if (map.empty()) {
        return err("clip: no tensors in " + path);
    }

    ClipTextEncoder m;
    m.cfg = cfg;

    RT_TRY(tok, take_mat(map, "embeddings.token_embedding.weight", cfg.vocab_size,
                         cfg.hidden_size));
    RT_TRY(pos, take_mat(map, "embeddings.position_embedding.weight",
                         cfg.max_position_embeddings, cfg.hidden_size));
    m.token_embedding = std::move(tok);
    m.position_embedding = std::move(pos);

    const std::size_t head_dim = cfg.hidden_size / cfg.n_heads;
    m.layers.reserve(cfg.n_layers);
    for (std::size_t i = 0; i < cfg.n_layers; ++i) {
        const std::string p = "encoder.layers." + std::to_string(i) + ".";
        ClipLayer layer;

        RT_TRY(n1w, take_vec(map, p + "layer_norm1.weight", cfg.hidden_size));
        RT_TRY(n1b, take_vec(map, p + "layer_norm1.bias", cfg.hidden_size));
        layer.norm1_weight = std::move(n1w);
        layer.norm1_bias = std::move(n1b);

        RT_TRY(q, take_linear(map, p + "self_attn.q_proj", cfg.hidden_size, cfg.hidden_size));
        RT_TRY(k, take_linear(map, p + "self_attn.k_proj", cfg.hidden_size, cfg.hidden_size));
        RT_TRY(v, take_linear(map, p + "self_attn.v_proj", cfg.hidden_size, cfg.hidden_size));
        RT_TRY(o, take_linear(map, p + "self_attn.out_proj", cfg.hidden_size, cfg.hidden_size));
        layer.attn.q = std::move(q);
        layer.attn.k = std::move(k);
        layer.attn.v = std::move(v);
        layer.attn.out = std::move(o);
        layer.attn.n_heads = cfg.n_heads;
        layer.attn.head_dim = head_dim;

        RT_TRY(n2w, take_vec(map, p + "layer_norm2.weight", cfg.hidden_size));
        RT_TRY(n2b, take_vec(map, p + "layer_norm2.bias", cfg.hidden_size));
        layer.norm2_weight = std::move(n2w);
        layer.norm2_bias = std::move(n2b);

        RT_TRY(fc1, take_linear(map, p + "mlp.fc1", cfg.intermediate_size, cfg.hidden_size));
        RT_TRY(fc2, take_linear(map, p + "mlp.fc2", cfg.hidden_size, cfg.intermediate_size));
        layer.mlp.fc1 = std::move(fc1);
        layer.mlp.fc2 = std::move(fc2);

        m.layers.push_back(std::move(layer));
    }

    RT_TRY(fw, take_vec(map, "final_layer_norm.weight", cfg.hidden_size));
    RT_TRY(fb, take_vec(map, "final_layer_norm.bias", cfg.hidden_size));
    m.final_norm_weight = std::move(fw);
    m.final_norm_bias = std::move(fb);

    return m;
}

// =============================================================================
// Forward
// =============================================================================

Result<ClipTextOutput> ClipTextEncoder::forward(const std::vector<std::uint32_t>& tokens) const {
    if (tokens.empty()) {
        return err("clip forward: empty token sequence");
    }
    if (tokens.size() > cfg.max_position_embeddings) {
        return err("clip forward: " + std::to_string(tokens.size()) +
                   " tokens exceeds the positional table's " +
                   std::to_string(cfg.max_position_embeddings) + " rows");
    }
    if (token_embedding.rows != cfg.vocab_size) {
        return err("clip forward: embedding table not loaded");
    }

    const std::size_t t = tokens.size();
    Mat x = Mat::zeros(t, cfg.hidden_size);
    for (std::size_t i = 0; i < t; ++i) {
        if (tokens[i] >= cfg.vocab_size) {
            return err("clip forward: token id " + std::to_string(tokens[i]) + " out of range");
        }
        const float* tok = token_embedding.row(tokens[i]).data();
        const float* pos = position_embedding.row(i).data();
        float* dst = x.row_mut(i).data();
        for (std::size_t c = 0; c < cfg.hidden_size; ++c) {
            dst[c] = tok[c] + pos[c];
        }
    }

    for (const ClipLayer& layer : layers) {
        x = layer.forward(x, cfg.layer_norm_eps);
    }

    ClipTextOutput out;
    out.sequence = clip_layer_norm(x, final_norm_weight, final_norm_bias, cfg.layer_norm_eps);

    // The EOT position, by argmax over the ids. Padding is EOT too, so the
    // first match is the real one and `argmax` finds it -- taking the last
    // position instead would read the tail of the padding.
    std::size_t eot = 0;
    std::uint32_t best = 0;
    for (std::size_t i = 0; i < t; ++i) {
        if (tokens[i] > best) {
            best = tokens[i];
            eot = i;
        }
    }
    out.eot_index = eot;
    out.pooled.assign(out.sequence.row(eot).begin(), out.sequence.row(eot).end());
    return out;
}

std::size_t ClipTextEncoder::parameter_count() const {
    std::size_t n = token_embedding.numel() + position_embedding.numel() +
                    final_norm_weight.size() + final_norm_bias.size();
    for (const ClipLayer& l : layers) {
        n += l.norm1_weight.size() + l.norm1_bias.size() + l.norm2_weight.size() +
             l.norm2_bias.size();
        for (const QLinear* p : {&l.attn.q, &l.attn.k, &l.attn.v, &l.attn.out, &l.mlp.fc1,
                                 &l.mlp.fc2}) {
            n += p->out_features * p->in_features + p->bias.size();
        }
    }
    return n;
}

void ClipTextEncoder::free_weights() {
    token_embedding = Mat();
    position_embedding = Mat();
    layers.clear();
}

}  // namespace rt

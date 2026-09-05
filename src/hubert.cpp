#include "rt/hubert.hpp"

#include <cassert>
#include <cmath>
#include <tuple>
#include <utility>

#include "rt/conv1d.hpp"
#include "rt/transformer6.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

HubertConfig HubertConfig::omnivoice_semantic() { return HubertConfig{}; }

std::size_t HubertConfig::downsample_factor() const {
    std::size_t n = 1;
    for (const std::size_t s : conv_stride) {
        n *= s;
    }
    return n;
}

std::size_t HubertConfig::feature_frames(std::size_t samples) const {
    std::size_t t = samples;
    for (std::size_t i = 0; i < conv_kernel.size(); ++i) {
        // No padding anywhere in the feature extractor, so a clip shorter than
        // the receptive field produces nothing rather than a short frame.
        if (t < conv_kernel[i]) {
            return 0;
        }
        t = (t - conv_kernel[i]) / conv_stride[i] + 1;
    }
    return t;
}

// =============================================================================
// Normalisation and activation
// =============================================================================

void layer_norm_rows(Mat& x, std::span<const float> weight, std::span<const float> bias,
                     float eps) {
    assert(weight.size() == x.cols && "hubert: layer norm width mismatch");
    assert(bias.size() == x.cols && "hubert: layer norm bias width mismatch");
    const auto n = static_cast<double>(x.cols);

    for (std::size_t r = 0; r < x.rows; ++r) {
        float* row = x.row_mut(r).data();
        double sum = 0.0;
        for (std::size_t c = 0; c < x.cols; ++c) {
            sum += row[c];
        }
        const double mean = sum / n;
        double var = 0.0;
        for (std::size_t c = 0; c < x.cols; ++c) {
            const double d = static_cast<double>(row[c]) - mean;
            var += d * d;
        }
        // Biased variance, which is what PyTorch's LayerNorm uses.
        const float inv = static_cast<float>(1.0 / std::sqrt(var / n + eps));
        for (std::size_t c = 0; c < x.cols; ++c) {
            row[c] = (static_cast<float>(static_cast<double>(row[c]) - mean)) * inv * weight[c] +
                     bias[c];
        }
    }
}

void group_norm_channels(Mat& x, std::span<const float> weight, std::span<const float> bias,
                         float eps) {
    assert(weight.size() == x.cols && "hubert: group norm width mismatch");
    const auto n = static_cast<double>(x.rows);
    if (x.rows == 0) {
        return;
    }

    // One group per channel, so each column is normalised against its own
    // statistics over time -- the transpose of what LayerNorm does.
    for (std::size_t c = 0; c < x.cols; ++c) {
        double sum = 0.0;
        for (std::size_t r = 0; r < x.rows; ++r) {
            sum += x.at(r, c);
        }
        const double mean = sum / n;
        double var = 0.0;
        for (std::size_t r = 0; r < x.rows; ++r) {
            const double d = static_cast<double>(x.at(r, c)) - mean;
            var += d * d;
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(var / n + eps));
        for (std::size_t r = 0; r < x.rows; ++r) {
            x.at_mut(r, c) =
                static_cast<float>(static_cast<double>(x.at(r, c)) - mean) * inv * weight[c] +
                bias[c];
        }
    }
}

void gelu_erf_inplace(Mat& x) {
    constexpr float kInvSqrt2 = 0.70710678118654752f;
    for (float& v : x.data) {
        v = 0.5f * v * (1.0f + std::erf(v * kInvSqrt2));
    }
}

// =============================================================================
// Feature extractor
// =============================================================================

Mat HubertFeatureLayer::forward(const Mat& x, float eps) const {
    Mat h = conv1d_dense(x, weight, weight.rows, kernel, {}, 1, 0, stride);
    if (group_norm) {
        group_norm_channels(h, norm_weight, norm_bias, eps);
    }
    gelu_erf_inplace(h);
    return h;
}

Result<Mat> HubertModel::extract_features(std::span<const float> samples) const {
    const std::size_t frames = config.feature_frames(samples.size());
    if (frames == 0) {
        return err("hubert: " + std::to_string(samples.size()) +
                   " samples are too few for the feature extractor");
    }

    // The waveform enters as a one-channel [T, 1] sequence.
    Mat h(std::vector<float>(samples.begin(), samples.end()), samples.size(), 1);
    for (const HubertFeatureLayer& layer : feature_layers) {
        h = layer.forward(h, config.group_norm_eps);
    }
    if (h.rows != frames) {
        return err("hubert: feature extractor produced " + std::to_string(h.rows) +
                   " frames where the length arithmetic says " + std::to_string(frames));
    }
    return h;
}

// =============================================================================
// Encoder
// =============================================================================

namespace {

/// `x @ weight^T + bias`, the ordinary affine layer.
[[nodiscard]] Mat affine(const Mat& x, const Mat& weight, std::span<const float> bias) {
    Mat out = x.matmul_bt(weight);
    if (!bias.empty()) {
        for (std::size_t r = 0; r < out.rows; ++r) {
            float* row = out.row_mut(r).data();
            for (std::size_t c = 0; c < out.cols; ++c) {
                row[c] += bias[c];
            }
        }
    }
    return out;
}

}  // namespace

Mat HubertEncoderLayer::forward(const Mat& x, const HubertConfig& cfg) const {
    const std::size_t heads = cfg.num_attention_heads;
    const std::size_t dim = cfg.head_dim();

    const Mat q = affine(x, q_weight, q_bias);
    const Mat k = affine(x, k_weight, k_bias);
    const Mat v = affine(x, v_weight, v_bias);

    // Self-attention over the whole clip: no causal mask, and with one item in
    // the batch there is no padding to mask either.
    const Mat context = bidirectional_gqa_attention(
        q, k, v, heads, heads, dim, 1.0f / std::sqrt(static_cast<float>(dim)));

    Mat h = affine(context, o_weight, o_bias);
    h.add_assign(x);
    // Post-norm: after the residual add, not before the sublayer.
    layer_norm_rows(h, attn_norm_weight, attn_norm_bias, cfg.layer_norm_eps);

    Mat ff = affine(h, fc1_weight, fc1_bias);
    gelu_erf_inplace(ff);
    ff = affine(ff, fc2_weight, fc2_bias);
    ff.add_assign(h);
    layer_norm_rows(ff, final_norm_weight, final_norm_bias, cfg.layer_norm_eps);
    return ff;
}

Result<Mat> HubertModel::mean_hidden_states(std::span<const float> samples) const {
    RT_TRY(features, extract_features(samples));

    // feature_projection: normalise the convolution output, then widen it.
    layer_norm_rows(features, proj_norm_weight, proj_norm_bias, config.layer_norm_eps);
    Mat h = affine(features, proj_weight, proj_bias);

    // The positional convolution is "same" padded at kernel/2, which for an
    // even kernel leaves one sample too many. HuBERT drops the last one rather
    // than pad asymmetrically.
    const std::size_t k = config.num_conv_pos_embeddings;
    Mat pos = conv1d_grouped(h, pos_conv_weight, config.hidden_size, k, pos_conv_bias,
                             config.num_conv_pos_embedding_groups, 1, k / 2);
    if (k % 2 == 0) {
        if (pos.rows != h.rows + 1) {
            return err("hubert: positional convolution produced " + std::to_string(pos.rows) +
                       " frames for " + std::to_string(h.rows) + " inputs");
        }
        pos.data.resize(h.rows * pos.cols);
        pos.rows = h.rows;
    }
    gelu_erf_inplace(pos);
    h.add_assign(pos);

    layer_norm_rows(h, encoder_norm_weight, encoder_norm_bias, config.layer_norm_eps);

    // The codec wants the average of every hidden state, so accumulate as we
    // go rather than keeping thirteen copies of a [T, 768] matrix alive.
    Mat sum = h;
    for (const HubertEncoderLayer& layer : layers) {
        h = layer.forward(h, config);
        sum.add_assign(h);
    }
    const float inv = 1.0f / static_cast<float>(layers.size() + 1);
    for (float& v : sum.data) {
        v *= inv;
    }
    return sum;
}

std::size_t HubertModel::parameter_count() const {
    std::size_t n = 0;
    const auto add_mat = [&n](const Mat& m) { n += m.data.size(); };
    const auto add_vec = [&n](const std::vector<float>& v) { n += v.size(); };

    for (const HubertFeatureLayer& l : feature_layers) {
        add_mat(l.weight);
        add_vec(l.norm_weight);
        add_vec(l.norm_bias);
    }
    add_vec(proj_norm_weight);
    add_vec(proj_norm_bias);
    add_mat(proj_weight);
    add_vec(proj_bias);
    add_mat(pos_conv_weight);
    add_vec(pos_conv_bias);
    add_vec(encoder_norm_weight);
    add_vec(encoder_norm_bias);
    for (const HubertEncoderLayer& l : layers) {
        add_mat(l.q_weight);
        add_mat(l.k_weight);
        add_mat(l.v_weight);
        add_mat(l.o_weight);
        add_vec(l.q_bias);
        add_vec(l.k_bias);
        add_vec(l.v_bias);
        add_vec(l.o_bias);
        add_vec(l.attn_norm_weight);
        add_vec(l.attn_norm_bias);
        add_mat(l.fc1_weight);
        add_mat(l.fc2_weight);
        add_vec(l.fc1_bias);
        add_vec(l.fc2_bias);
        add_vec(l.final_norm_weight);
        add_vec(l.final_norm_bias);
    }
    return n;
}

// =============================================================================
// Loading
// =============================================================================

namespace {

/// Read a `[out, in]` linear weight, checking both dimensions.
[[nodiscard]] Result<Mat> load_linear(const GgufFile& gguf, const std::string& name,
                                      std::size_t out_dim, std::size_t in_dim) {
    RT_TRY(m, load_gguf_conv_weight(gguf, name, out_dim));
    if (m.cols != in_dim) {
        return err("hubert: '" + name + "' is [" + std::to_string(m.rows) + ", " +
                   std::to_string(m.cols) + "], expected [" + std::to_string(out_dim) + ", " +
                   std::to_string(in_dim) + "]");
    }
    return m;
}

[[nodiscard]] Result<std::vector<float>> load_sized(const GgufFile& gguf, const std::string& name,
                                                    std::size_t n) {
    RT_TRY(v, load_gguf_vector(gguf, name));
    if (v.size() != n) {
        return err("hubert: '" + name + "' holds " + std::to_string(v.size()) +
                   " values, expected " + std::to_string(n));
    }
    return v;
}

}  // namespace

Result<HubertModel> HubertModel::load(const GgufFile& gguf, HubertConfig cfg,
                                      const std::string& prefix) {
    if (cfg.conv_dim.size() != cfg.conv_kernel.size() ||
        cfg.conv_dim.size() != cfg.conv_stride.size()) {
        return err("hubert: conv_dim, conv_kernel and conv_stride disagree in length");
    }
    if (cfg.num_attention_heads == 0 || cfg.hidden_size % cfg.num_attention_heads != 0) {
        return err("hubert: hidden size does not divide into attention heads");
    }

    HubertModel m;
    m.config = std::move(cfg);
    const HubertConfig& c = m.config;

    // -- feature extractor ---------------------------------------------------
    std::size_t in_channels = 1;
    for (std::size_t i = 0; i < c.conv_dim.size(); ++i) {
        const std::string base =
            prefix + ".feature_extractor.conv_layers." + std::to_string(i);
        HubertFeatureLayer layer;
        layer.kernel = c.conv_kernel[i];
        layer.stride = c.conv_stride[i];
        RT_TRY(w, load_linear(gguf, base + ".conv.weight", c.conv_dim[i],
                              in_channels * layer.kernel));
        layer.weight = std::move(w);

        // Only the first layer carries a norm under `feat_extract_norm="group"`.
        // Deciding from the file rather than from a config flag means a
        // checkpoint of the other variant is a load error, not silent drift.
        if (gguf.find_tensor(base + ".layer_norm.weight")) {
            RT_TRY(nw, load_sized(gguf, base + ".layer_norm.weight", c.conv_dim[i]));
            RT_TRY(nb, load_sized(gguf, base + ".layer_norm.bias", c.conv_dim[i]));
            layer.group_norm = true;
            layer.norm_weight = std::move(nw);
            layer.norm_bias = std::move(nb);
        } else if (i == 0) {
            return err("hubert: '" + base +
                       ".layer_norm.weight' is missing, so the first convolution has no "
                       "group norm to apply");
        }
        in_channels = c.conv_dim[i];
        m.feature_layers.push_back(std::move(layer));
    }

    // -- feature projection --------------------------------------------------
    const std::size_t feat_dim = c.conv_dim.back();
    RT_TRY(pn_w, load_sized(gguf, prefix + ".feature_projection.layer_norm.weight", feat_dim));
    RT_TRY(pn_b, load_sized(gguf, prefix + ".feature_projection.layer_norm.bias", feat_dim));
    RT_TRY(p_w, load_linear(gguf, prefix + ".feature_projection.projection.weight",
                            c.hidden_size, feat_dim));
    RT_TRY(p_b, load_sized(gguf, prefix + ".feature_projection.projection.bias", c.hidden_size));
    m.proj_norm_weight = std::move(pn_w);
    m.proj_norm_bias = std::move(pn_b);
    m.proj_weight = std::move(p_w);
    m.proj_bias = std::move(p_b);

    // -- positional convolution ---------------------------------------------
    const std::size_t groups = c.num_conv_pos_embedding_groups;
    if (groups == 0 || c.hidden_size % groups != 0) {
        return err("hubert: positional convolution groups do not divide the hidden size");
    }
    RT_TRY(pc_w, load_linear(gguf, prefix + ".encoder.pos_conv_embed.conv.weight", c.hidden_size,
                             (c.hidden_size / groups) * c.num_conv_pos_embeddings));
    RT_TRY(pc_b, load_sized(gguf, prefix + ".encoder.pos_conv_embed.conv.bias", c.hidden_size));
    m.pos_conv_weight = std::move(pc_w);
    m.pos_conv_bias = std::move(pc_b);

    RT_TRY(en_w, load_sized(gguf, prefix + ".encoder.layer_norm.weight", c.hidden_size));
    RT_TRY(en_b, load_sized(gguf, prefix + ".encoder.layer_norm.bias", c.hidden_size));
    m.encoder_norm_weight = std::move(en_w);
    m.encoder_norm_bias = std::move(en_b);

    // -- encoder layers ------------------------------------------------------
    m.layers.reserve(c.num_hidden_layers);
    for (std::size_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string base = prefix + ".encoder.layers." + std::to_string(i);
        HubertEncoderLayer layer;

        for (const auto& [name, weight, bias] :
             {std::tuple{"q_proj", &layer.q_weight, &layer.q_bias},
              std::tuple{"k_proj", &layer.k_weight, &layer.k_bias},
              std::tuple{"v_proj", &layer.v_weight, &layer.v_bias},
              std::tuple{"out_proj", &layer.o_weight, &layer.o_bias}}) {
            RT_TRY(w, load_linear(gguf, base + ".attention." + name + ".weight", c.hidden_size,
                                  c.hidden_size));
            RT_TRY(b, load_sized(gguf, base + ".attention." + name + ".bias", c.hidden_size));
            *weight = std::move(w);
            *bias = std::move(b);
        }

        RT_TRY(an_w, load_sized(gguf, base + ".layer_norm.weight", c.hidden_size));
        RT_TRY(an_b, load_sized(gguf, base + ".layer_norm.bias", c.hidden_size));
        layer.attn_norm_weight = std::move(an_w);
        layer.attn_norm_bias = std::move(an_b);

        RT_TRY(f1_w, load_linear(gguf, base + ".feed_forward.intermediate_dense.weight",
                                 c.intermediate_size, c.hidden_size));
        RT_TRY(f1_b, load_sized(gguf, base + ".feed_forward.intermediate_dense.bias",
                                c.intermediate_size));
        RT_TRY(f2_w, load_linear(gguf, base + ".feed_forward.output_dense.weight", c.hidden_size,
                                 c.intermediate_size));
        RT_TRY(f2_b, load_sized(gguf, base + ".feed_forward.output_dense.bias", c.hidden_size));
        layer.fc1_weight = std::move(f1_w);
        layer.fc1_bias = std::move(f1_b);
        layer.fc2_weight = std::move(f2_w);
        layer.fc2_bias = std::move(f2_b);

        RT_TRY(fn_w, load_sized(gguf, base + ".final_layer_norm.weight", c.hidden_size));
        RT_TRY(fn_b, load_sized(gguf, base + ".final_layer_norm.bias", c.hidden_size));
        layer.final_norm_weight = std::move(fn_w);
        layer.final_norm_bias = std::move(fn_b);

        m.layers.push_back(std::move(layer));
    }

    return m;
}

}  // namespace rt

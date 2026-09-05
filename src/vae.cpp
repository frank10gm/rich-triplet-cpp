#include "rt/vae.hpp"

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
#include "rt/result.hpp"
#include "rt/transformer3.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

VaeConfig VaeConfig::flux() {
    VaeConfig c;
    c.latent_channels = 16;
    c.block_out_channels = {128, 256, 512, 512};
    c.layers_per_block = 2;
    c.norm_groups = 32;
    c.norm_eps = 1e-6f;
    c.scaling_factor = 0.3611f;
    c.shift_factor = 0.1159f;
    return c;
}

std::size_t VaeConfig::downsample_factor() const {
    return block_out_channels.empty() ? 1 : (std::size_t{1} << (block_out_channels.size() - 1));
}

// =============================================================================
// Layers
// =============================================================================

Mat VaeResnetBlock::forward(const Mat& x, std::size_t h, std::size_t w,
                            const VaeConfig& cfg) const {
    assert(x.cols == in_channels && "VaeResnetBlock: channel mismatch");

    Mat hidden = group_norm(x, cfg.norm_groups, norm1_weight, norm1_bias, cfg.norm_eps);
    silu_inplace(hidden);
    hidden = conv2d_dense(hidden, h, w, conv1_weight, out_channels, 3, 3, conv1_bias, 1, 1, 1);

    group_norm_inplace(hidden, cfg.norm_groups, norm2_weight, norm2_bias, cfg.norm_eps);
    silu_inplace(hidden);
    hidden = conv2d_dense(hidden, h, w, conv2_weight, out_channels, 3, 3, conv2_bias, 1, 1, 1);

    // The shortcut exists only where the width changes; everywhere else the
    // residual is the input itself.
    if (shortcut_weight.rows > 0) {
        const Mat skip = conv2d_pointwise(x, shortcut_weight, shortcut_bias);
        for (std::size_t i = 0; i < hidden.data.size(); ++i) {
            hidden.data[i] += skip.data[i];
        }
    } else {
        for (std::size_t i = 0; i < hidden.data.size(); ++i) {
            hidden.data[i] += x.data[i];
        }
    }
    return hidden;
}

Mat VaeAttentionBlock::forward(const Mat& x, const VaeConfig& cfg) const {
    assert(x.cols == channels && "VaeAttentionBlock: channel mismatch");
    const std::size_t n = x.rows;

    const Mat normed = group_norm(x, cfg.norm_groups, norm_weight, norm_bias, cfg.norm_eps);
    const Mat q = conv2d_pointwise(normed, q_weight, q_bias);
    const Mat k = conv2d_pointwise(normed, k_weight, k_bias);
    const Mat v = conv2d_pointwise(normed, v_weight, v_bias);

    const float scale = 1.0f / std::sqrt(static_cast<float>(channels));

    // One head over `n` positions. The score matrix is n x n, which at a 64x64
    // latent is 4096 x 4096 -- 67 MB, and the only place in the decoder where
    // attention is affordable at all. It sits at the coarsest resolution for
    // exactly that reason.
    Mat out = Mat::zeros(n, channels);
    std::vector<float> scores(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float* qi = q.row(i).data();
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < n; ++j) {
            const float* kj = k.row(j).data();
            float acc = 0.0f;
            for (std::size_t c = 0; c < channels; ++c) {
                acc += qi[c] * kj[c];
            }
            scores[j] = acc * scale;
            max_score = std::max(max_score, scores[j]);
        }
        float denom = 0.0f;
        for (std::size_t j = 0; j < n; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            denom += scores[j];
        }
        const float inv = 1.0f / denom;
        float* orow = out.row_mut(i).data();
        for (std::size_t j = 0; j < n; ++j) {
            const float weight = scores[j] * inv;
            if (weight == 0.0f) {
                continue;
            }
            const float* vj = v.row(j).data();
            for (std::size_t c = 0; c < channels; ++c) {
                orow[c] += weight * vj[c];
            }
        }
    }

    Mat projected = conv2d_pointwise(out, out_weight, out_bias);
    for (std::size_t i = 0; i < projected.data.size(); ++i) {
        projected.data[i] += x.data[i];
    }
    return projected;
}

// =============================================================================
// Loading
// =============================================================================

namespace {

using TensorMap = std::unordered_map<std::string, SafeTensor>;

[[nodiscard]] Result<Mat> take_mat(TensorMap& map, const std::string& name, std::size_t rows,
                                   std::size_t cols) {
    const auto it = map.find(name);
    if (it == map.end()) {
        return err("vae: missing tensor " + name);
    }
    if (it->second.data.size() != rows * cols) {
        return err("vae: " + name + " has " + std::to_string(it->second.data.size()) +
                   " elements, expected " + std::to_string(rows * cols));
    }
    Mat m(std::move(it->second.data), rows, cols);
    map.erase(it);
    return m;
}

[[nodiscard]] Result<std::vector<float>> take_vec(TensorMap& map, const std::string& name,
                                                  std::size_t len) {
    const auto it = map.find(name);
    if (it == map.end()) {
        return err("vae: missing tensor " + name);
    }
    if (it->second.data.size() != len) {
        return err("vae: " + name + " has " + std::to_string(it->second.data.size()) +
                   " elements, expected " + std::to_string(len));
    }
    std::vector<float> v = std::move(it->second.data);
    map.erase(it);
    return v;
}

/// Which naming convention a checkpoint uses.
///
/// Two are in wide circulation for the same weights. `black-forest-labs` ships
/// `ae.safetensors` under the original LDM names -- `decoder.mid.block_1`,
/// `decoder.up.0.block.0`, `nin_shortcut` -- and diffusers ships
/// `vae/diffusion_pytorch_model.safetensors` under its own --
/// `decoder.mid_block.resnets.0`, `decoder.up_blocks.0.resnets.0`,
/// `conv_shortcut`. Repackaged files put either under either path, so the
/// convention is detected from the tensors present rather than from the
/// filename.
///
/// The difference that matters is not the spelling. **The two number the
/// upsampling levels in opposite directions**: diffusers' `up_blocks.0` is the
/// coarsest and the original's `up.0` is the finest. Loading one as the other
/// lines up for the two 512-channel levels and then fails the channel count on
/// the third -- which is the good case, because reversing the resnets without
/// reversing the upsamplers produces a blurred image and no error at all.
struct VaeNames {
    bool original = false;

    [[nodiscard]] std::string mid_resnet(std::size_t which) const {
        return original ? "decoder.mid.block_" + std::to_string(which + 1) + "."
                        : "decoder.mid_block.resnets." + std::to_string(which) + ".";
    }
    [[nodiscard]] std::string mid_attn() const {
        return original ? "decoder.mid.attn_1." : "decoder.mid_block.attentions.0.";
    }
    [[nodiscard]] const char* attn_norm() const { return original ? "norm" : "group_norm"; }
    [[nodiscard]] const char* attn_q() const { return original ? "q" : "to_q"; }
    [[nodiscard]] const char* attn_k() const { return original ? "k" : "to_k"; }
    [[nodiscard]] const char* attn_v() const { return original ? "v" : "to_v"; }
    [[nodiscard]] const char* attn_out() const { return original ? "proj_out" : "to_out.0"; }

    /// `exec` counts from the coarsest level, which is the order the decoder
    /// runs them in. The original convention stores them the other way round.
    [[nodiscard]] std::string level(std::size_t exec, std::size_t n_levels) const {
        return original ? "decoder.up." + std::to_string(n_levels - 1 - exec) + "."
                        : "decoder.up_blocks." + std::to_string(exec) + ".";
    }
    [[nodiscard]] std::string resnet(const std::string& level, std::size_t r) const {
        return level + (original ? "block." : "resnets.") + std::to_string(r) + ".";
    }
    [[nodiscard]] const char* shortcut() const {
        return original ? "nin_shortcut" : "conv_shortcut";
    }
    [[nodiscard]] std::string upsample(const std::string& level) const {
        return level + (original ? "upsample.conv" : "upsamplers.0.conv");
    }
    [[nodiscard]] const char* norm_out() const {
        return original ? "decoder.norm_out" : "decoder.conv_norm_out";
    }
};

/// One of the attention projections, under whichever name this file uses.
[[nodiscard]] Result<Mat> take_attn_mat(TensorMap& map, const std::string& prefix,
                                        const char* name, std::size_t c) {
    return take_mat(map, prefix + name + ".weight", c, c);
}

[[nodiscard]] Result<std::vector<float>> take_attn_vec(TensorMap& map, const std::string& prefix,
                                                       const char* name, std::size_t c) {
    return take_vec(map, prefix + name + ".bias", c);
}

[[nodiscard]] Result<VaeResnetBlock> load_resnet(TensorMap& map, const VaeNames& names,
                                                 const std::string& prefix, std::size_t c_in,
                                                 std::size_t c_out) {
    VaeResnetBlock b;
    b.in_channels = c_in;
    b.out_channels = c_out;

    RT_TRY(n1w, take_vec(map, prefix + "norm1.weight", c_in));
    RT_TRY(n1b, take_vec(map, prefix + "norm1.bias", c_in));
    RT_TRY(c1w, take_mat(map, prefix + "conv1.weight", c_out, c_in * 9));
    RT_TRY(c1b, take_vec(map, prefix + "conv1.bias", c_out));
    RT_TRY(n2w, take_vec(map, prefix + "norm2.weight", c_out));
    RT_TRY(n2b, take_vec(map, prefix + "norm2.bias", c_out));
    RT_TRY(c2w, take_mat(map, prefix + "conv2.weight", c_out, c_out * 9));
    RT_TRY(c2b, take_vec(map, prefix + "conv2.bias", c_out));

    b.norm1_weight = std::move(n1w);
    b.norm1_bias = std::move(n1b);
    b.conv1_weight = std::move(c1w);
    b.conv1_bias = std::move(c1b);
    b.norm2_weight = std::move(n2w);
    b.norm2_bias = std::move(n2b);
    b.conv2_weight = std::move(c2w);
    b.conv2_bias = std::move(c2b);

    if (c_in != c_out) {
        RT_TRY(sw, take_mat(map, prefix + names.shortcut() + ".weight", c_out, c_in));
        RT_TRY(sb, take_vec(map, prefix + names.shortcut() + ".bias", c_out));
        b.shortcut_weight = std::move(sw);
        b.shortcut_bias = std::move(sb);
    }
    return b;
}

}  // namespace

Result<VaeDecoder> VaeDecoder::load(const std::string& path, VaeConfig cfg) {
    if (cfg.block_out_channels.empty()) {
        return err("vae: block_out_channels is empty");
    }

    RT_TRY(header, parse_safetensors_header(path));
    const std::size_t data_offset = header.first;
    const std::vector<SafeTensorEntry>& entries = header.second;

    // Stream only what the decoder needs. The file also holds the encoder,
    // which is never used and is a third of its size.
    TensorMap map;
    for (const SafeTensorEntry& e : entries) {
        if (!e.name.starts_with("decoder.")) {
            continue;
        }
        RT_TRY(t, read_safetensor_from_file(path, data_offset, e, false));
        map.emplace(e.name, std::move(t));
    }
    if (map.empty()) {
        return err("vae: no decoder.* tensors in " + path);
    }

    VaeDecoder d;
    d.cfg = cfg;

    VaeNames names;
    names.original = map.contains("decoder.mid.block_1.conv1.weight");

    const std::size_t n_levels = cfg.block_out_channels.size();
    const std::size_t c_coarse = cfg.block_out_channels.back();
    const std::size_t c_fine = cfg.block_out_channels.front();

    RT_TRY(ciw, take_mat(map, "decoder.conv_in.weight", c_coarse, cfg.latent_channels * 9));
    RT_TRY(cib, take_vec(map, "decoder.conv_in.bias", c_coarse));
    d.conv_in_weight = std::move(ciw);
    d.conv_in_bias = std::move(cib);

    RT_TRY(mid1, load_resnet(map, names, names.mid_resnet(0), c_coarse, c_coarse));
    RT_TRY(mid2, load_resnet(map, names, names.mid_resnet(1), c_coarse, c_coarse));
    d.mid_resnet1 = std::move(mid1);
    d.mid_resnet2 = std::move(mid2);

    {
        const std::string p = names.mid_attn();
        VaeAttentionBlock a;
        a.channels = c_coarse;
        RT_TRY(nw, take_vec(map, p + names.attn_norm() + ".weight", c_coarse));
        RT_TRY(nb, take_vec(map, p + names.attn_norm() + ".bias", c_coarse));
        RT_TRY(qw, take_attn_mat(map, p, names.attn_q(), c_coarse));
        RT_TRY(qb, take_attn_vec(map, p, names.attn_q(), c_coarse));
        RT_TRY(kw, take_attn_mat(map, p, names.attn_k(), c_coarse));
        RT_TRY(kb, take_attn_vec(map, p, names.attn_k(), c_coarse));
        RT_TRY(vw, take_attn_mat(map, p, names.attn_v(), c_coarse));
        RT_TRY(vb, take_attn_vec(map, p, names.attn_v(), c_coarse));
        RT_TRY(ow, take_attn_mat(map, p, names.attn_out(), c_coarse));
        RT_TRY(ob, take_attn_vec(map, p, names.attn_out(), c_coarse));
        a.norm_weight = std::move(nw);
        a.norm_bias = std::move(nb);
        a.q_weight = std::move(qw);
        a.q_bias = std::move(qb);
        a.k_weight = std::move(kw);
        a.k_bias = std::move(kb);
        a.v_weight = std::move(vw);
        a.v_bias = std::move(vb);
        a.out_weight = std::move(ow);
        a.out_bias = std::move(ob);
        d.mid_attn = std::move(a);
    }

    // Walk the levels in execution order -- coarsest first -- and let `names`
    // work out which index that is in this file's convention.
    std::size_t c_prev = c_coarse;
    for (std::size_t i = 0; i < n_levels; ++i) {
        // Execution index `i` is level `n_levels - 1 - i` of the finest-first
        // `block_out_channels`.
        const std::size_t c_out = cfg.block_out_channels[n_levels - 1 - i];
        const std::string p = names.level(i, n_levels);

        VaeUpBlock block;
        for (std::size_t r = 0; r <= cfg.layers_per_block; ++r) {
            const std::size_t c_in = (r == 0) ? c_prev : c_out;
            RT_TRY(res, load_resnet(map, names, names.resnet(p, r), c_in, c_out));
            block.resnets.push_back(std::move(res));
        }
        // Every level upsamples except the finest, which is the last one.
        if (i + 1 < n_levels) {
            const std::string u = names.upsample(p);
            RT_TRY(uw, take_mat(map, u + ".weight", c_out, c_out * 9));
            RT_TRY(ub, take_vec(map, u + ".bias", c_out));
            block.upsample_weight = std::move(uw);
            block.upsample_bias = std::move(ub);
        }
        d.up_blocks.push_back(std::move(block));
        c_prev = c_out;
    }

    RT_TRY(onw, take_vec(map, std::string(names.norm_out()) + ".weight", c_fine));
    RT_TRY(onb, take_vec(map, std::string(names.norm_out()) + ".bias", c_fine));
    RT_TRY(ow2, take_mat(map, "decoder.conv_out.weight", 3, c_fine * 9));
    RT_TRY(ob2, take_vec(map, "decoder.conv_out.bias", 3));
    d.conv_out_norm_weight = std::move(onw);
    d.conv_out_norm_bias = std::move(onb);
    d.conv_out_weight = std::move(ow2);
    d.conv_out_bias = std::move(ob2);

    return d;
}

// =============================================================================
// Decoding
// =============================================================================

Result<Mat> VaeDecoder::decode(const Mat& z, std::size_t lat_h, std::size_t lat_w) const {
    if (z.cols != cfg.latent_channels) {
        return err("vae decode: latent has " + std::to_string(z.cols) + " channels, expected " +
                   std::to_string(cfg.latent_channels));
    }
    if (z.rows != lat_h * lat_w) {
        return err("vae decode: rows " + std::to_string(z.rows) + " != lat_h*lat_w");
    }
    if (cfg.block_out_channels.empty()) {
        return err("vae decode: block_out_channels is empty");
    }

    // Undo the training-time rescaling. Both constants, in this order.
    Mat x = z;
    const float inv_scale = 1.0f / cfg.scaling_factor;
    for (float& v : x.data) {
        v = v * inv_scale + cfg.shift_factor;
    }

    std::size_t h = lat_h;
    std::size_t w = lat_w;

    x = conv2d_dense(x, h, w, conv_in_weight, cfg.block_out_channels.back(), 3, 3, conv_in_bias,
                     1, 1, 1);

    x = mid_resnet1.forward(x, h, w, cfg);
    x = mid_attn.forward(x, cfg);
    x = mid_resnet2.forward(x, h, w, cfg);

    for (const VaeUpBlock& block : up_blocks) {
        for (const VaeResnetBlock& res : block.resnets) {
            x = res.forward(x, h, w, cfg);
        }
        if (block.upsamples()) {
            x = upsample_nearest2d(x, h, w, 2);
            h *= 2;
            w *= 2;
            x = conv2d_dense(x, h, w, block.upsample_weight, x.cols, 3, 3, block.upsample_bias, 1,
                             1, 1);
        }
    }

    group_norm_inplace(x, cfg.norm_groups, conv_out_norm_weight, conv_out_norm_bias, cfg.norm_eps);
    silu_inplace(x);
    x = conv2d_dense(x, h, w, conv_out_weight, 3, 3, 3, conv_out_bias, 1, 1, 1);
    return x;
}

Result<Mat> VaeDecoder::decode_tiled(const Mat& z, std::size_t lat_h, std::size_t lat_w,
                                     std::size_t tile, std::size_t overlap) const {
    if (tile == 0) {
        return err("vae decode_tiled: tile must be > 0");
    }
    if (overlap >= tile) {
        return err("vae decode_tiled: overlap must be smaller than tile");
    }
    if (lat_h <= tile && lat_w <= tile) {
        return decode(z, lat_h, lat_w);
    }

    const std::size_t f = cfg.downsample_factor();
    const std::size_t out_h = lat_h * f;
    const std::size_t out_w = lat_w * f;
    Mat acc = Mat::zeros(out_h * out_w, 3);
    std::vector<float> weight_sum(out_h * out_w, 0.0f);

    const std::size_t step = tile - overlap;
    for (std::size_t y0 = 0; y0 < lat_h; y0 += step) {
        const std::size_t th = std::min(tile, lat_h - y0);
        for (std::size_t x0 = 0; x0 < lat_w; x0 += step) {
            const std::size_t tw = std::min(tile, lat_w - x0);

            Mat sub = Mat::zeros(th * tw, z.cols);
            for (std::size_t y = 0; y < th; ++y) {
                const float* src = z.row((y0 + y) * lat_w + x0).data();
                std::copy(src, src + tw * z.cols, sub.row_mut(y * tw).data());
            }

            RT_TRY(pixels, decode(sub, th, tw));

            // A linear ramp over the overlap, applied only on the edges that
            // actually abut another tile. Ramping an image boundary would fade
            // the picture out at its own edges.
            const std::size_t ph = th * f;
            const std::size_t pw = tw * f;
            const std::size_t ramp = overlap * f;
            const bool ramp_top = y0 > 0;
            const bool ramp_left = x0 > 0;
            const bool ramp_bottom = y0 + th < lat_h;
            const bool ramp_right = x0 + tw < lat_w;

            for (std::size_t y = 0; y < ph; ++y) {
                float wy = 1.0f;
                if (ramp_top && y < ramp) {
                    wy = std::min(wy, static_cast<float>(y + 1) / static_cast<float>(ramp + 1));
                }
                if (ramp_bottom && y + ramp >= ph) {
                    wy = std::min(wy, static_cast<float>(ph - y) / static_cast<float>(ramp + 1));
                }
                for (std::size_t x = 0; x < pw; ++x) {
                    float wx = 1.0f;
                    if (ramp_left && x < ramp) {
                        wx = std::min(wx,
                                      static_cast<float>(x + 1) / static_cast<float>(ramp + 1));
                    }
                    if (ramp_right && x + ramp >= pw) {
                        wx = std::min(wx,
                                      static_cast<float>(pw - x) / static_cast<float>(ramp + 1));
                    }
                    const float blend = wy * wx;
                    const std::size_t dst = (y0 * f + y) * out_w + (x0 * f + x);
                    const float* src = pixels.row(y * pw + x).data();
                    float* out = acc.row_mut(dst).data();
                    for (std::size_t c = 0; c < 3; ++c) {
                        out[c] += blend * src[c];
                    }
                    weight_sum[dst] += blend;
                }
            }
            if (x0 + tw >= lat_w) {
                break;
            }
        }
        if (y0 + th >= lat_h) {
            break;
        }
    }

    for (std::size_t p = 0; p < acc.rows; ++p) {
        const float wsum = weight_sum[p];
        if (wsum <= 0.0f) {
            continue;
        }
        float* row = acc.row_mut(p).data();
        for (std::size_t c = 0; c < 3; ++c) {
            row[c] /= wsum;
        }
    }
    return acc;
}

std::size_t VaeDecoder::parameter_count() const {
    std::size_t n = conv_in_weight.numel() + conv_in_bias.size();

    const auto count_resnet = [](const VaeResnetBlock& b) {
        return b.norm1_weight.size() + b.norm1_bias.size() + b.conv1_weight.numel() +
               b.conv1_bias.size() + b.norm2_weight.size() + b.norm2_bias.size() +
               b.conv2_weight.numel() + b.conv2_bias.size() + b.shortcut_weight.numel() +
               b.shortcut_bias.size();
    };

    n += count_resnet(mid_resnet1) + count_resnet(mid_resnet2);
    n += mid_attn.norm_weight.size() + mid_attn.norm_bias.size() + mid_attn.q_weight.numel() +
         mid_attn.q_bias.size() + mid_attn.k_weight.numel() + mid_attn.k_bias.size() +
         mid_attn.v_weight.numel() + mid_attn.v_bias.size() + mid_attn.out_weight.numel() +
         mid_attn.out_bias.size();

    for (const VaeUpBlock& b : up_blocks) {
        for (const VaeResnetBlock& r : b.resnets) {
            n += count_resnet(r);
        }
        n += b.upsample_weight.numel() + b.upsample_bias.size();
    }

    n += conv_out_norm_weight.size() + conv_out_norm_bias.size() + conv_out_weight.numel() +
         conv_out_bias.size();
    return n;
}

}  // namespace rt

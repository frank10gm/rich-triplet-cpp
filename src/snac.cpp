#include "rt/snac.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <utility>

#include "rt/conv1d.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

SnacConfig SnacConfig::snac_24khz() {
    SnacConfig c;
    c.sampling_rate = 24000;
    // encoder_dim 48 * 2^len(encoder_rates=[2,4,8,8]) = 48 * 16.
    c.latent_dim = 768;
    c.decoder_dim = 1024;
    c.decoder_rates = {8, 8, 4, 2};
    c.vq_strides = {4, 2, 1};
    c.codebook_size = 4096;
    c.codebook_dim = 8;
    c.noise = true;
    c.depthwise = true;
    return c;
}

std::size_t SnacConfig::upsample_factor() const {
    std::size_t n = 1;
    for (const std::size_t r : decoder_rates) {
        n *= r;
    }
    return n;
}

std::size_t SnacConfig::frames_per_group() const {
    return vq_strides.empty() ? 1 : vq_strides.front();
}

// =============================================================================
// Helpers
// =============================================================================

Mat repeat_rows(const Mat& x, std::size_t factor) {
    if (factor <= 1) {
        return x;
    }
    Mat out = Mat::zeros(x.rows * factor, x.cols);
    for (std::size_t r = 0; r < x.rows; ++r) {
        const float* src = x.row(r).data();
        for (std::size_t k = 0; k < factor; ++k) {
            float* dst = out.row_mut(r * factor + k).data();
            for (std::size_t c = 0; c < x.cols; ++c) {
                dst[c] = src[c];
            }
        }
    }
    return out;
}

Result<Mat> load_weight_norm_mat(const TorchStateDict& sd, const std::string& prefix) {
    RT_TRY(g, sd.require(prefix + ".parametrizations.weight.original0"));
    RT_TRY(v, sd.require(prefix + ".parametrizations.weight.original1"));

    // g is [axis0, 1, 1] and v is [axis0, ...]. Whether axis 0 means output or
    // input channels depends on the layer, and deriving the group count from g
    // rather than assuming either one is what makes transposed convolutions
    // work with no special case.
    if (g->shape.empty() || v->shape.empty() || g->shape[0] != v->shape[0]) {
        return err("snac: weight-norm shapes disagree on axis 0 for '" + prefix + "'");
    }
    if (g->numel() != g->shape[0]) {
        return err("snac: weight-norm magnitude for '" + prefix + "' is not one per group");
    }

    std::vector<float> combined = weight_norm_combine(g->data, v->data);
    const std::size_t rows = v->shape[0];
    const std::size_t cols = v->inner_size();
    return Mat(std::move(combined), rows, cols);
}

Result<std::vector<float>> load_alpha(const TorchStateDict& sd, const std::string& name) {
    RT_TRY(t, sd.require(name));
    // Stored as [1, C, 1]; only the channel count matters here.
    return t->data;
}

namespace {

/// Read an optional bias, returning an empty vector when absent.
[[nodiscard]] std::vector<float> optional_bias(const TorchStateDict& sd, const std::string& name) {
    const TorchTensor* t = sd.find(name);
    return t == nullptr ? std::vector<float>{} : t->data;
}

}  // namespace

// =============================================================================
// SnacResidualUnit
// =============================================================================

Mat SnacResidualUnit::forward(const Mat& x) const {
    Mat y = snake1d(x, alpha1);

    // Padding of 3*dilation against a kernel of 7 preserves length, so the
    // skip connection lines up without a crop.
    const std::size_t padding = ((kernel - 1) * dilation) / 2;
    y = depthwise ? conv1d_depthwise(y, conv1_weight, conv1_bias, dilation, padding)
                  : conv1d_dense(y, conv1_weight, x.cols, kernel, conv1_bias, dilation, padding);
    assert(y.rows == x.rows && "snac: residual unit changed the sequence length");

    snake1d_inplace(y, alpha2);
    y = conv1d_pointwise(y, conv2_weight, conv2_bias);

    y.add_assign(x);
    return y;
}

// =============================================================================
// SnacNoiseBlock
// =============================================================================

Mat SnacNoiseBlock::forward(const Mat& x, SnacNoise mode, InitRng& rng) const {
    const Mat h = conv1d_pointwise(x, weight, {});
    Mat out = x;
    for (std::size_t t = 0; t < out.rows; ++t) {
        // One draw per timestep, shared across every channel -- the reference
        // samples shape [B, 1, T]. Drawing per channel would decorrelate what
        // the model was trained to expect.
        const float n = mode == SnacNoise::Zero ? 0.0f : rng.next_normal();
        if (n == 0.0f) {
            continue;
        }
        const float* hrow = h.row(t).data();
        float* orow = out.row_mut(t).data();
        for (std::size_t c = 0; c < out.cols; ++c) {
            orow[c] += n * hrow[c];
        }
    }
    return out;
}

// =============================================================================
// SnacDecoderBlock
// =============================================================================

Mat SnacDecoderBlock::forward(const Mat& x, SnacNoise mode, InitRng& rng) const {
    Mat h = snake1d(x, alpha);
    h = conv_transpose1d(h, up_weight, out_channels, kernel, up_bias, stride, padding,
                         output_padding);
    assert(h.rows == x.rows * stride && "snac: upsampling block broke the length multiple");

    if (noise) {
        h = noise->forward(h, mode, rng);
    }
    for (const SnacResidualUnit& unit : units) {
        h = unit.forward(h);
    }
    return h;
}

// =============================================================================
// SnacQuantizer
// =============================================================================

Result<Mat> SnacQuantizer::from_codes(
    const std::vector<std::vector<std::uint32_t>>& codes) const {
    if (codes.size() != levels.size()) {
        return err("snac: expected " + std::to_string(levels.size()) + " codebooks, got " +
                   std::to_string(codes.size()));
    }
    if (codes.back().empty()) {
        return err("snac: the finest codebook has no codes");
    }

    const std::size_t frames = codes.back().size();
    Mat z_q = Mat::zeros(frames, latent_dim);

    for (std::size_t i = 0; i < levels.size(); ++i) {
        const Level& level = levels[i];
        const std::vector<std::uint32_t>& ids = codes[i];

        if (ids.size() * level.stride != frames) {
            return err("snac: codebook " + std::to_string(i) + " has " +
                       std::to_string(ids.size()) + " codes at stride " +
                       std::to_string(level.stride) + ", which does not cover " +
                       std::to_string(frames) + " frames");
        }

        // Codebook lookup: one row of [codebook_dim] per code.
        Mat latents = Mat::zeros(ids.size(), level.codebook.cols);
        for (std::size_t t = 0; t < ids.size(); ++t) {
            if (ids[t] >= level.codebook.rows) {
                return err("snac: code " + std::to_string(ids[t]) + " at codebook " +
                           std::to_string(i) + " index " + std::to_string(t) +
                           " is out of range (codebook holds " +
                           std::to_string(level.codebook.rows) + ")");
            }
            const float* src = level.codebook.row(ids[t]).data();
            float* dst = latents.row_mut(t).data();
            for (std::size_t c = 0; c < level.codebook.cols; ++c) {
                dst[c] = src[c];
            }
        }

        Mat projected = conv1d_pointwise(latents, level.out_proj_weight, level.out_proj_bias);
        // Repeat, not tile: stride 4 turns [a, b] into [a,a,a,a,b,b,b,b].
        const Mat upsampled = repeat_rows(projected, level.stride);
        z_q.add_assign(upsampled);
    }

    return z_q;
}

// =============================================================================
// Loading
// =============================================================================

Result<SnacDecoder> SnacDecoder::load(const std::string& path, SnacConfig cfg) {
    RT_TRY(sd, load_torch_state_dict(path));

    SnacDecoder d;
    d.config = std::move(cfg);

    // ---- quantizer ----
    d.quantizer.latent_dim = d.config.latent_dim;
    for (std::size_t i = 0; i < d.config.vq_strides.size(); ++i) {
        const std::string base = "quantizer.quantizers." + std::to_string(i);
        RT_TRY(codebook, sd.require(base + ".codebook.weight",
                                    {d.config.codebook_size, d.config.codebook_dim}));
        RT_TRY(proj, load_weight_norm_mat(sd, base + ".out_proj"));
        if (proj.rows != d.config.latent_dim || proj.cols != d.config.codebook_dim) {
            return err("snac: codebook " + std::to_string(i) + " out_proj is " +
                       std::to_string(proj.rows) + "x" + std::to_string(proj.cols) +
                       ", expected " + std::to_string(d.config.latent_dim) + "x" +
                       std::to_string(d.config.codebook_dim));
        }

        SnacQuantizer::Level level;
        level.codebook = Mat(codebook->data, codebook->shape[0], codebook->shape[1]);
        level.out_proj_weight = std::move(proj);
        level.out_proj_bias = optional_bias(sd, base + ".out_proj.bias");
        level.stride = d.config.vq_strides[i];
        d.quantizer.levels.push_back(std::move(level));
    }

    // ---- decoder input ----
    // depthwise: model.0 filters at 768 channels, model.1 widens to 1024.
    // dense:     model.0 widens directly and there is no model.1.
    RT_TRY(in_conv, load_weight_norm_mat(sd, "decoder.model.0"));
    d.in_conv_weight = std::move(in_conv);
    d.in_conv_bias = optional_bias(sd, "decoder.model.0.bias");

    std::size_t next_module = 1;
    if (d.config.depthwise) {
        RT_TRY(mix, load_weight_norm_mat(sd, "decoder.model.1"));
        if (mix.rows != d.config.decoder_dim) {
            return err("snac: decoder.model.1 outputs " + std::to_string(mix.rows) +
                       " channels, expected " + std::to_string(d.config.decoder_dim));
        }
        d.in_mix_weight = std::move(mix);
        d.in_mix_bias = optional_bias(sd, "decoder.model.1.bias");
        next_module = 2;
    }

    // ---- upsampling blocks ----
    // With a NoiseBlock the residual units sit at block.{3,4,5}; without one
    // they shift down to block.{2,3,4}.
    const std::size_t unit_base = d.config.noise ? 3 : 2;
    std::size_t channels = d.config.decoder_dim;

    for (std::size_t i = 0; i < d.config.decoder_rates.size(); ++i) {
        const std::string base = "decoder.model." + std::to_string(next_module + i);
        const std::size_t stride = d.config.decoder_rates[i];
        const std::size_t out_channels = channels / 2;

        SnacDecoderBlock block;
        block.stride = stride;
        block.kernel = 2 * stride;
        block.padding = (stride + 1) / 2;  // ceil(stride / 2)
        block.output_padding = stride % 2;
        block.out_channels = out_channels;

        RT_TRY(alpha, load_alpha(sd, base + ".block.0.alpha"));
        if (alpha.size() != channels) {
            return err("snac: " + base + ".block.0.alpha has " + std::to_string(alpha.size()) +
                       " channels, expected " + std::to_string(channels));
        }
        block.alpha = std::move(alpha);

        RT_TRY(up, load_weight_norm_mat(sd, base + ".block.1"));
        if (up.rows != channels || up.cols != out_channels * block.kernel) {
            return err("snac: " + base + ".block.1 is " + std::to_string(up.rows) + "x" +
                       std::to_string(up.cols) + ", expected " + std::to_string(channels) + "x" +
                       std::to_string(out_channels * block.kernel) +
                       " ([Cin, Cout*K] -- transposed convolutions store input channels first)");
        }
        block.up_weight = std::move(up);
        block.up_bias = optional_bias(sd, base + ".block.1.bias");
        if (block.up_bias.size() != out_channels) {
            return err("snac: " + base + ".block.1.bias has " +
                       std::to_string(block.up_bias.size()) + " entries, expected " +
                       std::to_string(out_channels) + " (bias is per output channel)");
        }

        if (d.config.noise) {
            RT_TRY(noise_w, load_weight_norm_mat(sd, base + ".block.2.linear"));
            if (noise_w.rows != out_channels || noise_w.cols != out_channels) {
                return err("snac: " + base + ".block.2.linear is not square at " +
                           std::to_string(out_channels) + " channels");
            }
            SnacNoiseBlock nb;
            nb.weight = std::move(noise_w);
            block.noise = std::move(nb);
        }

        for (std::size_t u = 0; u < 3; ++u) {
            const std::string ub = base + ".block." + std::to_string(unit_base + u);
            SnacResidualUnit unit;
            unit.dilation = u == 0 ? 1 : (u == 1 ? 3 : 9);
            unit.kernel = 7;
            unit.depthwise = d.config.depthwise;

            RT_TRY(a1, load_alpha(sd, ub + ".block.0.alpha"));
            unit.alpha1 = std::move(a1);
            RT_TRY(c1, load_weight_norm_mat(sd, ub + ".block.1"));
            unit.conv1_weight = std::move(c1);
            unit.conv1_bias = optional_bias(sd, ub + ".block.1.bias");
            RT_TRY(a2, load_alpha(sd, ub + ".block.2.alpha"));
            unit.alpha2 = std::move(a2);
            RT_TRY(c2, load_weight_norm_mat(sd, ub + ".block.3"));
            unit.conv2_weight = std::move(c2);
            unit.conv2_bias = optional_bias(sd, ub + ".block.3.bias");

            if (unit.alpha1.size() != out_channels || unit.alpha2.size() != out_channels) {
                return err("snac: " + ub + " alphas do not match " +
                           std::to_string(out_channels) + " channels");
            }
            // The first convolution is grouped, the second is dense. Mixing
            // those up is a shape error rather than a silent one.
            const std::size_t want_c1_cols = d.config.depthwise ? unit.kernel
                                                                : out_channels * unit.kernel;
            if (unit.conv1_weight.rows != out_channels ||
                unit.conv1_weight.cols != want_c1_cols) {
                return err("snac: " + ub + ".block.1 is " +
                           std::to_string(unit.conv1_weight.rows) + "x" +
                           std::to_string(unit.conv1_weight.cols) + ", expected " +
                           std::to_string(out_channels) + "x" + std::to_string(want_c1_cols));
            }
            if (unit.conv2_weight.rows != out_channels ||
                unit.conv2_weight.cols != out_channels) {
                return err("snac: " + ub + ".block.3 is not a dense " +
                           std::to_string(out_channels) + "x" + std::to_string(out_channels) +
                           " pointwise convolution");
            }

            block.units.push_back(std::move(unit));
        }

        d.blocks.push_back(std::move(block));
        channels = out_channels;
    }

    // ---- output ----
    const std::size_t tail = next_module + d.config.decoder_rates.size();
    RT_TRY(out_alpha, load_alpha(sd, "decoder.model." + std::to_string(tail) + ".alpha"));
    if (out_alpha.size() != channels) {
        return err("snac: output alpha has " + std::to_string(out_alpha.size()) +
                   " channels, expected " + std::to_string(channels));
    }
    d.out_alpha = std::move(out_alpha);

    RT_TRY(out_w, load_weight_norm_mat(sd, "decoder.model." + std::to_string(tail + 1)));
    if (out_w.rows != 1 || out_w.cols != channels * 7) {
        return err("snac: output convolution is " + std::to_string(out_w.rows) + "x" +
                   std::to_string(out_w.cols) + ", expected 1x" + std::to_string(channels * 7));
    }
    d.out_weight = std::move(out_w);
    d.out_bias = optional_bias(sd, "decoder.model." + std::to_string(tail + 1) + ".bias");

    return d;
}

std::size_t SnacDecoder::parameter_count() const {
    std::size_t n = in_conv_weight.numel() + in_conv_bias.size() + in_mix_bias.size() +
                    out_alpha.size() + out_weight.numel() + out_bias.size();
    if (in_mix_weight) {
        n += in_mix_weight->numel();
    }
    for (const SnacQuantizer::Level& level : quantizer.levels) {
        n += level.codebook.numel() + level.out_proj_weight.numel() + level.out_proj_bias.size();
    }
    for (const SnacDecoderBlock& block : blocks) {
        n += block.alpha.size() + block.up_weight.numel() + block.up_bias.size();
        if (block.noise) {
            n += block.noise->weight.numel();
        }
        for (const SnacResidualUnit& unit : block.units) {
            n += unit.alpha1.size() + unit.conv1_weight.numel() + unit.conv1_bias.size() +
                 unit.alpha2.size() + unit.conv2_weight.numel() + unit.conv2_bias.size();
        }
    }
    return n;
}

// =============================================================================
// Decoding
// =============================================================================

Result<std::vector<float>> SnacDecoder::decode(
    const std::vector<std::vector<std::uint32_t>>& codes, SnacNoise mode,
    std::uint64_t seed) const {
    RT_TRY(z_q, quantizer.from_codes(codes));
    const std::size_t frames = z_q.rows;

    InitRng rng(seed);

    // decoder.model.0 -- filters at the latent width, kernel 7, padding 3.
    Mat h = config.depthwise
                ? conv1d_depthwise(z_q, in_conv_weight, in_conv_bias, 1, 3)
                : conv1d_dense(z_q, in_conv_weight, config.decoder_dim, 7, in_conv_bias, 1, 3);
    if (h.rows != frames) {
        return err("snac: input convolution changed the frame count");
    }

    // decoder.model.1 -- widens to decoder_dim.
    if (in_mix_weight) {
        h = conv1d_pointwise(h, *in_mix_weight, in_mix_bias);
    }

    for (const SnacDecoderBlock& block : blocks) {
        h = block.forward(h, mode, rng);
    }

    if (h.rows != frames * config.upsample_factor()) {
        return err("snac: decoder produced " + std::to_string(h.rows) + " samples for " +
                   std::to_string(frames) + " frames, expected " +
                   std::to_string(frames * config.upsample_factor()));
    }

    snake1d_inplace(h, out_alpha);
    const Mat mono = conv1d_dense(h, out_weight, 1, 7, out_bias, 1, 3);
    if (mono.cols != 1) {
        return err("snac: output convolution did not collapse to one channel");
    }

    // Tanh bounds the waveform to [-1, 1], which is what makes the int16
    // conversion downstream safe without clipping.
    std::vector<float> samples(mono.rows);
    for (std::size_t t = 0; t < mono.rows; ++t) {
        samples[t] = std::tanh(mono.at(t, 0));
    }
    return samples;
}

}  // namespace rt

#include "rt/omnivoice_codec.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <utility>

#include "rt/conv1d.hpp"

namespace rt {

// =============================================================================
// Config
// =============================================================================

OmniCodecConfig OmniCodecConfig::defaults() { return OmniCodecConfig{}; }

std::size_t OmniCodecConfig::upsample_factor() const {
    std::size_t n = 1;
    for (const std::size_t r : upsampling_ratios) {
        n *= r;
    }
    return n;
}

// =============================================================================
// GGUF helpers
// =============================================================================

namespace {

/// Decode any supported tensor to flat f32, in GGUF memory order.
[[nodiscard]] Result<std::vector<float>> tensor_to_f32(const GgufFile& gguf, std::size_t idx) {
    switch (gguf.tensor_info[idx].gguf_type) {
        case GgufType::F32:
            return gguf.decode_f32(idx);
        case GgufType::F16:
            return gguf.decode_f16_to_f32(idx);
        case GgufType::Q8_0:
            return gguf.decode_q8_0_to_f32(idx);
        case GgufType::Q4K:
            return gguf.decode_q4k_to_f32(idx);
        case GgufType::Q6K:
            return gguf.decode_q6k_to_f32(idx);
        case GgufType::Q5K:
            return gguf.decode_q5k_to_f32(idx);
        case GgufType::Q4_0:
            return gguf.decode_q4_0_to_f32(idx);
        default:
            return err(std::string("omnivoice codec: tensor '") + gguf.tensor_info[idx].name +
                       "' has unsupported type " +
                       gguf_type_name(gguf.tensor_info[idx].gguf_type));
    }
}

[[nodiscard]] Result<std::size_t> find(const GgufFile& gguf, const std::string& name) {
    const std::optional<std::size_t> idx = gguf.find_tensor(name);
    if (!idx) {
        return err("omnivoice codec: checkpoint has no tensor '" + name + "'");
    }
    return *idx;
}

}  // namespace

Result<std::vector<float>> load_gguf_vector(const GgufFile& gguf, const std::string& name) {
    RT_TRY(idx, find(gguf, name));
    return tensor_to_f32(gguf, idx);
}

Result<std::vector<float>> load_gguf_alpha(const GgufFile& gguf, const std::string& name) {
    // Stored as [1, C]; only the channel count matters here.
    return load_gguf_vector(gguf, name);
}

Result<Mat> load_gguf_conv_weight(const GgufFile& gguf, const std::string& name,
                                  std::size_t out_channels) {
    RT_TRY(idx, find(gguf, name));
    RT_TRY(values, tensor_to_f32(gguf, idx));

    if (out_channels == 0 || values.size() % out_channels != 0) {
        return err("omnivoice codec: tensor '" + name + "' holds " +
                   std::to_string(values.size()) + " values, which does not divide into " +
                   std::to_string(out_channels) + " output channels");
    }
    const std::size_t cols = values.size() / out_channels;
    // No transpose: GGUF reverses the dimension *listing*, not the buffer.
    return Mat(std::move(values), out_channels, cols);
}

// =============================================================================
// OmniResidualUnit
// =============================================================================

Mat OmniResidualUnit::forward(const Mat& x) const {
    const std::size_t channels = x.cols;
    Mat y = snake1d(x, alpha1);

    // Kernel 7 with padding 3*dilation preserves length exactly.
    y = conv1d_dense(y, conv1_weight, channels, 7, conv1_bias, dilation, 3 * dilation);
    assert(y.rows == x.rows && "omnivoice codec: residual unit changed the sequence length");

    snake1d_inplace(y, alpha2);
    y = conv1d_pointwise(y, conv2_weight, conv2_bias);

    y.add_assign(x);
    return y;
}

// =============================================================================
// OmniDecoderBlock
// =============================================================================

Mat OmniDecoderBlock::forward(const Mat& x) const {
    Mat h = snake1d(x, alpha);
    h = conv_transpose1d(h, up_weight, out_channels, kernel, up_bias, stride, padding,
                         output_padding);
    assert(h.rows == x.rows * stride && "omnivoice codec: block broke the length multiple");

    for (const OmniResidualUnit& unit : units) {
        h = unit.forward(h);
    }
    return h;
}

// =============================================================================
// OmniQuantizer
// =============================================================================

Result<Mat> OmniQuantizer::from_codes(
    const std::vector<std::vector<std::uint32_t>>& codes) const {
    if (codes.size() != levels.size()) {
        return err("omnivoice codec: expected " + std::to_string(levels.size()) +
                   " codebooks, got " + std::to_string(codes.size()));
    }
    if (codes.empty() || codes.front().empty()) {
        return err("omnivoice codec: no codes to decode");
    }

    const std::size_t frames = codes.front().size();
    Mat z = Mat::zeros(frames, latent_dim);

    for (std::size_t i = 0; i < levels.size(); ++i) {
        const Level& level = levels[i];
        const std::vector<std::uint32_t>& ids = codes[i];
        // Every codebook runs at the frame rate, so there is no stride to
        // reconcile -- just a length check.
        if (ids.size() != frames) {
            return err("omnivoice codec: codebook " + std::to_string(i) + " has " +
                       std::to_string(ids.size()) + " codes, expected " +
                       std::to_string(frames));
        }

        Mat latents = Mat::zeros(frames, level.codebook.cols);
        for (std::size_t t = 0; t < frames; ++t) {
            if (ids[t] >= level.codebook.rows) {
                return err("omnivoice codec: code " + std::to_string(ids[t]) + " at codebook " +
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

        const Mat projected =
            conv1d_pointwise(latents, level.project_out_weight, level.project_out_bias);
        z.add_assign(projected);
    }

    return z;
}

// =============================================================================
// Loading
// =============================================================================

Result<OmniCodecDecoder> OmniCodecDecoder::load(const std::string& path, OmniCodecConfig cfg) {
    RT_TRY(gguf, GgufFile::open(path));

    const auto arch = gguf.metadata.find("general.architecture");
    if (arch != gguf.metadata.end()) {
        const std::optional<std::string_view> name = arch->second.as_str();
        if (name && *name != "omnivoice-tokenizer") {
            return err("omnivoice codec: GGUF declares architecture '" + std::string(*name) +
                       "', expected 'omnivoice-tokenizer'");
        }
    }

    OmniCodecDecoder d;
    d.config = std::move(cfg);

    if (d.config.upsample_factor() != d.config.hop_length) {
        return err("omnivoice codec: upsampling ratios multiply to " +
                   std::to_string(d.config.upsample_factor()) + " but hop_length is " +
                   std::to_string(d.config.hop_length));
    }

    // ---- quantizer ----
    d.quantizer.latent_dim = d.config.latent_dim;
    for (std::size_t i = 0; i < d.config.n_codebooks; ++i) {
        const std::string base = "quantizer.quantizers." + std::to_string(i);
        OmniQuantizer::Level level;

        RT_TRY(embed, load_gguf_conv_weight(gguf, base + ".codebook.embed", d.config.codebook_size));
        if (embed.cols != d.config.codebook_dim) {
            return err("omnivoice codec: codebook " + std::to_string(i) + " is " +
                       std::to_string(embed.rows) + "x" + std::to_string(embed.cols) +
                       ", expected " + std::to_string(d.config.codebook_size) + "x" +
                       std::to_string(d.config.codebook_dim));
        }
        level.codebook = std::move(embed);

        RT_TRY(proj, load_gguf_conv_weight(gguf, base + ".project_out.weight", d.config.latent_dim));
        if (proj.cols != d.config.codebook_dim) {
            return err("omnivoice codec: codebook " + std::to_string(i) + " project_out is " +
                       std::to_string(proj.rows) + "x" + std::to_string(proj.cols) +
                       ", expected " + std::to_string(d.config.latent_dim) + "x" +
                       std::to_string(d.config.codebook_dim));
        }
        level.project_out_weight = std::move(proj);

        RT_TRY(bias, load_gguf_vector(gguf, base + ".project_out.bias"));
        level.project_out_bias = std::move(bias);

        d.quantizer.levels.push_back(std::move(level));
    }

    // ---- fc2: latent width down to the decoder's input width ----
    RT_TRY(fc2, load_gguf_conv_weight(gguf, "fc2.weight", d.config.decoder_in_dim));
    if (fc2.cols != d.config.latent_dim) {
        return err("omnivoice codec: fc2 is " + std::to_string(fc2.rows) + "x" +
                   std::to_string(fc2.cols) + ", expected " +
                   std::to_string(d.config.decoder_in_dim) + "x" +
                   std::to_string(d.config.latent_dim));
    }
    d.fc2_weight = std::move(fc2);
    RT_TRY(fc2_bias, load_gguf_vector(gguf, "fc2.bias"));
    d.fc2_bias = std::move(fc2_bias);

    // ---- acoustic_decoder ----
    RT_TRY(in_conv, load_gguf_conv_weight(gguf, "acoustic_decoder.conv1.weight",
                                          d.config.decoder_dim));
    if (in_conv.cols != d.config.decoder_in_dim * 7) {
        return err("omnivoice codec: acoustic_decoder.conv1 is " + std::to_string(in_conv.rows) +
                   "x" + std::to_string(in_conv.cols) + ", expected " +
                   std::to_string(d.config.decoder_dim) + "x" +
                   std::to_string(d.config.decoder_in_dim * 7));
    }
    d.in_conv_weight = std::move(in_conv);
    RT_TRY(in_bias, load_gguf_vector(gguf, "acoustic_decoder.conv1.bias"));
    d.in_conv_bias = std::move(in_bias);

    std::size_t channels = d.config.decoder_dim;
    for (std::size_t i = 0; i < d.config.upsampling_ratios.size(); ++i) {
        const std::string base = "acoustic_decoder.block." + std::to_string(i);
        const std::size_t stride = d.config.upsampling_ratios[i];
        const std::size_t out_channels = channels / 2;

        OmniDecoderBlock block;
        block.stride = stride;
        block.kernel = 2 * stride;
        block.padding = (stride + 1) / 2;  // ceil(stride / 2)
        block.output_padding = stride % 2;
        block.out_channels = out_channels;

        RT_TRY(alpha, load_gguf_alpha(gguf, base + ".snake1.alpha"));
        if (alpha.size() != channels) {
            return err("omnivoice codec: " + base + ".snake1.alpha has " +
                       std::to_string(alpha.size()) + " channels, expected " +
                       std::to_string(channels));
        }
        block.alpha = std::move(alpha);

        // A transposed convolution's weight is [Cin, Cout * K].
        RT_TRY(up, load_gguf_conv_weight(gguf, base + ".conv_t1.weight", channels));
        if (up.cols != out_channels * block.kernel) {
            return err("omnivoice codec: " + base + ".conv_t1 is " + std::to_string(up.rows) +
                       "x" + std::to_string(up.cols) + ", expected " + std::to_string(channels) +
                       "x" + std::to_string(out_channels * block.kernel));
        }
        block.up_weight = std::move(up);
        RT_TRY(up_bias, load_gguf_vector(gguf, base + ".conv_t1.bias"));
        if (up_bias.size() != out_channels) {
            return err("omnivoice codec: " + base + ".conv_t1.bias has " +
                       std::to_string(up_bias.size()) + " entries, expected " +
                       std::to_string(out_channels) + " (bias is per output channel)");
        }
        block.up_bias = std::move(up_bias);

        for (std::size_t u = 0; u < 3; ++u) {
            const std::string ub = base + ".res_unit" + std::to_string(u + 1);
            OmniResidualUnit unit;
            unit.dilation = u == 0 ? 1 : (u == 1 ? 3 : 9);

            RT_TRY(a1, load_gguf_alpha(gguf, ub + ".snake1.alpha"));
            unit.alpha1 = std::move(a1);
            RT_TRY(c1, load_gguf_conv_weight(gguf, ub + ".conv1.weight", out_channels));
            unit.conv1_weight = std::move(c1);
            RT_TRY(c1b, load_gguf_vector(gguf, ub + ".conv1.bias"));
            unit.conv1_bias = std::move(c1b);
            RT_TRY(a2, load_gguf_alpha(gguf, ub + ".snake2.alpha"));
            unit.alpha2 = std::move(a2);
            RT_TRY(c2, load_gguf_conv_weight(gguf, ub + ".conv2.weight", out_channels));
            unit.conv2_weight = std::move(c2);
            RT_TRY(c2b, load_gguf_vector(gguf, ub + ".conv2.bias"));
            unit.conv2_bias = std::move(c2b);

            // conv1 is a dense 7-tap, conv2 a pointwise mix. Mixing those up is
            // a shape error rather than a silent one.
            if (unit.conv1_weight.cols != out_channels * 7) {
                return err("omnivoice codec: " + ub + ".conv1 is " +
                           std::to_string(unit.conv1_weight.rows) + "x" +
                           std::to_string(unit.conv1_weight.cols) + ", expected " +
                           std::to_string(out_channels) + "x" + std::to_string(out_channels * 7));
            }
            if (unit.conv2_weight.cols != out_channels) {
                return err("omnivoice codec: " + ub + ".conv2 is not a pointwise " +
                           std::to_string(out_channels) + "x" + std::to_string(out_channels));
            }
            if (unit.alpha1.size() != out_channels || unit.alpha2.size() != out_channels) {
                return err("omnivoice codec: " + ub + " alphas do not match " +
                           std::to_string(out_channels) + " channels");
            }

            block.units.push_back(std::move(unit));
        }

        d.blocks.push_back(std::move(block));
        channels = out_channels;
    }

    // ---- output ----
    RT_TRY(out_alpha, load_gguf_alpha(gguf, "acoustic_decoder.snake1.alpha"));
    if (out_alpha.size() != channels) {
        return err("omnivoice codec: output alpha has " + std::to_string(out_alpha.size()) +
                   " channels, expected " + std::to_string(channels));
    }
    d.out_alpha = std::move(out_alpha);

    // One output channel, so GGUF elides the leading dimension entirely.
    RT_TRY(out_w, load_gguf_conv_weight(gguf, "acoustic_decoder.conv2.weight", 1));
    if (out_w.cols != channels * 7) {
        return err("omnivoice codec: acoustic_decoder.conv2 is 1x" + std::to_string(out_w.cols) +
                   ", expected 1x" + std::to_string(channels * 7));
    }
    d.out_weight = std::move(out_w);
    RT_TRY(out_bias, load_gguf_vector(gguf, "acoustic_decoder.conv2.bias"));
    d.out_bias = std::move(out_bias);

    return d;
}

std::size_t OmniCodecDecoder::parameter_count() const {
    std::size_t n = fc2_weight.numel() + fc2_bias.size() + in_conv_weight.numel() +
                    in_conv_bias.size() + out_alpha.size() + out_weight.numel() + out_bias.size();
    for (const OmniQuantizer::Level& level : quantizer.levels) {
        n += level.codebook.numel() + level.project_out_weight.numel() +
             level.project_out_bias.size();
    }
    for (const OmniDecoderBlock& block : blocks) {
        n += block.alpha.size() + block.up_weight.numel() + block.up_bias.size();
        for (const OmniResidualUnit& unit : block.units) {
            n += unit.alpha1.size() + unit.conv1_weight.numel() + unit.conv1_bias.size() +
                 unit.alpha2.size() + unit.conv2_weight.numel() + unit.conv2_bias.size();
        }
    }
    return n;
}

// =============================================================================
// Decoding
// =============================================================================

Result<std::vector<float>> OmniCodecDecoder::decode(
    const std::vector<std::vector<std::uint32_t>>& codes) const {
    RT_TRY(z, quantizer.from_codes(codes));
    const std::size_t frames = z.rows;

    // fc2 narrows the quantizer's output to the decoder's input width.
    Mat h = conv1d_pointwise(z, fc2_weight, fc2_bias);

    h = conv1d_dense(h, in_conv_weight, config.decoder_dim, 7, in_conv_bias, 1, 3);
    if (h.rows != frames) {
        return err("omnivoice codec: input convolution changed the frame count");
    }

    for (const OmniDecoderBlock& block : blocks) {
        h = block.forward(h);
    }

    if (h.rows != frames * config.hop_length) {
        return err("omnivoice codec: decoder produced " + std::to_string(h.rows) +
                   " samples for " + std::to_string(frames) + " frames, expected " +
                   std::to_string(frames * config.hop_length));
    }

    snake1d_inplace(h, out_alpha);
    const Mat mono = conv1d_dense(h, out_weight, 1, 7, out_bias, 1, 3);
    if (mono.cols != 1) {
        return err("omnivoice codec: output convolution did not collapse to one channel");
    }

    // The reference bounds its output with tanh, which is also what keeps the
    // int16 conversion downstream safe without clipping.
    std::vector<float> samples(mono.rows);
    for (std::size_t t = 0; t < mono.rows; ++t) {
        samples[t] = std::tanh(mono.at(t, 0));
    }
    return samples;
}

}  // namespace rt

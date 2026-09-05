#include "rt/omnivoice_codec.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

#include "rt/conv1d.hpp"
#include "rt/resample.hpp"

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
// Quantizer -- analysis
// =============================================================================

Result<std::vector<std::vector<std::uint32_t>>> OmniQuantizer::to_codes(const Mat& latents) const {
    if (levels.empty()) {
        return err("omnivoice codec: the quantizer has no levels");
    }
    if (latents.cols != latent_dim) {
        return err("omnivoice codec: latents are " + std::to_string(latents.cols) +
                   " wide, expected " + std::to_string(latent_dim));
    }
    for (const Level& level : levels) {
        if (level.project_in_weight.rows == 0) {
            return err(
                "omnivoice codec: this quantizer was loaded for synthesis only and has no "
                "project_in");
        }
    }

    std::vector<std::vector<std::uint32_t>> codes;
    codes.reserve(levels.size());

    Mat residual = latents;
    for (const Level& level : levels) {
        const std::size_t dim = level.codebook.cols;
        const Mat projected =
            conv1d_pointwise(residual, level.project_in_weight, level.project_in_bias);

        // Nearest entry by Euclidean distance. ||x||^2 is the same for every
        // candidate, so only -2 x.e + ||e||^2 decides -- which turns the search
        // into one gemm against the codebook plus a precomputed norm.
        std::vector<float> code_norm(level.codebook.rows, 0.0f);
        for (std::size_t e = 0; e < level.codebook.rows; ++e) {
            float sum = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                const float v = level.codebook.at(e, d);
                sum += v * v;
            }
            code_norm[e] = sum;
        }
        const Mat dots = projected.matmul_bt(level.codebook);

        std::vector<std::uint32_t> chosen(latents.rows, 0);
        for (std::size_t t = 0; t < latents.rows; ++t) {
            const float* row = dots.row(t).data();
            float best = std::numeric_limits<float>::infinity();
            std::uint32_t best_e = 0;
            for (std::size_t e = 0; e < level.codebook.rows; ++e) {
                const float d = code_norm[e] - 2.0f * row[e];
                if (d < best) {
                    best = d;
                    best_e = static_cast<std::uint32_t>(e);
                }
            }
            chosen[t] = best_e;
        }

        // Subtract what this level can represent, so the next one codes the
        // error rather than the signal again.
        Mat picked = Mat::zeros(latents.rows, dim);
        for (std::size_t t = 0; t < latents.rows; ++t) {
            for (std::size_t d = 0; d < dim; ++d) {
                picked.at_mut(t, d) = level.codebook.at(chosen[t], d);
            }
        }
        const Mat reconstructed =
            conv1d_pointwise(picked, level.project_out_weight, level.project_out_bias);
        for (std::size_t i = 0; i < residual.data.size(); ++i) {
            residual.data[i] -= reconstructed.data[i];
        }

        codes.push_back(std::move(chosen));
    }
    return codes;
}

// =============================================================================
// Acoustic encoder
// =============================================================================

Mat OmniEncoderBlock::forward(const Mat& x) const {
    Mat h = x;
    for (const OmniResidualUnit& unit : units) {
        h = unit.forward(h);
    }
    snake1d_inplace(h, alpha);
    h = conv1d_dense(h, down_weight, out_channels, kernel, down_bias, 1, padding, stride);
    return h;
}

Result<Mat> OmniAcousticEncoder::forward(std::span<const float> samples,
                                         std::size_t hop_length) const {
    if (hop_length == 0 || samples.size() % hop_length != 0) {
        return err("omnivoice codec: " + std::to_string(samples.size()) +
                   " samples do not divide into frames of " + std::to_string(hop_length));
    }
    const std::size_t frames = samples.size() / hop_length;
    if (frames == 0) {
        return err("omnivoice codec: the reference clip is shorter than one frame");
    }

    Mat h(std::vector<float>(samples.begin(), samples.end()), samples.size(), 1);
    h = conv1d_dense(h, in_weight, in_weight.rows, 7, in_bias, 1, 3);
    assert(h.rows == samples.size() && "omnivoice codec: the input convolution changed the length");

    for (const OmniEncoderBlock& block : blocks) {
        h = block.forward(h);
    }
    if (h.rows != frames) {
        return err("omnivoice codec: the acoustic encoder produced " + std::to_string(h.rows) +
                   " frames from " + std::to_string(samples.size()) + " samples, expected " +
                   std::to_string(frames));
    }

    snake1d_inplace(h, out_alpha);
    return conv1d_dense(h, out_weight, out_weight.rows, 3, out_bias, 1, 1);
}

// =============================================================================
// Semantic encoder
// =============================================================================

namespace {

/// ELU at alpha 1, the semantic stack's activation.
void elu_inplace(Mat& x) {
    for (float& v : x.data) {
        v = v > 0.0f ? v : std::expm1(v);
    }
}

}  // namespace

Mat OmniSemanticResidualUnit::forward(const Mat& x) const {
    Mat h = x;
    elu_inplace(h);
    h = conv1d_dense(h, conv1_weight, conv1_weight.rows, 3, {}, dilation, dilation);
    assert(h.rows == x.rows && "omnivoice codec: semantic residual unit changed the length");
    elu_inplace(h);
    h = conv1d_pointwise(h, conv2_weight, {});
    h.add_assign(x);
    return h;
}

Mat OmniSemanticBlock::forward(const Mat& x) const {
    Mat h = x;
    for (const OmniSemanticResidualUnit& unit : units) {
        h = unit.forward(h);
    }
    return conv1d_dense(h, conv_weight, out_channels, kernel, conv_bias, 1, padding, stride);
}

Mat OmniSemanticEncoder::forward(const Mat& x) const {
    Mat h = conv1d_dense(x, in_weight, in_weight.rows, 3, {}, 1, 1);
    for (const OmniSemanticBlock& block : blocks) {
        h = block.forward(h);
    }
    return h;
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


// =============================================================================
// Loading -- analysis
// =============================================================================

Result<OmniCodecEncoder> OmniCodecEncoder::load(const std::string& path, OmniCodecConfig cfg) {
    RT_TRY(gguf, GgufFile::open(path));

    const auto arch = gguf.metadata.find("general.architecture");
    if (arch != gguf.metadata.end()) {
        const std::optional<std::string_view> name = arch->second.as_str();
        if (name && *name != "omnivoice-tokenizer") {
            return err("omnivoice codec: GGUF declares architecture '" + std::string(*name) +
                       "', expected 'omnivoice-tokenizer'");
        }
    }

    OmniCodecEncoder e;
    e.config = std::move(cfg);
    const OmniCodecConfig& c = e.config;

    if (c.upsample_factor() != c.hop_length) {
        return err("omnivoice codec: upsampling ratios multiply to " +
                   std::to_string(c.upsample_factor()) + " but hop_length is " +
                   std::to_string(c.hop_length));
    }

    // ---- quantizer: codebooks plus both projections ----
    e.quantizer.latent_dim = c.latent_dim;
    for (std::size_t i = 0; i < c.n_codebooks; ++i) {
        const std::string base = "quantizer.quantizers." + std::to_string(i);
        OmniQuantizer::Level level;

        RT_TRY(embed, load_gguf_conv_weight(gguf, base + ".codebook.embed", c.codebook_size));
        if (embed.cols != c.codebook_dim) {
            return err("omnivoice codec: codebook " + std::to_string(i) + " is " +
                       std::to_string(embed.rows) + "x" + std::to_string(embed.cols));
        }
        level.codebook = std::move(embed);

        RT_TRY(pin, load_gguf_conv_weight(gguf, base + ".project_in.weight", c.codebook_dim));
        if (pin.cols != c.latent_dim) {
            return err("omnivoice codec: codebook " + std::to_string(i) + " project_in is " +
                       std::to_string(pin.rows) + "x" + std::to_string(pin.cols));
        }
        level.project_in_weight = std::move(pin);
        RT_TRY(pin_b, load_gguf_vector(gguf, base + ".project_in.bias"));
        level.project_in_bias = std::move(pin_b);

        RT_TRY(pout, load_gguf_conv_weight(gguf, base + ".project_out.weight", c.latent_dim));
        level.project_out_weight = std::move(pout);
        RT_TRY(pout_b, load_gguf_vector(gguf, base + ".project_out.bias"));
        level.project_out_bias = std::move(pout_b);

        e.quantizer.levels.push_back(std::move(level));
    }

    // ---- acoustic encoder ----
    RT_TRY(in_w, load_gguf_conv_weight(gguf, "acoustic_encoder.conv1.weight", c.encoder_dim));
    if (in_w.cols != 7) {
        return err("omnivoice codec: acoustic_encoder.conv1 has " + std::to_string(in_w.cols) +
                   " taps, expected 7 over one input channel");
    }
    e.acoustic.in_weight = std::move(in_w);
    RT_TRY(in_b, load_gguf_vector(gguf, "acoustic_encoder.conv1.bias"));
    e.acoustic.in_bias = std::move(in_b);

    std::size_t channels = c.encoder_dim;
    for (std::size_t i = 0; i < c.upsampling_ratios.size(); ++i) {
        const std::string base = "acoustic_encoder.block." + std::to_string(i);
        const std::size_t stride = c.upsampling_ratios[i];
        OmniEncoderBlock block;

        // Three residual units at dilations 1, 3 and 9, all on the block's
        // *input* width -- they run before the downsample.
        for (std::size_t u = 0; u < 3; ++u) {
            const std::string ub = base + ".res_unit" + std::to_string(u + 1);
            OmniResidualUnit unit;
            unit.dilation = u == 0 ? 1 : (u == 1 ? 3 : 9);

            RT_TRY(a1, load_gguf_alpha(gguf, ub + ".snake1.alpha"));
            if (a1.size() != channels) {
                return err("omnivoice codec: " + ub + ".snake1.alpha has " +
                           std::to_string(a1.size()) + " channels, expected " +
                           std::to_string(channels));
            }
            unit.alpha1 = std::move(a1);
            RT_TRY(c1, load_gguf_conv_weight(gguf, ub + ".conv1.weight", channels));
            unit.conv1_weight = std::move(c1);
            RT_TRY(c1b, load_gguf_vector(gguf, ub + ".conv1.bias"));
            unit.conv1_bias = std::move(c1b);

            RT_TRY(a2, load_gguf_alpha(gguf, ub + ".snake2.alpha"));
            unit.alpha2 = std::move(a2);
            RT_TRY(c2, load_gguf_conv_weight(gguf, ub + ".conv2.weight", channels));
            unit.conv2_weight = std::move(c2);
            RT_TRY(c2b, load_gguf_vector(gguf, ub + ".conv2.bias"));
            unit.conv2_bias = std::move(c2b);

            block.units.push_back(std::move(unit));
        }

        RT_TRY(alpha, load_gguf_alpha(gguf, base + ".snake1.alpha"));
        block.alpha = std::move(alpha);

        // Kernel 2s with padding ceil(s/2) is what turns a multiple of s into
        // exactly that multiple divided by s, for odd and even strides alike.
        block.out_channels = channels * 2;
        block.stride = stride;
        block.kernel = 2 * stride;
        block.padding = (stride + 1) / 2;
        RT_TRY(dw, load_gguf_conv_weight(gguf, base + ".conv1.weight", block.out_channels));
        if (dw.cols != channels * block.kernel) {
            return err("omnivoice codec: " + base + ".conv1 is " + std::to_string(dw.rows) + "x" +
                       std::to_string(dw.cols) + ", expected " +
                       std::to_string(block.out_channels) + "x" +
                       std::to_string(channels * block.kernel));
        }
        block.down_weight = std::move(dw);
        RT_TRY(db, load_gguf_vector(gguf, base + ".conv1.bias"));
        block.down_bias = std::move(db);

        channels = block.out_channels;
        e.acoustic.blocks.push_back(std::move(block));
    }

    RT_TRY(out_alpha, load_gguf_alpha(gguf, "acoustic_encoder.snake1.alpha"));
    if (out_alpha.size() != channels) {
        return err("omnivoice codec: acoustic_encoder.snake1.alpha has " +
                   std::to_string(out_alpha.size()) + " channels, expected " +
                   std::to_string(channels));
    }
    e.acoustic.out_alpha = std::move(out_alpha);
    RT_TRY(out_w, load_gguf_conv_weight(gguf, "acoustic_encoder.conv2.weight", c.decoder_in_dim));
    if (out_w.cols != channels * 3) {
        return err("omnivoice codec: acoustic_encoder.conv2 is " + std::to_string(out_w.rows) +
                   "x" + std::to_string(out_w.cols) + ", expected " +
                   std::to_string(c.decoder_in_dim) + "x" + std::to_string(channels * 3));
    }
    e.acoustic.out_weight = std::move(out_w);
    RT_TRY(out_b, load_gguf_vector(gguf, "acoustic_encoder.conv2.bias"));
    e.acoustic.out_bias = std::move(out_b);

    // ---- semantic model and its adapter ----
    RT_TRY(hubert, HubertModel::load(gguf, HubertConfig::omnivoice_semantic(), "semantic_model"));
    e.semantic = std::move(hubert);
    if (e.semantic.config.hidden_size != c.semantic_dim()) {
        return err("omnivoice codec: the semantic model is " +
                   std::to_string(e.semantic.config.hidden_size) + " wide but the quantizer "
                   "leaves " + std::to_string(c.semantic_dim()) + " for it");
    }

    const std::size_t sem = c.semantic_dim();
    RT_TRY(sem_in, load_gguf_conv_weight(gguf, "encoder_semantic.conv.weight", sem));
    if (sem_in.cols != sem * 3) {
        return err("omnivoice codec: encoder_semantic.conv is " + std::to_string(sem_in.rows) +
                   "x" + std::to_string(sem_in.cols));
    }
    e.semantic_adapter.in_weight = std::move(sem_in);

    // Both blocks run at stride 1 and keep the width, which is what
    // channel_ratios (1, 1) and strides (1, 1) mean.
    for (std::size_t i = 0; i < 2; ++i) {
        const std::string base = "encoder_semantic.conv_blocks." + std::to_string(i);
        OmniSemanticBlock block;
        block.out_channels = sem;

        for (std::size_t u = 0; u < 2; ++u) {
            const std::string ub = base + ".res_units." + std::to_string(u);
            OmniSemanticResidualUnit unit;
            unit.dilation = 1;
            RT_TRY(w1, load_gguf_conv_weight(gguf, ub + ".conv1.weight", sem));
            unit.conv1_weight = std::move(w1);
            RT_TRY(w2, load_gguf_conv_weight(gguf, ub + ".conv2.weight", sem));
            unit.conv2_weight = std::move(w2);
            block.units.push_back(std::move(unit));
        }

        RT_TRY(cw, load_gguf_conv_weight(gguf, base + ".conv.weight", sem));
        block.conv_weight = std::move(cw);
        RT_TRY(cb, load_gguf_vector(gguf, base + ".conv.bias"));
        block.conv_bias = std::move(cb);

        e.semantic_adapter.blocks.push_back(std::move(block));
    }

    // ---- fc: mix the two paths ----
    RT_TRY(fc, load_gguf_conv_weight(gguf, "fc.weight", c.latent_dim));
    if (fc.cols != c.latent_dim) {
        return err("omnivoice codec: fc is " + std::to_string(fc.rows) + "x" +
                   std::to_string(fc.cols) + ", expected a square " +
                   std::to_string(c.latent_dim));
    }
    e.fc_weight = std::move(fc);
    RT_TRY(fc_b, load_gguf_vector(gguf, "fc.bias"));
    e.fc_bias = std::move(fc_b);

    return e;
}

// =============================================================================
// Encoding
// =============================================================================

Result<std::vector<std::vector<std::uint32_t>>> OmniCodecEncoder::encode(
    std::span<const float> samples) const {
    const std::size_t hop = config.hop_length;
    const std::size_t frames = samples.size() / hop;
    if (frames == 0) {
        return err("omnivoice codec: the clip is shorter than one " + std::to_string(hop) +
                   "-sample frame");
    }
    // Drop the partial frame rather than pad it: a ragged tail would put the
    // acoustic and semantic paths' lengths out of step.
    const std::span<const float> trimmed = samples.subspan(0, frames * hop);

    // ---- semantic path ----
    const std::size_t semantic_rate = semantic.config.downsample_factor();
    std::vector<float> resampled =
        resample(trimmed, config.sample_rate, kSemanticSampleRate);

    // The reference pads by half the semantic hop on each side. It is not a
    // "same" padding of anything -- it is what makes the frame count come out
    // at twice the codec's rate, which the assertion below is the real check on.
    const std::size_t pad = semantic_rate / 2;
    std::vector<float> padded;
    padded.reserve(resampled.size() + 2 * pad);
    padded.insert(padded.end(), pad, 0.0f);
    padded.insert(padded.end(), resampled.begin(), resampled.end());
    padded.insert(padded.end(), pad, 0.0f);
    resampled.clear();
    resampled.shrink_to_fit();

    RT_TRY(hidden, semantic.mean_hidden_states(padded));
    if (hidden.rows != frames * 2) {
        return err("omnivoice codec: the semantic model produced " + std::to_string(hidden.rows) +
                   " frames where " + std::to_string(frames * 2) + " were expected");
    }

    // 50 Hz down to the codec's 25 Hz by keeping every other frame -- a plain
    // decimation, not an average.
    Mat semantic_features = Mat::from_fn(frames, hidden.cols, [&](std::size_t r, std::size_t c) {
        return hidden.at(r * 2, c);
    });
    const Mat sem = semantic_adapter.forward(semantic_features);
    if (sem.rows != frames) {
        return err("omnivoice codec: the semantic adapter changed the frame count");
    }

    // ---- acoustic path ----
    RT_TRY(aco, acoustic.forward(trimmed, hop));

    // ---- concatenate, mix, quantize ----
    if (aco.cols + sem.cols != config.latent_dim) {
        return err("omnivoice codec: the two paths are " + std::to_string(aco.cols) + " and " +
                   std::to_string(sem.cols) + " wide, which do not fill " +
                   std::to_string(config.latent_dim));
    }
    Mat joined = Mat::zeros(frames, config.latent_dim);
    for (std::size_t t = 0; t < frames; ++t) {
        float* row = joined.row_mut(t).data();
        // Acoustic first, then semantic: the order fc was trained with.
        for (std::size_t c = 0; c < aco.cols; ++c) {
            row[c] = aco.at(t, c);
        }
        for (std::size_t c = 0; c < sem.cols; ++c) {
            row[aco.cols + c] = sem.at(t, c);
        }
    }

    const Mat latents = conv1d_pointwise(joined, fc_weight, fc_bias);
    return quantizer.to_codes(latents);
}

std::size_t OmniCodecEncoder::parameter_count() const {
    std::size_t n = semantic.parameter_count();
    const auto add_mat = [&n](const Mat& m) { n += m.data.size(); };
    const auto add_vec = [&n](const std::vector<float>& v) { n += v.size(); };

    add_mat(acoustic.in_weight);
    add_vec(acoustic.in_bias);
    for (const OmniEncoderBlock& b : acoustic.blocks) {
        add_vec(b.alpha);
        add_mat(b.down_weight);
        add_vec(b.down_bias);
        for (const OmniResidualUnit& u : b.units) {
            add_vec(u.alpha1);
            add_vec(u.alpha2);
            add_mat(u.conv1_weight);
            add_mat(u.conv2_weight);
            add_vec(u.conv1_bias);
            add_vec(u.conv2_bias);
        }
    }
    add_vec(acoustic.out_alpha);
    add_mat(acoustic.out_weight);
    add_vec(acoustic.out_bias);

    add_mat(semantic_adapter.in_weight);
    for (const OmniSemanticBlock& b : semantic_adapter.blocks) {
        add_mat(b.conv_weight);
        add_vec(b.conv_bias);
        for (const OmniSemanticResidualUnit& u : b.units) {
            add_mat(u.conv1_weight);
            add_mat(u.conv2_weight);
        }
    }

    add_mat(fc_weight);
    add_vec(fc_bias);
    for (const OmniQuantizer::Level& l : quantizer.levels) {
        add_mat(l.codebook);
        add_mat(l.project_in_weight);
        add_vec(l.project_in_bias);
        add_mat(l.project_out_weight);
        add_vec(l.project_out_bias);
    }
    return n;
}

}  // namespace rt

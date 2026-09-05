#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include "rt/conv2d.hpp"
#include "rt/mat.hpp"
#include "rt/vae.hpp"

using namespace rt;

namespace {

[[nodiscard]] bool approx(float a, float b, float tol) { return std::fabs(a - b) < tol; }

/// Deterministic small signed values, spread by a hash of the index so that a
/// transposed or mis-strided weight cannot pass on a smooth ramp.
[[nodiscard]] Mat spread(std::size_t rows, std::size_t cols, float scale = 0.05f,
                         std::size_t salt = 0) {
    return Mat::from_fn(rows, cols, [&](std::size_t r, std::size_t c) {
        const auto i = static_cast<float>((r * 131 + c * 37 + salt * 7919) % 251);
        return (std::sin(i * 0.41f) * 0.7f + std::cos(i * 0.13f) * 0.3f) * scale;
    });
}

[[nodiscard]] std::vector<float> spread_vec(std::size_t n, float scale = 0.05f,
                                            std::size_t salt = 0) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto k = static_cast<float>((i * 53 + salt * 6151) % 241);
        v[i] = (std::sin(k * 0.29f) * 0.6f + std::cos(k * 0.17f) * 0.4f) * scale;
    }
    return v;
}

[[nodiscard]] VaeResnetBlock make_resnet(std::size_t c_in, std::size_t c_out, std::size_t salt) {
    VaeResnetBlock b;
    b.in_channels = c_in;
    b.out_channels = c_out;
    b.norm1_weight = std::vector<float>(c_in, 1.0f);
    b.norm1_bias = std::vector<float>(c_in, 0.0f);
    b.conv1_weight = spread(c_out, c_in * 9, 0.05f, salt);
    b.conv1_bias = spread_vec(c_out, 0.02f, salt + 1);
    b.norm2_weight = std::vector<float>(c_out, 1.0f);
    b.norm2_bias = std::vector<float>(c_out, 0.0f);
    b.conv2_weight = spread(c_out, c_out * 9, 0.05f, salt + 2);
    b.conv2_bias = spread_vec(c_out, 0.02f, salt + 3);
    if (c_in != c_out) {
        b.shortcut_weight = spread(c_out, c_in, 0.2f, salt + 4);
        b.shortcut_bias = spread_vec(c_out, 0.01f, salt + 5);
    }
    return b;
}

/// Build a decoder of the given shape with deterministic synthetic weights.
///
/// The real weights are a 10 GB download away, but every structural property
/// worth testing -- the level ordering, the upsample count, the latent
/// rescaling, the tile blending -- is independent of their values.
[[nodiscard]] VaeDecoder make_decoder(const VaeConfig& cfg) {
    VaeDecoder d;
    d.cfg = cfg;

    const std::size_t n_levels = cfg.block_out_channels.size();
    const std::size_t c_coarse = cfg.block_out_channels.back();
    const std::size_t c_fine = cfg.block_out_channels.front();

    d.conv_in_weight = spread(c_coarse, cfg.latent_channels * 9, 0.1f, 1);
    d.conv_in_bias = spread_vec(c_coarse, 0.02f, 2);

    d.mid_resnet1 = make_resnet(c_coarse, c_coarse, 10);
    d.mid_resnet2 = make_resnet(c_coarse, c_coarse, 20);

    d.mid_attn.channels = c_coarse;
    d.mid_attn.norm_weight = std::vector<float>(c_coarse, 1.0f);
    d.mid_attn.norm_bias = std::vector<float>(c_coarse, 0.0f);
    d.mid_attn.q_weight = spread(c_coarse, c_coarse, 0.15f, 30);
    d.mid_attn.q_bias = spread_vec(c_coarse, 0.01f, 31);
    d.mid_attn.k_weight = spread(c_coarse, c_coarse, 0.15f, 32);
    d.mid_attn.k_bias = spread_vec(c_coarse, 0.01f, 33);
    d.mid_attn.v_weight = spread(c_coarse, c_coarse, 0.15f, 34);
    d.mid_attn.v_bias = spread_vec(c_coarse, 0.01f, 35);
    d.mid_attn.out_weight = spread(c_coarse, c_coarse, 0.15f, 36);
    d.mid_attn.out_bias = spread_vec(c_coarse, 0.01f, 37);

    std::size_t c_prev = c_coarse;
    for (std::size_t i = 0; i < n_levels; ++i) {
        const std::size_t c_out = cfg.block_out_channels[n_levels - 1 - i];
        VaeUpBlock block;
        for (std::size_t r = 0; r <= cfg.layers_per_block; ++r) {
            const std::size_t c_in = (r == 0) ? c_prev : c_out;
            block.resnets.push_back(make_resnet(c_in, c_out, 100 + i * 10 + r));
        }
        if (i + 1 < n_levels) {
            block.upsample_weight = spread(c_out, c_out * 9, 0.05f, 200 + i);
            block.upsample_bias = spread_vec(c_out, 0.01f, 300 + i);
        }
        d.up_blocks.push_back(std::move(block));
        c_prev = c_out;
    }

    d.conv_out_norm_weight = std::vector<float>(c_fine, 1.0f);
    d.conv_out_norm_bias = std::vector<float>(c_fine, 0.0f);
    d.conv_out_weight = spread(3, c_fine * 9, 0.2f, 400);
    d.conv_out_bias = spread_vec(3, 0.05f, 401);
    return d;
}

[[nodiscard]] VaeConfig tiny_config() {
    VaeConfig c;
    c.latent_channels = 4;
    c.block_out_channels = {8, 16};
    c.layers_per_block = 1;
    c.norm_groups = 4;
    c.norm_eps = 1e-6f;
    c.scaling_factor = 0.3611f;
    c.shift_factor = 0.1159f;
    return c;
}

}  // namespace

// =============================================================================
// Config
// =============================================================================

TEST_CASE("the FLUX autoencoder config is the 16-channel one", "[vae]") {
    const VaeConfig c = VaeConfig::flux();
    REQUIRE(c.latent_channels == 16);
    REQUIRE(c.block_out_channels == std::vector<std::size_t>{128, 256, 512, 512});
    REQUIRE(c.layers_per_block == 2);
    REQUIRE(c.norm_groups == 32);
    REQUIRE(approx(c.norm_eps, 1e-6f, 1e-12f));
    // Both rescaling constants. SD 1.x has only the first, which is exactly why
    // the second is the one that goes missing.
    REQUIRE(approx(c.scaling_factor, 0.3611f, 1e-6f));
    REQUIRE(approx(c.shift_factor, 0.1159f, 1e-6f));
    REQUIRE(c.downsample_factor() == 8);
}

TEST_CASE("downsample_factor is one doubling per level after the first", "[vae]") {
    VaeConfig c;
    c.block_out_channels = {32};
    REQUIRE(c.downsample_factor() == 1);
    c.block_out_channels = {32, 64};
    REQUIRE(c.downsample_factor() == 2);
    c.block_out_channels = {32, 64, 128};
    REQUIRE(c.downsample_factor() == 4);
    c.block_out_channels = {128, 256, 512, 512};
    REQUIRE(c.downsample_factor() == 8);
}

// =============================================================================
// Shape
// =============================================================================

TEST_CASE("decode upsamples by exactly the configured factor", "[vae]") {
    const VaeConfig cfg = tiny_config();
    const VaeDecoder d = make_decoder(cfg);
    const Mat z = spread(6 * 5, 4, 1.0f, 77);

    const auto out = d.decode(z, 6, 5);
    REQUIRE(out.has_value());
    REQUIRE(out->cols == 3);
    REQUIRE(out->rows == (6 * 2) * (5 * 2));
}

TEST_CASE("a three-level decoder upsamples by four", "[vae]") {
    VaeConfig cfg = tiny_config();
    cfg.block_out_channels = {8, 8, 16};
    const VaeDecoder d = make_decoder(cfg);
    const Mat z = spread(4 * 4, 4, 1.0f, 78);

    const auto out = d.decode(z, 4, 4);
    REQUIRE(out.has_value());
    REQUIRE(out->rows == (4 * 4) * (4 * 4));
    REQUIRE(cfg.downsample_factor() == 4);
}

TEST_CASE("decode rejects a latent of the wrong shape", "[vae]") {
    const VaeDecoder d = make_decoder(tiny_config());
    REQUIRE_FALSE(d.decode(spread(16, 8, 1.0f, 1), 4, 4).has_value());  // wrong channels
    REQUIRE_FALSE(d.decode(spread(15, 4, 1.0f, 1), 4, 4).has_value());  // wrong row count
}

TEST_CASE("decode is deterministic", "[vae]") {
    const VaeDecoder d = make_decoder(tiny_config());
    const Mat z = spread(4 * 4, 4, 1.0f, 5);
    const auto a = d.decode(z, 4, 4);
    const auto b = d.decode(z, 4, 4);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    REQUIRE(a->data == b->data);
}

TEST_CASE("decode produces something image-like from a plausible latent", "[vae]") {
    const VaeDecoder d = make_decoder(tiny_config());
    const Mat z = spread(8 * 8, 4, 1.0f, 9);
    const auto out = d.decode(z, 8, 8);
    REQUIRE(out.has_value());
    for (const float v : out->data) {
        REQUIRE(std::isfinite(v));
    }
}

// =============================================================================
// Latent rescaling
// =============================================================================

TEST_CASE("decode applies z / scaling_factor + shift_factor", "[vae]") {
    // Two decoders with identical weights, one rescaling and one not. Feeding
    // the second the pre-rescaled latent must give the identical image -- which
    // pins both constants and the order of the two operations.
    VaeConfig scaled = tiny_config();
    VaeConfig plain = tiny_config();
    plain.scaling_factor = 1.0f;
    plain.shift_factor = 0.0f;

    VaeDecoder a = make_decoder(scaled);
    VaeDecoder b = make_decoder(scaled);
    b.cfg = plain;

    const Mat z = spread(5 * 4, 4, 1.0f, 11);
    Mat pre = z;
    for (float& v : pre.data) {
        v = v / scaled.scaling_factor + scaled.shift_factor;
    }

    const auto from_scaled = a.decode(z, 5, 4);
    const auto from_plain = b.decode(pre, 5, 4);
    REQUIRE(from_scaled.has_value());
    REQUIRE(from_plain.has_value());
    for (std::size_t i = 0; i < from_scaled->data.size(); ++i) {
        REQUIRE(approx(from_scaled->data[i], from_plain->data[i], 1e-4f));
    }
}

TEST_CASE("dropping the shift factor changes the image", "[vae]") {
    // It does not break it -- it shifts the colour, which is why the omission
    // survives a visual check.
    VaeConfig with = tiny_config();
    VaeConfig without = tiny_config();
    without.shift_factor = 0.0f;

    VaeDecoder a = make_decoder(with);
    VaeDecoder b = make_decoder(with);
    b.cfg = without;

    const Mat z = spread(4 * 4, 4, 1.0f, 13);
    const auto pa = a.decode(z, 4, 4);
    const auto pb = b.decode(z, 4, 4);
    REQUIRE(pa.has_value());
    REQUIRE(pb.has_value());

    float max_delta = 0.0f;
    for (std::size_t i = 0; i < pa->data.size(); ++i) {
        max_delta = std::max(max_delta, std::fabs(pa->data[i] - pb->data[i]));
    }
    REQUIRE(max_delta > 1e-4f);
}

// =============================================================================
// Blocks
// =============================================================================

TEST_CASE("a resnet block carries a shortcut only when the width changes", "[vae]") {
    const VaeResnetBlock same = make_resnet(8, 8, 1);
    const VaeResnetBlock wider = make_resnet(8, 16, 2);
    REQUIRE(same.shortcut_weight.rows == 0);
    REQUIRE(wider.shortcut_weight.rows == 16);
    REQUIRE(wider.shortcut_weight.cols == 8);
}

TEST_CASE("a zeroed resnet block is its own residual", "[vae]") {
    // With both convolutions zero, the block reduces to the identity plus the
    // convolution biases -- which pins the residual add to the *input*, not to
    // the normalized input.
    VaeConfig cfg = tiny_config();
    VaeResnetBlock b = make_resnet(8, 8, 3);
    b.conv1_weight = Mat::zeros(8, 8 * 9);
    b.conv2_weight = Mat::zeros(8, 8 * 9);
    b.conv1_bias.assign(8, 0.0f);
    b.conv2_bias.assign(8, 0.0f);

    const Mat x = spread(4 * 4, 8, 1.0f, 4);
    const Mat out = b.forward(x, 4, 4, cfg);
    for (std::size_t i = 0; i < out.data.size(); ++i) {
        REQUIRE(approx(out.data[i], x.data[i], 1e-5f));
    }
}

TEST_CASE("a zeroed attention block is its own residual", "[vae]") {
    const VaeConfig cfg = tiny_config();
    VaeAttentionBlock a;
    a.channels = 8;
    a.norm_weight = std::vector<float>(8, 1.0f);
    a.norm_bias = std::vector<float>(8, 0.0f);
    a.q_weight = Mat::zeros(8, 8);
    a.q_bias = std::vector<float>(8, 0.0f);
    a.k_weight = Mat::zeros(8, 8);
    a.k_bias = std::vector<float>(8, 0.0f);
    a.v_weight = Mat::zeros(8, 8);
    a.v_bias = std::vector<float>(8, 0.0f);
    a.out_weight = Mat::zeros(8, 8);
    a.out_bias = std::vector<float>(8, 0.0f);

    const Mat x = spread(3 * 3, 8, 1.0f, 6);
    const Mat out = a.forward(x, cfg);
    for (std::size_t i = 0; i < out.data.size(); ++i) {
        REQUIRE(approx(out.data[i], x.data[i], 1e-6f));
    }
}

TEST_CASE("attention averages the values when every score is equal", "[vae]") {
    // Zero q and k make every score zero, so the softmax is uniform and each
    // output row is the mean of v. That pins the softmax axis: averaging over
    // channels instead of positions would give a different, per-row answer.
    const VaeConfig cfg = tiny_config();
    VaeAttentionBlock a;
    a.channels = 8;
    a.norm_weight = std::vector<float>(8, 1.0f);
    a.norm_bias = std::vector<float>(8, 0.0f);
    a.q_weight = Mat::zeros(8, 8);
    a.q_bias = std::vector<float>(8, 0.0f);
    a.k_weight = Mat::zeros(8, 8);
    a.k_bias = std::vector<float>(8, 0.0f);
    a.v_weight = Mat::zeros(8, 8);
    a.v_bias = spread_vec(8, 1.0f, 12);  // v is a constant per channel
    a.out_weight = Mat::zeros(8, 8);
    for (std::size_t i = 0; i < 8; ++i) {
        a.out_weight.at_mut(i, i) = 1.0f;  // identity projection
    }
    a.out_bias = std::vector<float>(8, 0.0f);

    const Mat x = spread(4 * 4, 8, 1.0f, 14);
    const Mat out = a.forward(x, cfg);
    for (std::size_t r = 0; r < out.rows; ++r) {
        for (std::size_t c = 0; c < 8; ++c) {
            REQUIRE(approx(out.at(r, c), x.at(r, c) + a.v_bias[c], 1e-5f));
        }
    }
}

TEST_CASE("attention mixes information across distant positions", "[vae]") {
    // Perturbing one corner of the input must change the opposite corner of the
    // output. A convolution cannot do that at this distance; attention must.
    const VaeConfig cfg = tiny_config();
    const VaeDecoder d = make_decoder(cfg);

    Mat z = spread(6 * 6, 4, 1.0f, 15);
    const auto base = d.mid_attn.forward(
        conv2d_dense(z, 6, 6, d.conv_in_weight, 16, 3, 3, d.conv_in_bias, 1, 1, 1), cfg);

    z.at_mut(0, 0) += 5.0f;
    const auto poked = d.mid_attn.forward(
        conv2d_dense(z, 6, 6, d.conv_in_weight, 16, 3, 3, d.conv_in_bias, 1, 1, 1), cfg);

    const std::size_t far = 6 * 6 - 1;
    float delta = 0.0f;
    for (std::size_t c = 0; c < 16; ++c) {
        delta = std::max(delta, std::fabs(base.at(far, c) - poked.at(far, c)));
    }
    REQUIRE(delta > 1e-5f);
}

// =============================================================================
// Tiling
// =============================================================================

TEST_CASE("decode_tiled falls through to decode when the latent fits", "[vae]") {
    const VaeDecoder d = make_decoder(tiny_config());
    const Mat z = spread(6 * 6, 4, 1.0f, 21);
    const auto whole = d.decode(z, 6, 6);
    const auto tiled = d.decode_tiled(z, 6, 6, 8, 2);
    REQUIRE(whole.has_value());
    REQUIRE(tiled.has_value());
    REQUIRE(whole->data == tiled->data);
}

TEST_CASE("decode_tiled covers every output pixel exactly once", "[vae]") {
    // A decoder whose convolutions are all zero emits its output bias
    // everywhere, whatever the tiling. Any pixel the blend weights fail to
    // normalize shows up immediately as a value that is not that bias.
    VaeConfig cfg = tiny_config();
    VaeDecoder d = make_decoder(cfg);
    d.conv_in_weight = Mat::zeros(16, 4 * 9);
    d.conv_in_bias.assign(16, 0.0f);
    for (VaeUpBlock& b : d.up_blocks) {
        for (VaeResnetBlock& r : b.resnets) {
            r.conv1_weight = Mat::zeros(r.out_channels, r.in_channels * 9);
            r.conv2_weight = Mat::zeros(r.out_channels, r.out_channels * 9);
            r.conv1_bias.assign(r.out_channels, 0.0f);
            r.conv2_bias.assign(r.out_channels, 0.0f);
            if (r.shortcut_weight.rows > 0) {
                r.shortcut_weight = Mat::zeros(r.out_channels, r.in_channels);
                r.shortcut_bias.assign(r.out_channels, 0.0f);
            }
        }
        if (b.upsamples()) {
            b.upsample_weight = Mat::zeros(b.upsample_weight.rows, b.upsample_weight.cols);
            b.upsample_bias.assign(b.upsample_bias.size(), 0.0f);
        }
    }
    d.mid_resnet1.conv1_weight = Mat::zeros(16, 16 * 9);
    d.mid_resnet1.conv2_weight = Mat::zeros(16, 16 * 9);
    d.mid_resnet1.conv1_bias.assign(16, 0.0f);
    d.mid_resnet1.conv2_bias.assign(16, 0.0f);
    d.mid_resnet2.conv1_weight = Mat::zeros(16, 16 * 9);
    d.mid_resnet2.conv2_weight = Mat::zeros(16, 16 * 9);
    d.mid_resnet2.conv1_bias.assign(16, 0.0f);
    d.mid_resnet2.conv2_bias.assign(16, 0.0f);
    d.mid_attn.q_weight = Mat::zeros(16, 16);
    d.mid_attn.k_weight = Mat::zeros(16, 16);
    d.mid_attn.v_weight = Mat::zeros(16, 16);
    d.mid_attn.out_weight = Mat::zeros(16, 16);
    d.mid_attn.q_bias.assign(16, 0.0f);
    d.mid_attn.k_bias.assign(16, 0.0f);
    d.mid_attn.v_bias.assign(16, 0.0f);
    d.mid_attn.out_bias.assign(16, 0.0f);
    d.conv_out_weight = Mat::zeros(3, 8 * 9);
    d.conv_out_bias = {0.25f, -0.5f, 0.75f};

    const Mat z = spread(24 * 20, 4, 1.0f, 22);
    const auto tiled = d.decode_tiled(z, 24, 20, 8, 3);
    REQUIRE(tiled.has_value());
    REQUIRE(tiled->rows == (24 * 2) * (20 * 2));
    for (std::size_t r = 0; r < tiled->rows; ++r) {
        REQUIRE(approx(tiled->at(r, 0), 0.25f, 1e-5f));
        REQUIRE(approx(tiled->at(r, 1), -0.5f, 1e-5f));
        REQUIRE(approx(tiled->at(r, 2), 0.75f, 1e-5f));
    }
}

TEST_CASE("decode_tiled produces the right shape on a non-multiple latent", "[vae]") {
    const VaeDecoder d = make_decoder(tiny_config());
    const Mat z = spread(23 * 17, 4, 1.0f, 23);
    const auto tiled = d.decode_tiled(z, 23, 17, 8, 2);
    REQUIRE(tiled.has_value());
    REQUIRE(tiled->rows == (23 * 2) * (17 * 2));
    REQUIRE(tiled->cols == 3);
    for (const float v : tiled->data) {
        REQUIRE(std::isfinite(v));
    }
}

TEST_CASE("decode_tiled rejects an overlap that is not smaller than the tile", "[vae]") {
    const VaeDecoder d = make_decoder(tiny_config());
    const Mat z = spread(20 * 20, 4, 1.0f, 24);
    REQUIRE_FALSE(d.decode_tiled(z, 20, 20, 8, 8).has_value());
    REQUIRE_FALSE(d.decode_tiled(z, 20, 20, 0, 0).has_value());
}

// =============================================================================
// Bookkeeping
// =============================================================================

TEST_CASE("parameter_count adds up", "[vae]") {
    const VaeConfig cfg = tiny_config();
    const VaeDecoder d = make_decoder(cfg);

    // conv_in 16 x (4*9) + 16
    std::size_t expected = 16 * 36 + 16;
    // two mid resnets, 16 -> 16, no shortcut
    const std::size_t resnet_16 = 16 + 16 + 16 * 144 + 16 + 16 + 16 + 16 * 144 + 16;
    expected += 2 * resnet_16;
    // attention: norm + four 16x16 projections with biases
    expected += 16 + 16 + 4 * (16 * 16 + 16);
    // up level 0: two 16 -> 16 resnets plus a 16 x (16*9) upsample conv
    expected += 2 * resnet_16 + 16 * 144 + 16;
    // up level 1: 16 -> 8 (with shortcut) then 8 -> 8
    expected += 16 + 16 + 8 * 144 + 8 + 8 + 8 + 8 * 72 + 8 + 8 * 16 + 8;
    expected += 8 + 8 + 8 * 72 + 8 + 8 + 8 + 8 * 72 + 8;
    // conv_norm_out + conv_out
    expected += 8 + 8 + 3 * 72 + 3;

    REQUIRE(d.parameter_count() == expected);
}

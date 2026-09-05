#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "rt/flux.hpp"
#include "rt/init_rng.hpp"
#include "rt/mat.hpp"
#include "rt/qlinear.hpp"

using namespace rt;

namespace {

[[nodiscard]] bool approx(float a, float b, float tol) { return std::fabs(a - b) < tol; }

[[nodiscard]] Mat spread(std::size_t rows, std::size_t cols, float scale, std::size_t salt) {
    return Mat::from_fn(rows, cols, [&](std::size_t r, std::size_t c) {
        const auto i = static_cast<float>((r * 131 + c * 37 + salt * 7919) % 251);
        return (std::sin(i * 0.41f) * 0.7f + std::cos(i * 0.13f) * 0.3f) * scale;
    });
}

[[nodiscard]] std::vector<float> spread_vec(std::size_t n, float scale, std::size_t salt) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto k = static_cast<float>((i * 53 + salt * 6151) % 241);
        v[i] = (std::sin(k * 0.29f) * 0.6f + std::cos(k * 0.17f) * 0.4f) * scale;
    }
    return v;
}

[[nodiscard]] QLinear linear(std::size_t out, std::size_t in, std::size_t salt) {
    return QLinear::from_f32(spread(out, in, 0.1f, salt), spread_vec(out, 0.02f, salt + 1));
}

/// A FLUX of the real shape but a hundredth of the width. Every structural
/// property under test -- the patch order, the stream split, the RoPE axes, the
/// modulation arithmetic -- is independent of the widths.
[[nodiscard]] FluxConfig tiny_config() {
    FluxConfig c;
    c.in_channels = 4;
    c.hidden_size = 32;
    c.n_heads = 4;          // head_dim 8
    c.n_double_blocks = 2;
    c.n_single_blocks = 2;
    c.mlp_ratio = 2.0f;     // mlp_hidden 64
    c.context_dim = 12;
    c.pooled_dim = 6;
    c.axes_dim = {2, 2, 4};  // even, and sums to head_dim 8
    c.guidance_embed = false;
    c.patch_size = 2;
    return c;
}

[[nodiscard]] FluxModel make_model(const FluxConfig& cfg) {
    FluxModel m;
    m.cfg = cfg;
    const std::size_t hidden = cfg.hidden_size;
    const std::size_t head_dim = cfg.head_dim();
    const std::size_t mlp = cfg.mlp_hidden();

    m.img_in = linear(hidden, cfg.patch_dim(), 1);
    m.txt_in = linear(hidden, cfg.context_dim, 3);
    m.time_in_1 = linear(hidden, 256, 5);
    m.time_in_2 = linear(hidden, hidden, 7);
    m.vector_in_1 = linear(hidden, cfg.pooled_dim, 9);
    m.vector_in_2 = linear(hidden, hidden, 11);
    if (cfg.guidance_embed) {
        m.guidance_in_1 = linear(hidden, 256, 13);
        m.guidance_in_2 = linear(hidden, hidden, 15);
    }

    for (std::size_t i = 0; i < cfg.n_double_blocks; ++i) {
        FluxDoubleBlock b;
        b.img_mod = linear(6 * hidden, hidden, 20 + i * 20);
        b.img_qkv = linear(3 * hidden, hidden, 22 + i * 20);
        b.img_proj = linear(hidden, hidden, 24 + i * 20);
        b.img_mlp_in = linear(mlp, hidden, 26 + i * 20);
        b.img_mlp_out = linear(hidden, mlp, 28 + i * 20);
        b.img_norm.query_scale = std::vector<float>(head_dim, 1.0f);
        b.img_norm.key_scale = std::vector<float>(head_dim, 1.0f);
        b.txt_mod = linear(6 * hidden, hidden, 30 + i * 20);
        b.txt_qkv = linear(3 * hidden, hidden, 32 + i * 20);
        b.txt_proj = linear(hidden, hidden, 34 + i * 20);
        b.txt_mlp_in = linear(mlp, hidden, 36 + i * 20);
        b.txt_mlp_out = linear(hidden, mlp, 38 + i * 20);
        b.txt_norm.query_scale = std::vector<float>(head_dim, 1.0f);
        b.txt_norm.key_scale = std::vector<float>(head_dim, 1.0f);
        m.double_blocks.push_back(std::move(b));
    }

    for (std::size_t i = 0; i < cfg.n_single_blocks; ++i) {
        FluxSingleBlock b;
        b.modulation = linear(3 * hidden, hidden, 60 + i * 10);
        b.linear1 = linear(3 * hidden + mlp, hidden, 62 + i * 10);
        b.linear2 = linear(hidden, hidden + mlp, 64 + i * 10);
        b.norm.query_scale = std::vector<float>(head_dim, 1.0f);
        b.norm.key_scale = std::vector<float>(head_dim, 1.0f);
        m.single_blocks.push_back(std::move(b));
    }

    m.final_mod = linear(2 * hidden, hidden, 90);
    m.final_linear = linear(cfg.patch_dim(), hidden, 92);
    return m;
}

}  // namespace

// =============================================================================
// Config
// =============================================================================

TEST_CASE("the schnell and dev configs differ only in the guidance embedding", "[flux]") {
    const FluxConfig s = FluxConfig::schnell();
    const FluxConfig d = FluxConfig::dev();
    REQUIRE_FALSE(s.guidance_embed);
    REQUIRE(d.guidance_embed);
    REQUIRE(s.hidden_size == d.hidden_size);
    REQUIRE(s.n_double_blocks == d.n_double_blocks);
    REQUIRE(s.n_single_blocks == d.n_single_blocks);
}

TEST_CASE("the FLUX shape constants line up", "[flux]") {
    const FluxConfig c = FluxConfig::schnell();
    REQUIRE(c.hidden_size == 3072);
    REQUIRE(c.n_heads == 24);
    REQUIRE(c.head_dim() == 128);
    REQUIRE(c.n_double_blocks == 19);
    REQUIRE(c.n_single_blocks == 38);
    REQUIRE(c.mlp_hidden() == 12288);
    // 16 channels through a 2x2 patch is a 64-wide token.
    REQUIRE(c.patch_dim() == 64);
    REQUIRE(c.context_dim == 4096);  // T5-XXL
    REQUIRE(c.pooled_dim == 768);    // CLIP-L
    // The RoPE axes must fill a head exactly.
    std::size_t sum = 0;
    for (const std::size_t d : c.axes_dim) {
        sum += d;
    }
    REQUIRE(sum == c.head_dim());
    REQUIRE(c.axes_dim == std::vector<std::size_t>{16, 56, 56});
}

// =============================================================================
// Patching
// =============================================================================

TEST_CASE("patchify and unpatchify are inverses", "[flux]") {
    const Mat z = spread(8 * 6, 4, 1.0f, 100);
    const Mat tokens = flux_patchify(z, 8, 6, 2);
    REQUIRE(tokens.rows == 4 * 3);
    REQUIRE(tokens.cols == 4 * 4);
    const Mat back = flux_unpatchify(tokens, 8, 6, 4, 2);
    REQUIRE(back.rows == z.rows);
    REQUIRE(back.cols == z.cols);
    for (std::size_t i = 0; i < z.data.size(); ++i) {
        REQUIRE(approx(back.data[i], z.data[i], 1e-7f));
    }
}

TEST_CASE("patchify packs channel-major within a patch", "[flux]") {
    // A latent whose value encodes its own (channel, y, x) makes the packing
    // order readable directly.
    const std::size_t h = 4;
    const std::size_t w = 4;
    const std::size_t ch = 2;
    Mat z = Mat::zeros(h * w, ch);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            for (std::size_t c = 0; c < ch; ++c) {
                z.at_mut(y * w + x, c) = static_cast<float>(c * 100 + y * 10 + x);
            }
        }
    }
    const Mat tokens = flux_patchify(z, h, w, 2);
    // Patch (0, 0) holds channel 0's four pixels first, then channel 1's.
    const float* p0 = tokens.row(0).data();
    REQUIRE(approx(p0[0], 0.0f, 1e-6f));    // c0 (0,0)
    REQUIRE(approx(p0[1], 1.0f, 1e-6f));    // c0 (0,1)
    REQUIRE(approx(p0[2], 10.0f, 1e-6f));   // c0 (1,0)
    REQUIRE(approx(p0[3], 11.0f, 1e-6f));   // c0 (1,1)
    REQUIRE(approx(p0[4], 100.0f, 1e-6f));  // c1 (0,0)
    REQUIRE(approx(p0[7], 111.0f, 1e-6f));  // c1 (1,1)

    // Patch (0, 1) is two pixels to the right.
    const float* p1 = tokens.row(1).data();
    REQUIRE(approx(p1[0], 2.0f, 1e-6f));
    REQUIRE(approx(p1[3], 13.0f, 1e-6f));
}

TEST_CASE("patchify is not pixel-major", "[flux]") {
    // The plausible alternative ordering -- pixel-major within the patch --
    // agrees with the real one only when there is a single channel.
    const Mat z = spread(4 * 4, 3, 1.0f, 101);
    const Mat tokens = flux_patchify(z, 4, 4, 2);
    // Under channel-major, columns 0..3 are all channel 0. Under pixel-major
    // they would be pixel (0,0)'s three channels then pixel (0,1)'s first.
    REQUIRE(approx(tokens.at(0, 0), z.at(0, 0), 1e-7f));
    REQUIRE(approx(tokens.at(0, 1), z.at(1, 0), 1e-7f));
    REQUIRE_FALSE(approx(tokens.at(0, 1), z.at(0, 1), 1e-7f));
}

TEST_CASE("image ids carry the patch-grid coordinates on axes 1 and 2", "[flux]") {
    const Mat ids = flux_image_ids(8, 6, 2);
    REQUIRE(ids.rows == 4 * 3);
    REQUIRE(ids.cols == 3);
    for (std::size_t y = 0; y < 4; ++y) {
        for (std::size_t x = 0; x < 3; ++x) {
            const std::size_t r = y * 3 + x;
            REQUIRE(approx(ids.at(r, 0), 0.0f, 1e-7f));  // the video axis
            REQUIRE(approx(ids.at(r, 1), static_cast<float>(y), 1e-7f));
            REQUIRE(approx(ids.at(r, 2), static_cast<float>(x), 1e-7f));
        }
    }
}

// =============================================================================
// Timestep embedding
// =============================================================================

TEST_CASE("the timestep embedding puts cosines first", "[flux]") {
    const std::vector<float> e = flux_timestep_embedding(0.0f, 8);
    REQUIRE(e.size() == 8);
    // At t = 0 every argument is zero, so the cosine half is all ones and the
    // sine half is all zeros. Swapping the halves is immediately visible.
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(approx(e[i], 1.0f, 1e-6f));
        REQUIRE(approx(e[4 + i], 0.0f, 1e-6f));
    }
}

TEST_CASE("the timestep embedding scales t by 1000", "[flux]") {
    // The lowest frequency is 1, so its argument is exactly `1000 * t`.
    const std::vector<float> e = flux_timestep_embedding(0.001f, 8);
    REQUIRE(approx(e[0], std::cos(1.0f), 1e-5f));
    REQUIRE(approx(e[4], std::sin(1.0f), 1e-5f));
}

TEST_CASE("the timestep embedding separates nearby timesteps", "[flux]") {
    const std::vector<float> a = flux_timestep_embedding(0.25f, 256);
    const std::vector<float> b = flux_timestep_embedding(0.26f, 256);
    float delta = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        delta = std::max(delta, std::fabs(a[i] - b[i]));
    }
    REQUIRE(delta > 0.1f);
}

// =============================================================================
// RoPE
// =============================================================================

TEST_CASE("RoPE leaves zero positions untouched", "[flux]") {
    // Text tokens carry all-zero ids, so their rotation must be the identity.
    // If it is not, the prompt is rotated as though it sat at the image's
    // origin and the two streams stop agreeing on what position means.
    const Mat original = spread(5, 4 * 8, 1.0f, 200);
    Mat x = original;
    const Mat ids = Mat::zeros(5, 3);
    flux_apply_rope(x, ids, 4, 8, {2, 3, 3}, 10000.0f);
    for (std::size_t i = 0; i < x.data.size(); ++i) {
        REQUIRE(approx(x.data[i], original.data[i], 1e-6f));
    }
}

TEST_CASE("RoPE preserves the norm of every pair", "[flux]") {
    Mat x = spread(6, 4 * 8, 1.0f, 201);
    const Mat before = x;
    Mat ids = Mat::zeros(6, 3);
    for (std::size_t r = 0; r < 6; ++r) {
        ids.at_mut(r, 1) = static_cast<float>(r);
        ids.at_mut(r, 2) = static_cast<float>(r * 2);
    }
    flux_apply_rope(x, ids, 4, 8, {2, 3, 3}, 10000.0f);

    for (std::size_t r = 0; r < 6; ++r) {
        for (std::size_t h = 0; h < 4; ++h) {
            for (std::size_t p = 0; p < 4; ++p) {
                const std::size_t at = h * 8 + 2 * p;
                const float n0 = before.at(r, at) * before.at(r, at) +
                                 before.at(r, at + 1) * before.at(r, at + 1);
                const float n1 =
                    x.at(r, at) * x.at(r, at) + x.at(r, at + 1) * x.at(r, at + 1);
                REQUIRE(approx(n0, n1, 1e-4f));
            }
        }
    }
}

TEST_CASE("RoPE rotates adjacent pairs, not split halves", "[flux]") {
    // One head, one axis, dimension 2: a single pair at a known angle.
    Mat x = Mat::zeros(1, 2);
    x.at_mut(0, 0) = 1.0f;
    x.at_mut(0, 1) = 0.0f;
    Mat ids = Mat::zeros(1, 1);
    ids.at_mut(0, 0) = 1.0f;
    flux_apply_rope(x, ids, 1, 2, {2}, 10000.0f);
    // omega = theta^0 = 1, so the angle is exactly the position.
    REQUIRE(approx(x.at(0, 0), std::cos(1.0f), 1e-6f));
    REQUIRE(approx(x.at(0, 1), std::sin(1.0f), 1e-6f));
}

TEST_CASE("each RoPE axis rotates its own slice of the head", "[flux]") {
    // Moving only the h coordinate must leave the w slice alone.
    const std::size_t head_dim = 8;
    const std::vector<std::size_t> axes{2, 2, 4};
    Mat a = spread(1, head_dim, 1.0f, 202);
    Mat b = a;
    Mat ids_a = Mat::zeros(1, 3);
    Mat ids_b = Mat::zeros(1, 3);
    ids_b.at_mut(0, 1) = 3.0f;  // move h only

    flux_apply_rope(a, ids_a, 1, head_dim, axes, 10000.0f);
    flux_apply_rope(b, ids_b, 1, head_dim, axes, 10000.0f);

    // Axis 0 owns dims 0..1, axis 1 owns 2..3, axis 2 owns 4..7. Only axis 1's
    // slice may change.
    REQUIRE(approx(a.at(0, 0), b.at(0, 0), 1e-6f));
    REQUIRE(approx(a.at(0, 1), b.at(0, 1), 1e-6f));
    REQUIRE_FALSE(approx(a.at(0, 2), b.at(0, 2), 1e-5f));
    REQUIRE_FALSE(approx(a.at(0, 3), b.at(0, 3), 1e-5f));
    for (std::size_t d = 4; d < 8; ++d) {
        REQUIRE(approx(a.at(0, d), b.at(0, d), 1e-6f));
    }
}

TEST_CASE("flux_check_axes rejects axes that do not tile the head", "[flux]") {
    FluxConfig c = tiny_config();
    REQUIRE(flux_check_axes(c).has_value());

    // An odd axis leaves a dimension unrotated and shifts everything after it.
    c.axes_dim = {2, 3, 3};
    REQUIRE_FALSE(flux_check_axes(c).has_value());

    // And the axes still have to fill the head exactly.
    c.axes_dim = {2, 2, 2};
    REQUIRE_FALSE(flux_check_axes(c).has_value());

    REQUIRE(flux_check_axes(FluxConfig::schnell()).has_value());
    REQUIRE(flux_check_axes(FluxConfig::dev()).has_value());
}

// =============================================================================
// Schedule
// =============================================================================

TEST_CASE("the schedule runs from 1 to 0 with one more entry than steps", "[flux]") {
    const std::vector<float> s = flux_schedule(4, 1.0f);
    REQUIRE(s.size() == 5);
    REQUIRE(approx(s.front(), 1.0f, 1e-6f));
    REQUIRE(approx(s.back(), 0.0f, 1e-6f));
    for (std::size_t i = 0; i + 1 < s.size(); ++i) {
        REQUIRE(s[i] > s[i + 1]);
    }
}

TEST_CASE("a shift of 1 is a linear schedule", "[flux]") {
    const std::vector<float> s = flux_schedule(4, 1.0f);
    REQUIRE(approx(s[0], 1.0f, 1e-6f));
    REQUIRE(approx(s[1], 0.75f, 1e-6f));
    REQUIRE(approx(s[2], 0.5f, 1e-6f));
    REQUIRE(approx(s[3], 0.25f, 1e-6f));
    REQUIRE(approx(s[4], 0.0f, 1e-6f));
}

TEST_CASE("a shift above 1 pushes the schedule toward the noisy end", "[flux]") {
    const std::vector<float> plain = flux_schedule(8, 1.0f);
    const std::vector<float> shifted = flux_schedule(8, 3.0f);
    REQUIRE(approx(shifted.front(), 1.0f, 1e-6f));
    REQUIRE(approx(shifted.back(), 0.0f, 1e-6f));
    // Every interior sigma sits higher, so more steps are spent at high noise.
    for (std::size_t i = 1; i + 1 < plain.size(); ++i) {
        REQUIRE(shifted[i] > plain[i]);
    }
}

// =============================================================================
// Forward
// =============================================================================

TEST_CASE("the model returns a velocity the shape of its latent", "[flux]") {
    const FluxConfig cfg = tiny_config();
    const FluxModel m = make_model(cfg);
    const Mat z = spread(8 * 6, 4, 1.0f, 300);
    const Mat ctx = spread(5, cfg.context_dim, 1.0f, 301);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 302);

    const auto v = m.forward(z, 8, 6, ctx, pooled, 0.75f);
    REQUIRE(v.has_value());
    REQUIRE(v->rows == z.rows);
    REQUIRE(v->cols == z.cols);
    for (const float f : v->data) {
        REQUIRE(std::isfinite(f));
    }
}

TEST_CASE("the model rejects shapes it cannot patch", "[flux]") {
    const FluxConfig cfg = tiny_config();
    const FluxModel m = make_model(cfg);
    const Mat ctx = spread(5, cfg.context_dim, 1.0f, 303);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 304);

    // Odd latent dimensions cannot be split into 2x2 patches.
    REQUIRE_FALSE(m.forward(spread(7 * 6, 4, 1.0f, 1), 7, 6, ctx, pooled, 0.5f).has_value());
    // Wrong channel count.
    REQUIRE_FALSE(m.forward(spread(8 * 6, 5, 1.0f, 1), 8, 6, ctx, pooled, 0.5f).has_value());
    // Wrong context width.
    REQUIRE_FALSE(
        m.forward(spread(8 * 6, 4, 1.0f, 1), 8, 6, spread(5, 7, 1.0f, 1), pooled, 0.5f)
            .has_value());
    // Wrong pooled width.
    REQUIRE_FALSE(
        m.forward(spread(8 * 6, 4, 1.0f, 1), 8, 6, ctx, std::vector<float>(3, 0.0f), 0.5f)
            .has_value());
}

TEST_CASE("the velocity depends on the timestep", "[flux]") {
    const FluxConfig cfg = tiny_config();
    const FluxModel m = make_model(cfg);
    const Mat z = spread(4 * 4, 4, 1.0f, 305);
    const Mat ctx = spread(3, cfg.context_dim, 1.0f, 306);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 307);

    const auto a = m.forward(z, 4, 4, ctx, pooled, 0.9f);
    const auto b = m.forward(z, 4, 4, ctx, pooled, 0.1f);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    float delta = 0.0f;
    for (std::size_t i = 0; i < a->data.size(); ++i) {
        delta = std::max(delta, std::fabs(a->data[i] - b->data[i]));
    }
    REQUIRE(delta > 1e-5f);
}

TEST_CASE("the velocity depends on the prompt and on the pooled vector separately", "[flux]") {
    const FluxConfig cfg = tiny_config();
    const FluxModel m = make_model(cfg);
    const Mat z = spread(4 * 4, 4, 1.0f, 308);
    const Mat ctx = spread(3, cfg.context_dim, 1.0f, 309);
    const Mat ctx2 = spread(3, cfg.context_dim, 1.0f, 310);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 311);
    const std::vector<float> pooled2 = spread_vec(cfg.pooled_dim, 1.0f, 312);

    const auto base = m.forward(z, 4, 4, ctx, pooled, 0.5f);
    const auto other_text = m.forward(z, 4, 4, ctx2, pooled, 0.5f);
    const auto other_pooled = m.forward(z, 4, 4, ctx, pooled2, 0.5f);
    REQUIRE(base.has_value());
    REQUIRE(other_text.has_value());
    REQUIRE(other_pooled.has_value());

    float dt = 0.0f;
    float dp = 0.0f;
    for (std::size_t i = 0; i < base->data.size(); ++i) {
        dt = std::max(dt, std::fabs(base->data[i] - other_text->data[i]));
        dp = std::max(dp, std::fabs(base->data[i] - other_pooled->data[i]));
    }
    REQUIRE(dt > 1e-5f);
    REQUIRE(dp > 1e-5f);
}

TEST_CASE("the image stream is spatially aware", "[flux]") {
    // Perturbing one patch must change a distant patch's velocity. Only the
    // joint attention can carry that, so this fails if the streams never meet.
    const FluxConfig cfg = tiny_config();
    const FluxModel m = make_model(cfg);
    Mat z = spread(6 * 6, 4, 1.0f, 313);
    const Mat ctx = spread(3, cfg.context_dim, 1.0f, 314);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 315);

    const auto base = m.forward(z, 6, 6, ctx, pooled, 0.5f);
    z.at_mut(0, 0) += 3.0f;
    const auto poked = m.forward(z, 6, 6, ctx, pooled, 0.5f);
    REQUIRE(base.has_value());
    REQUIRE(poked.has_value());

    const std::size_t far = 6 * 6 - 1;
    float delta = 0.0f;
    for (std::size_t c = 0; c < 4; ++c) {
        delta = std::max(delta, std::fabs(base->at(far, c) - poked->at(far, c)));
    }
    REQUIRE(delta > 1e-6f);
}

TEST_CASE("the guidance embedding is used only when the config asks for it", "[flux]") {
    FluxConfig cfg = tiny_config();
    cfg.guidance_embed = true;
    const FluxModel m = make_model(cfg);
    const Mat z = spread(4 * 4, 4, 1.0f, 316);
    const Mat ctx = spread(3, cfg.context_dim, 1.0f, 317);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 318);

    const auto a = m.forward(z, 4, 4, ctx, pooled, 0.5f, 1.0f);
    const auto b = m.forward(z, 4, 4, ctx, pooled, 0.5f, 7.0f);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    float delta = 0.0f;
    for (std::size_t i = 0; i < a->data.size(); ++i) {
        delta = std::max(delta, std::fabs(a->data[i] - b->data[i]));
    }
    REQUIRE(delta > 1e-5f);

    // schnell has no guidance path, so the argument is inert there.
    const FluxModel s = make_model(tiny_config());
    const auto c = s.forward(z, 4, 4, ctx, pooled, 0.5f, 1.0f);
    const auto d = s.forward(z, 4, 4, ctx, pooled, 0.5f, 7.0f);
    REQUIRE(c.has_value());
    REQUIRE(d.has_value());
    REQUIRE(c->data == d->data);
}

// =============================================================================
// Sampling
// =============================================================================

TEST_CASE("sampling returns a latent of the right shape", "[flux]") {
    const FluxConfig cfg = tiny_config();
    const FluxModel m = make_model(cfg);
    const Mat ctx = spread(3, cfg.context_dim, 1.0f, 400);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 401);

    FluxSampleParams p;
    p.width = 64;
    p.height = 32;
    p.steps = 2;
    p.seed = 7;

    const auto z = flux_sample(m, ctx, pooled, p);
    REQUIRE(z.has_value());
    REQUIRE(z->rows == (32 / 8) * (64 / 8));
    REQUIRE(z->cols == cfg.in_channels);
    for (const float v : z->data) {
        REQUIRE(std::isfinite(v));
    }
}

TEST_CASE("sampling is reproducible from its seed", "[flux]") {
    const FluxConfig cfg = tiny_config();
    const FluxModel m = make_model(cfg);
    const Mat ctx = spread(3, cfg.context_dim, 1.0f, 402);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 403);

    FluxSampleParams p;
    p.width = 32;
    p.height = 32;
    p.steps = 2;
    p.seed = 99;

    const auto a = flux_sample(m, ctx, pooled, p);
    const auto b = flux_sample(m, ctx, pooled, p);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    REQUIRE(a->data == b->data);

    p.seed = 100;
    const auto c = flux_sample(m, ctx, pooled, p);
    REQUIRE(c.has_value());
    REQUIRE_FALSE(a->data == c->data);
}

TEST_CASE("sampling rejects dimensions that do not divide", "[flux]") {
    const FluxConfig cfg = tiny_config();
    const FluxModel m = make_model(cfg);
    const Mat ctx = spread(3, cfg.context_dim, 1.0f, 404);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 405);

    FluxSampleParams p;
    p.steps = 1;
    p.width = 20;  // not a multiple of 16
    p.height = 32;
    REQUIRE_FALSE(flux_sample(m, ctx, pooled, p).has_value());

    p.width = 32;
    p.steps = 0;
    REQUIRE_FALSE(flux_sample(m, ctx, pooled, p).has_value());
}

TEST_CASE("a zero-velocity model leaves the noise untouched", "[flux]") {
    // Zeroing the final projection makes every velocity zero, so Euler
    // integration must return exactly the initial noise. That pins the
    // integration to `x += dt * v` and nothing else.
    const FluxConfig cfg = tiny_config();
    FluxModel m = make_model(cfg);
    m.final_linear = QLinear::from_f32(Mat::zeros(cfg.patch_dim(), cfg.hidden_size),
                                       std::vector<float>(cfg.patch_dim(), 0.0f));

    const Mat ctx = spread(3, cfg.context_dim, 1.0f, 406);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 407);

    FluxSampleParams p;
    p.width = 32;
    p.height = 32;
    p.steps = 4;
    p.seed = 5;

    const auto z = flux_sample(m, ctx, pooled, p);
    REQUIRE(z.has_value());

    InitRng rng(5);
    for (std::size_t i = 0; i < z->data.size(); ++i) {
        REQUIRE(approx(z->data[i], rng.next_normal(), 1e-6f));
    }
}

// =============================================================================
// Bookkeeping
// =============================================================================

TEST_CASE("parameter_count adds up", "[flux]") {
    const FluxConfig cfg = tiny_config();
    const FluxModel m = make_model(cfg);
    const std::size_t h = cfg.hidden_size;
    const std::size_t mlp = cfg.mlp_hidden();
    const std::size_t hd = cfg.head_dim();

    const auto lin = [](std::size_t out, std::size_t in) { return out * in + out; };

    std::size_t expected = lin(h, cfg.patch_dim()) + lin(h, cfg.context_dim) + lin(h, 256) +
                           lin(h, h) + lin(h, cfg.pooled_dim) + lin(h, h) + lin(2 * h, h) +
                           lin(cfg.patch_dim(), h);
    const std::size_t one_stream =
        lin(6 * h, h) + lin(3 * h, h) + lin(h, h) + lin(mlp, h) + lin(h, mlp) + 2 * hd;
    expected += cfg.n_double_blocks * 2 * one_stream;
    expected += cfg.n_single_blocks *
                (lin(3 * h, h) + lin(3 * h + mlp, h) + lin(h, h + mlp) + 2 * hd);

    REQUIRE(m.parameter_count() == expected);
}

TEST_CASE("the real FLUX config comes out near twelve billion parameters", "[flux]") {
    // A shape check on the config rather than on loaded weights: the block
    // counts and widths multiply out to the published size, so a typo in any
    // one of them shows up here.
    const FluxConfig c = FluxConfig::schnell();
    const std::size_t h = c.hidden_size;
    const std::size_t mlp = c.mlp_hidden();
    const auto lin = [](std::size_t out, std::size_t in) { return out * in + out; };

    std::size_t n = lin(h, c.patch_dim()) + lin(h, c.context_dim) + lin(h, 256) + lin(h, h) +
                    lin(h, c.pooled_dim) + lin(h, h) + lin(2 * h, h) + lin(c.patch_dim(), h);
    n += c.n_double_blocks * 2 *
         (lin(6 * h, h) + lin(3 * h, h) + lin(h, h) + lin(mlp, h) + lin(h, mlp) +
          2 * c.head_dim());
    n += c.n_single_blocks *
         (lin(3 * h, h) + lin(3 * h + mlp, h) + lin(h, h + mlp) + 2 * c.head_dim());

    REQUIRE(n > 11'000'000'000ULL);
    REQUIRE(n < 13'000'000'000ULL);
}

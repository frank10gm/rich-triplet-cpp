#include <catch2/catch_test_macros.hpp>

#if RT_FEATURE_METAL

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "rt/flux.hpp"
#include "rt/mat.hpp"
#include "rt/metal_flux.hpp"
#include "rt/qlinear.hpp"

using namespace rt;

namespace {

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

/// The GEMM kernel is a 64x64 register-blocked tile with no bounds checks, so
/// every width here is a multiple of 64 and every contracted dimension a
/// multiple of 8 -- the same constraints the real FLUX widths satisfy.
[[nodiscard]] FluxConfig gpu_config() {
    FluxConfig c;
    c.in_channels = 16;   // patch_dim = 16 * 4 = 64
    c.hidden_size = 64;
    c.n_heads = 8;        // head_dim 8
    c.n_double_blocks = 2;
    c.n_single_blocks = 2;
    c.mlp_ratio = 2.0f;   // mlp_hidden 128
    c.context_dim = 64;
    c.pooled_dim = 64;
    c.axes_dim = {2, 2, 4};
    c.guidance_embed = false;
    c.patch_size = 2;
    return c;
}

[[nodiscard]] FluxModel make_model(const FluxConfig& cfg) {
    FluxModel m;
    m.cfg = cfg;
    const std::size_t h = cfg.hidden_size;
    const std::size_t hd = cfg.head_dim();
    const std::size_t mlp = cfg.mlp_hidden();

    m.img_in = linear(h, cfg.patch_dim(), 1);
    m.txt_in = linear(h, cfg.context_dim, 3);
    m.time_in_1 = linear(h, 256, 5);
    m.time_in_2 = linear(h, h, 7);
    m.vector_in_1 = linear(h, cfg.pooled_dim, 9);
    m.vector_in_2 = linear(h, h, 11);
    if (cfg.guidance_embed) {
        m.guidance_in_1 = linear(h, 256, 13);
        m.guidance_in_2 = linear(h, h, 15);
    }
    for (std::size_t i = 0; i < cfg.n_double_blocks; ++i) {
        FluxDoubleBlock b;
        b.img_mod = linear(6 * h, h, 20 + i * 20);
        b.img_qkv = linear(3 * h, h, 22 + i * 20);
        b.img_proj = linear(h, h, 24 + i * 20);
        b.img_mlp_in = linear(mlp, h, 26 + i * 20);
        b.img_mlp_out = linear(h, mlp, 28 + i * 20);
        b.img_norm.query_scale = spread_vec(hd, 0.3f, 40 + i);
        b.img_norm.key_scale = spread_vec(hd, 0.3f, 41 + i);
        for (float& v : b.img_norm.query_scale) v += 1.0f;
        for (float& v : b.img_norm.key_scale) v += 1.0f;
        b.txt_mod = linear(6 * h, h, 30 + i * 20);
        b.txt_qkv = linear(3 * h, h, 32 + i * 20);
        b.txt_proj = linear(h, h, 34 + i * 20);
        b.txt_mlp_in = linear(mlp, h, 36 + i * 20);
        b.txt_mlp_out = linear(h, mlp, 38 + i * 20);
        b.txt_norm.query_scale = spread_vec(hd, 0.3f, 42 + i);
        b.txt_norm.key_scale = spread_vec(hd, 0.3f, 43 + i);
        for (float& v : b.txt_norm.query_scale) v += 1.0f;
        for (float& v : b.txt_norm.key_scale) v += 1.0f;
        m.double_blocks.push_back(std::move(b));
    }
    for (std::size_t i = 0; i < cfg.n_single_blocks; ++i) {
        FluxSingleBlock b;
        b.modulation = linear(3 * h, h, 60 + i * 10);
        b.linear1 = linear(3 * h + mlp, h, 62 + i * 10);
        b.linear2 = linear(h, h + mlp, 64 + i * 10);
        b.norm.query_scale = spread_vec(hd, 0.3f, 70 + i);
        b.norm.key_scale = spread_vec(hd, 0.3f, 71 + i);
        for (float& v : b.norm.query_scale) v += 1.0f;
        for (float& v : b.norm.key_scale) v += 1.0f;
        m.single_blocks.push_back(std::move(b));
    }
    m.final_mod = linear(2 * h, h, 90);
    m.final_linear = linear(cfg.patch_dim(), h, 92);
    return m;
}

struct Deviation {
    float max_abs = 0.0f;
    /// Error energy as a fraction of signal energy.
    ///
    /// The right metric for a velocity field. A pointwise relative error is
    /// not: the field crosses zero all over, and a value that happens to land
    /// near zero reports an enormous relative deviation for an absolute one
    /// that is pure rounding.
    float rel_rms = 0.0f;
};

[[nodiscard]] Deviation compare(const Mat& a, const Mat& b) {
    REQUIRE(a.rows == b.rows);
    REQUIRE(a.cols == b.cols);
    Deviation d;
    double err_sq = 0.0;
    double ref_sq = 0.0;
    for (std::size_t i = 0; i < a.data.size(); ++i) {
        const double diff = static_cast<double>(a.data[i]) - b.data[i];
        err_sq += diff * diff;
        ref_sq += static_cast<double>(a.data[i]) * a.data[i];
        d.max_abs = std::max(d.max_abs, static_cast<float>(std::fabs(diff)));
    }
    d.rel_rms = ref_sq > 0.0 ? static_cast<float>(std::sqrt(err_sq / ref_sq)) : 0.0f;
    return d;
}

/// The GEMM contracts against a bfloat weight, which keeps eight mantissa
/// bits, so the two paths agree to a fraction of a percent and no further.
/// Measured at 0.5-0.7% RMS-relative across block counts -- and, importantly,
/// flat in the number of blocks: a structural disagreement would compound with
/// depth, and this does not.
constexpr float kBf16Tolerance = 0.02f;

}  // namespace

TEST_CASE("MetalFluxContext matches the CPU forward pass", "[metal][flux]") {
    const FluxConfig cfg = gpu_config();
    // Two copies: `create` consumes the weights of the model it uploads, so the
    // reference has to be a second build of the same deterministic weights.
    FluxModel gpu_model = make_model(cfg);
    const FluxModel cpu_model = make_model(cfg);

    const std::size_t lat_h = 8;
    const std::size_t lat_w = 6;
    const std::size_t n_txt = 5;
    const Mat z = spread(lat_h * lat_w, cfg.in_channels, 1.0f, 500);
    const Mat ctx = spread(n_txt, cfg.context_dim, 1.0f, 501);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 502);

    auto ctx_result = MetalFluxContext::create(gpu_model, lat_h, lat_w, n_txt);
    REQUIRE(ctx_result.has_value());
    const std::unique_ptr<MetalFluxContext>& engine = *ctx_result;

    const auto expected = cpu_model.forward(z, lat_h, lat_w, ctx, pooled, 0.75f);
    const auto got = engine->forward(z, lat_h, lat_w, ctx, pooled, 0.75f);
    REQUIRE(expected.has_value());
    REQUIRE(got.has_value());

    const Deviation d = compare(*expected, *got);
    INFO("max abs " << d.max_abs << ", rel rms " << d.rel_rms);
    REQUIRE(d.rel_rms < kBf16Tolerance);
}

TEST_CASE("MetalFluxContext is deterministic", "[metal][flux]") {
    const FluxConfig cfg = gpu_config();
    FluxModel model = make_model(cfg);
    const Mat z = spread(4 * 4, cfg.in_channels, 1.0f, 503);
    const Mat ctx = spread(3, cfg.context_dim, 1.0f, 504);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 505);

    auto engine = MetalFluxContext::create(model, 4, 4, 3);
    REQUIRE(engine.has_value());

    const auto a = (*engine)->forward(z, 4, 4, ctx, pooled, 0.5f);
    const auto b = (*engine)->forward(z, 4, 4, ctx, pooled, 0.5f);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    REQUIRE(a->data == b->data);
}

TEST_CASE("MetalFluxContext handles a latent smaller than its allocation", "[metal][flux]") {
    const FluxConfig cfg = gpu_config();
    FluxModel gpu_model = make_model(cfg);
    const FluxModel cpu_model = make_model(cfg);

    auto engine = MetalFluxContext::create(gpu_model, 12, 12, 8);
    REQUIRE(engine.has_value());

    // A token count that is not a tile multiple, so the pad rows are exercised.
    const std::size_t lat_h = 6;
    const std::size_t lat_w = 6;
    const std::size_t n_txt = 3;
    const Mat z = spread(lat_h * lat_w, cfg.in_channels, 1.0f, 506);
    const Mat ctx = spread(n_txt, cfg.context_dim, 1.0f, 507);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 508);
    REQUIRE((lat_h / 2) * (lat_w / 2) + n_txt == 12);  // not a multiple of 64

    const auto expected = cpu_model.forward(z, lat_h, lat_w, ctx, pooled, 0.3f);
    const auto got = (*engine)->forward(z, lat_h, lat_w, ctx, pooled, 0.3f);
    REQUIRE(expected.has_value());
    REQUIRE(got.has_value());

    const Deviation d = compare(*expected, *got);
    INFO("max abs " << d.max_abs << ", rel rms " << d.rel_rms);
    REQUIRE(d.rel_rms < kBf16Tolerance);
}

TEST_CASE("MetalFluxContext follows the guidance embedding when the config has one",
          "[metal][flux]") {
    FluxConfig cfg = gpu_config();
    cfg.guidance_embed = true;
    FluxModel gpu_model = make_model(cfg);
    const FluxModel cpu_model = make_model(cfg);

    const Mat z = spread(4 * 4, cfg.in_channels, 1.0f, 509);
    const Mat ctx = spread(3, cfg.context_dim, 1.0f, 510);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 511);

    auto engine = MetalFluxContext::create(gpu_model, 4, 4, 3);
    REQUIRE(engine.has_value());

    const auto expected = cpu_model.forward(z, 4, 4, ctx, pooled, 0.5f, 3.5f);
    const auto got = (*engine)->forward(z, 4, 4, ctx, pooled, 0.5f, 3.5f);
    REQUIRE(expected.has_value());
    REQUIRE(got.has_value());
    const Deviation d = compare(*expected, *got);
    INFO("max abs " << d.max_abs << ", rel rms " << d.rel_rms);
    REQUIRE(d.rel_rms < kBf16Tolerance);

    // And the guidance value actually reaches the graph.
    const auto other = (*engine)->forward(z, 4, 4, ctx, pooled, 0.5f, 1.0f);
    REQUIRE(other.has_value());
    REQUIRE_FALSE(other->data == got->data);
}

TEST_CASE("MetalFluxContext survives activations large enough to saturate tanh",
          "[metal][flux]") {
    // The GELU's cubic term grows fast: a pre-activation of 27 -- ordinary in a
    // real FLUX MLP -- puts the tanh argument near 724, and Metal's fast-math
    // tanh evaluates exp(2x), which is inf there. inf/inf is NaN, one NaN in a
    // residual stream poisons every later block, and the image comes out black.
    //
    // The synthetic weights the other cases use never reach that magnitude, so
    // this one drives the activations up on purpose.
    const FluxConfig cfg = gpu_config();
    FluxModel gpu_model = make_model(cfg);
    const FluxModel cpu_model = make_model(cfg);

    const std::size_t lat_h = 4;
    const std::size_t lat_w = 4;
    const std::size_t n_txt = 3;
    Mat z = spread(lat_h * lat_w, cfg.in_channels, 40.0f, 600);
    const Mat ctx = spread(n_txt, cfg.context_dim, 40.0f, 601);
    std::vector<float> pooled = spread_vec(cfg.pooled_dim, 40.0f, 602);

    auto engine = MetalFluxContext::create(gpu_model, lat_h, lat_w, n_txt);
    REQUIRE(engine.has_value());

    const auto expected = cpu_model.forward(z, lat_h, lat_w, ctx, pooled, 0.5f);
    const auto got = (*engine)->forward(z, lat_h, lat_w, ctx, pooled, 0.5f);
    REQUIRE(expected.has_value());
    REQUIRE(got.has_value());

    for (const float v : got->data) {
        REQUIRE(std::isfinite(v));
    }
    const Deviation d = compare(*expected, *got);
    INFO("max abs " << d.max_abs << ", rel rms " << d.rel_rms);
    REQUIRE(d.rel_rms < kBf16Tolerance);
}

TEST_CASE("MetalFluxContext rejects work it has no scratch for", "[metal][flux]") {
    const FluxConfig cfg = gpu_config();
    FluxModel model = make_model(cfg);
    auto engine = MetalFluxContext::create(model, 4, 4, 3);
    REQUIRE(engine.has_value());

    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 512);
    // Larger latent than the allocation.
    REQUIRE_FALSE((*engine)
                      ->forward(spread(8 * 8, cfg.in_channels, 1.0f, 1), 8, 8,
                                spread(3, cfg.context_dim, 1.0f, 1), pooled, 0.5f)
                      .has_value());
    // Longer prompt than the allocation.
    REQUIRE_FALSE((*engine)
                      ->forward(spread(4 * 4, cfg.in_channels, 1.0f, 1), 4, 4,
                                spread(9, cfg.context_dim, 1.0f, 1), pooled, 0.5f)
                      .has_value());
    // A latent that cannot be patched.
    REQUIRE_FALSE((*engine)
                      ->forward(spread(3 * 4, cfg.in_channels, 1.0f, 1), 3, 4,
                                spread(3, cfg.context_dim, 1.0f, 1), pooled, 0.5f)
                      .has_value());
}

TEST_CASE("MetalFluxContext refuses a config its GEMM cannot tile", "[metal][flux]") {
    FluxConfig cfg = gpu_config();
    cfg.hidden_size = 96;  // not a multiple of 64
    cfg.n_heads = 8;
    FluxModel model = make_model(cfg);
    REQUIRE_FALSE(MetalFluxContext::create(model, 4, 4, 3).has_value());
}

TEST_CASE("flux_sample_metal reproduces the CPU sampler's trajectory", "[metal][flux]") {
    const FluxConfig cfg = gpu_config();
    FluxModel gpu_model = make_model(cfg);
    const FluxModel cpu_model = make_model(cfg);

    const Mat ctx = spread(4, cfg.context_dim, 1.0f, 513);
    const std::vector<float> pooled = spread_vec(cfg.pooled_dim, 1.0f, 514);

    FluxSampleParams p;
    p.width = 64;
    p.height = 32;
    p.steps = 2;
    p.seed = 11;

    auto engine = MetalFluxContext::create(gpu_model, 32 / 8, 64 / 8, 4);
    REQUIRE(engine.has_value());

    const auto cpu = flux_sample(cpu_model, ctx, pooled, p);
    const auto gpu = flux_sample_metal(**engine, cfg, ctx, pooled, p);
    REQUIRE(cpu.has_value());
    REQUIRE(gpu.has_value());

    const Deviation d = compare(*cpu, *gpu);
    INFO("max abs " << d.max_abs << ", rel rms " << d.rel_rms);
    // The latent is mostly the initial noise after two steps, so the agreement
    // is tighter here than on a raw velocity.
    REQUIRE(d.rel_rms < kBf16Tolerance / 2.0f);
}

#endif  // RT_FEATURE_METAL

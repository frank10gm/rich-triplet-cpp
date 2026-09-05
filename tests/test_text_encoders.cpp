#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rt/clip_text.hpp"
#include "rt/mat.hpp"
#include "rt/qlinear.hpp"
#include "rt/t5.hpp"

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

[[nodiscard]] QLinear linear(std::size_t out, std::size_t in, std::size_t salt,
                             bool with_bias = false) {
    return QLinear::from_f32(spread(out, in, 0.2f, salt),
                             with_bias ? spread_vec(out, 0.05f, salt + 1) : std::vector<float>{});
}

[[nodiscard]] T5Config t5_tiny() {
    T5Config c;
    c.vocab_size = 40;
    c.d_model = 16;
    c.d_ff = 32;
    c.n_layers = 2;
    c.n_heads = 4;
    c.d_kv = 4;
    c.rel_attn_buckets = 32;
    c.rel_attn_max_distance = 128;
    return c;
}

[[nodiscard]] T5Encoder make_t5(const T5Config& cfg) {
    T5Encoder e;
    e.cfg = cfg;
    e.token_embedding = spread(cfg.vocab_size, cfg.d_model, 1.0f, 1);
    e.rel_bias_table = spread(cfg.rel_attn_buckets, cfg.n_heads, 0.5f, 2);
    const std::size_t inner = cfg.n_heads * cfg.d_kv;
    for (std::size_t i = 0; i < cfg.n_layers; ++i) {
        T5Block b;
        b.norm1 = std::vector<float>(cfg.d_model, 1.0f);
        b.norm2 = std::vector<float>(cfg.d_model, 1.0f);
        b.attn.q = linear(inner, cfg.d_model, 10 + i * 8);
        b.attn.k = linear(inner, cfg.d_model, 11 + i * 8);
        b.attn.v = linear(inner, cfg.d_model, 12 + i * 8);
        b.attn.o = linear(cfg.d_model, inner, 13 + i * 8);
        b.attn.n_heads = cfg.n_heads;
        b.attn.d_kv = cfg.d_kv;
        b.ff.wi_0 = linear(cfg.d_ff, cfg.d_model, 14 + i * 8);
        b.ff.wi_1 = linear(cfg.d_ff, cfg.d_model, 15 + i * 8);
        b.ff.wo = linear(cfg.d_model, cfg.d_ff, 16 + i * 8);
        e.blocks.push_back(std::move(b));
    }
    e.final_norm = std::vector<float>(cfg.d_model, 1.0f);
    return e;
}

[[nodiscard]] ClipTextConfig clip_tiny() {
    ClipTextConfig c;
    c.vocab_size = 40;
    c.hidden_size = 16;
    c.intermediate_size = 32;
    c.n_layers = 2;
    c.n_heads = 4;
    c.max_position_embeddings = 12;
    c.eot_token_id = 39;
    return c;
}

[[nodiscard]] ClipTextEncoder make_clip(const ClipTextConfig& cfg) {
    ClipTextEncoder m;
    m.cfg = cfg;
    m.token_embedding = spread(cfg.vocab_size, cfg.hidden_size, 1.0f, 3);
    m.position_embedding = spread(cfg.max_position_embeddings, cfg.hidden_size, 0.3f, 4);
    const std::size_t head_dim = cfg.hidden_size / cfg.n_heads;
    for (std::size_t i = 0; i < cfg.n_layers; ++i) {
        ClipLayer l;
        l.norm1_weight = std::vector<float>(cfg.hidden_size, 1.0f);
        l.norm1_bias = std::vector<float>(cfg.hidden_size, 0.0f);
        l.norm2_weight = std::vector<float>(cfg.hidden_size, 1.0f);
        l.norm2_bias = std::vector<float>(cfg.hidden_size, 0.0f);
        l.attn.q = linear(cfg.hidden_size, cfg.hidden_size, 20 + i * 8, true);
        l.attn.k = linear(cfg.hidden_size, cfg.hidden_size, 21 + i * 8, true);
        l.attn.v = linear(cfg.hidden_size, cfg.hidden_size, 22 + i * 8, true);
        l.attn.out = linear(cfg.hidden_size, cfg.hidden_size, 23 + i * 8, true);
        l.attn.n_heads = cfg.n_heads;
        l.attn.head_dim = head_dim;
        l.mlp.fc1 = linear(cfg.intermediate_size, cfg.hidden_size, 24 + i * 8, true);
        l.mlp.fc2 = linear(cfg.hidden_size, cfg.intermediate_size, 25 + i * 8, true);
        m.layers.push_back(std::move(l));
    }
    m.final_norm_weight = std::vector<float>(cfg.hidden_size, 1.0f);
    m.final_norm_bias = std::vector<float>(cfg.hidden_size, 0.0f);
    return m;
}

}  // namespace

// =============================================================================
// QLinear
// =============================================================================

TEST_CASE("QLinear agrees across its three weight formats", "[qlinear]") {
    const Mat w = spread(6, 8, 1.0f, 5);
    const std::vector<float> b = spread_vec(6, 0.5f, 6);
    const Mat x = spread(4, 8, 1.0f, 7);

    const QLinear f = QLinear::from_f32(w, b);
    const QLinear h = QLinear::from_bf16(mat_to_bf16(w), b);

    const Mat rf = f.forward(x);
    const Mat rh = h.forward(x);
    REQUIRE(rf.rows == 4);
    REQUIRE(rf.cols == 6);
    for (std::size_t i = 0; i < rf.data.size(); ++i) {
        // BF16 keeps 8 mantissa bits, so agreement is to about 1%.
        REQUIRE(approx(rf.data[i], rh.data[i], 0.05f));
    }
}

TEST_CASE("QLinear applies its bias once per row", "[qlinear]") {
    const Mat w = spread(3, 5, 1.0f, 8);
    const std::vector<float> b{1.0f, -2.0f, 0.5f};
    const Mat x = spread(4, 5, 1.0f, 9);
    const Mat plain = QLinear::from_f32(w).forward(x);
    const Mat biased = QLinear::from_f32(w, b).forward(x);
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            REQUIRE(approx(biased.at(r, c), plain.at(r, c) + b[c], 1e-5f));
        }
    }
}

TEST_CASE("QLinear reports whether a weight is present", "[qlinear]") {
    QLinear empty;
    REQUIRE_FALSE(empty.loaded());
    QLinear loaded = QLinear::from_f32(spread(2, 2, 1.0f, 1));
    REQUIRE(loaded.loaded());
    loaded.free_weight();
    REQUIRE_FALSE(loaded.loaded());
}

// =============================================================================
// T5 relative position bias
// =============================================================================

TEST_CASE("t5_relative_bucket splits the table by sign", "[t5]") {
    // Bidirectional: negative offsets take the low half, positive the high.
    REQUIRE(t5_relative_bucket(0, 32, 128) == 0);
    REQUIRE(t5_relative_bucket(-1, 32, 128) == 1);
    REQUIRE(t5_relative_bucket(-7, 32, 128) == 7);
    REQUIRE(t5_relative_bucket(1, 32, 128) == 17);
    REQUIRE(t5_relative_bucket(7, 32, 128) == 23);
}

TEST_CASE("t5_relative_bucket is exact below max_exact and logarithmic above", "[t5]") {
    // num_buckets 32 halves to 16, so the first 8 offsets are exact.
    for (long n = 0; n < 8; ++n) {
        REQUIRE(t5_relative_bucket(-n, 32, 128) == static_cast<std::size_t>(n));
    }
    // Beyond that, buckets grow slowly and never leave the half.
    std::size_t previous = 7;
    for (long n = 8; n < 400; ++n) {
        const std::size_t b = t5_relative_bucket(-n, 32, 128);
        REQUIRE(b >= previous);
        REQUIRE(b < 16);
        previous = b;
    }
    // Everything past max_distance saturates on the last bucket of the half.
    REQUIRE(t5_relative_bucket(-100000, 32, 128) == 15);
    REQUIRE(t5_relative_bucket(100000, 32, 128) == 31);
}

TEST_CASE("t5_position_bias is not symmetric", "[t5]") {
    // The bias distinguishes "before" from "after". A symmetric bias would make
    // the encoder blind to word order at range, which still produces fluent
    // embeddings.
    const Mat table = spread(32, 4, 1.0f, 30);
    const Mat bias = t5_position_bias(table, 6, 4, 32, 128);
    REQUIRE(bias.rows == 4 * 6);
    REQUIRE(bias.cols == 6);

    bool asymmetric = false;
    for (std::size_t q = 0; q < 6; ++q) {
        for (std::size_t k = 0; k < 6; ++k) {
            if (std::fabs(bias.at(0 * 6 + q, k) - bias.at(0 * 6 + k, q)) > 1e-6f) {
                asymmetric = true;
            }
        }
    }
    REQUIRE(asymmetric);
}

TEST_CASE("t5_position_bias uses key minus query, in that order", "[t5]") {
    // A table that is zero except in one bucket pins the sign convention: with
    // bucket 17 set (relative position +1), the bias must land one step to the
    // *right* of the diagonal.
    Mat table = Mat::zeros(32, 1);
    table.at_mut(17, 0) = 1.0f;
    const Mat bias = t5_position_bias(table, 4, 1, 32, 128);
    for (std::size_t q = 0; q < 4; ++q) {
        for (std::size_t k = 0; k < 4; ++k) {
            const float expected = (static_cast<long>(k) - static_cast<long>(q) == 1) ? 1.0f : 0.0f;
            REQUIRE(approx(bias.at(q, k), expected, 1e-6f));
        }
    }
}

TEST_CASE("t5_position_bias depends on the head", "[t5]") {
    const Mat table = spread(32, 4, 1.0f, 31);
    const Mat bias = t5_position_bias(table, 5, 4, 32, 128);
    bool differs = false;
    for (std::size_t q = 0; q < 5; ++q) {
        for (std::size_t k = 0; k < 5; ++k) {
            if (std::fabs(bias.at(0 * 5 + q, k) - bias.at(1 * 5 + q, k)) > 1e-6f) {
                differs = true;
            }
        }
    }
    REQUIRE(differs);
}

// =============================================================================
// T5 forward
// =============================================================================

TEST_CASE("t5_rms_norm normalizes without subtracting the mean", "[t5]") {
    // A constant row has zero variance but a nonzero RMS, so RMSNorm leaves it
    // at magnitude one rather than at zero. LayerNorm would zero it.
    Mat x = Mat::zeros(2, 8);
    for (std::size_t c = 0; c < 8; ++c) {
        x.at_mut(0, c) = 3.0f;
        x.at_mut(1, c) = -0.5f;
    }
    const std::vector<float> w(8, 1.0f);
    const Mat out = t5_rms_norm(x, w, 1e-6f);
    for (std::size_t c = 0; c < 8; ++c) {
        REQUIRE(approx(out.at(0, c), 1.0f, 1e-4f));
        REQUIRE(approx(out.at(1, c), -1.0f, 1e-4f));
    }
}

TEST_CASE("the T5 encoder produces one row per token", "[t5]") {
    const T5Config cfg = t5_tiny();
    const T5Encoder e = make_t5(cfg);
    const std::vector<std::uint32_t> tokens{3, 11, 7, 29, 1};
    const auto out = e.forward(tokens);
    REQUIRE(out.has_value());
    REQUIRE(out->rows == tokens.size());
    REQUIRE(out->cols == cfg.d_model);
    for (const float v : out->data) {
        REQUIRE(std::isfinite(v));
    }
}

TEST_CASE("the T5 encoder is bidirectional", "[t5]") {
    // Changing the last token must move the first token's embedding. A causal
    // encoder could not do that, and CLIP next door genuinely cannot.
    const T5Encoder e = make_t5(t5_tiny());
    std::vector<std::uint32_t> tokens{3, 11, 7, 29, 1};
    const auto base = e.forward(tokens);
    tokens.back() = 22;
    const auto poked = e.forward(tokens);
    REQUIRE(base.has_value());
    REQUIRE(poked.has_value());

    float delta = 0.0f;
    for (std::size_t c = 0; c < base->cols; ++c) {
        delta = std::max(delta, std::fabs(base->at(0, c) - poked->at(0, c)));
    }
    REQUIRE(delta > 1e-5f);
}

TEST_CASE("the T5 encoder depends on token order", "[t5]") {
    const T5Encoder e = make_t5(t5_tiny());
    const auto a = e.forward({3, 11, 7});
    const auto b = e.forward({7, 11, 3});
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    float delta = 0.0f;
    for (std::size_t i = 0; i < a->data.size(); ++i) {
        delta = std::max(delta, std::fabs(a->data[i] - b->data[i]));
    }
    REQUIRE(delta > 1e-4f);
}

TEST_CASE("T5 attention scores carry no 1/sqrt(d_kv)", "[t5]") {
    // Scale the q projection by sqrt(d_kv) and the output must be unchanged
    // *only if* the implementation divides by it. It does not, so the outputs
    // must differ -- which is the check that catches the scaling being added.
    const T5Config cfg = t5_tiny();
    T5Encoder plain = make_t5(cfg);
    T5Encoder scaled = make_t5(cfg);
    const float s = std::sqrt(static_cast<float>(cfg.d_kv));
    for (T5Block& b : scaled.blocks) {
        Mat w = b.attn.q.f32;
        for (float& v : w.data) {
            v *= s;
        }
        b.attn.q = QLinear::from_f32(std::move(w));
    }

    const auto a = plain.forward({3, 11, 7, 29});
    const auto c = scaled.forward({3, 11, 7, 29});
    REQUIRE(a.has_value());
    REQUIRE(c.has_value());
    float delta = 0.0f;
    for (std::size_t i = 0; i < a->data.size(); ++i) {
        delta = std::max(delta, std::fabs(a->data[i] - c->data[i]));
    }
    REQUIRE(delta > 1e-4f);
}

TEST_CASE("the T5 encoder rejects out-of-range tokens and empty input", "[t5]") {
    const T5Encoder e = make_t5(t5_tiny());
    REQUIRE_FALSE(e.forward({}).has_value());
    REQUIRE_FALSE(e.forward({0, 1, 999}).has_value());
}

TEST_CASE("the XXL config is the one FLUX uses", "[t5]") {
    const T5Config c = T5Config::xxl();
    REQUIRE(c.d_model == 4096);
    REQUIRE(c.d_ff == 10240);
    REQUIRE(c.n_layers == 24);
    REQUIRE(c.n_heads == 64);
    REQUIRE(c.d_kv == 64);
    REQUIRE(c.vocab_size == 32128);
    // 64 heads of 64 is 4096 -- the inner width equals d_model here, which is
    // not true of every T5 size and is worth pinning.
    REQUIRE(c.n_heads * c.d_kv == c.d_model);
}

// =============================================================================
// CLIP
// =============================================================================

TEST_CASE("the CLIP-L config matches the checkpoint FLUX ships with", "[clip]") {
    const ClipTextConfig c = ClipTextConfig::large();
    REQUIRE(c.hidden_size == 768);
    REQUIRE(c.intermediate_size == 3072);
    REQUIRE(c.n_layers == 12);
    REQUIRE(c.n_heads == 12);
    REQUIRE(c.max_position_embeddings == 77);
    REQUIRE(c.vocab_size == 49408);
    REQUIRE(c.eot_token_id == 49407);
    // EOT is the highest id, which is what makes argmax pooling work.
    REQUIRE(c.eot_token_id == c.vocab_size - 1);
}

TEST_CASE("the CLIP encoder is causal", "[clip]") {
    // Changing the last token must leave the first token's row untouched.
    const ClipTextEncoder m = make_clip(clip_tiny());
    std::vector<std::uint32_t> tokens{2, 5, 9, 14};
    const auto base = m.forward(tokens);
    tokens.back() = 21;
    const auto poked = m.forward(tokens);
    REQUIRE(base.has_value());
    REQUIRE(poked.has_value());
    for (std::size_t c = 0; c < base->sequence.cols; ++c) {
        REQUIRE(approx(base->sequence.at(0, c), poked->sequence.at(0, c), 1e-5f));
    }
}

TEST_CASE("CLIP pooling reads the argmax token, not the last position", "[clip]") {
    const ClipTextConfig cfg = clip_tiny();
    const ClipTextEncoder m = make_clip(cfg);
    // EOT (39) in the middle, padding after it.
    const std::vector<std::uint32_t> tokens{2, 5, 39, 9, 9};
    const auto out = m.forward(tokens);
    REQUIRE(out.has_value());
    REQUIRE(out->eot_index == 2);
    for (std::size_t c = 0; c < cfg.hidden_size; ++c) {
        REQUIRE(approx(out->pooled[c], out->sequence.at(2, c), 1e-7f));
    }
    // The last position holds something else entirely.
    float delta = 0.0f;
    for (std::size_t c = 0; c < cfg.hidden_size; ++c) {
        delta = std::max(delta, std::fabs(out->pooled[c] - out->sequence.at(4, c)));
    }
    REQUIRE(delta > 1e-4f);
}

TEST_CASE("the CLIP encoder uses its positional table", "[clip]") {
    // The same token at two positions must embed differently.
    const ClipTextEncoder m = make_clip(clip_tiny());
    const auto a = m.forward({7, 7, 7});
    REQUIRE(a.has_value());
    float delta = 0.0f;
    for (std::size_t c = 0; c < a->sequence.cols; ++c) {
        delta = std::max(delta, std::fabs(a->sequence.at(0, c) - a->sequence.at(1, c)));
    }
    REQUIRE(delta > 1e-4f);
}

TEST_CASE("the CLIP encoder refuses a sequence longer than its positional table", "[clip]") {
    const ClipTextEncoder m = make_clip(clip_tiny());
    std::vector<std::uint32_t> long_seq(13, 5);
    REQUIRE_FALSE(m.forward(long_seq).has_value());
    REQUIRE_FALSE(m.forward({}).has_value());
    REQUIRE_FALSE(m.forward({0, 999}).has_value());
}

TEST_CASE("clip_layer_norm subtracts the mean", "[clip]") {
    // Unlike T5's RMSNorm next door, a constant row normalizes to zero.
    Mat x = Mat::zeros(1, 8);
    for (std::size_t c = 0; c < 8; ++c) {
        x.at_mut(0, c) = 3.0f;
    }
    const std::vector<float> w(8, 1.0f);
    const std::vector<float> b(8, 0.0f);
    const Mat out = clip_layer_norm(x, w, b, 1e-5f);
    for (std::size_t c = 0; c < 8; ++c) {
        REQUIRE(approx(out.at(0, c), 0.0f, 1e-4f));
    }
}

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

#include "gguf_writer.hpp"
#include "rt/transformer4.hpp"

using namespace rt;
using rt::testing::GgufWriter;

namespace {

/// A tiny Gemma 3 config: small enough to run in a unit test, but with the
/// 5-local:1-global layer pattern and a sliding window intact.
Config4 tiny_cfg() {
    Config4 c;
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.num_hidden_layers = 4;
    c.num_attention_heads = 2;
    c.num_key_value_heads = 1;
    c.intermediate_size = 64;
    c.head_dim = 16;
    c.sliding_window = 8;
    c.rope_theta_local = 10000.0f;
    c.rope_theta_global = 1000000.0f;
    c.rope_freq_scale_local = 1.0f;
    c.rope_freq_scale_global = 1.0f;
    c.rms_norm_eps = 1e-6f;
    c.query_pre_attn_scalar = 16.0f;  // matches head_dim at this scale
    c.eos_token_id = 1;
    c.max_position_embeddings = 128;
    return c;
}

void require_all_finite(const Mat& m, const char* what) {
    for (std::size_t r = 0; r < m.rows; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            INFO(what << "[" << r << "," << c << "] = " << m.at(r, c));
            REQUIRE(std::isfinite(m.at(r, c)));
        }
    }
}

/// Greedy argmax over row `row`.
std::size_t argmax_row(const Mat& m, std::size_t row) {
    std::size_t best = 0;
    for (std::size_t c = 1; c < m.cols; ++c) {
        if (m.at(row, best) < m.at(row, c)) {
            best = c;
        }
    }
    return best;
}

/// `n` values stepping by `step`, small enough to keep a forward pass tame.
std::vector<float> ramp(std::size_t n, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>(i % 17) * step - 8.0f * step;
    }
    return v;
}

std::vector<float> ones(std::size_t n) { return std::vector<float>(n, 1.0f); }

/// f32 values as little-endian bytes, for the GGUF writer.
std::vector<std::uint8_t> f32_bytes(const std::vector<float>& values) {
    std::vector<std::uint8_t> bytes(values.size() * 4);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

}  // namespace

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

TEST_CASE("config4 global layer pattern", "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    // 5 local layers, then 1 global, repeating.
    for (std::size_t i = 0; i < 24; ++i) {
        INFO("layer " << i);
        REQUIRE(cfg.is_global_layer(i) == (i % 6 == 5));
    }
}

TEST_CASE("gemma3 1b and 4b config values", "[transformer4]") {
    const Config4 c1 = Config4::gemma3_1b();
    REQUIRE(c1.vocab_size == 262144);
    REQUIRE(c1.hidden_size == 1152);
    REQUIRE(c1.num_hidden_layers == 26);
    REQUIRE(c1.num_attention_heads == 4);
    REQUIRE(c1.num_key_value_heads == 1);
    REQUIRE(c1.head_dim == 256);
    REQUIRE(c1.sliding_window == 512u);
    REQUIRE(c1.rope_freq_scale_global == 1.0f);

    const Config4 c4 = Config4::gemma3_4b();
    REQUIRE(c4.vocab_size == 262208);
    REQUIRE(c4.hidden_size == 2560);
    REQUIRE(c4.num_hidden_layers == 34);
    REQUIRE(c4.num_attention_heads == 8);
    REQUIRE(c4.num_key_value_heads == 4);
    REQUIRE(c4.head_dim == 256);
    REQUIRE(c4.sliding_window == 1024u);
    // rope_scaling.factor = 8 becomes a frequency scale of 1/8.
    REQUIRE(c4.rope_freq_scale_global == 0.125f);
}

// -----------------------------------------------------------------------------
// Layers
// -----------------------------------------------------------------------------

TEST_CASE("attention distinguishes local from global layers", "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(5);
    const Gemma3Attention local(cfg, 0, rng);
    const Gemma3Attention global(cfg, 5, rng);

    REQUIRE(local.sliding_window.has_value());
    REQUIRE_FALSE(global.sliding_window.has_value());
    REQUIRE(local.rope_theta == cfg.rope_theta_local);
    REQUIRE(global.rope_theta == cfg.rope_theta_global);
    // Both share the query_pre_attn_scalar-derived scale, not 1/sqrt(head_dim).
    REQUIRE(local.attn_scale == 1.0f / std::sqrt(cfg.query_pre_attn_scalar));
}

TEST_CASE("attention forward preserves shape", "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(99);
    const Gemma3Attention attn(cfg, 0, rng);
    const std::size_t t = 6;
    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(t, cfg.hidden_size, [&](std::size_t r, std::size_t c) {
            return static_cast<float>(r * cfg.hidden_size + c) * 0.01f;
        }));
    const Mat out = attn.forward(x).data();
    REQUIRE(out.rows == t);
    REQUIRE(out.cols == cfg.hidden_size);
    require_all_finite(out, "attn");
}

TEST_CASE("mlp forward shape", "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(17);
    const Gemma3Mlp mlp(cfg, rng);
    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(4, cfg.hidden_size, [&](std::size_t r, std::size_t c) {
            return static_cast<float>(r * cfg.hidden_size + c) * 0.01f - 0.5f;
        }));
    const Mat out = mlp.forward(x).data();
    REQUIRE(out.rows == 4);
    REQUIRE(out.cols == cfg.hidden_size);
}

TEST_CASE("block output shape", "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(22);
    const Gemma3Block block(cfg, 0, rng);
    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(5, cfg.hidden_size, [](std::size_t r, std::size_t c) {
            return static_cast<float>(r + c) * 0.01f;
        }));
    const Mat out = block.forward(x).data();
    REQUIRE(out.rows == 5);
    REQUIRE(out.cols == cfg.hidden_size);
}

TEST_CASE("per-head norm and rope preserve shape", "[transformer4]") {
    const std::size_t t = 4, n_heads = 3, head_dim = 8;
    const RmsNorm2 norm(head_dim, 1e-6f);
    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(t, n_heads * head_dim, [&](std::size_t r, std::size_t c) {
            return static_cast<float>(r * n_heads * head_dim + c) * 0.05f - 0.4f;
        }));

    const Mat normed = apply_per_head_norm(x, norm, t, n_heads, head_dim).data();
    REQUIRE(normed.rows == t);
    REQUIRE(normed.cols == n_heads * head_dim);

    const Mat roped =
        apply_rope_to_all_heads(x, n_heads, t, head_dim, 10000.0f, 1.0f).data();
    REQUIRE(roped.rows == t);
    REQUIRE(roped.cols == n_heads * head_dim);
    // RoPE is an orthogonal rotation, so each head's norm is unchanged.
    for (std::size_t h = 0; h < n_heads; ++h) {
        for (std::size_t r = 0; r < t; ++r) {
            float before = 0.0f, after = 0.0f;
            for (std::size_t i = 0; i < head_dim; ++i) {
                const float b = x.data().at(r, h * head_dim + i);
                const float a = roped.at(r, h * head_dim + i);
                before += b * b;
                after += a * a;
            }
            INFO("head " << h << " row " << r);
            REQUIRE(std::fabs(std::sqrt(before) - std::sqrt(after)) < 1e-4f);
        }
    }
}

TEST_CASE("rope at offset zero matches rope from position zero", "[transformer4]") {
    const std::size_t t = 3, n_heads = 2, head_dim = 8;
    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(t, n_heads * head_dim, [&](std::size_t r, std::size_t c) {
            return std::sin(static_cast<float>(r * 16 + c) * 0.1f);
        }));
    const Mat a = apply_rope_to_all_heads(x, n_heads, t, head_dim, 10000.0f, 1.0f).data();
    const Mat b = apply_rope_at_offset(x, n_heads, t, head_dim, 10000.0f, 0, 1.0f).data();
    REQUIRE(a.data == b.data);
}

// -----------------------------------------------------------------------------
// Model
// -----------------------------------------------------------------------------

TEST_CASE("model forward shape and finiteness", "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(42);
    const Gemma3Model model(cfg, rng);
    const Mat logits = model.forward({0, 5, 10, 2}).data();
    REQUIRE(logits.rows == 4);
    REQUIRE(logits.cols == cfg.vocab_size);
    require_all_finite(logits, "logits");
}

TEST_CASE("model parameter list covers every layer", "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(1);
    const Gemma3Model model(cfg, rng);
    // embed + final norm + the weight-tied lm_head, plus 13 tensors per block
    // (4 norms, 6 attention, 3 MLP).
    REQUIRE(model.parameters().size() == 3 + cfg.num_hidden_layers * 13);
    // lm_head is tied to the embedding, so the same node appears at both ends.
    REQUIRE(model.parameters().front().id() == model.parameters().back().id());
}

TEST_CASE("loss is finite and positive, and backward reaches the parameters",
          "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(3);
    const Gemma3Model model(cfg, rng);
    const std::vector<std::size_t> inputs{0, 1, 2, 3};
    const std::vector<std::size_t> targets{1, 2, 3, 4};

    const TensorNode loss = model.loss_tokens(inputs, targets);
    const float value = loss.data().at(0, 0);
    INFO("loss = " << value);
    REQUIRE(std::isfinite(value));
    REQUIRE(value > 0.0f);
    // A ~uniform distribution over 64 tokens gives about ln(64) = 4.16.
    REQUIRE(value < 10.0f);

    loss.backward();
    bool any_nonzero = false;
    for (const TensorNode& p : model.parameters()) {
        for (float g : p.grad().data) {
            REQUIRE(std::isfinite(g));
            any_nonzero = any_nonzero || std::fabs(g) > 1e-9f;
        }
    }
    REQUIRE(any_nonzero);
}

// -----------------------------------------------------------------------------
// KV cache
// -----------------------------------------------------------------------------

TEST_CASE("layer kv cache appends and reports length", "[transformer4]") {
    Gemma3LayerKvCache cache(2, 4, 16);
    REQUIRE(cache.seq_len == 0);

    const Mat k = Mat::from_fn(3, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c);
    });
    const Mat v = k.scale(-1.0f);
    cache.append(k, v);
    REQUIRE(cache.seq_len == 3);
    REQUIRE(cache.k_filled().rows == 3);
    REQUIRE(cache.k_filled().at(2, 7) == k.at(2, 7));
    REQUIRE(cache.v_filled().at(1, 3) == v.at(1, 3));

    cache.append(k, v);
    REQUIRE(cache.seq_len == 6);
    REQUIRE(cache.k.at(3, 0) == k.at(0, 0));
}

TEST_CASE("layer kv cache windowed views", "[transformer4]") {
    Gemma3LayerKvCache cache(1, 4, 16);
    const Mat k = Mat::from_fn(6, 4, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c);
    });
    cache.append(k, k);

    // A window larger than the cache returns everything.
    REQUIRE(cache.k_last(10).rows == 6);
    // Otherwise it returns the trailing rows.
    const Mat last3 = cache.k_last(3);
    REQUIRE(last3.rows == 3);
    REQUIRE(last3.at(0, 0) == k.at(3, 0));
    REQUIRE(cache.v_last(2).at(0, 1) == k.at(4, 1));
}

TEST_CASE("kv cache construction, clear and free", "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    Gemma3KvCache cache(cfg, 32);
    REQUIRE(cache.layers.size() == cfg.num_hidden_layers);
    REQUIRE(cache.layers[0].k.rows == 32);
    REQUIRE(cache.layers[0].k.cols == cfg.num_key_value_heads * cfg.head_dim);

    cache.layers[0].seq_len = 5;
    cache.clear();
    REQUIRE(cache.layers[0].seq_len == 0);

    cache.free();
    REQUIRE(cache.layers[0].k.numel() == 0);

    // The cache is capped by max_position_embeddings.
    const Gemma3KvCache capped(cfg, 1'000'000);
    REQUIRE(capped.layers[0].k.rows == cfg.max_position_embeddings);
}

TEST_CASE("cached attention grows the cache and keeps its shape", "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(8);
    const Gemma3Attention attn(cfg, 0, rng);
    Gemma3LayerKvCache cache(cfg.num_key_value_heads, cfg.head_dim, 32);

    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(1, cfg.hidden_size, [&](std::size_t, std::size_t c) {
            return static_cast<float>(c) * 0.02f - 0.3f;
        }));

    for (std::size_t step = 0; step < 3; ++step) {
        const Mat out = attn.forward_cached(x, cache).data();
        REQUIRE(out.rows == 1);
        REQUIRE(out.cols == cfg.hidden_size);
        REQUIRE(cache.seq_len == step + 1);
        require_all_finite(out, "cached attn");
    }
}

TEST_CASE("cached forward matches the uncached forward", "[transformer4]") {
    // Prefilling the whole prompt through the cache must reproduce the last
    // row of the plain forward pass.
    const Config4 cfg = tiny_cfg();
    InitRng rng(42);
    const Gemma3Model model(cfg, rng);
    const std::vector<std::size_t> prompt{0, 1, 2, 3};

    const Mat full = model.forward(prompt).data();
    Gemma3KvCache cache(cfg, prompt.size() + 4);
    const Mat cached = model.forward_cached(prompt, cache);

    REQUIRE(cached.rows == 1);
    REQUIRE(cached.cols == cfg.vocab_size);
    for (std::size_t c = 0; c < cfg.vocab_size; ++c) {
        INFO("logit " << c);
        REQUIRE(std::fabs(cached.at(0, c) - full.at(full.rows - 1, c)) < 1e-3f);
    }
    // In particular the greedy pick must agree.
    REQUIRE(argmax_row(cached, 0) == argmax_row(full, full.rows - 1));
}

TEST_CASE("cached decode stays finite past the sliding window", "[transformer4]") {
    // A prompt longer than the window exercises the windowed branch of the
    // cached attention on local layers.
    const Config4 cfg = tiny_cfg();  // window = 8
    InitRng rng(11);
    const Gemma3Model model(cfg, rng);

    std::vector<std::size_t> prompt(20);
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = i % cfg.vocab_size;
    }

    Gemma3KvCache cache(cfg, prompt.size() + 8);
    Mat logits = model.forward_cached(prompt, cache);
    require_all_finite(logits, "prefill logits");

    // Then decode a few tokens one at a time.
    for (std::size_t step = 0; step < 4; ++step) {
        const std::size_t next = argmax_row(logits, 0);
        REQUIRE(next < cfg.vocab_size);
        logits = model.forward_cached({next}, cache);
        require_all_finite(logits, "decode logits");
    }
    REQUIRE(cache.layers[0].seq_len == prompt.size() + 4);
}

TEST_CASE("inference model allocates no weights", "[transformer4]") {
    const Gemma3Model model = Gemma3Model::new_for_inference(Config4::gemma3_4b());
    REQUIRE(model.embed_tokens.data().numel() == 0);
    REQUIRE(model.layers.size() == 34);
    REQUIRE(model.layers[0].self_attn.q_proj.weight.data().numel() == 0);
    REQUIRE(model.layers[0].mlp.gate_proj.weight.data().numel() == 0);
    // The layer pattern still holds.
    REQUIRE_FALSE(model.layers[5].self_attn.sliding_window.has_value());
    REQUIRE(model.layers[4].self_attn.sliding_window.has_value());
}

// -----------------------------------------------------------------------------
// Sampling
// -----------------------------------------------------------------------------

TEST_CASE("sample_token is greedy at temperature zero", "[transformer4]") {
    const Mat logits({0.1f, 5.0f, 0.3f, 2.0f}, 1, 4);
    LcgRng rng(0);
    REQUIRE(sample_token(logits, 0, SamplingParams::greedy(), {}, rng) == 1);
}

TEST_CASE("sample_token applies the repetition penalty", "[transformer4]") {
    // Token 1 wins outright, but penalizing it hands the pick to token 3.
    const Mat logits({0.1f, 5.0f, 0.3f, 4.0f}, 1, 4);
    SamplingParams params = SamplingParams::greedy();
    params.repetition_penalty = 2.0f;

    LcgRng rng(0);
    REQUIRE(sample_token(logits, 0, params, {1}, rng) == 3);

    // Only the last 64 generated tokens count, so an older occurrence of
    // token 1 does not penalize it.
    std::vector<std::size_t> long_history(70, 2);
    long_history[0] = 1;
    LcgRng rng2(0);
    REQUIRE(sample_token(logits, 0, params, long_history, rng2) == 1);
}

TEST_CASE("sample_token honours top_k", "[transformer4]") {
    // With top_k = 1 only the argmax survives, so any draw returns it.
    const Mat logits({1.0f, 3.0f, 2.0f, 0.5f}, 1, 4);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 1;
    for (std::uint64_t seed = 0; seed < 5; ++seed) {
        LcgRng rng(seed);
        REQUIRE(sample_token(logits, 0, params, {}, rng) == 1);
    }
}

TEST_CASE("sample_token honours top_p", "[transformer4]") {
    // One token holds nearly all the mass, so a small top_p keeps only it.
    const Mat logits({20.0f, 0.0f, 0.0f, 0.0f}, 1, 4);
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_p = 0.5f;
    for (std::uint64_t seed = 0; seed < 5; ++seed) {
        LcgRng rng(seed);
        REQUIRE(sample_token(logits, 0, params, {}, rng) == 0);
    }
}

TEST_CASE("sample_token is reproducible for a given seed", "[transformer4]") {
    const Mat logits = Mat::from_fn(1, 32, [](std::size_t, std::size_t c) {
        return std::sin(static_cast<float>(c) * 0.7f) * 2.0f;
    });
    const SamplingParams params = SamplingParams::creative(1234);

    LcgRng a(params.seed), b(params.seed);
    for (int i = 0; i < 8; ++i) {
        REQUIRE(sample_token(logits, 0, params, {}, a) ==
                sample_token(logits, 0, params, {}, b));
    }
}

TEST_CASE("bf16 conversion helper drops its input", "[transformer4]") {
    const std::vector<float> values{1.0f, -2.5f, 0.125f, 100.0f};
    const std::vector<std::uint16_t> bits = f32s_to_bf16_and_drop(values);
    REQUIRE(bits.size() == values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        INFO("value " << i);
        REQUIRE(std::fabs(bf16_to_f32(bits[i]) - values[i]) < std::fabs(values[i]) * 0.01f + 1e-6f);
    }
}

// -----------------------------------------------------------------------------
// Generation
// -----------------------------------------------------------------------------

TEST_CASE("generate_streaming produces tokens", "[transformer4]") {
    InitRng rng(42);
    const Gemma3Model model(tiny_cfg(), rng);
    std::vector<std::size_t> generated;
    model.generate_streaming({0, 1, 2}, 5, 1.0f, 0, 42,
                             [&](std::size_t tok) { generated.push_back(tok); });
    REQUIRE_FALSE(generated.empty());
    REQUIRE(generated.size() <= 5);
}

TEST_CASE("generate_cached_streaming produces tokens", "[transformer4]") {
    InitRng rng(7);
    Gemma3Model model(tiny_cfg(), rng);
    std::vector<std::size_t> generated;
    model.generate_cached_streaming({0, 1, 2}, 5, 1.0f, 0, 1.0f, 1.0f, 42, false, 0,
                                    [&](std::size_t tok) { generated.push_back(tok); });
    REQUIRE_FALSE(generated.empty());
    REQUIRE(generated.size() <= 5);
}

// -----------------------------------------------------------------------------
// NgramDraftEngine
// -----------------------------------------------------------------------------

TEST_CASE("ngram draft engine finds a repeated continuation", "[transformer4]") {
    NgramDraftEngine ng(4, 4);
    REQUIRE(ng.draft().empty());

    // History: 10 20 30 40 10 20 30 40 50 60.
    ng.record_many({10, 20, 30, 40, 10, 20, 30, 40, 50, 60});
    // The suffixes ending at 60 -- [30,40,50,60], [40,50,60], [50,60] -- appear
    // only once each, so there is nothing to draft from.
    REQUIRE(ng.draft().empty());

    // [60, 10] is likewise unique.
    ng.record(10);
    REQUIRE(ng.draft().empty());

    // Now [10, 20] matches at index 0, and what followed it was 30 40 10 20.
    ng.record(20);
    REQUIRE(ng.draft() == std::vector<std::size_t>{30, 40, 10, 20});
}

// -----------------------------------------------------------------------------
// Binary weight cache
// -----------------------------------------------------------------------------

TEST_CASE("save_cache and load_cache round-trip the weights", "[transformer4]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(11);
    const Gemma3Model source(cfg, rng);

    // The cache only carries BF16 projections and f32 norms, so the source has
    // to be in that form for the round trip to cover the projections.
    Gemma3Model saved = Gemma3Model::new_for_inference(cfg);
    saved.embed_bf16 = source.embed_tokens.data().to_bf16();
    saved.norm.gamma.set_data(source.norm.gamma.data());
    const auto copy_linear = [](Linear2& d, const Linear2& s) {
        d.load_bf16(*s.weight.data().to_bf16().data, s.out_features, s.in_features);
    };
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const Gemma3Block& s = source.layers[i];
        Gemma3Block& d = saved.layers[i];
        d.input_layernorm.gamma.set_data(s.input_layernorm.gamma.data());
        d.post_attention_layernorm.gamma.set_data(s.post_attention_layernorm.gamma.data());
        d.pre_feedforward_layernorm.gamma.set_data(s.pre_feedforward_layernorm.gamma.data());
        d.post_feedforward_layernorm.gamma.set_data(s.post_feedforward_layernorm.gamma.data());
        d.self_attn.q_norm.gamma.set_data(s.self_attn.q_norm.gamma.data());
        d.self_attn.k_norm.gamma.set_data(s.self_attn.k_norm.gamma.data());
        copy_linear(d.self_attn.q_proj, s.self_attn.q_proj);
        copy_linear(d.self_attn.k_proj, s.self_attn.k_proj);
        copy_linear(d.self_attn.v_proj, s.self_attn.v_proj);
        copy_linear(d.self_attn.o_proj, s.self_attn.o_proj);
        copy_linear(d.mlp.gate_proj, s.mlp.gate_proj);
        copy_linear(d.mlp.up_proj, s.mlp.up_proj);
        copy_linear(d.mlp.down_proj, s.mlp.down_proj);
    }
    // The head is weight-tied, exactly as the loaders leave it.
    saved.lm_head.load_bf16_shared(saved.embed_bf16->data, cfg.vocab_size, cfg.hidden_size);

    const std::string path = "/tmp/rt_g3cache_test.bin";
    REQUIRE(saved.save_cache(path).has_value());

    Gemma3Model loaded = Gemma3Model::new_for_inference(cfg);
    const Result<bool> ok = loaded.load_cache(path);
    REQUIRE(ok.has_value());
    REQUIRE(*ok);

    // The embedding comes back as the same BF16 bits, tied to lm_head.
    REQUIRE(loaded.embed_bf16.has_value());
    REQUIRE(loaded.embed_bf16->rows == cfg.vocab_size);
    REQUIRE(loaded.embed_bf16->cols == cfg.hidden_size);
    REQUIRE(*loaded.embed_bf16->data == *saved.embed_bf16->data);
    REQUIRE(loaded.lm_head.bf16_weight.has_value());

    const Linear2& want = saved.layers[2].mlp.down_proj;
    const Linear2& got = loaded.layers[2].mlp.down_proj;
    REQUIRE(got.bf16_weight.has_value());
    REQUIRE(*got.bf16_weight->data == *want.bf16_weight->data);
    REQUIRE(loaded.layers[3].self_attn.q_norm.gamma.data().data ==
            saved.layers[3].self_attn.q_norm.gamma.data().data);

    // Both models must now agree token for token.
    Gemma3KvCache cache_a(cfg, 16);
    Gemma3KvCache cache_b(cfg, 16);
    const std::vector<std::size_t> prompt{3, 9, 17};
    REQUIRE(argmax_row(saved.forward_cached(prompt, cache_a), 0) ==
            argmax_row(loaded.forward_cached(prompt, cache_b), 0));

    std::remove(path.c_str());
}

TEST_CASE("load_cache reports a bad magic instead of failing", "[transformer4]") {
    const std::string path = "/tmp/rt_g3cache_bad.bin";
    {
        std::ofstream out(path, std::ios::binary);
        out << "NOTACACHE";
    }
    Gemma3Model model = Gemma3Model::new_for_inference(tiny_cfg());
    const Result<bool> ok = model.load_cache(path);
    REQUIRE(ok.has_value());
    REQUIRE_FALSE(*ok);
    std::remove(path.c_str());

    // A missing file is likewise a "no cache", not an error.
    const Result<bool> missing = model.load_cache("/tmp/rt_g3cache_does_not_exist.bin");
    REQUIRE(missing.has_value());
    REQUIRE_FALSE(*missing);
}

// -----------------------------------------------------------------------------
// Quantization
// -----------------------------------------------------------------------------

TEST_CASE("quantize_for_inference keeps the f32 weights", "[transformer4]") {
    InitRng rng(5);
    Gemma3Model model(tiny_cfg(), rng);
    model.quantize_for_inference();
    for (const Gemma3Block& layer : model.layers) {
        REQUIRE(layer.self_attn.q_proj.q4_weight.has_value());
        REQUIRE(layer.mlp.down_proj.q4_weight.has_value());
        // Gradients still need the f32 copy, so it stays.
        REQUIRE(layer.self_attn.q_proj.weight.data().numel() > 0);
    }
    REQUIRE(model.lm_head.q4_weight.has_value());
}

TEST_CASE("quantize_inference_free_f32 drops the f32 weights", "[transformer4]") {
    InitRng rng(5);
    Gemma3Model model(tiny_cfg(), rng);
    model.quantize_inference_free_f32();
    for (const Gemma3Block& layer : model.layers) {
        REQUIRE(layer.self_attn.q_proj.q4_weight.has_value());
        REQUIRE(layer.self_attn.q_proj.weight.data().numel() == 0);
    }
    REQUIRE(model.lm_head.q4_weight.has_value());
    // lm_head is weight-tied, so freeing its f32 copy also empties
    // `embed_tokens` -- a model quantized this way can no longer look up
    // embeddings, and the GGUF paths set `embed_bf16` instead.
    REQUIRE(model.embed_tokens.data().numel() == 0);
}

// -----------------------------------------------------------------------------
// GGUF weight loading
// -----------------------------------------------------------------------------

TEST_CASE("load_weights_from_gguf maps blk.* names onto the model", "[transformer4]") {
    Config4 cfg = tiny_cfg();
    cfg.num_hidden_layers = 1;
    // Q4_K needs 256-element blocks, but this file is written as F32, so the
    // small dimensions are fine.
    const std::size_t h = cfg.hidden_size;
    const std::size_t inter = cfg.intermediate_size;
    const std::size_t d = cfg.head_dim;
    const std::size_t nq = cfg.num_attention_heads;
    const std::size_t nkv = cfg.num_key_value_heads;

    GgufWriter w;
    w.add_meta_str("general.architecture", "gemma3");
    // GGUF shapes are [in, out] -- the transpose of our [out, in].
    w.add_tensor("token_embd.weight", {h, cfg.vocab_size}, GgufType::F32, f32_bytes(ramp(cfg.vocab_size * h, 0.01f)));
    w.add_tensor("output_norm.weight", {h}, GgufType::F32, f32_bytes(ones(h)));
    w.add_tensor("blk.0.attn_norm.weight", {h}, GgufType::F32, f32_bytes(ones(h)));
    w.add_tensor("blk.0.post_attn_norm.weight", {h}, GgufType::F32, f32_bytes(ones(h)));
    w.add_tensor("blk.0.ffn_pre_norm.weight", {h}, GgufType::F32, f32_bytes(ones(h)));
    w.add_tensor("blk.0.ffn_post_norm.weight", {h}, GgufType::F32, f32_bytes(ones(h)));
    w.add_tensor("blk.0.attn_q_norm.weight", {d}, GgufType::F32, f32_bytes(ones(d)));
    w.add_tensor("blk.0.attn_k_norm.weight", {d}, GgufType::F32, f32_bytes(ones(d)));
    w.add_tensor("blk.0.attn_q.weight", {h, nq * d}, GgufType::F32, f32_bytes(ramp(nq * d * h, 0.002f)));
    w.add_tensor("blk.0.attn_k.weight", {h, nkv * d}, GgufType::F32, f32_bytes(ramp(nkv * d * h, 0.003f)));
    w.add_tensor("blk.0.attn_v.weight", {h, nkv * d}, GgufType::F32, f32_bytes(ramp(nkv * d * h, 0.004f)));
    w.add_tensor("blk.0.attn_output.weight", {nq * d, h}, GgufType::F32, f32_bytes(ramp(nq * d * h, 0.005f)));
    w.add_tensor("blk.0.ffn_gate.weight", {h, inter}, GgufType::F32, f32_bytes(ramp(inter * h, 0.006f)));
    w.add_tensor("blk.0.ffn_up.weight", {h, inter}, GgufType::F32, f32_bytes(ramp(inter * h, 0.007f)));
    w.add_tensor("blk.0.ffn_down.weight", {inter, h}, GgufType::F32, f32_bytes(ramp(inter * h, 0.008f)));
    // The vision tower is skipped without complaint.
    w.add_tensor("v.blk.0.attn_q.weight", {4, 4}, GgufType::F32, f32_bytes(ramp(16, 0.1f)));

    const std::string path = "/tmp/rt_gemma3_load.gguf";
    w.write(path);

    Gemma3Model model = Gemma3Model::new_for_inference(cfg);
    REQUIRE(model.load_weights_from_gguf(path).has_value());

    // The embedding loaded as BF16 and lm_head is tied to it.
    REQUIRE(model.embed_bf16.has_value());
    REQUIRE(model.embed_bf16->rows == cfg.vocab_size);
    REQUIRE(model.embed_bf16->cols == h);
    REQUIRE(model.lm_head.bf16_weight.has_value());
    REQUIRE(model.lm_head.bf16_weight->data == model.embed_bf16->data);

    // GGUF stores norm gammas as 1 + weight, so a file of ones becomes zeros.
    const Mat& gamma = model.layers[0].input_layernorm.gamma.data();
    REQUIRE(gamma.cols == h);
    for (std::size_t c = 0; c < h; ++c) {
        REQUIRE(std::fabs(gamma.at(0, c)) < 1e-6f);
    }

    // The projections transposed into [out, in].
    REQUIRE(model.layers[0].self_attn.q_proj.weight.data().rows == nq * d);
    REQUIRE(model.layers[0].self_attn.q_proj.weight.data().cols == h);
    REQUIRE(model.layers[0].mlp.down_proj.weight.data().rows == h);
    REQUIRE(model.layers[0].mlp.down_proj.weight.data().cols == inter);

    Gemma3KvCache cache(cfg, 8);
    require_all_finite(model.forward_cached({1, 2, 3}, cache), "gguf logits");
    std::remove(path.c_str());
}

TEST_CASE("load_weights_from_gguf honours an explicit output.weight", "[transformer4]") {
    Config4 cfg = tiny_cfg();
    cfg.num_hidden_layers = 1;
    const std::size_t h = cfg.hidden_size;

    GgufWriter w;
    w.add_tensor("token_embd.weight", {h, cfg.vocab_size}, GgufType::F32, f32_bytes(ramp(cfg.vocab_size * h, 0.01f)));
    w.add_tensor("output.weight", {h, cfg.vocab_size}, GgufType::F32, f32_bytes(ramp(cfg.vocab_size * h, 0.02f)));

    const std::string path = "/tmp/rt_gemma3_untied.gguf";
    w.write(path);

    Gemma3Model model = Gemma3Model::new_for_inference(cfg);
    REQUIRE(model.load_weights_from_gguf(path).has_value());

    REQUIRE(model.embed_bf16.has_value());
    // output.weight loaded into the f32 slot, and the final weight-tying step
    // was skipped because the file supplied its own head.
    REQUIRE(model.lm_head.weight.data().rows == cfg.vocab_size);
    REQUIRE(model.lm_head.weight.data().cols == h);
    // token_embd is read first and ties lm_head to the embedding as it goes, so
    // the BF16 weight from that tying survives -- and since `fused_linear`
    // prefers BF16 over f32, an *f32* output.weight is shadowed by it. This
    // matches the Rust reference; a BF16 or quantized output.weight replaces
    // the tied one properly.
    REQUIRE(model.lm_head.bf16_weight.has_value());
    REQUIRE(model.lm_head.bf16_weight->data == model.embed_bf16->data);
    std::remove(path.c_str());
}

TEST_CASE("load_weights_from_dir rejects a directory with no shards", "[transformer4]") {
    Gemma3Model model = Gemma3Model::new_for_inference(tiny_cfg());
    const Result<void> missing = model.load_weights_from_dir("/tmp/rt_no_such_dir_12345");
    REQUIRE_FALSE(missing.has_value());

    const std::string dir = "/tmp/rt_empty_shard_dir";
    std::filesystem::create_directories(dir);
    const Result<void> empty = model.load_weights_from_dir(dir);
    REQUIRE_FALSE(empty.has_value());
    REQUIRE(empty.error().find("no .safetensors files") != std::string::npos);
    std::filesystem::remove_all(dir);
}

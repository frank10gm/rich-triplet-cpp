#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

#include "rt/transformer4.hpp"

using namespace rt;

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

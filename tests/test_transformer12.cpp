#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <set>

#include "rt/transformer.hpp"
#include "rt/transformer2.hpp"

using namespace rt;

namespace {

/// A very small config, so the scalar model stays fast enough to test.
Config tiny_cfg() {
    Config c;
    c.vocab_size = 12;
    c.context_length = 8;
    c.d_model = 8;
    c.n_layers = 2;
    c.n_heads = 2;
    return c;
}

}  // namespace

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

TEST_CASE("config nano and derived sizes", "[transformer]") {
    const Config c = Config::nano(65);
    REQUIRE(c.vocab_size == 65);
    REQUIRE(c.context_length == 64);
    REQUIRE(c.d_model == 64);
    REQUIRE(c.n_layers == 2);
    REQUIRE(c.n_heads == 2);
    REQUIRE(c.d_head() == 32);
    REQUIRE(c.param_count() > 0);

    const Config t = tiny_cfg();
    REQUIRE(t.d_head() == 4);
}

// -----------------------------------------------------------------------------
// Scalar transformer
// -----------------------------------------------------------------------------

TEST_CASE("scalar embedding adds token and position vectors", "[transformer]") {
    const Config cfg = tiny_cfg();
    InitRng rng(1);
    const Embedding embed(cfg, rng);

    REQUIRE(embed.token_embed.size() == cfg.vocab_size);
    REQUIRE(embed.pos_embed.size() == cfg.context_length);
    REQUIRE(embed.parameters().size() ==
            (cfg.vocab_size + cfg.context_length) * cfg.d_model);

    const auto out = embed.forward({3, 7});
    REQUIRE(out.size() == 2);
    REQUIRE(out[0].size() == cfg.d_model);
    for (std::size_t d = 0; d < cfg.d_model; ++d) {
        REQUIRE(std::fabs(out[0][d].val() -
                          (embed.token_embed[3][d].val() + embed.pos_embed[0][d].val())) < 1e-6f);
    }
}

TEST_CASE("scalar attention head is causal", "[transformer]") {
    const Config cfg = tiny_cfg();
    InitRng rng(2);
    const AttentionHead head(cfg.d_model, cfg.d_head(), rng);

    // Build two sequences that agree on the first token; with a causal mask,
    // position 0's output cannot depend on later tokens.
    std::vector<std::vector<Value>> x1, x2;
    for (std::size_t t = 0; t < 3; ++t) {
        std::vector<Value> a, b;
        for (std::size_t d = 0; d < cfg.d_model; ++d) {
            const float v = static_cast<float>(t * cfg.d_model + d) * 0.01f;
            a.emplace_back(v);
            b.emplace_back(t == 0 ? v : v + 1.0f);
        }
        x1.push_back(std::move(a));
        x2.push_back(std::move(b));
    }

    const auto o1 = head.forward(x1);
    const auto o2 = head.forward(x2);
    REQUIRE(o1.size() == 3);
    REQUIRE(o1[0].size() == cfg.d_head());
    for (std::size_t d = 0; d < cfg.d_head(); ++d) {
        INFO("position 0 dim " << d);
        REQUIRE(std::fabs(o1[0][d].val() - o2[0][d].val()) < 1e-5f);
    }
}

TEST_CASE("scalar multi-head attention and block shapes", "[transformer]") {
    const Config cfg = tiny_cfg();
    InitRng rng(3);
    const MultiHeadAttention mha(cfg, rng);
    const TransformerBlock block(cfg, rng);

    std::vector<std::vector<Value>> x;
    for (std::size_t t = 0; t < 3; ++t) {
        std::vector<Value> row;
        for (std::size_t d = 0; d < cfg.d_model; ++d) {
            row.emplace_back(static_cast<float>(t + d) * 0.05f);
        }
        x.push_back(std::move(row));
    }

    const auto attn_out = mha.forward(x);
    REQUIRE(attn_out.size() == 3);
    REQUIRE(attn_out[0].size() == cfg.d_model);

    const auto block_out = block.forward(x);
    REQUIRE(block_out.size() == 3);
    REQUIRE(block_out[0].size() == cfg.d_model);
    for (const auto& row : block_out) {
        for (const Value& v : row) {
            REQUIRE(std::isfinite(v.val()));
        }
    }
}

TEST_CASE("scalar gpt forward, loss and backward", "[transformer]") {
    const Config cfg = tiny_cfg();
    InitRng rng(4);
    const Gpt model(cfg, rng);

    const std::vector<std::size_t> inputs{1, 2, 3};
    const std::vector<std::size_t> targets{2, 3, 4};

    const auto logits = model.forward(inputs);
    REQUIRE(logits.size() == 3);
    REQUIRE(logits[0].size() == cfg.vocab_size);

    const Value loss = model.loss(inputs, targets);
    INFO("loss = " << loss.val());
    REQUIRE(std::isfinite(loss.val()));
    REQUIRE(loss.val() > 0.0f);
    // An untrained model is roughly uniform: log(12) is about 2.48.
    REQUIRE(loss.val() < 2.0f * std::log(static_cast<float>(cfg.vocab_size)));

    loss.backward();
    bool any_nonzero = false;
    for (const Value& p : model.parameters()) {
        REQUIRE(std::isfinite(p.grad()));
        any_nonzero = any_nonzero || std::fabs(p.grad()) > 1e-9f;
    }
    REQUIRE(any_nonzero);
}

// -----------------------------------------------------------------------------
// Tensor transformer
// -----------------------------------------------------------------------------

TEST_CASE("embedding2 gathers and scatters gradients", "[transformer2]") {
    const Config cfg = tiny_cfg();
    InitRng rng(5);
    const Embedding2 embed(cfg, rng);

    const std::vector<std::size_t> ids{3, 3, 7};
    const TensorNode out = embed.forward(ids);
    REQUIRE(out.data().rows == 3);
    REQUIRE(out.data().cols == cfg.d_model);
    REQUIRE(std::fabs(out.data().at(0, 0) - (embed.token_embed.data().at(3, 0) +
                                             embed.pos_embed.data().at(0, 0))) < 1e-6f);

    out.seed_grad_ones();
    out.call_backward_fn();
    // Token 3 appears twice, so its row accumulates twice the gradient.
    REQUIRE(std::fabs(embed.token_embed.grad().at(3, 0) - 2.0f) < 1e-6f);
    REQUIRE(std::fabs(embed.token_embed.grad().at(7, 0) - 1.0f) < 1e-6f);
    // Every position row is used once.
    for (std::size_t r = 0; r < 3; ++r) {
        REQUIRE(std::fabs(embed.pos_embed.grad().at(r, 0) - 1.0f) < 1e-6f);
    }
}

TEST_CASE("concat_heads splits the gradient back per head", "[transformer2]") {
    const std::size_t t = 2, n = 3, dh = 2;
    std::vector<TensorNode> heads;
    for (std::size_t h = 0; h < n; ++h) {
        heads.push_back(TensorNode::leaf(Mat::from_fn(t, dh, [&](std::size_t r, std::size_t c) {
            return static_cast<float>(h * 10 + r * dh + c);
        })));
    }

    const TensorNode concat = MultiHeadAttention2::concat_heads(heads, t, n, dh);
    REQUIRE(concat.data().rows == t);
    REQUIRE(concat.data().cols == n * dh);
    REQUIRE(concat.data().at(1, 4) == heads[2].data().at(1, 0));

    concat.seed_grad_ones();
    concat.call_backward_fn();
    for (const TensorNode& h : heads) {
        for (float g : h.grad().data) {
            REQUIRE(g == 1.0f);
        }
    }
}

TEST_CASE("gpt2 forward, loss and backward", "[transformer2]") {
    const Config cfg = tiny_cfg();
    InitRng rng(6);
    const Gpt2 model(cfg, rng);

    const std::vector<std::size_t> inputs{1, 2, 3, 4};
    const std::vector<std::size_t> targets{2, 3, 4, 5};

    const Mat logits = model.forward(inputs).data();
    REQUIRE(logits.rows == 4);
    REQUIRE(logits.cols == cfg.vocab_size);

    const TensorNode loss = model.loss(inputs, targets);
    REQUIRE(loss.data().rows == 1);
    REQUIRE(loss.data().cols == 1);
    const float value = loss.data().at(0, 0);
    INFO("loss = " << value);
    REQUIRE(std::isfinite(value));
    REQUIRE(value > 0.0f);

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

TEST_CASE("gpt2 tie_weights shares one node", "[transformer2]") {
    const Config cfg = tiny_cfg();
    InitRng rng(7);
    Gpt2 model(cfg, rng);
    REQUIRE(model.lm_head.weight.id() != model.embed.token_embed.id());

    model.tie_weights();
    REQUIRE(model.lm_head.weight.id() == model.embed.token_embed.id());
    // Writing through one view is visible through the other.
    model.embed.token_embed.set_grad(Mat::ones(cfg.vocab_size, cfg.d_model));
    REQUIRE(model.lm_head.weight.grad().at(0, 0) == 1.0f);
}

TEST_CASE("gpt2 kv cache append and eviction", "[transformer2]") {
    HeadKvCache cache(2, 3);  // d_head = 2, at most 3 rows
    const std::vector<float> a{1, 2}, b{3, 4}, c{5, 6}, d{7, 8};

    cache.append_k(a);
    cache.append_k(b);
    REQUIRE(cache.k.rows == 2);
    REQUIRE(cache.k.at(1, 0) == 3.0f);

    cache.append_k(c);
    REQUIRE(cache.k.rows == 3);

    // At capacity, the oldest row is dropped.
    cache.append_k(d);
    REQUIRE(cache.k.rows == 3);
    REQUIRE(cache.k.at(0, 0) == 3.0f);
    REQUIRE(cache.k.at(2, 0) == 7.0f);

    cache.append_v(a);
    REQUIRE(cache.v.rows == 1);
}

TEST_CASE("gpt2 cached generation matches the uncached forward", "[transformer2]") {
    // Prefilling the prompt through the cache must reproduce the plain
    // forward's last-row logits, so greedy decoding picks the same token.
    const Config cfg = tiny_cfg();
    InitRng rng(8);
    const Gpt2 model(cfg, rng);
    const std::vector<std::size_t> prompt{1, 2, 3};

    const Mat full = model.forward(prompt).data();
    std::size_t expected = 0;
    for (std::size_t c = 1; c < full.cols; ++c) {
        if (full.at(full.rows - 1, expected) < full.at(full.rows - 1, c)) {
            expected = c;
        }
    }

    const std::vector<std::size_t> got = model.generate_cached(prompt, 1, 0.0f);
    REQUIRE(got.size() == 1);
    REQUIRE(got[0] == expected);
}

TEST_CASE("gpt2 streaming generation yields the requested count", "[transformer2]") {
    const Config cfg = tiny_cfg();
    InitRng rng(9);
    const Gpt2 model(cfg, rng);

    std::vector<std::size_t> tokens;
    model.generate_cached_streaming({1, 2}, 4, 0.8f, 3,
                                    [&](std::size_t t) { tokens.push_back(t); });
    REQUIRE(tokens.size() == 4);
    for (std::size_t t : tokens) {
        REQUIRE(t < cfg.vocab_size);
    }
}

TEST_CASE("gpt2 samplers", "[transformer2]") {
    const Mat logits({0.1f, 5.0f, 0.3f, 2.0f}, 1, 4);

    // Greedy at temperature <= 0.
    REQUIRE(gpt2_sample_token(logits, 0, 0.0f) == 1);
    REQUIRE(gpt2_sample_token_topk(logits, 0, -1.0f, 2) == 1);

    // gpt2_sample_token keeps the argmax after temperature scaling.
    REQUIRE(gpt2_sample_token(logits, 0, 0.5f) == 1);

    // top_k = 1 leaves only the argmax, so every draw returns it.
    for (int i = 0; i < 5; ++i) {
        REQUIRE(gpt2_sample_token_topk(logits, 0, 1.0f, 1) == 1);
    }
    // top_k = 0 means the whole vocabulary.
    REQUIRE(gpt2_sample_token_topk(logits, 0, 1.0f, 0) < 4);
}

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include "rt/transformer5.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

namespace {

/// Deterministic activations for a [T, n_heads * head_dim] tensor.
[[nodiscard]] Mat heads(std::size_t rows, std::size_t n_heads, std::size_t head_dim) {
    return Mat::from_fn(rows, n_heads * head_dim, [&](std::size_t r, std::size_t c) {
        return 0.01f * static_cast<float>(r * n_heads * head_dim + c) - 0.5f;
    });
}

}  // namespace

// =============================================================================
// Config
// =============================================================================

TEST_CASE("Config5::orpheus_3b matches the published GGUF metadata", "[transformer5]") {
    const Config5 c = Config5::orpheus_3b();
    REQUIRE(c.num_hidden_layers == 28);
    REQUIRE(c.hidden_size == 3072);
    REQUIRE(c.num_attention_heads == 24);
    REQUIRE(c.num_key_value_heads == 8);
    REQUIRE(c.head_dim == 128);
    REQUIRE(c.intermediate_size == 8192);
    REQUIRE(c.rope_theta == 500000.0f);
    REQUIRE(c.vocab_size == 156940);
    // 24 query heads over 8 KV heads: three queries share each KV head.
    REQUIRE(c.gqa_group() == 3);
    // head_dim * n_heads reconstructs hidden_size, so the projections line up.
    REQUIRE(c.head_dim * c.num_attention_heads == c.hidden_size);
    // Interleaved is the GGUF convention; see RopePairing.
    REQUIRE(c.rope_pairing == RopePairing::Interleaved);
}

TEST_CASE("inv_freq decreases with dimension", "[transformer5]") {
    const Config5 c = Config5::orpheus_3b();
    const std::vector<float> f = c.inv_freq();
    REQUIRE(f.size() == c.head_dim / 2);
    REQUIRE(approx(f.front(), 1.0f));  // theta^0
    for (std::size_t i = 1; i < f.size(); ++i) {
        REQUIRE(f[i] < f[i - 1]);
    }
}

TEST_CASE("rope_freq_divisors divide rather than multiply", "[transformer5]") {
    // `rope_freqs.weight` runs from 1.0 up to 32.0. Treating those as
    // multipliers would raise the low-frequency bands by 32x instead of
    // lowering them -- a factor of 1024 the wrong way, on exactly the
    // dimensions that carry long-range position.
    Config5 c = Config5::orpheus_3b();
    const std::vector<float> plain = c.inv_freq();

    c.rope_freq_divisors.assign(c.head_dim / 2, 1.0f);
    c.rope_freq_divisors.back() = 32.0f;
    const std::vector<float> scaled = c.inv_freq();

    REQUIRE(approx(scaled.front(), plain.front()));
    REQUIRE(scaled.back() < plain.back());
    REQUIRE(std::fabs(scaled.back() - plain.back() / 32.0f) < 1e-12f);
}

TEST_CASE("a zero divisor is ignored rather than dividing by zero", "[transformer5]") {
    Config5 c = Config5::orpheus_3b();
    c.rope_freq_divisors.assign(c.head_dim / 2, 0.0f);
    for (const float f : c.inv_freq()) {
        REQUIRE(std::isfinite(f));
        REQUIRE(f > 0.0f);
    }
}

// =============================================================================
// RoPE
// =============================================================================

TEST_CASE("llama_rope at position 0 is the identity", "[transformer5]") {
    const Config5 c = Config5::orpheus_3b();
    const std::vector<float> f = c.inv_freq();
    const Mat x = heads(1, 2, 128);
    for (const RopePairing p : {RopePairing::Interleaved, RopePairing::HalfSplit}) {
        const Mat out = llama_rope(x, 2, 128, 0, f, p);
        for (std::size_t i = 0; i < x.data.size(); ++i) {
            REQUIRE(approx(out.data[i], x.data[i]));
        }
    }
}

TEST_CASE("llama_rope preserves the norm of every rotated pair", "[transformer5]") {
    // A rotation changes direction, never length. This holds for both
    // conventions and is the cheapest check that the sin/cos signs are
    // consistent.
    const Config5 c = Config5::orpheus_3b();
    const std::vector<float> f = c.inv_freq();
    constexpr std::size_t head_dim = 128;
    constexpr std::size_t half = head_dim / 2;
    const Mat x = heads(3, 2, head_dim);

    for (const RopePairing p : {RopePairing::Interleaved, RopePairing::HalfSplit}) {
        const Mat out = llama_rope(x, 2, head_dim, 5, f, p);
        for (std::size_t row = 0; row < x.rows; ++row) {
            for (std::size_t h = 0; h < 2; ++h) {
                for (std::size_t i = 0; i < half; ++i) {
                    const std::size_t c0 = p == RopePairing::Interleaved ? h * head_dim + 2 * i
                                                                        : h * head_dim + i;
                    const std::size_t c1 = p == RopePairing::Interleaved ? c0 + 1 : c0 + half;
                    const float before = x.at(row, c0) * x.at(row, c0) +
                                         x.at(row, c1) * x.at(row, c1);
                    const float after = out.at(row, c0) * out.at(row, c0) +
                                        out.at(row, c1) * out.at(row, c1);
                    REQUIRE(std::fabs(before - after) < 1e-3f);
                }
            }
        }
    }
}

TEST_CASE("llama_rope pairings are genuinely different", "[transformer5]") {
    // The regression test for the bug that produced this enum. GGUF permutes
    // llama Q/K weight rows so that interleaved rotation reproduces
    // HuggingFace's half-split `rotate_half`; applying half-split to those
    // permuted weights rotates the wrong pairs together.
    //
    // The reason it is worth a test rather than a comment: both conventions
    // use the same angles, so at low positions -- where every rotation is
    // near identity -- they agree to several decimals. They only separate as
    // position grows, which in a speech model means the first few tokens come
    // out right and everything after is noise.
    const Config5 c = Config5::orpheus_3b();
    const std::vector<float> f = c.inv_freq();
    const Mat x = heads(1, 1, 128);

    const Mat inter = llama_rope(x, 1, 128, 40, f, RopePairing::Interleaved);
    const Mat split = llama_rope(x, 1, 128, 40, f, RopePairing::HalfSplit);

    std::size_t differing = 0;
    for (std::size_t i = 0; i < inter.data.size(); ++i) {
        if (std::fabs(inter.data[i] - split.data[i]) > 1e-3f) {
            ++differing;
        }
    }
    REQUIRE(differing > 0);
}

TEST_CASE("llama_rope rotates a unit pair by the expected angle", "[transformer5]") {
    // One head, head_dim 2, so there is a single pair at frequency inv_freq[0]
    // == 1 and the rotation angle is just the position.
    Config5 c;
    c.head_dim = 2;
    c.rope_theta = 10000.0f;
    const std::vector<float> f = c.inv_freq();
    REQUIRE(approx(f[0], 1.0f));

    const Mat x({1.0f, 0.0f}, 1, 2);
    const Mat out = llama_rope(x, 1, 2, 3, f, RopePairing::Interleaved);
    // (1, 0) rotated by 3 radians -> (cos 3, sin 3).
    REQUIRE(approx(out.at(0, 0), std::cos(3.0f)));
    REQUIRE(approx(out.at(0, 1), std::sin(3.0f)));
}

TEST_CASE("llama_rope advances the angle with the row offset", "[transformer5]") {
    Config5 c;
    c.head_dim = 2;
    const std::vector<float> f = c.inv_freq();
    const Mat x = Mat::from_fn(3, 2, [](std::size_t, std::size_t col) {
        return col == 0 ? 1.0f : 0.0f;
    });
    // Rows are at absolute positions offset..offset+2.
    const Mat out = llama_rope(x, 1, 2, 10, f, RopePairing::Interleaved);
    for (std::size_t r = 0; r < 3; ++r) {
        const float angle = static_cast<float>(10 + r) * f[0];
        REQUIRE(approx(out.at(r, 0), std::cos(angle)));
        REQUIRE(approx(out.at(r, 1), std::sin(angle)));
    }
}

// =============================================================================
// KV cache
// =============================================================================

TEST_CASE("LlamaKvCache preallocates and appends", "[transformer5]") {
    Config5 c = Config5::orpheus_3b();
    c.num_hidden_layers = 2;
    LlamaKvCache cache(c, 16);
    REQUIRE(cache.layers.size() == 2);

    const std::size_t kv_dim = c.num_key_value_heads * c.head_dim;
    REQUIRE(cache.layers[0].k.rows == 16);
    REQUIRE(cache.layers[0].k.cols == kv_dim);
    REQUIRE(cache.layers[0].seq_len == 0);

    const Mat k = Mat::ones(3, kv_dim);
    const Mat v = Mat::ones(3, kv_dim).scale(2.0f);
    cache.layers[0].append(k, v);
    REQUIRE(cache.layers[0].seq_len == 3);
    REQUIRE(approx(cache.layers[0].k.at(2, 0), 1.0f));
    REQUIRE(approx(cache.layers[0].v.at(2, 0), 2.0f));
    // Rows past seq_len stay zero.
    REQUIRE(approx(cache.layers[0].k.at(3, 0), 0.0f));

    cache.layers[0].append(k, v);
    REQUIRE(cache.layers[0].seq_len == 6);

    cache.clear();
    REQUIRE(cache.layers[0].seq_len == 0);
    cache.free();
    REQUIRE(cache.layers[0].k.rows == 0);
}

// =============================================================================
// Sampling
// =============================================================================

TEST_CASE("sample_token_large_vocab picks the argmax when greedy", "[transformer5]") {
    Mat logits = Mat::zeros(1, 100);
    logits.at_mut(0, 57) = 5.0f;
    SamplingParams p;
    p.temperature = 0.0f;
    LcgRng rng(1);
    REQUIRE(sample_token_large_vocab(logits, 0, p, {}, rng) == 57);
}

TEST_CASE("sample_token_large_vocab honours the vocabulary mask", "[transformer5]") {
    // The highest logit sits outside the allowed range, so it must not win.
    Mat logits = Mat::zeros(1, 100);
    logits.at_mut(0, 5) = 100.0f;   // masked out
    logits.at_mut(0, 60) = 1.0f;    // best allowed
    SamplingParams p;
    p.temperature = 0.0f;
    p.allowed_min = 50;
    p.allowed_max = 70;
    LcgRng rng(1);
    REQUIRE(sample_token_large_vocab(logits, 0, p, {}, rng) == 60);

    // allowed_extra reaches back outside the range, for stop markers.
    p.allowed_extra = {5};
    LcgRng rng2(1);
    REQUIRE(sample_token_large_vocab(logits, 0, p, {}, rng2) == 5);
}

TEST_CASE("sample_token_large_vocab never draws a masked token", "[transformer5]") {
    Mat logits = Mat::from_fn(1, 500, [](std::size_t, std::size_t c) {
        return static_cast<float>(c % 7);
    });
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_p = 1.0f;
    p.allowed_min = 100;
    p.allowed_max = 120;
    LcgRng rng(99);
    for (int i = 0; i < 200; ++i) {
        const std::size_t id = sample_token_large_vocab(logits, 0, p, {}, rng);
        REQUIRE(id >= 100);
        REQUIRE(id < 120);
    }
}

TEST_CASE("sample_token_large_vocab respects top-k", "[transformer5]") {
    Mat logits = Mat::zeros(1, 1000);
    logits.at_mut(0, 10) = 10.0f;
    logits.at_mut(0, 20) = 9.0f;
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 2;
    p.top_p = 1.0f;
    LcgRng rng(3);
    for (int i = 0; i < 100; ++i) {
        const std::size_t id = sample_token_large_vocab(logits, 0, p, {}, rng);
        REQUIRE((id == 10 || id == 20));
    }
}

TEST_CASE("sample_token_large_vocab collapses to one token at top_p near zero",
          "[transformer5]") {
    Mat logits = Mat::zeros(1, 1000);
    logits.at_mut(0, 42) = 20.0f;
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_p = 0.01f;
    LcgRng rng(5);
    for (int i = 0; i < 50; ++i) {
        REQUIRE(sample_token_large_vocab(logits, 0, p, {}, rng) == 42);
    }
}

TEST_CASE("sample_token_large_vocab applies the repetition penalty", "[transformer5]") {
    Mat logits = Mat::zeros(1, 100);
    logits.at_mut(0, 7) = 2.0f;
    logits.at_mut(0, 8) = 1.9f;
    SamplingParams p;
    p.temperature = 0.0f;
    p.repetition_penalty = 2.0f;
    LcgRng rng(1);
    // Unpenalised, 7 wins.
    REQUIRE(sample_token_large_vocab(logits, 0, p, {}, rng) == 7);
    // Having already produced 7, its positive logit is divided and 8 takes it.
    REQUIRE(sample_token_large_vocab(logits, 0, p, {7}, rng) == 8);
}

TEST_CASE("sample_token_large_vocab is reproducible for a seed", "[transformer5]") {
    Mat logits = Mat::from_fn(1, 2000, [](std::size_t, std::size_t c) {
        return std::sin(static_cast<float>(c) * 0.01f) * 3.0f;
    });
    SamplingParams p;
    p.temperature = 0.9f;
    p.top_p = 0.9f;

    std::vector<std::size_t> first;
    LcgRng a(1234);
    for (int i = 0; i < 20; ++i) {
        first.push_back(sample_token_large_vocab(logits, 0, p, {}, a));
    }
    LcgRng b(1234);
    for (const std::size_t want : first) {
        REQUIRE(sample_token_large_vocab(logits, 0, p, {}, b) == want);
    }
}

#include <catch2/catch_test_macros.hpp>

#if RT_FEATURE_METAL

#include <cmath>

#include "rt/metal_decode.hpp"

using namespace rt;

namespace {

/// A small Gemma 3 config that still exercises both attention modes: with a
/// window of 8 and the 5-local:1-global pattern, layer 5 is global.
Config4 tiny_cfg() {
    Config4 c;
    c.vocab_size = 256;
    c.hidden_size = 64;
    c.num_hidden_layers = 6;
    c.num_attention_heads = 2;
    c.num_key_value_heads = 1;
    c.intermediate_size = 128;
    c.head_dim = 32;
    c.sliding_window = 8;
    c.rope_theta_local = 10000.0f;
    c.rope_theta_global = 1000000.0f;
    c.rope_freq_scale_local = 1.0f;
    c.rope_freq_scale_global = 0.125f;
    c.rms_norm_eps = 1e-6f;
    c.query_pre_attn_scalar = 32.0f;
    c.eos_token_id = 1;
    c.max_position_embeddings = 128;
    return c;
}

/// Copy `src`'s weights into `dst` as BF16.
///
/// The GPU path only reads BF16 or Q4_K, so the CPU reference is built from the
/// same BF16 bits: that way the comparison isolates the GPU kernels rather than
/// bf16 rounding.
void copy_as_bf16(Gemma3Model& dst, const Gemma3Model& src) {
    const Config4& cfg = src.config;
    dst.embed_bf16 = src.embed_tokens.data().to_bf16();
    dst.norm.gamma.set_data(src.norm.gamma.data());

    const auto copy_linear = [](Linear2& d, const Linear2& s) {
        d.load_bf16(*s.weight.data().to_bf16().data, s.out_features, s.in_features);
    };

    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const Gemma3Block& s = src.layers[i];
        Gemma3Block& d = dst.layers[i];
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
    // lm_head is weight-tied, so it shares the embedding's bits.
    dst.lm_head.load_bf16_shared(dst.embed_bf16->data, cfg.vocab_size, cfg.hidden_size);
}

std::size_t argmax(const std::vector<float>& v) {
    return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

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

TEST_CASE("metal decode step matches the CPU model", "[metal][decode]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(77);
    const Gemma3Model source(cfg, rng);

    // Two models from the same BF16 weights: one the GPU consumes (and
    // destroys, since the context frees CPU weights as it uploads), one the CPU
    // reference keeps.
    Gemma3Model gpu_model = Gemma3Model::new_for_inference(cfg);
    Gemma3Model ref = Gemma3Model::new_for_inference(cfg);
    copy_as_bf16(gpu_model, source);
    copy_as_bf16(ref, source);

    const std::vector<std::size_t> prompt{5, 11, 23, 42};
    Gemma3KvCache cache(cfg, 64);
    const Mat cpu_prefill = ref.forward_cached(prompt, cache);

    const std::unique_ptr<MetalDecodeContext> ctx = MetalDecodeContext::create(gpu_model, 4);
    REQUIRE(ctx->lm_head_vocab() == cfg.vocab_size);

    // The GPU picks up where CPU prefill left off.
    ctx->sync_kv_from_cpu(cache);

    const std::size_t next = argmax_row(cpu_prefill, 0);
    const Mat cpu_logits = ref.forward_cached({next}, cache);
    const std::vector<float> gpu_logits = ctx->decode_step(next, prompt.size());

    REQUIRE(gpu_logits.size() == cfg.vocab_size);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < gpu_logits.size(); ++i) {
        REQUIRE(std::isfinite(gpu_logits[i]));
        max_err = std::max(max_err, std::fabs(gpu_logits[i] - cpu_logits.at(0, i)));
    }
    // The GPU keeps its KV cache in half precision where the CPU uses f32, so
    // the two agree closely rather than exactly.
    INFO("decode max error " << max_err);
    REQUIRE(max_err < 0.01f);
    // What matters for generation is that they pick the same token.
    REQUIRE(argmax(gpu_logits) == argmax_row(cpu_logits, 0));
}

TEST_CASE("metal batch decode runs", "[metal][decode]") {
    const Config4 cfg = tiny_cfg();
    InitRng rng(78);
    const Gemma3Model source(cfg, rng);

    Gemma3Model gpu_model = Gemma3Model::new_for_inference(cfg);
    copy_as_bf16(gpu_model, source);

    const std::unique_ptr<MetalDecodeContext> ctx = MetalDecodeContext::create(gpu_model, 4);

    // Nothing was prefilled, so the batch starts at position 0.
    const std::vector<std::size_t> tokens{5, 11, 23};
    const std::vector<float> logits = ctx->decode_step_batch(tokens, 0);
    REQUIRE(logits.size() == tokens.size() * ctx->lm_head_vocab());
    for (float v : logits) {
        REQUIRE(std::isfinite(v));
    }

    // Each row should be a different token's logits, not a repeat.
    const std::size_t v = ctx->lm_head_vocab();
    REQUIRE(std::vector<float>(logits.begin(), logits.begin() + v) !=
            std::vector<float>(logits.begin() + v, logits.begin() + 2 * v));
}

TEST_CASE("metal decode step advances the KV cache", "[metal][decode]") {
    // Two consecutive decode steps must see different context, so their logits
    // differ even when the same token is fed twice.
    const Config4 cfg = tiny_cfg();
    InitRng rng(79);
    const Gemma3Model source(cfg, rng);

    Gemma3Model gpu_model = Gemma3Model::new_for_inference(cfg);
    copy_as_bf16(gpu_model, source);

    const std::unique_ptr<MetalDecodeContext> ctx = MetalDecodeContext::create(gpu_model, 0);
    const std::vector<float> first = ctx->decode_step(7, 0);
    const std::vector<float> second = ctx->decode_step(7, 1);
    REQUIRE(first.size() == second.size());
    REQUIRE(first != second);
}

#endif  // RT_FEATURE_METAL

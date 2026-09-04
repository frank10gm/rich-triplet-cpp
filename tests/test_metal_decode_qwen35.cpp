#include <catch2/catch_test_macros.hpp>

#if RT_FEATURE_METAL

#include <algorithm>
#include <cmath>

#include "rt/metal_decode_qwen35.hpp"

using namespace rt;

namespace {

/// A tiny hybrid config. With `full_attention_interval = 4` and four layers,
/// layer 3 is softmax attention and the other three are DeltaNet, so both
/// encode paths run.
ConfigQwen35 tiny_cfg() {
    ConfigQwen35 c;
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.num_hidden_layers = 4;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.head_dim = 16;
    c.linear_num_key_heads = 2;
    c.linear_num_value_heads = 4;
    c.linear_key_head_dim = 8;
    c.linear_value_head_dim = 8;
    c.linear_conv_kernel_dim = 4;
    c.intermediate_size = 64;
    c.rms_norm_eps = 1e-6f;
    c.rope_theta = 10000.0f;
    c.partial_rotary_factor = 0.25f;
    c.full_attention_interval = 4;
    c.max_position_embeddings = 128;
    c.eos_token_id = 1;
    c.tie_word_embeddings = true;
    return c;
}

void fill_weights(Qwen35Model& model, std::uint64_t seed) {
    InitRng rng(seed);
    const auto fill_linear = [&](Linear2& l) {
        l.weight.set_data(Mat(rng.normal_vec(l.out_features * l.in_features, 0.05f),
                              l.out_features, l.in_features));
        l.bias.set_data(Mat::zeros(0, 0));
    };
    const auto fill_vec = [&](std::vector<float>& v, float scale) {
        for (float& x : v) {
            x = rng.next_normal() * scale;
        }
    };

    const std::size_t h = model.config.hidden_size;
    model.embed_tokens.set_data(
        Mat(rng.normal_vec(model.config.vocab_size * h, 0.05f), model.config.vocab_size, h));
    model.norm.gamma.set_data(Mat(rng.normal_vec(h, 0.1f), 1, h));

    for (Qwen35Block& block : model.layers) {
        block.input_layernorm.gamma.set_data(Mat(rng.normal_vec(h, 0.1f), 1, h));
        block.post_attention_layernorm.gamma.set_data(Mat(rng.normal_vec(h, 0.1f), 1, h));
        fill_linear(block.mlp.gate_proj);
        fill_linear(block.mlp.up_proj);
        fill_linear(block.mlp.down_proj);

        if (auto* dn = std::get_if<Qwen35DeltaNet>(&block.token_mixer)) {
            fill_linear(dn->in_proj_qkv);
            fill_linear(dn->in_proj_z);
            fill_linear(dn->in_proj_a);
            fill_linear(dn->in_proj_b);
            fill_linear(dn->out_proj);
            fill_vec(dn->conv1d_weight, 0.3f);
            fill_vec(dn->a_log, 0.2f);
            fill_vec(dn->dt_bias, 0.2f);
            fill_vec(dn->norm_weight, 0.3f);
        } else {
            auto& fa = std::get<Qwen35FullAttention>(block.token_mixer);
            fill_linear(fa.q_proj);
            fill_linear(fa.k_proj);
            fill_linear(fa.v_proj);
            fill_linear(fa.o_proj);
            fa.q_norm.gamma.set_data(Mat(rng.normal_vec(fa.head_dim, 0.1f), 1, fa.head_dim));
            fa.k_norm.gamma.set_data(Mat(rng.normal_vec(fa.head_dim, 0.1f), 1, fa.head_dim));
        }
    }
}

/// Copy `src`'s weights into `dst` as BF16.
///
/// The GPU only reads BF16 or Q4_K, so the CPU reference is built from the same
/// BF16 bits: the comparison then isolates the GPU kernels rather than bf16
/// rounding.
void copy_as_bf16(Qwen35Model& dst, const Qwen35Model& src) {
    const ConfigQwen35& cfg = src.config;
    dst.embed_bf16 = src.embed_tokens.data().to_bf16();
    dst.norm.gamma.set_data(src.norm.gamma.data());

    const auto copy_linear = [](Linear2& d, const Linear2& s) {
        d.load_bf16(*s.weight.data().to_bf16().data, s.out_features, s.in_features);
    };

    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const Qwen35Block& s = src.layers[i];
        Qwen35Block& d = dst.layers[i];
        d.input_layernorm.gamma.set_data(s.input_layernorm.gamma.data());
        d.post_attention_layernorm.gamma.set_data(s.post_attention_layernorm.gamma.data());
        copy_linear(d.mlp.gate_proj, s.mlp.gate_proj);
        copy_linear(d.mlp.up_proj, s.mlp.up_proj);
        copy_linear(d.mlp.down_proj, s.mlp.down_proj);

        if (const auto* sdn = std::get_if<Qwen35DeltaNet>(&s.token_mixer)) {
            auto& ddn = std::get<Qwen35DeltaNet>(d.token_mixer);
            copy_linear(ddn.in_proj_qkv, sdn->in_proj_qkv);
            copy_linear(ddn.in_proj_z, sdn->in_proj_z);
            copy_linear(ddn.in_proj_a, sdn->in_proj_a);
            copy_linear(ddn.in_proj_b, sdn->in_proj_b);
            copy_linear(ddn.out_proj, sdn->out_proj);
            ddn.conv1d_weight = sdn->conv1d_weight;
            ddn.a_log = sdn->a_log;
            ddn.dt_bias = sdn->dt_bias;
            ddn.norm_weight = sdn->norm_weight;
        } else {
            const auto& sfa = std::get<Qwen35FullAttention>(s.token_mixer);
            auto& dfa = std::get<Qwen35FullAttention>(d.token_mixer);
            copy_linear(dfa.q_proj, sfa.q_proj);
            copy_linear(dfa.k_proj, sfa.k_proj);
            copy_linear(dfa.v_proj, sfa.v_proj);
            copy_linear(dfa.o_proj, sfa.o_proj);
            dfa.q_norm.gamma.set_data(sfa.q_norm.gamma.data());
            dfa.k_norm.gamma.set_data(sfa.k_norm.gamma.data());
        }
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

TEST_CASE("qwen35 metal decode step matches the CPU model", "[metal][qwen35]") {
    const ConfigQwen35 cfg = tiny_cfg();
    Qwen35Model source = Qwen35Model::new_for_inference(cfg);
    fill_weights(source, 91);

    Qwen35Model gpu_model = Qwen35Model::new_for_inference(cfg);
    Qwen35Model ref = Qwen35Model::new_for_inference(cfg);
    copy_as_bf16(gpu_model, source);
    copy_as_bf16(ref, source);

    const std::vector<std::size_t> prompt{5, 11, 23, 42};
    Qwen35Cache cache(cfg, 64);
    const Mat cpu_prefill = ref.prefill(prompt, cache);

    const std::unique_ptr<MetalDecodeContextQwen35> ctx =
        MetalDecodeContextQwen35::create(gpu_model);
    REQUIRE(ctx->lm_head_vocab() == cfg.vocab_size);

    // The GPU picks up the DeltaNet state and KV cache left by CPU prefill.
    ctx->sync_state_from_cpu(cache);

    const std::size_t next = argmax_row(cpu_prefill, 0);
    const Mat cpu_logits = ref.decode_step(next, cache);
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

TEST_CASE("qwen35 metal decode advances its state", "[metal][qwen35]") {
    // Both the DeltaNet recurrent state and the attention KV cache carry over,
    // so feeding the same token twice must give different logits.
    const ConfigQwen35 cfg = tiny_cfg();
    Qwen35Model source = Qwen35Model::new_for_inference(cfg);
    fill_weights(source, 92);

    Qwen35Model gpu_model = Qwen35Model::new_for_inference(cfg);
    copy_as_bf16(gpu_model, source);

    const std::unique_ptr<MetalDecodeContextQwen35> ctx =
        MetalDecodeContextQwen35::create(gpu_model);
    const std::vector<float> first = ctx->decode_step(7, 0);
    const std::vector<float> second = ctx->decode_step(7, 1);
    REQUIRE(first.size() == second.size());
    REQUIRE(first != second);
    for (float v : second) {
        REQUIRE(std::isfinite(v));
    }
}

#endif  // RT_FEATURE_METAL

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "gguf_writer.hpp"
#include "rt/transformer_qwen35.hpp"

using namespace rt;

namespace {

/// A tiny hybrid config: small dimensions, but the 4-layer full-attention
/// interval, the DeltaNet head expansion (nv = 2 * nk) and partial RoPE all
/// stay in play.
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

/// Fill a model's zero-placeholder weights with deterministic pseudo-random
/// values so the forward pass produces something meaningful.
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
    model.lm_head.weight = model.embed_tokens;
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

void require_all_finite(const Mat& m, const char* what) {
    for (std::size_t i = 0; i < m.data.size(); ++i) {
        INFO(what << "[" << i << "] = " << m.data[i]);
        REQUIRE(std::isfinite(m.data[i]));
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

TEST_CASE("qwen35 full-attention layer pattern", "[qwen35]") {
    const ConfigQwen35 cfg = ConfigQwen35::qwen35_4b();
    // Every 4th layer, counting from 1: layers 3, 7, 11, ...
    for (std::size_t i = 0; i < 32; ++i) {
        INFO("layer " << i);
        REQUIRE(cfg.is_full_attention_layer(i) == ((i + 1) % 4 == 0));
    }
    REQUIRE(cfg.is_full_attention_layer(3));
    REQUIRE_FALSE(cfg.is_full_attention_layer(0));
}

TEST_CASE("qwen35 config sizes", "[qwen35]") {
    const ConfigQwen35 c4 = ConfigQwen35::qwen35_4b();
    REQUIRE(c4.hidden_size == 2560);
    REQUIRE(c4.intermediate_size == 9216);
    REQUIRE(c4.tie_word_embeddings);
    // key_dim * 2 + value_dim = 16*128*2 + 32*128
    REQUIRE(c4.deltanet_qkv_dim() == 16 * 128 * 2 + 32 * 128);
    // 25% of 256
    REQUIRE(c4.rope_dim() == 64);

    const ConfigQwen35 c9 = ConfigQwen35::qwen35_9b();
    REQUIRE(c9.hidden_size == 4096);
    REQUIRE_FALSE(c9.tie_word_embeddings);

    const ConfigQwen35 c08 = ConfigQwen35::qwen35_0_8b();
    REQUIRE(c08.hidden_size == 1024);
    REQUIRE(c08.num_hidden_layers == 24);
    REQUIRE(c08.linear_num_value_heads == 16);
}

// -----------------------------------------------------------------------------
// Scalar helpers
// -----------------------------------------------------------------------------

TEST_CASE("qwen35 activations", "[qwen35]") {
    REQUIRE(std::fabs(qwen_sigmoid(0.0f) - 0.5f) < 1e-6f);
    REQUIRE(qwen_sigmoid(20.0f) > 0.999f);
    REQUIRE(std::fabs(qwen_silu(0.0f)) < 1e-6f);

    // softplus(0) = ln 2, and above 20 it passes x straight through so exp
    // cannot overflow.
    REQUIRE(std::fabs(qwen_softplus(0.0f) - std::log(2.0f)) < 1e-6f);
    REQUIRE(qwen_softplus(100.0f) == 100.0f);
    REQUIRE(std::isfinite(qwen_softplus(1000.0f)));
}

TEST_CASE("l2_normalize_heads normalizes each head separately", "[qwen35]") {
    std::vector<float> x{3.0f, 4.0f, 0.0f, 0.0f, 6.0f, 8.0f, 0.0f, 0.0f};
    l2_normalize_heads(x, 2, 4);
    // Each head had norm 5 and 10 respectively.
    REQUIRE(std::fabs(x[0] - 0.6f) < 1e-6f);
    REQUIRE(std::fabs(x[1] - 0.8f) < 1e-6f);
    REQUIRE(std::fabs(x[4] - 0.6f) < 1e-6f);
    REQUIRE(std::fabs(x[5] - 0.8f) < 1e-6f);

    // An all-zero head must not divide by zero.
    std::vector<float> zeros(4, 0.0f);
    l2_normalize_heads(zeros, 1, 4);
    for (float v : zeros) {
        REQUIRE(std::isfinite(v));
    }
}

TEST_CASE("apply_per_head_norm_raw uses the (1 + gamma) variant", "[qwen35]") {
    RmsNorm2 norm(4, 1e-6f);
    norm.gamma.set_data(Mat::zeros(1, 4));  // gamma = 0 -> scale of 1

    std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> original = x;
    apply_per_head_norm_raw(x, norm, 1, 4);

    float sum_sq = 0.0f;
    for (float v : original) {
        sum_sq += v * v;
    }
    const float rms = std::sqrt(sum_sq / 4.0f + 1e-6f);
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(std::fabs(x[i] - original[i] / rms) < 1e-5f);
    }
}

TEST_CASE("apply_partial_rope only touches the rotated prefix", "[qwen35]") {
    const std::size_t n_heads = 2, head_dim = 8, rope_dim = 4;
    std::vector<float> x(n_heads * head_dim);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i) + 1.0f;
    }
    const std::vector<float> original = x;

    apply_partial_rope(x, n_heads, head_dim, rope_dim, 10000.0f, 5);

    for (std::size_t h = 0; h < n_heads; ++h) {
        // Dimensions at or past rope_dim are untouched.
        for (std::size_t j = rope_dim; j < head_dim; ++j) {
            INFO("head " << h << " dim " << j);
            REQUIRE(x[h * head_dim + j] == original[h * head_dim + j]);
        }
        // The rotated prefix preserves its L2 norm.
        float before = 0.0f, after = 0.0f;
        for (std::size_t j = 0; j < rope_dim; ++j) {
            before += original[h * head_dim + j] * original[h * head_dim + j];
            after += x[h * head_dim + j] * x[h * head_dim + j];
        }
        REQUIRE(std::fabs(before - after) < 1e-3f);
    }

    // Position 0 is the identity rotation.
    std::vector<float> at_zero = original;
    apply_partial_rope(at_zero, n_heads, head_dim, rope_dim, 10000.0f, 0);
    REQUIRE(at_zero == original);
}

TEST_CASE("qwen gqa attention over an empty cache returns zeros", "[qwen35]") {
    const std::vector<float> q(8, 1.0f);
    const Mat empty = Mat::zeros(4, 8);
    const std::vector<float> out = qwen_gqa_attention_cached(q, empty, empty, 0, 0, 2, 2, 4, 1.0f);
    REQUIRE(out.size() == 8);
    for (float v : out) {
        REQUIRE(v == 0.0f);
    }
}

TEST_CASE("qwen gqa attention with one cached key returns that value", "[qwen35]") {
    // A single key means softmax gives it weight 1, so the output is V.
    const std::size_t nq = 2, nkv = 1, d = 4;
    Mat k = Mat::zeros(1, nkv * d);
    Mat v = Mat::from_fn(1, nkv * d, [](std::size_t, std::size_t c) {
        return static_cast<float>(c) + 1.0f;
    });
    const std::vector<float> q(nq * d, 0.5f);

    const std::vector<float> out = qwen_gqa_attention_cached(q, k, v, 0, 1, nq, nkv, d, 1.0f);
    for (std::size_t h = 0; h < nq; ++h) {
        for (std::size_t j = 0; j < d; ++j) {
            INFO("head " << h << " dim " << j);
            REQUIRE(std::fabs(out[h * d + j] - v.at(0, j)) < 1e-6f);
        }
    }
}

// -----------------------------------------------------------------------------
// Caches
// -----------------------------------------------------------------------------

TEST_CASE("qwen35 cache picks the right kind per layer", "[qwen35]") {
    const ConfigQwen35 cfg = tiny_cfg();
    Qwen35Cache cache(cfg, 32);
    REQUIRE(cache.layers.size() == cfg.num_hidden_layers);
    // Layers 0-2 are DeltaNet, layer 3 is full attention.
    REQUIRE(std::holds_alternative<DeltaNetState>(cache.layers[0]));
    REQUIRE(std::holds_alternative<DeltaNetState>(cache.layers[2]));
    REQUIRE(std::holds_alternative<FullAttnKvCache>(cache.layers[3]));

    const auto& state = std::get<DeltaNetState>(cache.layers[0]);
    REQUIRE(state.state.size() ==
            cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim);
    REQUIRE(state.conv_state.size() ==
            cfg.deltanet_qkv_dim() * (cfg.linear_conv_kernel_dim - 1));

    const auto& kv = std::get<FullAttnKvCache>(cache.layers[3]);
    REQUIRE(kv.k.rows == 32);
    REQUIRE(kv.k.cols == cfg.num_key_value_heads * cfg.head_dim);
    REQUIRE(kv.seq_len == 0);
}

TEST_CASE("deltanet conv1d shifts its history", "[qwen35]") {
    ConfigQwen35 cfg = tiny_cfg();
    const Qwen35DeltaNet dn = Qwen35DeltaNet::new_for_inference(cfg);
    DeltaNetState state(cfg);

    // With an all-zero kernel the output is silu(0) = 0, but the history must
    // still advance.
    const std::vector<float> input(dn.qkv_dim, 1.0f);
    const std::vector<float> out = dn.apply_conv1d(input, state);
    REQUIRE(out.size() == dn.qkv_dim);
    for (float v : out) {
        REQUIRE(std::fabs(v) < 1e-6f);
    }
    // The newest sample lands in the last history slot.
    const std::size_t hist = cfg.linear_conv_kernel_dim - 1;
    REQUIRE(state.conv_state[hist - 1] == 1.0f);
    REQUIRE(state.conv_state[0] == 0.0f);

    (void)dn.apply_conv1d(input, state);
    REQUIRE(state.conv_state[hist - 2] == 1.0f);
}

// -----------------------------------------------------------------------------
// Model
// -----------------------------------------------------------------------------

TEST_CASE("qwen35 inference model allocates no weights", "[qwen35]") {
    const Qwen35Model model = Qwen35Model::new_for_inference(ConfigQwen35::qwen35_0_8b());
    REQUIRE(model.layers.size() == 24);
    REQUIRE(model.embed_tokens.data().numel() == 0);
    // Layer 3 is the first full-attention layer.
    REQUIRE(std::holds_alternative<Qwen35DeltaNet>(model.layers[0].token_mixer));
    REQUIRE(std::holds_alternative<Qwen35FullAttention>(model.layers[3].token_mixer));
}

TEST_CASE("qwen35 prefill and decode produce finite logits", "[qwen35]") {
    const ConfigQwen35 cfg = tiny_cfg();
    Qwen35Model model = Qwen35Model::new_for_inference(cfg);
    fill_weights(model, 1234);

    const std::vector<std::size_t> prompt{3, 9, 17, 25, 40};
    Qwen35Cache cache(cfg, prompt.size() + 8);

    Mat logits = model.prefill(prompt, cache);
    REQUIRE(logits.rows == 1);
    REQUIRE(logits.cols == cfg.vocab_size);
    require_all_finite(logits, "prefill logits");

    // The DeltaNet state and the KV cache must both have advanced.
    REQUIRE(std::get<FullAttnKvCache>(cache.layers[3]).seq_len == prompt.size());
    bool state_nonzero = false;
    for (float v : std::get<DeltaNetState>(cache.layers[0]).state) {
        state_nonzero = state_nonzero || v != 0.0f;
    }
    REQUIRE(state_nonzero);

    for (int step = 0; step < 3; ++step) {
        logits = model.decode_step(7, cache);
        require_all_finite(logits, "decode logits");
    }
    REQUIRE(std::get<FullAttnKvCache>(cache.layers[3]).seq_len == prompt.size() + 3);
}

TEST_CASE("qwen35 token-by-token decode tracks batched prefill", "[qwen35]") {
    // Both paths drive the same recurrence, so the logits agree -- but only to
    // within float rounding: the reference associates the delta-rule multiply
    // differently in decode than in prefill, and that is reproduced here, so
    // the two are close rather than bit-identical.
    const ConfigQwen35 cfg = tiny_cfg();
    Qwen35Model model = Qwen35Model::new_for_inference(cfg);
    fill_weights(model, 4321);

    const std::vector<std::size_t> prompt{2, 11, 30, 44};

    Qwen35Cache batched(cfg, 16);
    const Mat from_prefill = model.prefill(prompt, batched);

    Qwen35Cache stepped(cfg, 16);
    Mat from_decode;
    for (std::size_t tok : prompt) {
        from_decode = model.decode_step(tok, stepped);
    }

    REQUIRE(from_decode.cols == from_prefill.cols);
    for (std::size_t c = 0; c < from_prefill.cols; ++c) {
        INFO("logit " << c);
        REQUIRE(std::fabs(from_decode.at(0, c) - from_prefill.at(0, c)) < 1e-3f);
    }
}

// -----------------------------------------------------------------------------
// Generation
// -----------------------------------------------------------------------------

TEST_CASE("qwen35 generate_cached_streaming produces tokens", "[qwen35]") {
    Qwen35Model model = Qwen35Model::new_for_inference(tiny_cfg());
    fill_weights(model, 21);

    std::vector<std::size_t> generated;
    model.generate_cached_streaming({0, 1, 2}, 5, 1.0f, 0, 1.0f, 1.0f, 42, false,
                                    [&](std::size_t tok) { generated.push_back(tok); });
    REQUIRE_FALSE(generated.empty());
    REQUIRE(generated.size() <= 5);
    for (std::size_t tok : generated) {
        REQUIRE(tok < model.config.vocab_size);
    }
}

// -----------------------------------------------------------------------------
// GGUF weight loading
// -----------------------------------------------------------------------------

namespace {

std::vector<float> qramp(std::size_t n, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>(i % 13) * step - 6.0f * step;
    }
    return v;
}

std::vector<std::uint8_t> qf32_bytes(const std::vector<float>& values) {
    std::vector<std::uint8_t> bytes(values.size() * 4);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

/// Write a GGUF holding one DeltaNet layer and one full-attention layer, using
/// llama.cpp's tensor names.
void write_qwen35_gguf(const std::string& path, const ConfigQwen35& cfg) {
    using rt::testing::GgufWriter;
    const std::size_t h = cfg.hidden_size;
    const std::size_t inter = cfg.intermediate_size;
    const std::size_t d = cfg.head_dim;
    const std::size_t nq = cfg.num_attention_heads;
    const std::size_t nkv = cfg.num_key_value_heads;
    const std::size_t qkv = cfg.deltanet_qkv_dim();
    const std::size_t nv = cfg.linear_num_value_heads;
    const std::size_t vdim = nv * cfg.linear_value_head_dim;

    GgufWriter w;
    w.add_meta_str("general.architecture", "qwen3next");
    // GGUF shapes are [in, out].
    w.add_tensor("token_embd.weight", {h, cfg.vocab_size}, GgufType::F32,
                 qf32_bytes(qramp(cfg.vocab_size * h, 0.01f)));
    w.add_tensor("output_norm.weight", {h}, GgufType::F32,
                 qf32_bytes(std::vector<float>(h, 1.0f)));

    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        w.add_tensor(p + "attn_norm.weight", {h}, GgufType::F32,
                     qf32_bytes(std::vector<float>(h, 1.0f)));
        w.add_tensor(p + "post_attention_norm.weight", {h}, GgufType::F32,
                     qf32_bytes(std::vector<float>(h, 1.0f)));
        w.add_tensor(p + "ffn_gate.weight", {h, inter}, GgufType::F32,
                     qf32_bytes(qramp(inter * h, 0.002f)));
        w.add_tensor(p + "ffn_up.weight", {h, inter}, GgufType::F32,
                     qf32_bytes(qramp(inter * h, 0.003f)));
        w.add_tensor(p + "ffn_down.weight", {inter, h}, GgufType::F32,
                     qf32_bytes(qramp(inter * h, 0.004f)));

        if (cfg.is_full_attention_layer(i)) {
            w.add_tensor(p + "attn_q.weight", {h, nq * d * 2}, GgufType::F32,
                         qf32_bytes(qramp(nq * d * 2 * h, 0.005f)));
            w.add_tensor(p + "attn_k.weight", {h, nkv * d}, GgufType::F32,
                         qf32_bytes(qramp(nkv * d * h, 0.006f)));
            w.add_tensor(p + "attn_v.weight", {h, nkv * d}, GgufType::F32,
                         qf32_bytes(qramp(nkv * d * h, 0.007f)));
            w.add_tensor(p + "attn_output.weight", {nq * d, h}, GgufType::F32,
                         qf32_bytes(qramp(nq * d * h, 0.008f)));
            w.add_tensor(p + "attn_q_norm.weight", {d}, GgufType::F32,
                         qf32_bytes(std::vector<float>(d, 1.0f)));
            w.add_tensor(p + "attn_k_norm.weight", {d}, GgufType::F32,
                         qf32_bytes(std::vector<float>(d, 1.0f)));
        } else {
            w.add_tensor(p + "attn_qkv.weight", {h, qkv}, GgufType::F32,
                         qf32_bytes(qramp(qkv * h, 0.005f)));
            w.add_tensor(p + "attn_gate.weight", {h, vdim}, GgufType::F32,
                         qf32_bytes(qramp(vdim * h, 0.006f)));
            w.add_tensor(p + "ssm_alpha.weight", {h, nv}, GgufType::F32,
                         qf32_bytes(qramp(nv * h, 0.007f)));
            w.add_tensor(p + "ssm_beta.weight", {h, nv}, GgufType::F32,
                         qf32_bytes(qramp(nv * h, 0.008f)));
            w.add_tensor(p + "ssm_out.weight", {vdim, h}, GgufType::F32,
                         qf32_bytes(qramp(vdim * h, 0.009f)));
            w.add_tensor(p + "ssm_a", {nv}, GgufType::F32,
                         qf32_bytes(std::vector<float>(nv, 0.5f)));
            w.add_tensor(p + "ssm_dt.bias", {nv}, GgufType::F32,
                         qf32_bytes(std::vector<float>(nv, 0.1f)));
            w.add_tensor(p + "ssm_norm.weight", {cfg.linear_value_head_dim}, GgufType::F32,
                         qf32_bytes(std::vector<float>(cfg.linear_value_head_dim, 1.0f)));
            w.add_tensor(p + "ssm_conv1d.weight", {qkv * cfg.linear_conv_kernel_dim},
                         GgufType::F32,
                         qf32_bytes(qramp(qkv * cfg.linear_conv_kernel_dim, 0.05f)));
        }
    }
    REQUIRE(w.write(path));
}

}  // namespace

TEST_CASE("qwen35 load_weights_from_gguf maps llama.cpp names", "[qwen35]") {
    const ConfigQwen35 cfg = tiny_cfg();
    const std::string path = "/tmp/rt_qwen35_load.gguf";
    write_qwen35_gguf(path, cfg);

    Qwen35Model model = Qwen35Model::new_for_inference(cfg);
    REQUIRE(model.load_weights_from_gguf(path).has_value());

    // The embedding loaded as BF16, and the tied head shares its bits.
    REQUIRE(model.embed_bf16.has_value());
    REQUIRE(model.embed_bf16->rows == cfg.vocab_size);
    REQUIRE(model.lm_head.bf16_weight.has_value());
    REQUIRE(model.lm_head.bf16_weight->data == model.embed_bf16->data);

    // GGUF norms are 1 + weight, so a file of ones becomes zeros.
    const Mat& gamma = model.layers[0].input_layernorm.gamma.data();
    for (std::size_t c = 0; c < gamma.cols; ++c) {
        REQUIRE(std::fabs(gamma.at(0, c)) < 1e-6f);
    }

    // attn_qkv is the DeltaNet fused projection, transposed into [out, in].
    const auto& dn = std::get<Qwen35DeltaNet>(model.layers[0].token_mixer);
    REQUIRE(dn.in_proj_qkv.weight.data().rows == cfg.deltanet_qkv_dim());
    REQUIRE(dn.in_proj_qkv.weight.data().cols == cfg.hidden_size);
    REQUIRE(dn.conv1d_weight.size() == cfg.deltanet_qkv_dim() * cfg.linear_conv_kernel_dim);
    REQUIRE(dn.a_log.size() == cfg.linear_num_value_heads);
    REQUIRE(dn.dt_bias.size() == cfg.linear_num_value_heads);
    // The gated norm weight is stored raw, so ones stay ones.
    REQUIRE(dn.norm_weight.size() == cfg.linear_value_head_dim);
    REQUIRE(dn.norm_weight[0] == 1.0f);

    // Layer 3 is the full-attention layer; q_proj carries query and gate.
    const auto& fa = std::get<Qwen35FullAttention>(model.layers[3].token_mixer);
    REQUIRE(fa.q_proj.weight.data().rows == cfg.num_attention_heads * cfg.head_dim * 2);
    REQUIRE(fa.o_proj.weight.data().rows == cfg.hidden_size);

    Qwen35Cache cache(cfg, 8);
    require_all_finite(model.prefill({1, 2, 3}, cache), "gguf prefill logits");
    require_all_finite(model.decode_step(4, cache), "gguf decode logits");

    std::remove(path.c_str());
}

TEST_CASE("qwen35 load_weights_from_dir rejects a directory with no shards", "[qwen35]") {
    Qwen35Model model = Qwen35Model::new_for_inference(tiny_cfg());
    REQUIRE_FALSE(model.load_weights_from_dir("/tmp/rt_no_such_qwen_dir").has_value());

    const std::string dir = "/tmp/rt_empty_qwen_dir";
    std::filesystem::create_directories(dir);
    const Result<void> empty = model.load_weights_from_dir(dir);
    REQUIRE_FALSE(empty.has_value());
    REQUIRE(empty.error().find("no .safetensors files") != std::string::npos);
    std::filesystem::remove_all(dir);
}

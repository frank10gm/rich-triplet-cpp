#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <unistd.h>

#include "rt/transformer3.hpp"

using namespace rt;

namespace {

/// A tiny GPT-OSS: real structure (GQA, alternating windows, MoE) at a size a
/// unit test can run.
Config3 tiny_cfg() {
    Config3 c;
    c.vocab_size = 16;
    c.hidden_size = 8;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;
    c.intermediate_size = 8;
    c.num_local_experts = 4;
    c.experts_per_token = 2;
    c.max_position_embeddings = 64;
    c.rope_theta = 150000.0f;
    c.rms_norm_eps = 1e-5f;
    c.swiglu_limit = 7.0f;
    c.sliding_window = 3;
    return c;
}

struct TempPath {
    std::filesystem::path path;
    explicit TempPath(const char* stem)
        : path(std::filesystem::temp_directory_path() /
               (std::string(stem) + std::to_string(::getpid()) + ".safetensors")) {}
    ~TempPath() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    [[nodiscard]] std::string str() const { return path.string(); }
};

/// Serialize one f32 tensor into a safetensors blob.
std::vector<std::uint8_t> make_safetensors(const std::string& name,
                                           const std::vector<std::size_t>& shape,
                                           const std::vector<float>& values) {
    std::string header = "{\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        header += (i ? "," : "") + std::to_string(shape[i]);
    }
    header += "],\"data_offsets\":[0," + std::to_string(values.size() * 4) + "]}}";

    std::vector<std::uint8_t> out;
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::uint8_t>((header.size() >> (8 * i)) & 0xff));
    }
    out.insert(out.end(), header.begin(), header.end());
    const auto* raw = reinterpret_cast<const std::uint8_t*>(values.data());
    out.insert(out.end(), raw, raw + values.size() * 4);
    return out;
}

void require_all_finite(const Mat& m) {
    for (float v : m.data) {
        REQUIRE(std::isfinite(v));
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

TEST_CASE("gpt-oss config sizes", "[transformer3]") {
    const Config3 c20 = Config3::gpt_oss_20b();
    REQUIRE(c20.hidden_size == 2880);
    REQUIRE(c20.num_hidden_layers == 24);
    REQUIRE(c20.num_attention_heads == 64);
    REQUIRE(c20.num_key_value_heads == 8);
    REQUIRE(c20.num_local_experts == 32);
    REQUIRE(c20.experts_per_token == 4);
    REQUIRE(c20.d_head() == 45);  // 2880 / 64
    REQUIRE(c20.sliding_window == 128u);

    const Config3 c120 = Config3::gpt_oss_120b();
    REQUIRE(c120.hidden_size == 7168);
    REQUIRE(c120.num_local_experts == 128);
    REQUIRE(c120.d_head() == 56);  // 7168 / 128
}

// -----------------------------------------------------------------------------
// Attention
// -----------------------------------------------------------------------------

TEST_CASE("gpt-oss attention shape, both window modes", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(1);
    const GptOssAttention full(cfg, rng, std::nullopt);
    const GptOssAttention windowed(cfg, rng, cfg.sliding_window);

    REQUIRE_FALSE(full.sliding_window.has_value());
    REQUIRE(windowed.sliding_window == 3u);
    // max_position_embeddings above 4096 would enable YaRN; the tiny config
    // stays below it.
    REQUIRE_FALSE(full.use_yarn);

    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(5, cfg.hidden_size, [&](std::size_t r, std::size_t c) {
            return static_cast<float>(r * cfg.hidden_size + c) * 0.05f - 0.3f;
        }));

    for (const GptOssAttention* attn : {&full, &windowed}) {
        const Mat out = attn->forward(x).data();
        REQUIRE(out.rows == 5);
        REQUIRE(out.cols == cfg.hidden_size);
        require_all_finite(out);
    }
    REQUIRE(full.parameters().size() == 8);  // four Linears, weight + bias each
}

TEST_CASE("gpt-oss rope is norm-preserving per head", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(2);
    const GptOssAttention attn(cfg, rng, std::nullopt);

    const std::size_t t = 3, n_heads = 2, dh = 4;
    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(t, n_heads * dh, [&](std::size_t r, std::size_t c) {
            return std::sin(static_cast<float>(r * 8 + c) * 0.3f);
        }));

    const Mat roped = attn.apply_rope_to_all_heads(x, n_heads, t, dh).data();
    for (std::size_t h = 0; h < n_heads; ++h) {
        for (std::size_t r = 0; r < t; ++r) {
            float before = 0.0f, after = 0.0f;
            for (std::size_t i = 0; i < dh; ++i) {
                const float b = x.data().at(r, h * dh + i);
                const float a = roped.at(r, h * dh + i);
                before += b * b;
                after += a * a;
            }
            INFO("head " << h << " row " << r);
            REQUIRE(std::fabs(std::sqrt(before) - std::sqrt(after)) < 1e-4f);
        }
    }

    // Row 0 is at position 0, where the rotation is the identity.
    for (std::size_t c = 0; c < n_heads * dh; ++c) {
        REQUIRE(std::fabs(roped.at(0, c) - x.data().at(0, c)) < 1e-6f);
    }
}

TEST_CASE("gpt-oss yarn engages past the trained context", "[transformer3]") {
    Config3 cfg = tiny_cfg();
    cfg.max_position_embeddings = 131072;
    InitRng rng(3);
    const GptOssAttention attn(cfg, rng, std::nullopt);
    REQUIRE(attn.use_yarn);
    REQUIRE(attn.original_ctx == 4096);

    // YaRN still leaves position 0 as the identity.
    const TensorNode x = TensorNode::leaf(Mat::from_fn(2, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.1f;
    }));
    const Mat roped = attn.apply_rope_to_all_heads(x, 2, 2, 4).data();
    for (std::size_t c = 0; c < 8; ++c) {
        REQUIRE(std::fabs(roped.at(0, c) - x.data().at(0, c)) < 1e-6f);
    }
    require_all_finite(roped);
}

// -----------------------------------------------------------------------------
// MoE
// -----------------------------------------------------------------------------

TEST_CASE("moe routes to exactly experts_per_token experts", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(4);
    const MoELayer moe(cfg, rng);

    REQUIRE(moe.experts.size() == cfg.num_local_experts);
    REQUIRE(moe.experts_per_token == 2);
    // router (weight + bias) plus 6 tensors per expert.
    REQUIRE(moe.parameters().size() == 2 + cfg.num_local_experts * 6);

    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(3, cfg.hidden_size, [&](std::size_t r, std::size_t c) {
            return static_cast<float>(r * cfg.hidden_size + c) * 0.07f - 0.2f;
        }));
    const Mat out = moe.forward(x).data();
    REQUIRE(out.rows == 3);
    REQUIRE(out.cols == cfg.hidden_size);
    require_all_finite(out);
}

TEST_CASE("moe with one expert reduces to that expert", "[transformer3]") {
    // With a single expert selected out of a single expert, the renormalized
    // weight is 1 and the output is exactly that expert's.
    Config3 cfg = tiny_cfg();
    cfg.num_local_experts = 1;
    cfg.experts_per_token = 1;
    InitRng rng(5);
    const MoELayer moe(cfg, rng);

    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(1, cfg.hidden_size, [&](std::size_t, std::size_t c) {
            return static_cast<float>(c) * 0.1f - 0.4f;
        }));
    const Mat moe_out = moe.forward(x).data();
    const Mat direct = moe.experts[0].forward(x).data();
    for (std::size_t c = 0; c < cfg.hidden_size; ++c) {
        INFO("dim " << c);
        REQUIRE(std::fabs(moe_out.at(0, c) - direct.at(0, c)) < 1e-5f);
    }
}

// -----------------------------------------------------------------------------
// Block and model
// -----------------------------------------------------------------------------

TEST_CASE("gpt-oss block alternates attention type by index", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(6);
    const GptOssBlock even(cfg, 0, rng);
    const GptOssBlock odd(cfg, 1, rng);
    REQUIRE_FALSE(even.self_attn.sliding_window.has_value());
    REQUIRE(odd.self_attn.sliding_window == 3u);

    const TensorNode x = TensorNode::leaf(
        Mat::from_fn(4, cfg.hidden_size, [](std::size_t r, std::size_t c) {
            return static_cast<float>(r + c) * 0.05f;
        }));
    const Mat out = even.forward(x).data();
    REQUIRE(out.rows == 4);
    REQUIRE(out.cols == cfg.hidden_size);
}

TEST_CASE("gpt-oss model forward, loss and backward", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(7);
    const GptOssModel model(cfg, rng);

    const Mat logits = model.forward({1, 2, 3}).data();
    REQUIRE(logits.rows == 3);
    REQUIRE(logits.cols == cfg.vocab_size);
    require_all_finite(logits);

    const TensorNode loss = model.loss({1, 2, 3}, {2, 3, 4});
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

    // predict_next is the argmax of the last logits row.
    std::size_t expected = 0;
    for (std::size_t c = 1; c < logits.cols; ++c) {
        if (logits.at(2, expected) < logits.at(2, c)) {
            expected = c;
        }
    }
    REQUIRE(model.predict_next({1, 2, 3}) == expected);
}

TEST_CASE("gpt-oss forward_batch runs each sequence", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(8);
    const GptOssModel model(cfg, rng);

    const auto outs = model.forward_batch({{1, 2}, {3, 4, 5}});
    REQUIRE(outs.size() == 2);
    REQUIRE(outs[0].data().rows == 2);
    REQUIRE(outs[1].data().rows == 3);
}

TEST_CASE("gpt-oss tie_weights shares one node", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(9);
    GptOssModel model(cfg, rng);
    REQUIRE(model.lm_head.weight.id() != model.embed_tokens.id());
    model.tie_weights();
    REQUIRE(model.lm_head.weight.id() == model.embed_tokens.id());
}

TEST_CASE("gpt-oss cached generation matches the uncached forward", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(10);
    const GptOssModel model(cfg, rng);
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

    // Streaming yields the same first token and the requested count.
    std::vector<std::size_t> streamed;
    model.generate_cached_streaming(prompt, 3, 0.0f,
                                    [&](std::size_t t) { streamed.push_back(t); });
    REQUIRE(streamed.size() == 3);
    REQUIRE(streamed[0] == expected);
}

TEST_CASE("gpt-oss generate_with_params honours eos and length", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(11);
    const GptOssModel model(cfg, rng);

    SamplingParams params = SamplingParams::greedy();
    const std::vector<std::size_t> greedy = model.generate_with_params({1, 2}, 4, params);
    REQUIRE(greedy.size() <= 4);
    REQUIRE_FALSE(greedy.empty());

    // Setting EOS to the first token the model would emit stops it immediately.
    params.eos_token_id = greedy[0];
    const std::vector<std::size_t> stopped = model.generate_with_params({1, 2}, 4, params);
    REQUIRE(stopped.size() == 1);
    REQUIRE(stopped[0] == greedy[0]);
}

TEST_CASE("gpt-oss quantizes its weight matrices", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(12);
    const GptOssModel model(cfg, rng);

    const Q4QuantStats stats = model.quantize_all_linear_weights();
    REQUIRE(stats.n_tensors > 0);
    REQUIRE(stats.q4_bytes < stats.f32_bytes);
    REQUIRE(stats.compression_ratio > 4.0f);
    // The model still runs, on the dequantized approximation.
    require_all_finite(model.forward({1, 2}).data());
}

// -----------------------------------------------------------------------------
// Sampling
// -----------------------------------------------------------------------------

TEST_CASE("sample_token_full penalties and filters", "[transformer3]") {
    const Mat logits({1.0f, 4.0f, 2.0f, 3.0f}, 1, 4);
    LcgRng53 rng(0);

    // Greedy.
    REQUIRE(sample_token_full(logits, 0, SamplingParams::greedy(), {}, rng) == 1);

    // The repetition penalty divides, so a seen token loses to the runner-up.
    SamplingParams rep = SamplingParams::greedy();
    rep.repetition_penalty = 2.0f;
    REQUIRE(sample_token_full(logits, 0, rep, {1}, rng) == 3);

    // The frequency penalty scales with the count.
    SamplingParams freq = SamplingParams::greedy();
    freq.frequency_penalty = 1.0f;
    REQUIRE(sample_token_full(logits, 0, freq, {1, 1}, rng) == 3);

    // The presence penalty is a flat subtraction.
    SamplingParams pres = SamplingParams::greedy();
    pres.presence_penalty = 2.0f;
    REQUIRE(sample_token_full(logits, 0, pres, {1}, rng) == 3);

    // top_k = 1 leaves only the argmax.
    SamplingParams topk;
    topk.temperature = 1.0f;
    topk.top_k = 1;
    for (int i = 0; i < 5; ++i) {
        REQUIRE(sample_token_full(logits, 0, topk, {}, rng) == 1);
    }

    // A tiny top_p keeps only the most likely token.
    SamplingParams topp;
    topp.temperature = 1.0f;
    topp.top_p = 0.01f;
    for (int i = 0; i < 5; ++i) {
        REQUIRE(sample_token_full(logits, 0, topp, {}, rng) == 1);
    }
}

// -----------------------------------------------------------------------------
// safetensors
// -----------------------------------------------------------------------------

TEST_CASE("safetensors parses an f32 tensor", "[transformer3]") {
    const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const auto blob = make_safetensors("model.embed_tokens.weight", {2, 3}, values);

    const auto tensors = parse_safetensors(blob);
    REQUIRE(tensors.has_value());
    REQUIRE(tensors->size() == 1);
    REQUIRE((*tensors)[0].name == "model.embed_tokens.weight");
    REQUIRE((*tensors)[0].shape == std::vector<std::size_t>{2, 3});
    REQUIRE((*tensors)[0].data == values);
    REQUIRE_FALSE((*tensors)[0].bf16_data.has_value());
}

TEST_CASE("safetensors handles BF16 and the skip-conversion path", "[transformer3]") {
    const std::vector<float> values{1.0f, -2.0f, 0.5f, 8.0f};
    std::vector<std::uint8_t> raw;
    for (float v : values) {
        const std::uint16_t bits = f32_to_bf16(v);
        raw.push_back(static_cast<std::uint8_t>(bits & 0xff));
        raw.push_back(static_cast<std::uint8_t>(bits >> 8));
    }

    std::string header = "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0," +
                         std::to_string(raw.size()) + "]}}";
    std::vector<std::uint8_t> blob;
    for (int i = 0; i < 8; ++i) {
        blob.push_back(static_cast<std::uint8_t>((header.size() >> (8 * i)) & 0xff));
    }
    blob.insert(blob.end(), header.begin(), header.end());
    blob.insert(blob.end(), raw.begin(), raw.end());

    const auto converted = parse_safetensors(blob);
    REQUIRE(converted.has_value());
    REQUIRE((*converted)[0].bf16_data.has_value());
    REQUIRE((*converted)[0].data.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(std::fabs((*converted)[0].data[i] - values[i]) < 0.05f);
    }

    // The skip variant keeps the bits but leaves the f32 form empty.
    const auto skipped = parse_safetensors_skip_bf16_f32(blob);
    REQUIRE(skipped.has_value());
    REQUIRE((*skipped)[0].data.empty());
    REQUIRE((*skipped)[0].bf16_data->size() == 4);
    REQUIRE((*skipped)[0].bf16_data == (*converted)[0].bf16_data);
}

TEST_CASE("safetensors rejects malformed blobs", "[transformer3]") {
    REQUIRE_FALSE(parse_safetensors({1, 2, 3}).has_value());

    // A header length past the end of the file.
    std::vector<std::uint8_t> bad(16, 0);
    bad[0] = 0xff;
    REQUIRE_FALSE(parse_safetensors(bad).has_value());
}

TEST_CASE("safetensors header and single-tensor read", "[transformer3]") {
    TempPath tmp("rt_st_");
    const std::vector<float> values{1.5f, 2.5f, 3.5f, 4.5f};
    {
        const auto blob = make_safetensors("w", {2, 2}, values);
        std::ofstream f(tmp.str(), std::ios::binary);
        f.write(reinterpret_cast<const char*>(blob.data()),
                static_cast<std::streamsize>(blob.size()));
    }

    const auto header = parse_safetensors_header(tmp.str());
    REQUIRE(header.has_value());
    const auto& [data_start, entries] = *header;
    REQUIRE(entries.size() == 1);
    REQUIRE(entries[0].name == "w");
    REQUIRE(entries[0].dtype == "F32");
    REQUIRE(entries[0].shape == std::vector<std::size_t>{2, 2});
    REQUIRE(entries[0].byte_end - entries[0].byte_start == 16);

    const auto tensor = read_safetensor_from_file(tmp.str(), data_start, entries[0], false);
    REQUIRE(tensor.has_value());
    REQUIRE(tensor->data == values);

    REQUIRE_FALSE(parse_safetensors_header("/definitely/not/here.safetensors").has_value());
}

TEST_CASE("load_into_model applies tensors by name", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(13);
    GptOssModel model(cfg, rng);

    std::vector<SafeTensor> tensors;
    // The embedding table.
    tensors.push_back({"model.embed_tokens.weight",
                       {cfg.vocab_size, cfg.hidden_size},
                       std::vector<float>(cfg.vocab_size * cfg.hidden_size, 0.25f),
                       std::nullopt});
    // A layer norm.
    tensors.push_back({"model.layers.0.input_layernorm.weight",
                       {cfg.hidden_size},
                       std::vector<float>(cfg.hidden_size, 2.0f),
                       std::nullopt});
    // An expert projection.
    tensors.push_back({"model.layers.1.mlp.experts.2.up_proj.weight",
                       {cfg.intermediate_size, cfg.hidden_size},
                       std::vector<float>(cfg.intermediate_size * cfg.hidden_size, 0.5f),
                       std::nullopt});
    // Names the model does not know are skipped rather than failing.
    tensors.push_back({"model.rotary_emb.inv_freq", {4}, {1, 2, 3, 4}, std::nullopt});
    tensors.push_back({"model.layers.99.input_layernorm.weight", {cfg.hidden_size},
                       std::vector<float>(cfg.hidden_size, 7.0f), std::nullopt});

    load_into_model(model, tensors);

    REQUIRE(model.embed_tokens.data().at(0, 0) == 0.25f);
    REQUIRE(model.layers[0].input_layernorm.gamma.data().at(0, 0) == 2.0f);
    REQUIRE(model.layers[1].mlp.experts[2].up_proj.weight.data().at(0, 0) == 0.5f);
    // Layer 1's norm was never touched.
    REQUIRE(model.layers[1].input_layernorm.gamma.data().at(0, 0) == 1.0f);
}

TEST_CASE("load_weights_from_dir reports an empty directory", "[transformer3]") {
    const Config3 cfg = tiny_cfg();
    InitRng rng(14);
    GptOssModel model(cfg, rng);

    REQUIRE_FALSE(model.load_weights_from_dir("/definitely/not/here").has_value());

    const auto empty_dir = std::filesystem::temp_directory_path() /
                           ("rt_st_empty_" + std::to_string(::getpid()));
    std::filesystem::create_directory(empty_dir);
    const auto result = model.load_weights_from_dir(empty_dir.string());
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().find("no .safetensors files") != std::string::npos);
    std::filesystem::remove(empty_dir);
}

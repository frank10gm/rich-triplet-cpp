// =============================================================================
// Qwen 3.5 -- weight loading and the generation loop
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "rt/transformer3.hpp"
#include "rt/transformer4.hpp"
#include "rt/transformer_qwen35.hpp"

#if RT_FEATURE_METAL
#include "rt/metal_decode_qwen35.hpp"
#endif

namespace rt {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// Qwen 3.5's second stop token, alongside `config.eos_token_id`.
constexpr std::size_t kImEnd = 248046;

/// Root-mean-square of one row, for tracking how the residual stream grows.
[[nodiscard]] float row_rms(const Mat& m, std::size_t row) {
    double sum = 0.0;
    for (std::size_t c = 0; c < m.cols; ++c) {
        const double v = static_cast<double>(m.at(row, c));
        sum += v * v;
    }
    return m.cols == 0 ? 0.0f
                       : static_cast<float>(std::sqrt(sum / static_cast<double>(m.cols)));
}

void print_top5(const char* prefix, const Mat& logits) {
    std::vector<std::pair<std::size_t, float>> indexed(logits.data.size());
    for (std::size_t i = 0; i < logits.data.size(); ++i) {
        indexed[i] = {i, logits.data[i]};
    }
    const std::size_t k = std::min<std::size_t>(5, indexed.size());
    std::partial_sort(indexed.begin(), indexed.begin() + static_cast<std::ptrdiff_t>(k),
                      indexed.end(), [](const auto& a, const auto& b) {
                          // total_cmp ordering: NaNs sort last rather than
                          // poisoning the comparison.
                          return b.second < a.second;
                      });
    std::fprintf(stderr, "%s", prefix);
    for (std::size_t i = 0; i < k; ++i) {
        std::fprintf(stderr, "%s(%zu, %.4f)", i == 0 ? "[" : ", ", indexed[i].first,
                     static_cast<double>(indexed[i].second));
    }
    std::fprintf(stderr, "]\n");
}

}  // namespace

std::vector<float> Qwen35Model::embed_token(std::size_t tok) const {
    const std::size_t h = config.hidden_size;
    std::vector<float> out(h);
    if (embed_bf16) {
        const std::vector<std::uint16_t>& bits = *embed_bf16->data;
        for (std::size_t c = 0; c < h; ++c) {
            out[c] = bf16_to_f32(bits[tok * h + c]);
        }
    } else {
        const Mat& te = embed_tokens.data();
        for (std::size_t c = 0; c < h; ++c) {
            out[c] = te.at(tok, c);
        }
    }
    return out;
}

// =============================================================================
// Generation
// =============================================================================

void Qwen35Model::generate_cached_streaming(const std::vector<std::size_t>& token_ids,
                                            std::size_t max_new, float temperature,
                                            std::size_t top_k, float top_p,
                                            float repetition_penalty, std::uint64_t seed,
                                            bool debug,
                                            const std::function<void(std::size_t)>& callback) {
    SamplingParams params;
    params.temperature = temperature;
    params.top_k = top_k;
    params.top_p = top_p;
    params.repetition_penalty = repetition_penalty;
    params.seed = seed;
    params.eos_token_id = config.eos_token_id;

    const std::size_t h = config.hidden_size;
    LcgRng rng(seed);
    std::vector<std::size_t> seen;

    Qwen35Cache cache(config, token_ids.size() + max_new);

    // ----- Prefill: the whole prompt in one batched pass -----
    const Clock::time_point prefill_start = Clock::now();
    const std::size_t t_len = token_ids.size();

    std::vector<float> embed_data;
    embed_data.reserve(t_len * h);
    for (std::size_t tok : token_ids) {
        const std::vector<float> row = embed_token(tok);
        embed_data.insert(embed_data.end(), row.begin(), row.end());
    }
    TensorNode x = TensorNode::leaf(Mat(std::move(embed_data), t_len, h));

    if (debug) {
        const Mat& xd = x.data();
        std::fprintf(stderr, "[ Qwen3.5-dbg ] embed rms=%.4f first5=[", 
                     static_cast<double>(row_rms(xd, t_len - 1)));
        for (std::size_t c = 0; c < std::min<std::size_t>(5, h); ++c) {
            std::fprintf(stderr, "%s%.4f", c == 0 ? "" : ", ",
                         static_cast<double>(xd.at(t_len - 1, c)));
        }
        std::fprintf(stderr, "]\n");
    }

    for (std::size_t i = 0; i < layers.size(); ++i) {
        x = layers[i].forward_prefill(x, cache.layers[i]);
        if (debug && (i < 3 || i == layers.size() - 1)) {
            std::fprintf(stderr, "[ Qwen3.5-dbg ] layer %zu (%s) h_rms=%.4f\n", i,
                         config.is_full_attention_layer(i) ? "full" : "delta",
                         static_cast<double>(row_rms(x.data(), t_len - 1)));
        }
    }

    // Only the last row feeds lm_head.
    const Mat last_row =
        Mat::from_fn(1, h, [&](std::size_t, std::size_t c) { return x.data().at(t_len - 1, c); });
    const TensorNode normed = norm.forward_gemma3(TensorNode::leaf(last_row));
    const Mat logits = lm_head.forward(normed).data();

    const double prefill_ms = ms_since(prefill_start);
    std::fprintf(stderr, "[ Qwen3.5 ] Prefill: %zu tokens in %.0f ms (%.1f tok/s)\n", t_len,
                 prefill_ms, static_cast<double>(t_len) / (prefill_ms / 1000.0));

    if (debug) {
        print_top5("[ Qwen3.5-dbg ] prefill top-5: ", logits);
    }

    std::size_t first = sample_token(logits, 0, params, seen, rng);
    callback(first);
    seen.push_back(first);
    if (first == config.eos_token_id || first == kImEnd) {
        return;
    }

#if RT_FEATURE_METAL
    constexpr bool use_metal = true;
    std::unique_ptr<MetalDecodeContextQwen35> metal_ctx = MetalDecodeContextQwen35::create(*this);
    metal_ctx->print_memory_stats();
    {
        // Metal JIT-compiles the shaders on the first dispatch; a dummy step
        // pays that cost before the real decode loop starts.
        const Clock::time_point t_warmup = Clock::now();
        (void)metal_ctx->decode_step(0, 0);
        std::fprintf(stderr, "[ Qwen3.5-Metal ] GPU warmup in %.0f ms\n", ms_since(t_warmup));
    }
    metal_ctx->sync_state_from_cpu(cache);
#else
    constexpr bool use_metal = false;
#endif

    const Clock::time_point decode_start = Clock::now();
    std::size_t prev = first;
    std::size_t n_decoded = 1;
    [[maybe_unused]] std::size_t metal_seq_len = t_len;

    for (std::size_t step = 1; step < max_new; ++step) {
        Mat step_logits(std::vector<float>{}, 0, 0);
#if RT_FEATURE_METAL
        {
            std::vector<float> logits_vec = metal_ctx->decode_step(prev, metal_seq_len);
            ++metal_seq_len;
            const std::size_t n = logits_vec.size();
            step_logits = Mat(std::move(logits_vec), 1, n);
        }
#else
        {
            TensorNode xs = TensorNode::leaf(Mat(embed_token(prev), 1, h));
            for (std::size_t i = 0; i < layers.size(); ++i) {
                xs = layers[i].forward_cached(xs, cache.layers[i]);
            }
            step_logits = lm_head.forward(norm.forward_gemma3(xs)).data();
        }
#endif

        prev = sample_token(step_logits, 0, params, seen, rng);
        callback(prev);
        seen.push_back(prev);
        ++n_decoded;

        if (prev == config.eos_token_id || prev == kImEnd) {
            break;
        }
    }

    const double decode_ms = ms_since(decode_start);
    std::fprintf(stderr, "[ Qwen3.5 ] Decode (%s): %zu tokens in %.0f ms (%.1f tok/s, %.0f ms/tok)\n",
                 use_metal ? "Metal" : "CPU", n_decoded, decode_ms,
                 static_cast<double>(n_decoded) / (decode_ms / 1000.0),
                 decode_ms / static_cast<double>(n_decoded));
}

// =============================================================================
// GGUF weight loading
// =============================================================================

Result<void> Qwen35Model::load_weights_from_gguf(const std::string& path) {
    std::fprintf(stderr, "[ GGUF ] Opening %s...\n", path.c_str());
    RT_TRY(gguf, GgufFile::open(path));
    std::fprintf(stderr, "[ GGUF ] Found %zu tensors.\n", gguf.tensor_info.size());

    if (const auto arch = gguf.metadata.find("general.architecture"); arch != gguf.metadata.end()) {
        if (const auto s = arch->second.as_str()) {
            std::fprintf(stderr, "[ GGUF ] Architecture: %.*s\n", static_cast<int>(s->size()),
                         s->data());
        }
    }

    const std::size_t n_tensors = gguf.tensor_info.size();
    std::size_t loaded = 0;
    bool lm_head_explicitly_loaded = false;

    /// The small tensors, as f32. Q4_K_M stores the DeltaNet scalars as Q8_0
    /// and some norms as Q6_K, so both decode here too.
    const auto load_f32 = [&gguf](std::size_t i) -> Result<std::vector<float>> {
        switch (gguf.tensor_info[i].gguf_type) {
            case GgufType::F32:
                return gguf.decode_f32(i);
            case GgufType::F16:
                return gguf.decode_f16_to_f32(i);
            case GgufType::Q8_0:
                return gguf.decode_q8_0_to_f32(i);
            case GgufType::Q6K:
                return gguf.decode_q6k_to_f32(i);
            default:
                return err("expected f32/f16/q8_0 for " + gguf.tensor_info[i].name);
        }
    };

    /// GGUF stores norm gammas as `1 + HF_weight`; `forward_gemma3` adds the 1
    /// back, so it comes off here.
    const auto set_norm = [&](std::size_t i, const TensorNode& gamma) -> Result<void> {
        RT_TRY(f32s, load_f32(i));
        for (float& v : f32s) {
            v -= 1.0f;
        }
        const std::size_t n = f32s.size();
        gamma.set_data(Mat(std::move(f32s), 1, n));
        return {};
    };

    for (std::size_t idx = 0; idx < n_tensors; ++idx) {
        const std::string name = gguf.tensor_info[idx].name;
        const GgufType gtype = gguf.tensor_info[idx].gguf_type;

        // ---- token embedding ----
        if (name == "token_embd.weight") {
            const std::vector<std::size_t>& shape = gguf.tensor_info[idx].shape;
            const std::size_t vocab = shape.size() == 2 ? shape[1] : shape[0];
            const std::size_t hidden = shape.size() == 2 ? shape[0] : 1;
            std::fprintf(stderr, "[ GGUF ] token_embd: type=%s vocab=%zu hidden=%zu\n",
                         gguf_type_name(gtype), vocab, hidden);

            std::vector<std::uint16_t> bits;
            switch (gtype) {
                case GgufType::Bf16: {
                    RT_TRY(raw, gguf.decode_bf16(idx));
                    bits = std::move(raw);
                    break;
                }
                case GgufType::F16: {
                    RT_TRY(f32s, gguf.decode_f16_to_f32(idx));
                    bits = f32s_to_bf16_and_drop(std::move(f32s));
                    break;
                }
                case GgufType::F32: {
                    RT_TRY(f32s, gguf.decode_f32(idx));
                    bits = f32s_to_bf16_and_drop(std::move(f32s));
                    break;
                }
                case GgufType::Q4K: {
                    RT_TRY(f32s, gguf.decode_q4k_to_f32(idx));
                    bits = f32s_to_bf16_and_drop(std::move(f32s));
                    break;
                }
                case GgufType::Q6K: {
                    RT_TRY(f32s, gguf.decode_q6k_to_f32(idx));
                    bits = f32s_to_bf16_and_drop(std::move(f32s));
                    break;
                }
                case GgufType::Q4_0: {
                    RT_TRY(f32s, gguf.decode_q4_0_to_f32(idx));
                    bits = f32s_to_bf16_and_drop(std::move(f32s));
                    break;
                }
                default:
                    std::fprintf(stderr, "[ GGUF ] Warning: unsupported embed type %s\n",
                                 gguf_type_name(gtype));
                    continue;
            }
            model_set_embed_bf16_raw(embed_tokens, embed_bf16, lm_head, std::move(bits), vocab,
                                     hidden, config.tie_word_embeddings);
            ++loaded;
            continue;
        }

        // ---- lm_head, for non-tied models ----
        if (name == "output.weight") {
            std::fprintf(stderr, "[ GGUF ] Loading explicit output.weight for lm_head (type=%s)\n",
                         gguf_type_name(gtype));
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, lm_head));
            lm_head_explicitly_loaded = true;
            ++loaded;
            continue;
        }

        if (name == "output_norm.weight") {
            RT_TRY_VOID(set_norm(idx, norm.gamma));
            ++loaded;
            continue;
        }

        // ---- per-layer tensors: blk.{i}.* ----
        constexpr std::string_view kBlkPrefix = "blk.";
        if (!name.starts_with(kBlkPrefix)) {
            continue;
        }
        const std::string rest = name.substr(kBlkPrefix.size());
        const std::size_t dot = rest.find('.');
        if (dot == std::string::npos) {
            continue;
        }
        std::size_t layer_idx = 0;
        try {
            layer_idx = static_cast<std::size_t>(std::stoul(rest.substr(0, dot)));
        } catch (const std::exception&) {
            continue;
        }
        if (layer_idx >= layers.size()) {
            continue;
        }
        const std::string field = rest.substr(dot + 1);
        Qwen35Block& layer = layers[layer_idx];
        auto* fa = std::get_if<Qwen35FullAttention>(&layer.token_mixer);
        auto* dn = std::get_if<Qwen35DeltaNet>(&layer.token_mixer);

        // ---- layer norms and the MLP, present on every layer ----
        if (field == "attn_norm.weight") {
            RT_TRY_VOID(set_norm(idx, layer.input_layernorm.gamma));
            ++loaded;
        } else if (field == "ffn_norm.weight" || field == "post_attention_norm.weight") {
            RT_TRY_VOID(set_norm(idx, layer.post_attention_layernorm.gamma));
            ++loaded;
        } else if (field == "ffn_gate.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.mlp.gate_proj));
            ++loaded;
        } else if (field == "ffn_up.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.mlp.up_proj));
            ++loaded;
        } else if (field == "ffn_down.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.mlp.down_proj));
            ++loaded;

            // ---- full-attention layers ----
        } else if (field == "attn_q.weight") {
            if (fa != nullptr) RT_TRY_VOID(load_linear_from_gguf(gguf, idx, fa->q_proj));
            ++loaded;
        } else if (field == "attn_k.weight") {
            if (fa != nullptr) RT_TRY_VOID(load_linear_from_gguf(gguf, idx, fa->k_proj));
            ++loaded;
        } else if (field == "attn_v.weight") {
            if (fa != nullptr) RT_TRY_VOID(load_linear_from_gguf(gguf, idx, fa->v_proj));
            ++loaded;
        } else if (field == "attn_output.weight") {
            if (fa != nullptr) RT_TRY_VOID(load_linear_from_gguf(gguf, idx, fa->o_proj));
            ++loaded;
        } else if (field == "attn_q_norm.weight") {
            if (fa != nullptr) RT_TRY_VOID(set_norm(idx, fa->q_norm.gamma));
            ++loaded;
        } else if (field == "attn_k_norm.weight") {
            if (fa != nullptr) RT_TRY_VOID(set_norm(idx, fa->k_norm.gamma));
            ++loaded;

            // ---- DeltaNet layers ----
            //
            // llama.cpp names these after the state-space family: attn_qkv is
            // the fused QKV projection and attn_gate the output gate.
        } else if (field == "ssm_in.weight" || field == "attn_qkv.weight") {
            if (dn != nullptr) RT_TRY_VOID(load_linear_from_gguf(gguf, idx, dn->in_proj_qkv));
            ++loaded;
        } else if (field == "ssm_gate.weight" || field == "attn_gate.weight") {
            if (dn != nullptr) RT_TRY_VOID(load_linear_from_gguf(gguf, idx, dn->in_proj_z));
            ++loaded;
        } else if (field == "ssm_out.weight") {
            if (dn != nullptr) RT_TRY_VOID(load_linear_from_gguf(gguf, idx, dn->out_proj));
            ++loaded;
        } else if (field == "ssm_alpha.weight") {
            if (dn != nullptr) RT_TRY_VOID(load_linear_from_gguf(gguf, idx, dn->in_proj_a));
            ++loaded;
        } else if (field == "ssm_beta.weight") {
            if (dn != nullptr) RT_TRY_VOID(load_linear_from_gguf(gguf, idx, dn->in_proj_b));
            ++loaded;
        } else if (field == "ssm_a.weight" || field == "ssm_a") {
            if (dn != nullptr) {
                RT_TRY(f32s, load_f32(idx));
                dn->a_log = std::move(f32s);
            }
            ++loaded;
        } else if (field == "ssm_dt.bias") {
            if (dn != nullptr) {
                RT_TRY(f32s, load_f32(idx));
                dn->dt_bias = std::move(f32s);
            }
            ++loaded;
        } else if (field == "ssm_norm.weight") {
            if (dn != nullptr) {
                // The gated RMSNorm multiplies by the weight directly (it is
                // initialized to 1), so unlike the other norms this one is
                // stored raw and needs no adjustment.
                RT_TRY(f32s, load_f32(idx));
                dn->norm_weight = std::move(f32s);
            }
            ++loaded;
        } else if (field == "ssm_conv1d.weight") {
            if (dn != nullptr) {
                // GGUF shape is [qkv_dim, 1, kernel_size]; we keep it flat.
                RT_TRY(f32s, load_f32(idx));
                dn->conv1d_weight = std::move(f32s);
            }
            ++loaded;
        } else {
            std::fprintf(stderr, "[ GGUF ] Unknown blk tensor: blk.%zu.%s\n", layer_idx,
                         field.c_str());
        }
    }

    if (config.tie_word_embeddings && !lm_head_explicitly_loaded && embed_bf16) {
        lm_head.load_bf16_shared(embed_bf16->data, embed_bf16->rows, embed_bf16->cols);
        std::fprintf(stderr, "[ GGUF ] lm_head weight-tied to embed_tokens (BF16)\n");
    }

    std::fprintf(stderr, "[ GGUF ] Loaded %zu / %zu tensors for Qwen3.5-%s\n", loaded, n_tensors,
                 config.hidden_size == 2560 ? "4B" : "9B");
    return {};
}

// =============================================================================
// safetensors weight loading
// =============================================================================

bool Qwen35Model::apply_safetensor(SafeTensor t) {
    // Text-only checkpoints use "model."; multimodal ones nest the text tower
    // under "model.language_model.".
    std::string inner;
    if (t.name.starts_with("model.language_model.")) {
        inner = t.name.substr(std::string_view("model.language_model.").size());
    } else if (t.name.starts_with("model.")) {
        inner = t.name.substr(std::string_view("model.").size());
    } else {
        return false;
    }

    const auto get_f32 = [](const SafeTensor& st) {
        if (st.bf16_data) {
            std::vector<float> out(st.bf16_data->size());
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] = bf16_to_f32((*st.bf16_data)[i]);
            }
            return out;
        }
        return st.data;
    };

    const auto set_linear = [](Linear2& linear, SafeTensor st) {
        const std::size_t rows = st.shape.size() >= 2 ? st.shape[0] : 1;
        const std::size_t cols = st.shape.size() >= 2 ? st.shape[1] : st.shape[0];
        if (st.bf16_data) {
            linear.load_bf16(std::move(*st.bf16_data), rows, cols);
        } else {
            linear.weight.set_data(Mat(std::move(st.data), rows, cols));
        }
    };

    if (inner == "embed_tokens.weight") {
        const std::size_t vocab = t.shape[0];
        const std::size_t hidden = t.shape[1];
        std::fprintf(stderr, "[ Safetensors ] embed_tokens: vocab=%zu hidden=%zu\n", vocab,
                     hidden);
        std::vector<std::uint16_t> bits = t.bf16_data ? std::move(*t.bf16_data)
                                                      : f32s_to_bf16_and_drop(std::move(t.data));
        model_set_embed_bf16_raw(embed_tokens, embed_bf16, lm_head, std::move(bits), vocab, hidden,
                                 config.tie_word_embeddings);
        return true;
    }

    if (inner == "norm.weight") {
        // safetensors stores gamma raw and zero-initialized, and
        // `forward_gemma3` applies (1 + gamma), so nothing is subtracted here.
        std::vector<float> f32s = get_f32(t);
        const std::size_t n = f32s.size();
        norm.gamma.set_data(Mat(std::move(f32s), 1, n));
        return true;
    }

    if (inner == "lm_head.weight") {
        set_linear(lm_head, std::move(t));
        return true;
    }

    constexpr std::string_view kLayerPrefix = "layers.";
    if (!inner.starts_with(kLayerPrefix)) {
        return false;
    }
    const std::string rest = inner.substr(kLayerPrefix.size());
    const std::size_t dot = rest.find('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::size_t layer_idx = 0;
    try {
        layer_idx = static_cast<std::size_t>(std::stoul(rest.substr(0, dot)));
    } catch (const std::exception&) {
        return false;
    }
    if (layer_idx >= layers.size()) {
        return false;
    }
    const std::string field = rest.substr(dot + 1);
    Qwen35Block& layer = layers[layer_idx];
    auto* fa = std::get_if<Qwen35FullAttention>(&layer.token_mixer);
    auto* dn = std::get_if<Qwen35DeltaNet>(&layer.token_mixer);

    const auto set_gamma = [&](const TensorNode& gamma) {
        std::vector<float> f32s = get_f32(t);
        const std::size_t n = f32s.size();
        gamma.set_data(Mat(std::move(f32s), 1, n));
        return true;
    };

    if (field == "input_layernorm.weight") return set_gamma(layer.input_layernorm.gamma);
    if (field == "post_attention_layernorm.weight")
        return set_gamma(layer.post_attention_layernorm.gamma);

    if (field == "mlp.gate_proj.weight") {
        set_linear(layer.mlp.gate_proj, std::move(t));
        return true;
    }
    if (field == "mlp.up_proj.weight") {
        set_linear(layer.mlp.up_proj, std::move(t));
        return true;
    }
    if (field == "mlp.down_proj.weight") {
        set_linear(layer.mlp.down_proj, std::move(t));
        return true;
    }

    // A tensor addressed at the wrong layer kind still counts as matched: the
    // name is known, it just does not apply to this layer.
    if (field == "self_attn.q_proj.weight") {
        if (fa != nullptr) set_linear(fa->q_proj, std::move(t));
        return true;
    }
    if (field == "self_attn.k_proj.weight") {
        if (fa != nullptr) set_linear(fa->k_proj, std::move(t));
        return true;
    }
    if (field == "self_attn.v_proj.weight") {
        if (fa != nullptr) set_linear(fa->v_proj, std::move(t));
        return true;
    }
    if (field == "self_attn.o_proj.weight") {
        if (fa != nullptr) set_linear(fa->o_proj, std::move(t));
        return true;
    }
    if (field == "self_attn.q_norm.weight") {
        if (fa != nullptr) set_gamma(fa->q_norm.gamma);
        return true;
    }
    if (field == "self_attn.k_norm.weight") {
        if (fa != nullptr) set_gamma(fa->k_norm.gamma);
        return true;
    }

    if (field == "linear_attn.in_proj_qkv.weight") {
        if (dn != nullptr) set_linear(dn->in_proj_qkv, std::move(t));
        return true;
    }
    if (field == "linear_attn.in_proj_z.weight") {
        if (dn != nullptr) set_linear(dn->in_proj_z, std::move(t));
        return true;
    }
    if (field == "linear_attn.in_proj_a.weight") {
        if (dn != nullptr) set_linear(dn->in_proj_a, std::move(t));
        return true;
    }
    if (field == "linear_attn.in_proj_b.weight") {
        if (dn != nullptr) set_linear(dn->in_proj_b, std::move(t));
        return true;
    }
    if (field == "linear_attn.out_proj.weight") {
        if (dn != nullptr) set_linear(dn->out_proj, std::move(t));
        return true;
    }
    if (field == "linear_attn.A_log") {
        if (dn != nullptr) dn->a_log = get_f32(t);
        return true;
    }
    if (field == "linear_attn.dt_bias") {
        if (dn != nullptr) dn->dt_bias = get_f32(t);
        return true;
    }
    if (field == "linear_attn.norm.weight") {
        // Raw gamma, like the other safetensors norms.
        if (dn != nullptr) dn->norm_weight = get_f32(t);
        return true;
    }
    if (field == "linear_attn.conv1d.weight") {
        // [qkv_dim, 1, kernel_size], flattened.
        if (dn != nullptr) dn->conv1d_weight = get_f32(t);
        return true;
    }

    // Vision, MTP and other heads are skipped without complaint.
    return false;
}

Result<void> Qwen35Model::load_weights_from_dir(const std::string& dir) {
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    if (ec) {
        return err("cannot read dir " + dir + ": " + ec.message());
    }

    std::size_t loaded_shards = 0;
    std::size_t loaded_tensors = 0;
    std::size_t matched = 0;

    for (const std::filesystem::directory_entry& entry : it) {
        const std::filesystem::path path = entry.path();
        if (path.extension() != ".safetensors") {
            continue;
        }

        RT_TRY(header, parse_safetensors_header(path.string()));
        const auto& [data_start, entries] = header;
        for (const SafeTensorEntry& te : entries) {
            RT_TRY(t, read_safetensor_from_file(path.string(), data_start, te, true));
            ++loaded_tensors;
            if (apply_safetensor(std::move(t))) {
                ++matched;
            }
        }
        ++loaded_shards;
    }

    if (loaded_shards == 0) {
        return err("no .safetensors files found in " + dir);
    }

    // A tied checkpoint has no lm_head tensor of its own.
    if (config.tie_word_embeddings && !lm_head.bf16_weight && embed_bf16) {
        lm_head.load_bf16_shared(embed_bf16->data, embed_bf16->rows, embed_bf16->cols);
    }

    std::fprintf(stderr, "Qwen3.5: loaded %zu tensors (%zu matched) from %zu shards in %s\n",
                 loaded_tensors, matched, loaded_shards, dir.c_str());
    return {};
}

}  // namespace rt

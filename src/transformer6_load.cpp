// =============================================================================
// OmniVoice LM weight loading from GGUF
// =============================================================================
//
// The `omnivoice-lm` architecture maps onto the model fields as:
//
//   llm.embed_tokens.weight                    -> text_embed
//   audio_embeddings.weight                    -> audio_embed
//   audio_heads.weight                         -> audio_head
//   llm.norm.weight                            -> norm.gamma
//   llm.layers.{i}.input_layernorm.weight      -> input_layernorm.gamma
//   llm.layers.{i}.post_attention_layernorm.*  -> post_attention_layernorm.gamma
//   llm.layers.{i}.self_attn.{q,k,v,o}_proj.*  -> self_attn projections
//   llm.layers.{i}.self_attn.{q,k}_norm.weight -> per-head Q/K norms
//   llm.layers.{i}.mlp.{gate,up,down}_proj.*   -> mlp projections
//
// Norm gammas are stored verbatim -- the `1 + gamma` convention belongs to
// Gemma alone.
//
// The two audio tensors are both `[8200, 1024]`, which is
// `num_audio_codebook * audio_vocab_size` rows: eight blocks of 1025, one per
// codebook, each holding 1024 codes plus a mask token.

#include <cstdio>
#include <string_view>

#include "rt/transformer4.hpp"
#include "rt/transformer6.hpp"

namespace rt {

namespace {

[[nodiscard]] Result<void> load_norm(const GgufFile& gguf, std::size_t idx, RmsNorm2& norm,
                                     std::size_t expect) {
    const GgufType gtype = gguf.tensor_info[idx].gguf_type;
    std::vector<float> values;
    switch (gtype) {
        case GgufType::F32: {
            RT_TRY(v, gguf.decode_f32(idx));
            values = std::move(v);
            break;
        }
        case GgufType::F16: {
            RT_TRY(v, gguf.decode_f16_to_f32(idx));
            values = std::move(v);
            break;
        }
        default:
            return err(std::string("omnivoice lm: norm '") + gguf.tensor_info[idx].name +
                       "' has unsupported type " + gguf_type_name(gtype));
    }
    if (values.size() != expect) {
        return err(std::string("omnivoice lm: norm '") + gguf.tensor_info[idx].name + "' has " +
                   std::to_string(values.size()) + " entries, expected " +
                   std::to_string(expect));
    }
    norm.gamma.set_data(Mat(std::move(values), 1, expect));
    return {};
}

/// Read an embedding table into BF16, whatever it was stored as.
[[nodiscard]] Result<MatBf16> load_embedding(const GgufFile& gguf, std::size_t idx,
                                             std::size_t rows, std::size_t cols) {
    const GgufType gtype = gguf.tensor_info[idx].gguf_type;
    const auto widen = [&](Result<std::vector<float>> decoded) -> Result<MatBf16> {
        RT_TRY(f32s, std::move(decoded));
        if (f32s.size() != rows * cols) {
            return err(std::string("omnivoice lm: '") + gguf.tensor_info[idx].name + "' holds " +
                       std::to_string(f32s.size()) + " values, expected " +
                       std::to_string(rows * cols));
        }
        return MatBf16(f32s_to_bf16_and_drop(std::move(f32s)), rows, cols);
    };

    switch (gtype) {
        case GgufType::Bf16: {
            RT_TRY(bits, gguf.decode_bf16(idx));
            return MatBf16(std::move(bits), rows, cols);
        }
        case GgufType::F16:
            return widen(gguf.decode_f16_to_f32(idx));
        case GgufType::F32:
            return widen(gguf.decode_f32(idx));
        case GgufType::Q8_0:
            return widen(gguf.decode_q8_0_to_f32(idx));
        case GgufType::Q4K:
            return widen(gguf.decode_q4k_to_f32(idx));
        case GgufType::Q6K:
            return widen(gguf.decode_q6k_to_f32(idx));
        case GgufType::Q5K:
            return widen(gguf.decode_q5k_to_f32(idx));
        case GgufType::Q4_0:
            return widen(gguf.decode_q4_0_to_f32(idx));
        default:
            return err(std::string("omnivoice lm: '") + gguf.tensor_info[idx].name +
                       "' has unsupported type " + gguf_type_name(gtype));
    }
}

[[nodiscard]] bool split_layer_name(std::string_view name, std::size_t& layer,
                                    std::string_view& field) {
    constexpr std::string_view kPrefix = "llm.layers.";
    if (!name.starts_with(kPrefix)) {
        return false;
    }
    const std::size_t dot = name.find('.', kPrefix.size());
    if (dot == std::string_view::npos) {
        return false;
    }
    std::size_t value = 0;
    for (const char ch : name.substr(kPrefix.size(), dot - kPrefix.size())) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        value = value * 10 + static_cast<std::size_t>(ch - '0');
    }
    layer = value;
    field = name.substr(dot + 1);
    return true;
}

[[nodiscard]] std::size_t meta_u64(const GgufFile& gguf, const std::string& key,
                                   std::size_t fallback) {
    const auto it = gguf.metadata.find(key);
    if (it == gguf.metadata.end()) {
        return fallback;
    }
    const std::optional<std::uint64_t> v = it->second.as_u64();
    return v ? static_cast<std::size_t>(*v) : fallback;
}

[[nodiscard]] float meta_f32(const GgufFile& gguf, const std::string& key, float fallback) {
    const auto it = gguf.metadata.find(key);
    if (it == gguf.metadata.end()) {
        return fallback;
    }
    const std::optional<float> v = it->second.as_f32();
    return v ? *v : fallback;
}

}  // namespace

Result<OmniLm> OmniLm::load(const std::string& path, Config6 cfg) {
    RT_TRY(gguf, GgufFile::open(path));

    const auto arch = gguf.metadata.find("general.architecture");
    if (arch != gguf.metadata.end()) {
        const std::optional<std::string_view> name = arch->second.as_str();
        if (name && *name != "omnivoice-lm") {
            return err("omnivoice lm: GGUF declares architecture '" + std::string(*name) +
                       "', expected 'omnivoice-lm'");
        }
    }

    // Geometry comes from the file; a mismatch is an error rather than a
    // silent reinterpretation.
    cfg.num_hidden_layers = meta_u64(gguf, "omnivoice-lm.block_count", cfg.num_hidden_layers);
    cfg.hidden_size = meta_u64(gguf, "omnivoice-lm.embedding_length", cfg.hidden_size);
    cfg.num_attention_heads =
        meta_u64(gguf, "omnivoice-lm.attention.head_count", cfg.num_attention_heads);
    cfg.num_key_value_heads =
        meta_u64(gguf, "omnivoice-lm.attention.head_count_kv", cfg.num_key_value_heads);
    cfg.intermediate_size =
        meta_u64(gguf, "omnivoice-lm.feed_forward_length", cfg.intermediate_size);
    cfg.head_dim = meta_u64(gguf, "omnivoice-lm.attention.key_length", cfg.head_dim);
    cfg.text_vocab_size = meta_u64(gguf, "omnivoice-lm.vocab_size", cfg.text_vocab_size);
    cfg.rope_theta = meta_f32(gguf, "omnivoice-lm.rope.freq_base", cfg.rope_theta);
    cfg.rms_norm_eps =
        meta_f32(gguf, "omnivoice-lm.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps);

    cfg.num_audio_codebook = meta_u64(gguf, "omnivoice.num_audio_codebook", cfg.num_audio_codebook);
    cfg.audio_vocab_size = meta_u64(gguf, "omnivoice.audio_vocab_size", cfg.audio_vocab_size);
    cfg.audio_mask_id = meta_u64(gguf, "omnivoice.audio_mask_id", cfg.audio_mask_id);
    cfg.text_start = meta_u64(gguf, "omnivoice.special.text_start", cfg.text_start);
    cfg.text_end = meta_u64(gguf, "omnivoice.special.text_end", cfg.text_end);
    cfg.lang_start = meta_u64(gguf, "omnivoice.special.lang_start", cfg.lang_start);
    cfg.lang_end = meta_u64(gguf, "omnivoice.special.lang_end", cfg.lang_end);
    cfg.instruct_start = meta_u64(gguf, "omnivoice.special.instruct_start", cfg.instruct_start);
    cfg.instruct_end = meta_u64(gguf, "omnivoice.special.instruct_end", cfg.instruct_end);
    cfg.denoise = meta_u64(gguf, "omnivoice.special.denoise", cfg.denoise);

    if (cfg.audio_mask_id >= cfg.audio_vocab_size) {
        return err("omnivoice lm: mask id " + std::to_string(cfg.audio_mask_id) +
                   " is outside the audio vocabulary of " +
                   std::to_string(cfg.audio_vocab_size));
    }

    std::fprintf(stderr,
                 "[ GGUF ] omnivoice-lm: %zu layers, hidden %zu, %zu/%zu heads, ffn %zu, "
                 "text vocab %zu, %zu codebooks x %zu\n",
                 cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
                 cfg.num_key_value_heads, cfg.intermediate_size, cfg.text_vocab_size,
                 cfg.num_audio_codebook, cfg.audio_vocab_size);

    OmniLm model(std::move(cfg));

    std::size_t loaded = 0;
    for (std::size_t idx = 0; idx < gguf.tensor_info.size(); ++idx) {
        const std::string& name = gguf.tensor_info[idx].name;

        if (name == "llm.embed_tokens.weight") {
            RT_TRY(table, load_embedding(gguf, idx, model.config.text_vocab_size,
                                         model.config.hidden_size));
            model.text_embed = std::move(table);
            ++loaded;
            continue;
        }
        if (name == "audio_embeddings.weight") {
            RT_TRY(table, load_embedding(gguf, idx, model.config.audio_table_size(),
                                         model.config.hidden_size));
            model.audio_embed = std::move(table);
            ++loaded;
            continue;
        }
        if (name == "audio_heads.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, model.audio_head));
            ++loaded;
            continue;
        }
        if (name == "llm.norm.weight") {
            RT_TRY_VOID(load_norm(gguf, idx, model.norm, model.config.hidden_size));
            ++loaded;
            continue;
        }

        std::size_t layer_idx = 0;
        std::string_view field;
        if (!split_layer_name(name, layer_idx, field)) {
            std::fprintf(stderr, "[ GGUF ] Note: ignoring unrecognised tensor '%s'\n",
                         name.c_str());
            continue;
        }
        if (layer_idx >= model.layers.size()) {
            return err("omnivoice lm: tensor '" + name + "' names layer " +
                       std::to_string(layer_idx) + " but the model has " +
                       std::to_string(model.layers.size()));
        }
        OmniBlock& layer = model.layers[layer_idx];

        if (field == "input_layernorm.weight") {
            RT_TRY_VOID(load_norm(gguf, idx, layer.input_layernorm, model.config.hidden_size));
        } else if (field == "post_attention_layernorm.weight") {
            RT_TRY_VOID(
                load_norm(gguf, idx, layer.post_attention_layernorm, model.config.hidden_size));
        } else if (field == "self_attn.q_norm.weight") {
            // Per head, so this is head_dim wide rather than hidden_size.
            RT_TRY_VOID(load_norm(gguf, idx, layer.self_attn.q_norm, model.config.head_dim));
        } else if (field == "self_attn.k_norm.weight") {
            RT_TRY_VOID(load_norm(gguf, idx, layer.self_attn.k_norm, model.config.head_dim));
        } else if (field == "self_attn.q_proj.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.self_attn.q_proj));
        } else if (field == "self_attn.k_proj.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.self_attn.k_proj));
        } else if (field == "self_attn.v_proj.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.self_attn.v_proj));
        } else if (field == "self_attn.o_proj.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.self_attn.o_proj));
        } else if (field == "mlp.gate_proj.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.mlp.gate_proj));
        } else if (field == "mlp.up_proj.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.mlp.up_proj));
        } else if (field == "mlp.down_proj.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.mlp.down_proj));
        } else {
            std::fprintf(stderr, "[ GGUF ] Note: ignoring unrecognised field '%s'\n",
                         name.c_str());
            continue;
        }
        ++loaded;
    }

    if (!model.text_embed) {
        return err("omnivoice lm: GGUF has no llm.embed_tokens.weight");
    }
    if (!model.audio_embed) {
        return err("omnivoice lm: GGUF has no audio_embeddings.weight");
    }

    model.refresh_inv_freq();
    std::fprintf(stderr, "[ GGUF ] Loaded %zu tensors, %.2f GB resident\n", loaded,
                 static_cast<double>(model.weight_bytes()) / 1e9);
    return model;
}

}  // namespace rt

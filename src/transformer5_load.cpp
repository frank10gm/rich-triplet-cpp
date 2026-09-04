// =============================================================================
// Llama 3.2 weight loading from GGUF
// =============================================================================
//
// GGUF's llama naming maps onto the model fields as:
//
//   token_embd.weight          -> embed_bf16
//   output.weight              -> lm_head          (separate, not weight-tied)
//   output_norm.weight         -> norm.gamma
//   rope_freqs.weight          -> config.rope_freq_divisors
//   blk.{i}.attn_norm.weight   -> layers[i].input_layernorm.gamma
//   blk.{i}.attn_q.weight      -> layers[i].self_attn.q_proj
//   blk.{i}.attn_k.weight      -> layers[i].self_attn.k_proj
//   blk.{i}.attn_v.weight      -> layers[i].self_attn.v_proj
//   blk.{i}.attn_output.weight -> layers[i].self_attn.o_proj
//   blk.{i}.ffn_norm.weight    -> layers[i].post_attention_layernorm.gamma
//   blk.{i}.ffn_gate.weight    -> layers[i].mlp.gate_proj
//   blk.{i}.ffn_up.weight      -> layers[i].mlp.up_proj
//   blk.{i}.ffn_down.weight    -> layers[i].mlp.down_proj
//
// **Norm gammas are stored verbatim.** Gemma's GGUF files hold `1 + gamma` and
// its loader subtracts one; doing that here scales every norm by `gamma - 1`
// and the model emits garbage from the first token. The mistake is loud, but
// only if you know to look for it.

#include <cstdio>
#include <string_view>

#include "rt/transformer4.hpp"
#include "rt/transformer5.hpp"

namespace rt {

namespace {

/// Set an RMSNorm gamma from an f32 or F16 GGUF tensor.
[[nodiscard]] Result<void> load_norm_from_gguf(const GgufFile& gguf, std::size_t idx,
                                               RmsNorm2& norm, std::size_t expect) {
    const GgufType gtype = gguf.tensor_info[idx].gguf_type;
    std::vector<float> values;
    switch (gtype) {
        case GgufType::F32: {
            RT_TRY(f32s, gguf.decode_f32(idx));
            values = std::move(f32s);
            break;
        }
        case GgufType::F16: {
            RT_TRY(f32s, gguf.decode_f16_to_f32(idx));
            values = std::move(f32s);
            break;
        }
        default:
            return err(std::string("llama: norm tensor '") + gguf.tensor_info[idx].name +
                       "' has unsupported type " + gguf_type_name(gtype));
    }
    if (values.size() != expect) {
        return err(std::string("llama: norm tensor '") + gguf.tensor_info[idx].name + "' has " +
                   std::to_string(values.size()) + " entries, expected " +
                   std::to_string(expect));
    }
    // Verbatim -- no `1 + gamma` adjustment. See the file header.
    norm.gamma.set_data(Mat(std::move(values), 1, expect));
    return {};
}

/// Parse `blk.{i}.` and return the layer index plus the remaining field name.
[[nodiscard]] bool split_block_name(std::string_view name, std::size_t& layer,
                                    std::string_view& field) {
    constexpr std::string_view kPrefix = "blk.";
    if (!name.starts_with(kPrefix)) {
        return false;
    }
    const std::size_t dot = name.find('.', kPrefix.size());
    if (dot == std::string_view::npos) {
        return false;
    }
    const std::string_view digits = name.substr(kPrefix.size(), dot - kPrefix.size());
    std::size_t value = 0;
    for (const char ch : digits) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        value = value * 10 + static_cast<std::size_t>(ch - '0');
    }
    layer = value;
    field = name.substr(dot + 1);
    return true;
}

/// Read a u64 metadata value, or fall back.
[[nodiscard]] std::size_t meta_u64(const GgufFile& gguf, const std::string& key,
                                   std::size_t fallback) {
    const auto it = gguf.metadata.find(key);
    if (it == gguf.metadata.end()) {
        return fallback;
    }
    const std::optional<std::uint64_t> v = it->second.as_u64();
    return v ? static_cast<std::size_t>(*v) : fallback;
}

/// Read an f32 metadata value, or fall back.
[[nodiscard]] float meta_f32(const GgufFile& gguf, const std::string& key, float fallback) {
    const auto it = gguf.metadata.find(key);
    if (it == gguf.metadata.end()) {
        return fallback;
    }
    const std::optional<float> v = it->second.as_f32();
    return v ? *v : fallback;
}

}  // namespace

Result<void> LlamaModel::load_weights_from_gguf(const std::string& path) {
    RT_TRY(gguf, GgufFile::open(path));

    // ---- sanity-check the metadata against the config ----
    const auto arch = gguf.metadata.find("general.architecture");
    if (arch != gguf.metadata.end()) {
        const std::optional<std::string_view> name = arch->second.as_str();
        if (name && *name != "llama") {
            return err("llama: GGUF declares architecture '" + std::string(*name) +
                       "', expected 'llama'");
        }
    }

    const std::size_t file_layers = meta_u64(gguf, "llama.block_count", config.num_hidden_layers);
    const std::size_t file_hidden =
        meta_u64(gguf, "llama.embedding_length", config.hidden_size);
    const std::size_t file_heads =
        meta_u64(gguf, "llama.attention.head_count", config.num_attention_heads);
    const std::size_t file_kv_heads =
        meta_u64(gguf, "llama.attention.head_count_kv", config.num_key_value_heads);
    const std::size_t file_ffn =
        meta_u64(gguf, "llama.feed_forward_length", config.intermediate_size);
    const std::size_t file_head_dim =
        meta_u64(gguf, "llama.attention.key_length", config.head_dim);

    // A geometry mismatch would otherwise show up as a shape error deep in the
    // first matmul, or worse, as a model that runs and produces noise.
    const auto mismatch = [](const char* what, std::size_t got,
                             std::size_t want) -> std::unexpected<std::string> {
        return err(std::string("llama: GGUF says ") + what + " is " + std::to_string(got) +
                   " but the config says " + std::to_string(want));
    };
    if (file_layers != config.num_hidden_layers) {
        return mismatch("block_count", file_layers, config.num_hidden_layers);
    }
    if (file_hidden != config.hidden_size) {
        return mismatch("embedding_length", file_hidden, config.hidden_size);
    }
    if (file_heads != config.num_attention_heads) {
        return mismatch("head_count", file_heads, config.num_attention_heads);
    }
    if (file_kv_heads != config.num_key_value_heads) {
        return mismatch("head_count_kv", file_kv_heads, config.num_key_value_heads);
    }
    if (file_ffn != config.intermediate_size) {
        return mismatch("feed_forward_length", file_ffn, config.intermediate_size);
    }
    if (file_head_dim != config.head_dim) {
        return mismatch("attention.key_length", file_head_dim, config.head_dim);
    }

    config.rope_theta = meta_f32(gguf, "llama.rope.freq_base", config.rope_theta);
    config.rms_norm_eps =
        meta_f32(gguf, "llama.attention.layer_norm_rms_epsilon", config.rms_norm_eps);
    const std::size_t file_vocab = meta_u64(gguf, "llama.vocab_size", config.vocab_size);
    if (file_vocab != config.vocab_size) {
        return mismatch("vocab_size", file_vocab, config.vocab_size);
    }

    std::fprintf(stderr,
                 "[ GGUF ] llama: %zu layers, hidden %zu, %zu/%zu heads, ffn %zu, vocab %zu, "
                 "theta %.0f\n",
                 config.num_hidden_layers, config.hidden_size, config.num_attention_heads,
                 config.num_key_value_heads, config.intermediate_size, config.vocab_size,
                 static_cast<double>(config.rope_theta));

    // Norm epsilon reaches the layers through their constructors, so refresh
    // the ones already built from the file's value.
    norm.eps = config.rms_norm_eps;
    for (LlamaBlock& layer : layers) {
        layer.input_layernorm.eps = config.rms_norm_eps;
        layer.post_attention_layernorm.eps = config.rms_norm_eps;
    }

    // ---- tensors ----
    std::size_t loaded = 0;
    bool have_embed = false;
    bool have_output = false;

    for (std::size_t idx = 0; idx < gguf.tensor_info.size(); ++idx) {
        const std::string& name = gguf.tensor_info[idx].name;
        const GgufType gtype = gguf.tensor_info[idx].gguf_type;
        const std::vector<std::size_t>& shape = gguf.tensor_info[idx].shape;

        if (name == "token_embd.weight") {
            // GGUF stores [hidden, vocab]; the lookup table wants [vocab, hidden].
            const std::size_t vocab = shape.size() == 2 ? shape[1] : shape[0];
            const std::size_t hidden = shape.size() == 2 ? shape[0] : 1;
            if (vocab != config.vocab_size || hidden != config.hidden_size) {
                return err("llama: token_embd is " + std::to_string(vocab) + "x" +
                           std::to_string(hidden) + ", expected " +
                           std::to_string(config.vocab_size) + "x" +
                           std::to_string(config.hidden_size));
            }
            std::fprintf(stderr, "[ GGUF ] token_embd: type=%s vocab=%zu hidden=%zu\n",
                         gguf_type_name(gtype), vocab, hidden);

            const auto set_from_f32 = [&](Result<std::vector<float>> decoded) -> Result<void> {
                RT_TRY(f32s, std::move(decoded));
                embed_bf16 = MatBf16(f32s_to_bf16_and_drop(std::move(f32s)), vocab, hidden);
                return {};
            };

            switch (gtype) {
                case GgufType::Bf16: {
                    RT_TRY(bits, gguf.decode_bf16(idx));
                    embed_bf16 = MatBf16(std::move(bits), vocab, hidden);
                    break;
                }
                case GgufType::F16:
                    RT_TRY_VOID(set_from_f32(gguf.decode_f16_to_f32(idx)));
                    break;
                case GgufType::F32:
                    RT_TRY_VOID(set_from_f32(gguf.decode_f32(idx)));
                    break;
                case GgufType::Q4_0:
                    RT_TRY_VOID(set_from_f32(gguf.decode_q4_0_to_f32(idx)));
                    break;
                case GgufType::Q4K:
                    RT_TRY_VOID(set_from_f32(gguf.decode_q4k_to_f32(idx)));
                    break;
                case GgufType::Q6K:
                    RT_TRY_VOID(set_from_f32(gguf.decode_q6k_to_f32(idx)));
                    break;
                default:
                    return err(std::string("llama: token_embd has unsupported type ") +
                               gguf_type_name(gtype));
            }
            have_embed = true;
            ++loaded;
            continue;
        }

        if (name == "output.weight") {
            std::fprintf(stderr, "[ GGUF ] output.weight (lm_head): type=%s\n",
                         gguf_type_name(gtype));
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, lm_head));
            have_output = true;
            ++loaded;
            continue;
        }

        if (name == "output_norm.weight") {
            RT_TRY_VOID(load_norm_from_gguf(gguf, idx, norm, config.hidden_size));
            ++loaded;
            continue;
        }

        if (name == "rope_freqs.weight") {
            // Per-dimension RoPE divisors. Loading these is what keeps long
            // sequences coherent; see the header on why they are divisors.
            RT_TRY(divisors, gguf.decode_f32(idx));
            const std::size_t want = config.head_dim / 2;
            if (divisors.size() != want) {
                return err("llama: rope_freqs has " + std::to_string(divisors.size()) +
                           " entries, expected head_dim/2 = " + std::to_string(want));
            }
            std::fprintf(stderr, "[ GGUF ] rope_freqs: %zu divisors, %.4f..%.4f\n",
                         divisors.size(), static_cast<double>(divisors.front()),
                         static_cast<double>(divisors.back()));
            config.rope_freq_divisors = std::move(divisors);
            ++loaded;
            continue;
        }

        std::size_t layer_idx = 0;
        std::string_view field;
        if (!split_block_name(name, layer_idx, field)) {
            std::fprintf(stderr, "[ GGUF ] Note: ignoring unrecognised tensor '%s'\n",
                         name.c_str());
            continue;
        }
        if (layer_idx >= layers.size()) {
            return err("llama: tensor '" + name + "' names layer " + std::to_string(layer_idx) +
                       " but the model has " + std::to_string(layers.size()));
        }
        LlamaBlock& layer = layers[layer_idx];

        if (field == "attn_norm.weight") {
            RT_TRY_VOID(
                load_norm_from_gguf(gguf, idx, layer.input_layernorm, config.hidden_size));
        } else if (field == "ffn_norm.weight") {
            RT_TRY_VOID(load_norm_from_gguf(gguf, idx, layer.post_attention_layernorm,
                                            config.hidden_size));
        } else if (field == "attn_q.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.self_attn.q_proj));
        } else if (field == "attn_k.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.self_attn.k_proj));
        } else if (field == "attn_v.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.self_attn.v_proj));
        } else if (field == "attn_output.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.self_attn.o_proj));
        } else if (field == "ffn_gate.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.mlp.gate_proj));
        } else if (field == "ffn_up.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.mlp.up_proj));
        } else if (field == "ffn_down.weight") {
            RT_TRY_VOID(load_linear_from_gguf(gguf, idx, layer.mlp.down_proj));
        } else {
            std::fprintf(stderr, "[ GGUF ] Note: ignoring unrecognised field '%s'\n", name.c_str());
            continue;
        }
        ++loaded;
    }

    if (!have_embed) {
        return err("llama: GGUF has no token_embd.weight");
    }
    if (!have_output) {
        // Llama 3.2 3B ties its lm_head to the embedding, but Orpheus does
        // not, and a tied fallback would need an f32 copy of a 964 MB table.
        return err("llama: GGUF has no output.weight; weight-tied lm_head is not supported");
    }

    // The divisors are read from the same tensor loop that fills the layers,
    // and every layer holds a pointer into the cache, so recompute now.
    rebind_inv_freq();

    std::fprintf(stderr, "[ GGUF ] Loaded %zu tensors, %.2f GB resident\n", loaded,
                 static_cast<double>(weight_bytes()) / 1e9);
    return {};
}


// =============================================================================
// Embedded tokenizer
// =============================================================================

Result<HfBpeTokenizer> load_gguf_tokenizer(const GgufFile& gguf) {
    const auto read_strings = [&](const char* key) -> Result<std::vector<std::string>> {
        const auto it = gguf.metadata.find(key);
        if (it == gguf.metadata.end()) {
            return err(std::string("llama: GGUF has no '") + key + "'");
        }
        const std::vector<GgufMetaValue>* array = it->second.as_array();
        if (array == nullptr) {
            return err(std::string("llama: GGUF '") + key + "' is not an array");
        }
        std::vector<std::string> out;
        out.reserve(array->size());
        for (const GgufMetaValue& entry : *array) {
            const std::optional<std::string_view> text = entry.as_str();
            if (!text) {
                return err(std::string("llama: GGUF '") + key + "' holds a non-string entry");
            }
            out.emplace_back(*text);
        }
        return out;
    };

    RT_TRY(tokens, read_strings("tokenizer.ggml.tokens"));
    RT_TRY(merges, read_strings("tokenizer.ggml.merges"));

    // `tokenizer.ggml.model` says which family; `pre` says which
    // pre-tokenizer regex within it.
    std::string model = "gpt2";
    if (const auto it = gguf.metadata.find("tokenizer.ggml.model");
        it != gguf.metadata.end()) {
        if (const std::optional<std::string_view> text = it->second.as_str()) {
            model = std::string(*text);
        }
    }
    std::string pre = "default";
    if (const auto it = gguf.metadata.find("tokenizer.ggml.pre"); it != gguf.metadata.end()) {
        if (const std::optional<std::string_view> text = it->second.as_str()) {
            pre = std::string(*text);
        }
    }

    if (model != "gpt2") {
        return err("llama: GGUF tokenizer model is '" + model +
                   "'; only byte-level 'gpt2' vocabularies are supported here");
    }

    const HfBpeTokenizer::PreTokenizer kind = pre == "llama-bpe"
                                                  ? HfBpeTokenizer::PreTokenizer::Llama3
                                                  : HfBpeTokenizer::PreTokenizer::Gpt2;

    std::fprintf(stderr, "[ GGUF ] tokenizer: model=%s pre=%s vocab=%zu merges=%zu\n",
                 model.c_str(), pre.c_str(), tokens.size(), merges.size());

    return HfBpeTokenizer::from_vocab_and_merges(std::move(tokens), merges,
                                                 /*byte_level=*/true, kind);
}

}  // namespace rt

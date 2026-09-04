// =============================================================================
// Gemma 3 -- weight loading, the binary cache, and the generation loops
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "rt/transformer3.hpp"
#include "rt/transformer4.hpp"

#if RT_FEATURE_METAL
#include "rt/metal_decode.hpp"
#endif

#if defined(__APPLE__)
#include <malloc/malloc.h>
#include <unistd.h>
#endif

namespace rt {

// =============================================================================
// Weight-loading helpers
// =============================================================================

std::vector<std::uint16_t> f32s_to_bf16_and_drop(std::vector<float> f32s) {
    std::vector<std::uint16_t> bits(f32s.size());
    for (std::size_t i = 0; i < f32s.size(); ++i) {
        bits[i] = f32_to_bf16(f32s[i]);
    }
    f32s.clear();
    f32s.shrink_to_fit();
    return bits;
}

#if defined(__APPLE__)
void release_memory_to_os() { malloc_zone_pressure_relief(malloc_default_zone(), 0); }
#else
void release_memory_to_os() {}
#endif

void print_rss(const std::string& label) {
#if defined(__APPLE__)
    // `ps` is the most reliable source of a real RSS number on macOS.
    const std::string cmd = "ps -o rss= -p " + std::to_string(getpid());
    if (std::FILE* pipe = ::popen(cmd.c_str(), "r")) {
        char buf[64] = {};
        const bool got = std::fgets(buf, sizeof(buf), pipe) != nullptr;
        ::pclose(pipe);
        if (got) {
            try {
                const double mb = std::stod(buf) / 1024.0;
                std::fprintf(stderr, "[ RSS ] %s: %.1f MB\n", label.c_str(), mb);
                return;
            } catch (const std::exception&) {
                // Fall through to the failure message.
            }
        }
    }
#endif
    std::fprintf(stderr, "[ RSS ] %s: (failed to read)\n", label.c_str());
}

void model_set_embed_bf16_raw(TensorNode& embed_tokens, std::optional<MatBf16>& embed_bf16,
                              Linear2& lm_head, std::vector<std::uint16_t> bits,
                              std::size_t vocab, std::size_t hidden, bool tie_weights) {
    // Wrap the bits once so the embedding and lm_head share one allocation.
    auto shared = std::make_shared<std::vector<std::uint16_t>>(std::move(bits));
    embed_bf16 = MatBf16(shared, vocab, hidden);
    // The f32 table becomes a placeholder: lookups go through embed_bf16 from
    // here on, and dropping it frees ~2.5 GB at 4B scale.
    embed_tokens.set_data(Mat::zeros(0, 0));
    if (tie_weights) {
        lm_head.load_bf16_shared(std::move(shared), vocab, hidden);
    }
}

namespace {

/// Gemma 3's own embedding setup: always weight-tied.
void model_set_embed_bf16(Gemma3Model& model, std::vector<std::uint16_t> bits, std::size_t vocab,
                          std::size_t hidden) {
    model_set_embed_bf16_raw(model.embed_tokens, model.embed_bf16, model.lm_head, std::move(bits),
                             vocab, hidden, true);
}

}  // namespace

Result<void> load_linear_from_gguf(const GgufFile& gguf, std::size_t idx, Linear2& linear) {
    const GgufType gtype = gguf.tensor_info[idx].gguf_type;
    const std::vector<std::size_t>& shape = gguf.tensor_info[idx].shape;
    // GGUF stores a weight column-major as [in_features, out_features]; our
    // Linear2 wants [out_features, in_features].
    const std::size_t rows = shape.size() >= 2 ? shape[1] : 1;
    const std::size_t cols = shape.size() >= 2 ? shape[0] : shape[0];

    const auto load_as_bf16 = [&](Result<std::vector<float>> decoded) -> Result<void> {
        RT_TRY(f32s, std::move(decoded));
        linear.load_bf16(f32s_to_bf16_and_drop(std::move(f32s)), rows, cols);
        return {};
    };

    switch (gtype) {
        case GgufType::Q4_0: {
            // Repack Q4_0 as Q4_K: smaller (0.56 vs 0.63 bytes/element) and the
            // only quantized form with both a Metal GEMV and an SDOT CPU path.
            RT_TRY(q4, gguf.decode_q4_0_to_q4mat(idx));
            linear.q4k_weight = Q4KMat::from_q4mat(q4);
            linear.bf16_weight.reset();
            linear.weight.set_data(Mat::zeros(0, 0));
            break;
        }
        case GgufType::Bf16: {
            RT_TRY(bits, gguf.decode_bf16(idx));
            linear.load_bf16(std::move(bits), rows, cols);
            break;
        }
        case GgufType::F16:
            RT_TRY_VOID(load_as_bf16(gguf.decode_f16_to_f32(idx)));
            break;
        case GgufType::F32: {
            RT_TRY(f32s, gguf.decode_f32(idx));
            linear.weight.set_data(Mat(std::move(f32s), rows, cols));
            break;
        }
        case GgufType::Q4K: {
            // Keep the raw blocks and dequantize during the matmul: ~3.5x less
            // RAM than BF16, and no decode pass at load time.
            RT_TRY(q4k, gguf.decode_q4k_to_q4kmat(idx));
            linear.q4k_weight = std::move(q4k);
            linear.bf16_weight.reset();
            linear.weight.set_data(Mat::zeros(0, 0));
            break;
        }
        case GgufType::Q6K:
            RT_TRY_VOID(load_as_bf16(gguf.decode_q6k_to_f32(idx)));
            break;
        case GgufType::Q8_0:
            RT_TRY_VOID(load_as_bf16(gguf.decode_q8_0_to_f32(idx)));
            break;
        case GgufType::Q5K:
            RT_TRY_VOID(load_as_bf16(gguf.decode_q5k_to_f32(idx)));
            break;
        default:
            std::fprintf(stderr, "[ GGUF ] Warning: unsupported type %s for tensor %s, skipping\n",
                         gguf_type_name(gtype), gguf.tensor_info[idx].name.c_str());
            break;
    }
    return {};
}

// =============================================================================
// safetensors -> model field mapping
// =============================================================================

namespace {

/// Set a node's f32 data directly. Used for the small tensors: norms, and the
/// embedding when it did not arrive as BF16.
bool set_node(const TensorNode& node, const std::vector<float>& data, std::size_t rows,
              std::size_t cols) {
    if (data.size() != rows * cols) {
        return false;
    }
    node.set_data(Mat(data, rows, cols));
    return true;
}

/// A safetensor's values as f32, whether it arrived as f32 or as raw BF16 bits.
/// Always allocates, so this is for small tensors only.
std::vector<float> st_to_f32(const SafeTensor& t) {
    if (!t.data.empty()) {
        return t.data;
    }
    if (t.bf16_data) {
        std::vector<float> out(t.bf16_data->size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = bf16_to_f32((*t.bf16_data)[i]);
        }
        return out;
    }
    return {};
}

/// Set a Linear2's weight, preferring BF16 storage when the tensor was BF16 on
/// disk. Takes the tensor by value so the bits move rather than copy.
bool set_linear(Linear2& linear, SafeTensor t, std::size_t rows, std::size_t cols) {
    if (t.bf16_data && t.bf16_data->size() == rows * cols) {
        linear.load_bf16(std::move(*t.bf16_data), rows, cols);
        return true;
    }
    if (t.data.size() != rows * cols) {
        return false;
    }
    linear.weight.set_data(Mat(std::move(t.data), rows, cols));
    return true;
}

/// Route one safetensor to its model field. Returns whether it matched.
bool apply_tensor(Gemma3Model& model, SafeTensor t) {
    // Multimodal checkpoints prefix the text tower with "language_model.".
    constexpr std::string_view kLmPrefix = "language_model.";
    const std::string name = t.name.starts_with(kLmPrefix) ? t.name.substr(kLmPrefix.size())
                                                           : t.name;

    if (name == "model.embed_tokens.weight") {
        // Store as BF16: the 262K x 2560 table would be 2.7 GB as f32.
        const std::size_t vocab = t.shape[0];
        const std::size_t hidden = t.shape[1];
        if (t.bf16_data && t.bf16_data->size() == vocab * hidden) {
            model_set_embed_bf16(model, std::move(*t.bf16_data), vocab, hidden);
            return true;
        }
        return set_node(model.embed_tokens, t.data, vocab, hidden);
    }
    if (name == "model.norm.weight") {
        const std::vector<float> f32s = st_to_f32(t);
        return set_node(model.norm.gamma, f32s, 1, f32s.size());
    }
    if (name == "lm_head.weight") {
        const std::vector<float> f32s = st_to_f32(t);
        return set_node(model.lm_head.weight, f32s, t.shape[0], t.shape[1]);
    }

    constexpr std::string_view kLayerPrefix = "model.layers.";
    if (!name.starts_with(kLayerPrefix)) {
        return false;
    }
    const std::string rest = name.substr(kLayerPrefix.size());
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
    if (layer_idx >= model.layers.size()) {
        return false;
    }
    const std::string field = rest.substr(dot + 1);
    Gemma3Block& layer = model.layers[layer_idx];

    // Norm weights are small, so BF16 converts on the fly.
    const auto set_norm = [&](const TensorNode& gamma) {
        const std::vector<float> f32s = st_to_f32(t);
        return set_node(gamma, f32s, 1, f32s.size());
    };
    if (field == "input_layernorm.weight") return set_norm(layer.input_layernorm.gamma);
    if (field == "post_attention_layernorm.weight")
        return set_norm(layer.post_attention_layernorm.gamma);
    if (field == "pre_feedforward_layernorm.weight")
        return set_norm(layer.pre_feedforward_layernorm.gamma);
    if (field == "post_feedforward_layernorm.weight")
        return set_norm(layer.post_feedforward_layernorm.gamma);
    if (field == "self_attn.q_norm.weight") return set_norm(layer.self_attn.q_norm.gamma);
    if (field == "self_attn.k_norm.weight") return set_norm(layer.self_attn.k_norm.gamma);

    // Projection weights: move the BF16 bits into the Linear2.
    const auto set_proj = [&](Linear2& linear) {
        const std::size_t rows = t.shape[0];
        const std::size_t cols = t.shape[1];
        return set_linear(linear, std::move(t), rows, cols);
    };
    if (field == "self_attn.q_proj.weight") return set_proj(layer.self_attn.q_proj);
    if (field == "self_attn.k_proj.weight") return set_proj(layer.self_attn.k_proj);
    if (field == "self_attn.v_proj.weight") return set_proj(layer.self_attn.v_proj);
    if (field == "self_attn.o_proj.weight") return set_proj(layer.self_attn.o_proj);
    if (field == "mlp.gate_proj.weight") return set_proj(layer.mlp.gate_proj);
    if (field == "mlp.up_proj.weight") return set_proj(layer.mlp.up_proj);
    if (field == "mlp.down_proj.weight") return set_proj(layer.mlp.down_proj);
    return false;
}

}  // namespace

Result<void> Gemma3Model::load_weights_from_dir(const std::string& dir) {
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

        // Stream the shard: read the header, then seek and read one tensor at a
        // time. Only a single tensor (~200 MB at most) is resident, rather than
        // the whole ~4.6 GB file.
        RT_TRY(header, parse_safetensors_header(path.string()));
        const auto& [data_start, entries] = header;

        for (const SafeTensorEntry& te : entries) {
            RT_TRY(t, read_safetensor_from_file(path.string(), data_start, te, true));
            ++loaded_tensors;
            if (apply_tensor(*this, std::move(t))) {
                ++matched;
            }
        }
        ++loaded_shards;
    }

    if (loaded_shards == 0) {
        return err("no .safetensors files found in " + dir);
    }

    std::printf("Gemma3: loaded %zu tensors (%zu matched) from %zu shards in %s\n", loaded_tensors,
                matched, loaded_shards, dir.c_str());
    return {};
}

// =============================================================================
// Quantization
// =============================================================================

void Gemma3Model::quantize_for_inference() {
    for (Gemma3Block& layer : layers) {
        layer.self_attn.q_proj.quantize();
        layer.self_attn.k_proj.quantize();
        layer.self_attn.v_proj.quantize();
        layer.self_attn.o_proj.quantize();
        layer.mlp.gate_proj.quantize();
        layer.mlp.up_proj.quantize();
        layer.mlp.down_proj.quantize();
    }
    lm_head.quantize();
}

void Gemma3Model::quantize_inference_free_f32() {
    std::fprintf(stderr, "[ Gemma3 ] Quantizing weights to INT4 and freeing f32 copies...\n");
    const std::size_t n_layers = layers.size();
    for (std::size_t i = 0; i < n_layers; ++i) {
        Gemma3Block& layer = layers[i];
        layer.self_attn.q_proj.quantize_and_free_f32();
        layer.self_attn.k_proj.quantize_and_free_f32();
        layer.self_attn.v_proj.quantize_and_free_f32();
        layer.self_attn.o_proj.quantize_and_free_f32();
        layer.mlp.gate_proj.quantize_and_free_f32();
        layer.mlp.up_proj.quantize_and_free_f32();
        layer.mlp.down_proj.quantize_and_free_f32();
        if ((i + 1) % 8 == 0 || i + 1 == n_layers) {
            std::fprintf(stderr, "[ Gemma3 ] Quantized %zu/%zu layers\n", i + 1, n_layers);
        }
    }
    lm_head.quantize_and_free_f32();
    std::fprintf(stderr, "[ Gemma3 ] Quantization done.\n");
}

// =============================================================================
// GGUF weight loading
// =============================================================================

Result<void> Gemma3Model::load_weights_from_gguf(const std::string& path) {
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
    // An explicit output.weight means the model is not weight-tied, so the
    // tying step at the end must be skipped.
    bool lm_head_explicitly_loaded = false;

    /// F32 and F16 norm tensors, as f32.
    const auto load_f32 = [&gguf](std::size_t i) -> Result<std::vector<float>> {
        switch (gguf.tensor_info[i].gguf_type) {
            case GgufType::F32:
                return gguf.decode_f32(i);
            case GgufType::F16:
                return gguf.decode_f16_to_f32(i);
            default:
                return err("expected f32/f16 for norm tensor " + gguf.tensor_info[i].name);
        }
    };

    /// GGUF stores norm gammas as `1 + HF_weight`; `forward_gemma3` adds the 1
    /// back, so subtract it here.
    const auto set_norm = [&](std::size_t i, const TensorNode& gamma) -> Result<void> {
        RT_TRY(f32s, load_f32(i));
        std::vector<float> adjusted(f32s.size());
        for (std::size_t j = 0; j < f32s.size(); ++j) {
            adjusted[j] = f32s[j] - 1.0f;
        }
        const std::size_t n = adjusted.size();
        gamma.set_data(Mat(std::move(adjusted), 1, n));
        return {};
    };

    for (std::size_t idx = 0; idx < n_tensors; ++idx) {
        const std::string name = gguf.tensor_info[idx].name;
        const GgufType gtype = gguf.tensor_info[idx].gguf_type;

        // ---- token embedding ----
        if (name == "token_embd.weight") {
            const std::vector<std::size_t>& shape = gguf.tensor_info[idx].shape;
            // GGUF stores [hidden, vocab]; we want [vocab, hidden].
            const std::size_t vocab = shape.size() == 2 ? shape[1] : shape[0];
            const std::size_t hidden = shape.size() == 2 ? shape[0] : 1;
            std::fprintf(stderr, "[ GGUF ] token_embd: type=%s vocab=%zu hidden=%zu\n",
                         gguf_type_name(gtype), vocab, hidden);

            const auto set_from_f32 = [&](Result<std::vector<float>> decoded) -> Result<void> {
                RT_TRY(f32s, std::move(decoded));
                model_set_embed_bf16(*this, f32s_to_bf16_and_drop(std::move(f32s)), vocab, hidden);
                return {};
            };

            switch (gtype) {
                case GgufType::Bf16: {
                    RT_TRY(bits, gguf.decode_bf16(idx));
                    model_set_embed_bf16(*this, std::move(bits), vocab, hidden);
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
                    std::fprintf(stderr,
                                 "[ GGUF ] Warning: token_embd type %s not supported, skipping\n",
                                 gguf_type_name(gtype));
                    break;
            }
            ++loaded;
            continue;
        }

        // ---- lm_head, present separately even in some weight-tied files ----
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
            // The vision tower (v.*) and multimodal projector (mm.*) are not
            // part of the text model, so their tensors are silently skipped.
            if (!name.starts_with("v.") && !name.starts_with("mm.")) {
                std::fprintf(stderr, "[ GGUF ] Skipping unknown top-level tensor: %s\n",
                             name.c_str());
            }
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
        Gemma3Block& layer = layers[layer_idx];

        const TensorNode* norm_target = nullptr;
        if (field == "attn_norm.weight") {
            norm_target = &layer.input_layernorm.gamma;
        } else if (field == "post_attn_norm.weight" || field == "post_attention_norm.weight") {
            norm_target = &layer.post_attention_layernorm.gamma;
        } else if (field == "ffn_pre_norm.weight" || field == "ffn_norm.weight") {
            norm_target = &layer.pre_feedforward_layernorm.gamma;
        } else if (field == "ffn_post_norm.weight" || field == "post_ffw_norm.weight") {
            norm_target = &layer.post_feedforward_layernorm.gamma;
        } else if (field == "attn_q_norm.weight") {
            norm_target = &layer.self_attn.q_norm.gamma;
        } else if (field == "attn_k_norm.weight") {
            norm_target = &layer.self_attn.k_norm.gamma;
        }
        if (norm_target != nullptr) {
            RT_TRY_VOID(set_norm(idx, *norm_target));
            ++loaded;
        } else {
            Linear2* proj = nullptr;
            if (field == "attn_q.weight") {
                proj = &layer.self_attn.q_proj;
            } else if (field == "attn_k.weight") {
                proj = &layer.self_attn.k_proj;
            } else if (field == "attn_v.weight") {
                proj = &layer.self_attn.v_proj;
            } else if (field == "attn_output.weight") {
                proj = &layer.self_attn.o_proj;
            } else if (field == "ffn_gate.weight") {
                proj = &layer.mlp.gate_proj;
            } else if (field == "ffn_up.weight") {
                proj = &layer.mlp.up_proj;
            } else if (field == "ffn_down.weight") {
                proj = &layer.mlp.down_proj;
            }
            if (proj == nullptr) {
                std::fprintf(stderr, "[ GGUF ] Unknown blk tensor: blk.%zu.%s\n", layer_idx,
                             field.c_str());
            } else {
                RT_TRY_VOID(load_linear_from_gguf(gguf, idx, *proj));
                ++loaded;
            }
        }

        if (loaded % 50 == 0 && loaded > 0) {
            std::fprintf(stderr, "[ GGUF ] Loaded %zu/%zu tensors...\n", loaded, n_tensors);
        }
    }

    // Weight tying, unless the file carried its own output.weight.
    if (lm_head_explicitly_loaded) {
        std::fprintf(
            stderr,
            "[ GGUF ] lm_head loaded from explicit output.weight - skipping weight tying.\n");
    } else if (embed_bf16) {
        std::fprintf(stderr,
                     "[ GGUF ] embed_bf16 set: %zux%zu - applying weight tying to lm_head.\n",
                     embed_bf16->rows, embed_bf16->cols);
        lm_head.load_bf16_shared(embed_bf16->data, embed_bf16->rows, embed_bf16->cols);
    } else {
        std::fprintf(stderr,
                     "[ GGUF ] WARNING: embed_bf16 is None - token embeddings not loaded!\n");
    }

    // ---- Diagnostics: layer-0 norm gammas ----
    const auto print_head = [](const char* label, const Mat& g, std::size_t n) {
        std::fprintf(stderr, "%s", label);
        for (std::size_t c = 0; c < std::min(n, g.cols); ++c) {
            std::fprintf(stderr, "%s%.6f", c == 0 ? "[" : ", ", static_cast<double>(g.at(0, c)));
        }
        std::fprintf(stderr, "]\n");
    };
    std::fprintf(stderr, "[ GGUF ] Norm gamma diagnostics (layer 0, first 5 values):\n");
    {
        const Mat& g = layers[0].input_layernorm.gamma.data();
        print_head("  input_layernorm gamma (stored, after -1 fix): ", g, 5);
        Mat eff = g;
        for (float& v : eff.data) {
            v += 1.0f;
        }
        print_head("  input_layernorm effective scale (1+gamma):    ", eff, 5);
    }
    {
        const Mat& g = layers[0].post_attention_layernorm.gamma.data();
        print_head("  post_attn_layernorm gamma (stored):           ", g, 5);
        Mat eff = g;
        for (float& v : eff.data) {
            v += 1.0f;
        }
        print_head("  post_attn_layernorm effective scale (1+gamma): ", eff, 5);
        // A norm whose effective scale collapses toward zero would silently
        // erase the residual stream, so the whole distribution is summarized.
        double sum = 0.0;
        float lo = eff.data.empty() ? 0.0f : eff.data[0];
        float hi = lo;
        std::size_t near_zero = 0;
        for (float v : eff.data) {
            sum += static_cast<double>(v);
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            if (std::fabs(v) < 0.1f) {
                ++near_zero;
            }
        }
        const double mean = eff.data.empty() ? 0.0 : sum / static_cast<double>(eff.data.size());
        std::fprintf(stderr,
                     "  post_attn_layernorm eff-scale stats: mean=%.4f min=%.4f max=%.4f "
                     "near-zero(<0.1)=%zu/%zu\n",
                     mean, static_cast<double>(lo), static_cast<double>(hi), near_zero,
                     eff.data.size());
    }
    print_head("  q_norm gamma (stored):                        ",
               layers[0].self_attn.q_norm.gamma.data(), 5);
    std::fprintf(stderr, "[ GGUF ] Done. Loaded %zu tensors.\n", loaded);
    return {};
}

// =============================================================================
// Binary weight cache
// =============================================================================
//
// Format:  MAGIC(8) | N_RECORDS(u32le) | record* | EOF
// Record:  name_len(u32le) | name(utf8) | dtype(u8: 0=f32, 1=bf16) |
//          rows(u32le) | cols(u32le) | data(rows*cols * dtype_bytes)
//
// The point is to skip safetensors JSON parsing and BF16 conversion on every
// subsequent run: the records land in memory in exactly the layout the model
// wants.

namespace {

constexpr char kCacheMagic[8] = {'G', '3', 'C', 'A', 'C', 'H', 'E', '1'};

void write_u32(std::ostream& out, std::uint32_t v) {
    const std::uint8_t bytes[4] = {static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v >> 8),
                                   static_cast<std::uint8_t>(v >> 16),
                                   static_cast<std::uint8_t>(v >> 24)};
    out.write(reinterpret_cast<const char*>(bytes), 4);
}

/// One record's header plus a pointer to its payload, which stays owned by the
/// model -- nothing is copied until it is written.
struct CacheRecord {
    std::string name;
    std::uint8_t dtype;  // 0 = f32, 1 = bf16
    std::size_t rows;
    std::size_t cols;
    const void* data;
    std::size_t byte_len;
};

}  // namespace

Result<void> Gemma3Model::save_cache(const std::string& path) const {
    std::vector<CacheRecord> records;

    if (embed_bf16) {
        records.push_back({"model.embed_tokens.weight", 1, embed_bf16->rows, embed_bf16->cols,
                           embed_bf16->data->data(), embed_bf16->data->size() * 2});
    }
    {
        const Mat& g = norm.gamma.data();
        records.push_back({"model.norm.weight", 0, g.rows, g.cols, g.data.data(),
                           g.data.size() * 4});
    }

    for (std::size_t i = 0; i < layers.size(); ++i) {
        const Gemma3Block& layer = layers[i];
        const std::string prefix = "model.layers." + std::to_string(i);

        const auto push_norm = [&](const std::string& suffix, const TensorNode& node) {
            const Mat& g = node.data();
            records.push_back(
                {prefix + suffix, 0, g.rows, g.cols, g.data.data(), g.data.size() * 4});
        };
        push_norm(".input_layernorm.weight", layer.input_layernorm.gamma);
        push_norm(".post_attention_layernorm.weight", layer.post_attention_layernorm.gamma);
        push_norm(".pre_feedforward_layernorm.weight", layer.pre_feedforward_layernorm.gamma);
        push_norm(".post_feedforward_layernorm.weight", layer.post_feedforward_layernorm.gamma);
        push_norm(".self_attn.q_norm.weight", layer.self_attn.q_norm.gamma);
        push_norm(".self_attn.k_norm.weight", layer.self_attn.k_norm.gamma);

        // Only BF16 projections are cached; a quantized model has nothing to
        // write back in this format.
        const auto push_bf16 = [&](const std::string& suffix, const Linear2& lin) {
            if (lin.bf16_weight) {
                const MatBf16& b = *lin.bf16_weight;
                records.push_back({prefix + suffix, 1, b.rows, b.cols, b.data->data(),
                                   b.data->size() * 2});
            }
        };
        push_bf16(".self_attn.q_proj.weight", layer.self_attn.q_proj);
        push_bf16(".self_attn.k_proj.weight", layer.self_attn.k_proj);
        push_bf16(".self_attn.v_proj.weight", layer.self_attn.v_proj);
        push_bf16(".self_attn.o_proj.weight", layer.self_attn.o_proj);
        push_bf16(".mlp.gate_proj.weight", layer.mlp.gate_proj);
        push_bf16(".mlp.up_proj.weight", layer.mlp.up_proj);
        push_bf16(".mlp.down_proj.weight", layer.mlp.down_proj);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return err("cannot create cache file " + path);
    }
    out.write(kCacheMagic, 8);
    write_u32(out, static_cast<std::uint32_t>(records.size()));
    for (const CacheRecord& r : records) {
        write_u32(out, static_cast<std::uint32_t>(r.name.size()));
        out.write(r.name.data(), static_cast<std::streamsize>(r.name.size()));
        out.put(static_cast<char>(r.dtype));
        write_u32(out, static_cast<std::uint32_t>(r.rows));
        write_u32(out, static_cast<std::uint32_t>(r.cols));
        out.write(static_cast<const char*>(r.data), static_cast<std::streamsize>(r.byte_len));
    }
    if (!out) {
        return err("write failed for cache file " + path);
    }
    return {};
}

Result<bool> Gemma3Model::load_cache(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "[ Cache ] No cache file at %s\n", path.c_str());
        return false;
    }
    std::error_code ec;
    const std::uintmax_t file_len = std::filesystem::file_size(path, ec);
    std::fprintf(stderr, "[ Cache ] Streaming cache file %s (%llu MB)...\n", path.c_str(),
                 static_cast<unsigned long long>(ec ? 0 : file_len / 1048576));

    char magic[8] = {};
    in.read(magic, 8);
    if (!in || std::memcmp(magic, kCacheMagic, 8) != 0) {
        std::fprintf(stderr, "[ Cache ] Bad magic, ignoring cache.\n");
        return false;
    }

    const auto read_u32 = [&in]() -> Result<std::uint32_t> {
        std::uint8_t b[4];
        in.read(reinterpret_cast<char*>(b), 4);
        if (!in) {
            return err("unexpected end of cache file");
        }
        return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
               (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
    };

    RT_TRY(n_records, read_u32());
    std::fprintf(stderr, "[ Cache ] Loading %u records...\n", n_records);

    for (std::uint32_t i = 0; i < n_records; ++i) {
        RT_TRY(name_len, read_u32());
        std::string name(name_len, '\0');
        in.read(name.data(), name_len);
        const int dtype = in.get();
        if (!in) {
            return err("unexpected end of cache file");
        }
        RT_TRY(rows, read_u32());
        RT_TRY(cols, read_u32());
        const std::size_t n_elems = static_cast<std::size_t>(rows) * cols;

        SafeTensor t;
        t.name = std::move(name);
        t.shape = {rows, cols};
        if (dtype == 0) {
            t.data.resize(n_elems);
            in.read(reinterpret_cast<char*>(t.data.data()),
                    static_cast<std::streamsize>(n_elems * 4));
        } else {
            // BF16 goes straight into u16 storage -- no f32 round trip.
            std::vector<std::uint16_t> bits(n_elems);
            in.read(reinterpret_cast<char*>(bits.data()),
                    static_cast<std::streamsize>(n_elems * 2));
            t.bf16_data = std::move(bits);
        }
        if (!in) {
            return err("unexpected end of cache file");
        }
        apply_tensor(*this, std::move(t));
    }
    return true;
}

// =============================================================================
// NgramDraftEngine
// =============================================================================

std::vector<std::size_t> NgramDraftEngine::draft() const {
    const std::size_t len = history_.size();
    if (len < 2) {
        return {};
    }

    // Longest match first: a 4-gram match is a much stronger signal than a
    // 2-gram one, so shorter contexts are only a fallback.
    for (std::size_t n = std::min(max_n_, len); n >= 2; --n) {
        for (std::size_t start = 0; start + n < len; ++start) {
            if (!std::equal(history_.begin() + static_cast<std::ptrdiff_t>(start),
                            history_.begin() + static_cast<std::ptrdiff_t>(start + n),
                            history_.end() - static_cast<std::ptrdiff_t>(n))) {
                continue;
            }
            const std::size_t cont_start = start + n;
            const std::size_t cont_end = std::min(cont_start + max_draft_, len);
            if (cont_start < cont_end) {
                return {history_.begin() + static_cast<std::ptrdiff_t>(cont_start),
                        history_.begin() + static_cast<std::ptrdiff_t>(cont_end)};
            }
        }
    }
    return {};
}

// =============================================================================
// Generation
// =============================================================================

void Gemma3Model::generate_streaming(const std::vector<std::size_t>& token_ids,
                                     std::size_t max_new, float temperature, std::size_t top_k,
                                     std::uint64_t seed,
                                     const std::function<void(std::size_t)>& callback) const {
    SamplingParams params;
    params.temperature = temperature;
    params.top_k = top_k;
    params.top_p = 1.0f;
    params.repetition_penalty = 1.0f;
    params.seed = seed;
    params.eos_token_id = config.eos_token_id;

    LcgRng rng(seed);
    std::vector<std::size_t> seen = token_ids;

    const TensorNode prompt_logits = forward(token_ids);
    const Mat& logits = prompt_logits.data();

    std::size_t next = sample_token(logits, logits.rows - 1, params, seen, rng);
    callback(next);
    seen.push_back(next);
    if (params.eos_token_id == next) {
        return;
    }

    std::size_t prev = next;
    for (std::size_t i = 1; i < max_new; ++i) {
        const TensorNode step_logits = forward({prev});
        prev = sample_token(step_logits.data(), 0, params, seen, rng);
        callback(prev);
        seen.push_back(prev);
        if (params.eos_token_id == prev) {
            break;
        }
    }
}

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// Log the largest logits and the gap between the top two.
///
/// A collapsing gap is the signature of a broken forward pass -- RoPE drift,
/// for instance -- long before the text itself looks wrong.
void print_logit_debug(const char* prefix, const Mat& logits) {
    std::vector<std::pair<std::size_t, float>> indexed(logits.data.size());
    for (std::size_t i = 0; i < logits.data.size(); ++i) {
        indexed[i] = {i, logits.data[i]};
    }
    const std::size_t k = std::min<std::size_t>(5, indexed.size());
    std::partial_sort(indexed.begin(), indexed.begin() + static_cast<std::ptrdiff_t>(k),
                      indexed.end(),
                      [](const auto& a, const auto& b) { return b.second < a.second; });
    const float l_max = indexed.empty() ? 0.0f : indexed[0].second;
    const float gap = indexed.size() >= 2 ? indexed[0].second - indexed[1].second : 0.0f;
    std::fprintf(stderr, "%slogit_max=%.2f  gap=%.2f  top-5: ", prefix, static_cast<double>(l_max),
                 static_cast<double>(gap));
    for (std::size_t i = 0; i < k; ++i) {
        std::fprintf(stderr, "%s(%zu, %.2f)", i == 0 ? "[" : ", ", indexed[i].first,
                     static_cast<double>(indexed[i].second));
    }
    std::fprintf(stderr, "]\n");
}

#if !RT_FEATURE_METAL
/// Root-mean-square of a matrix's values, for tracking residual-stream growth.
/// Only the CPU decode path can see the intermediate hidden states.
[[nodiscard]] float rms(const Mat& m) {
    double sum = 0.0;
    for (float v : m.data) {
        sum += static_cast<double>(v) * static_cast<double>(v);
    }
    return m.data.empty() ? 0.0f
                          : static_cast<float>(std::sqrt(sum / static_cast<double>(m.data.size())));
}
#endif

}  // namespace

void Gemma3Model::generate_cached_streaming(const std::vector<std::size_t>& token_ids,
                                            std::size_t max_new, float temperature,
                                            std::size_t top_k, float top_p,
                                            float repetition_penalty, std::uint64_t seed,
                                            bool debug, std::size_t draft_len,
                                            const std::function<void(std::size_t)>& callback) {
    SamplingParams params;
    params.temperature = temperature;
    params.top_k = top_k;
    params.top_p = top_p;
    params.repetition_penalty = repetition_penalty;
    params.seed = seed;
    params.eos_token_id = config.eos_token_id;

    print_rss("before KV cache alloc");

    NgramDraftEngine ngram(4, draft_len);
    // Seeding with the prompt lets the very first drafts match against it.
    ngram.record_many(token_ids);

    // Gemma 3 has two stop tokens: 1 (<eos>) and 106 (<end_of_turn>).
    const auto is_eos = [](std::size_t tok) { return tok == 1 || tok == 106; };

#if RT_FEATURE_METAL
    constexpr bool use_metal = true;
    // With Metal the CPU cache only serves prefill and is freed afterwards, so
    // it only needs to hold the prompt.
    const std::size_t cache_cap = token_ids.size();
#else
    constexpr bool use_metal = false;
    const std::size_t cache_cap = token_ids.size() + max_new;
#endif
    Gemma3KvCache cache(config, cache_cap);
    const std::size_t h = config.hidden_size;
    const float scale = std::sqrt(static_cast<float>(h));
    LcgRng rng(seed);

    // `seen` holds only *generated* tokens. Including the prompt would penalize
    // the end-of-turn token that the chat template already contains, making the
    // model far less likely to stop on its own.
    std::vector<std::size_t> seen;

    // ----- Prefill -----
    const std::size_t t_prompt = token_ids.size();
    const Clock::time_point prefill_start = Clock::now();

    std::vector<float> embed_data(t_prompt * h);
    for (std::size_t r = 0; r < t_prompt; ++r) {
        const std::size_t tok = token_ids[r];
        if (embed_bf16) {
            const std::vector<std::uint16_t>& bits = *embed_bf16->data;
            for (std::size_t c = 0; c < h; ++c) {
                embed_data[r * h + c] = bf16_to_f32(bits[tok * h + c]) * scale;
            }
        } else {
            const Mat& te = embed_tokens.data();
            for (std::size_t c = 0; c < h; ++c) {
                embed_data[r * h + c] = te.at(tok, c) * scale;
            }
        }
    }

    TensorNode x = TensorNode::leaf(Mat(std::move(embed_data), t_prompt, h));
    for (std::size_t i = 0; i < layers.size(); ++i) {
        x = layers[i].forward_cached(x, cache.layers[i]);
    }
    // lm_head only runs on the last row, avoiding a T x vocab matmul.
    const Mat last_row =
        Mat::from_fn(1, h, [&](std::size_t, std::size_t c) { return x.data().at(t_prompt - 1, c); });
    const TensorNode normed_final = norm.forward_gemma3(TensorNode::leaf(last_row));
    const TensorNode logits_node = lm_head.forward(normed_final);

    const double prefill_ms = ms_since(prefill_start);
    std::fprintf(stderr, "[ Gemma3 ] Prefill: %zu tokens in %.0f ms (%.1f tok/s)\n", t_prompt,
                 prefill_ms, static_cast<double>(t_prompt) / (prefill_ms / 1000.0));
    print_rss("after prefill");

    if (debug) {
        print_logit_debug("[ Gemma3-dbg ] prefill ", logits_node.data());
    }

    const std::size_t first = sample_token(logits_node.data(), 0, params, seen, rng);

    // Backward closures hold reference cycles, so the prefill graph has to be
    // broken explicitly rather than left to the shared_ptr refcounts.
    x.free_graph();
    logits_node.free_graph();
    release_memory_to_os();

    callback(first);
    seen.push_back(first);
    ngram.record(first);
    if (is_eos(first)) {
        std::fprintf(stderr, "[ Gemma3 ] EOS after first token - generation complete.\n");
        return;
    }

#if RT_FEATURE_METAL
    // The GPU path encodes the whole forward pass into one command buffer,
    // removing ~120 ms of per-dispatch overhead per step.
    std::unique_ptr<MetalDecodeContext> metal_ctx = MetalDecodeContext::create(*this, draft_len);
    metal_ctx->print_metal_memory();
    print_rss("after MetalDecodeContext::create");
    {
        // Metal JIT-compiles the shaders on first dispatch, which would
        // otherwise stall the first real step by ~25 s. The warmup writes to
        // KV position 0, so it has to run before the cache is synced.
        const Clock::time_point t_warmup = Clock::now();
        (void)metal_ctx->decode_step(0, 0);
        std::fprintf(stderr, "[ Metal ] GPU warmup in %.0f ms\n", ms_since(t_warmup));
    }
    print_rss("after GPU warmup");
    metal_ctx->sync_kv_from_cpu(cache);

    // Everything now lives in GPU buffers: ~3 GB of layer weights and ~0.5 GB
    // of CPU KV cache can go.
    clear_cpu_weights();
    cache.free();
    release_memory_to_os();
    print_rss("after clear_cpu_weights + cache.free");
#endif

    print_rss("decode loop start");

    // ----- Decode loop -----
    const Clock::time_point decode_start = Clock::now();
    std::size_t prev = first;
    std::size_t n_decoded = 1;

    // On the Metal path the CPU cache is gone, so the sequence length is
    // tracked here rather than read back from it.
    [[maybe_unused]] std::size_t metal_seq_len = t_prompt;

    double t_embed_ms = 0.0;
    double t_layers_ms = 0.0;
    double t_lmhead_ms = 0.0;
    [[maybe_unused]] double t_metal_ms = 0.0;
    bool profile_printed = false;

    std::uint64_t spec_attempts = 0;
    std::uint64_t spec_accepted = 0;
    std::uint64_t spec_drafted = 0;

    std::size_t step = 1;
    bool hit_eos = false;

    while (step < max_new && !hit_eos) {
        bool did_speculate = false;

#if RT_FEATURE_METAL
        if (draft_len > 0) {
            const std::vector<std::size_t> draft = ngram.draft();
            if (!draft.empty()) {
                const std::size_t k = std::min(draft.size(), max_new - step);
                // Row i of the batch predicts what follows verify_tokens[i], so
                // the batch is [prev, draft[0], ..., draft[k-2]]: k rows that
                // check drafts 0..k-1.
                std::vector<std::size_t> verify_tokens;
                verify_tokens.reserve(k);
                verify_tokens.push_back(prev);
                for (std::size_t i = 0; i + 1 < k; ++i) {
                    verify_tokens.push_back(draft[i]);
                }

                const Clock::time_point t_m = Clock::now();
                const std::vector<float> all_logits =
                    metal_ctx->decode_step_batch(verify_tokens, metal_seq_len);
                t_metal_ms += ms_since(t_m);

                const std::size_t vocab = metal_ctx->lm_head_vocab();
                ++spec_attempts;
                spec_drafted += k;

                std::size_t accepted = 0;
                for (std::size_t i = 0; i < k; ++i) {
                    const auto row_begin =
                        all_logits.begin() + static_cast<std::ptrdiff_t>(i * vocab);
                    const Mat row_logits(
                        std::vector<float>(row_begin,
                                           row_begin + static_cast<std::ptrdiff_t>(vocab)),
                        1, vocab);
                    const std::size_t sampled = sample_token(row_logits, 0, params, seen, rng);

                    callback(sampled);
                    seen.push_back(sampled);
                    ngram.record(sampled);
                    prev = sampled;
                    ++accepted;
                    if (sampled != draft[i]) {
                        // Diverged: the sampled token still counts, but every
                        // later draft was conditioned on a token we rejected.
                        break;
                    }
                }

                spec_accepted += accepted;
                n_decoded += accepted;
                metal_seq_len += accepted;
                step += accepted;
                if (is_eos(prev)) {
                    hit_eos = true;
                }

                if (!profile_printed) {
                    profile_printed = true;
                    std::fprintf(
                        stderr,
                        "[ Gemma3 ] Step-1 (Metal speculative, %zu->%zu accepted): %.1fms\n", k,
                        accepted, t_metal_ms);
                }
                did_speculate = true;
            }
        }
#endif

        if (hit_eos) {
            break;
        }
        if (did_speculate) {
            continue;
        }

        Mat logits_data(std::vector<float>{}, 0, 0);
#if RT_FEATURE_METAL
        {
            const Clock::time_point t_m = Clock::now();
            std::vector<float> logits_vec = metal_ctx->decode_step(prev, metal_seq_len);
            t_metal_ms += ms_since(t_m);
            ++metal_seq_len;
            const std::size_t n = logits_vec.size();
            logits_data = Mat(std::move(logits_vec), 1, n);
        }
#else
        {
            const Clock::time_point t0 = Clock::now();
            std::vector<float> embed_vec(h);
            if (embed_bf16) {
                const std::vector<std::uint16_t>& bits = *embed_bf16->data;
                for (std::size_t c = 0; c < h; ++c) {
                    embed_vec[c] = bf16_to_f32(bits[prev * h + c]) * scale;
                }
            } else {
                const Mat& te = embed_tokens.data();
                for (std::size_t c = 0; c < h; ++c) {
                    embed_vec[c] = te.at(prev, c) * scale;
                }
            }
            TensorNode xs = TensorNode::leaf(Mat(std::move(embed_vec), 1, h));
            t_embed_ms += ms_since(t0);

            const Clock::time_point t1 = Clock::now();
            for (std::size_t i = 0; i < layers.size(); ++i) {
                xs = layers[i].forward_cached(xs, cache.layers[i]);
            }
            t_layers_ms += ms_since(t1);

            const Clock::time_point t2 = Clock::now();
            const TensorNode normed_x = norm.forward_gemma3(xs);
            const TensorNode step_logits = lm_head.forward(normed_x);
            t_lmhead_ms += ms_since(t2);

            if (debug) {
                char prefix[128];
                std::snprintf(prefix, sizeof(prefix),
                              "[ Gemma3-dbg ] step=%zu prev_tok=%zu  h_rms=%.2f  hn_rms=%.2f  ",
                              step, prev, static_cast<double>(rms(xs.data())),
                              static_cast<double>(rms(normed_x.data())));
                print_logit_debug(prefix, step_logits.data());
            }
            logits_data = step_logits.data();
        }
#endif

        prev = sample_token(logits_data, 0, params, seen, rng);
        callback(prev);
        seen.push_back(prev);
        ngram.record(prev);
        ++n_decoded;
        ++step;

        if (!profile_printed) {
            profile_printed = true;
            if constexpr (use_metal) {
                std::fprintf(stderr, "[ Gemma3 ] Step-1 (Metal graph): %.1fms\n", t_metal_ms);
            } else {
                std::fprintf(stderr,
                             "[ Gemma3 ] Step-1 breakdown: embed=%.1fms  layers=%.1fms  "
                             "lm_head=%.1fms\n",
                             t_embed_ms, t_layers_ms, t_lmhead_ms);
            }
        }

        if (is_eos(prev)) {
            break;
        }
    }

    const double decode_ms = ms_since(decode_start);
    std::fprintf(stderr, "[ Gemma3 ] Decode:  %zu tokens in %.0f ms (%.1f tok/s)\n", n_decoded,
                 decode_ms, static_cast<double>(n_decoded) / std::max(decode_ms / 1000.0, 1e-9));
    if (n_decoded > 1) {
        const double n = static_cast<double>(n_decoded);
        if constexpr (use_metal) {
            std::fprintf(stderr, "[ Gemma3 ] Avg/step (Metal graph): %.1fms\n", t_metal_ms / n);
        } else {
            std::fprintf(stderr,
                         "[ Gemma3 ] Avg/step: embed=%.1fms  layers=%.1fms  lm_head=%.1fms\n",
                         t_embed_ms / n, t_layers_ms / n, t_lmhead_ms / n);
        }
    }
    if (spec_attempts > 0) {
        const double accept_rate =
            static_cast<double>(spec_accepted) / static_cast<double>(spec_drafted);
        std::fprintf(stderr, "[ Gemma3 ] Speculative: %llu attempts, %llu/%llu accepted (%.0f%%)\n",
                     static_cast<unsigned long long>(spec_attempts),
                     static_cast<unsigned long long>(spec_accepted),
                     static_cast<unsigned long long>(spec_drafted), accept_rate * 100.0);
    }
}

}  // namespace rt

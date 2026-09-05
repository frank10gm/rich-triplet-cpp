#include "rt/metal_omnivoice.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "rt/omnivoice_msl.hpp"

namespace rt {

namespace {

// A GEMM tile is 64x64, so both the sequence length and every weight's row
// count are rounded up to that. The pad rows are zero and stay harmless: every
// operator except attention is row-independent, and attention is dispatched
// over the true length.
constexpr std::size_t kTile = 64;

// The attention kernel keeps one query's scores in threadgroup memory.
constexpr std::size_t kMaxAttentionLength = 2048;

[[nodiscard]] std::size_t round_up(std::size_t v, std::size_t to) {
    return (v + to - 1) / to * to;
}

[[nodiscard]] std::uint16_t f32_bits_to_bf16(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

}  // namespace

// =============================================================================
// Impl
// =============================================================================

struct MetalOmniContext::Impl {
    Config6 config;
    std::size_t max_tokens = 0;
    std::size_t max_padded = 0;
    /// Audio head rows, padded to a tile multiple.
    std::size_t head_rows = 0;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> gemm = nil;
    id<MTLComputePipelineState> rms_norm = nil;
    id<MTLComputePipelineState> head_rms_norm = nil;
    id<MTLComputePipelineState> rope = nil;
    id<MTLComputePipelineState> attention = nil;
    id<MTLComputePipelineState> silu_mul = nil;
    id<MTLComputePipelineState> add = nil;
    id<MTLComputePipelineState> zero = nil;

    struct Layer {
        id<MTLBuffer> input_norm = nil;
        id<MTLBuffer> post_norm = nil;
        id<MTLBuffer> q_norm = nil;
        id<MTLBuffer> k_norm = nil;
        id<MTLBuffer> q = nil;
        id<MTLBuffer> k = nil;
        id<MTLBuffer> v = nil;
        id<MTLBuffer> o = nil;
        id<MTLBuffer> gate = nil;
        id<MTLBuffer> up = nil;
        id<MTLBuffer> down = nil;
    };
    std::vector<Layer> layers;

    id<MTLBuffer> final_norm = nil;
    id<MTLBuffer> head = nil;
    id<MTLBuffer> inv_freq = nil;

    /// The embedding tables stay in CPU memory and are shared with the model
    /// rather than copied -- `MatBf16` is reference-counted, so this costs a
    /// pointer. Only `T` rows are read per pass, and the text table is 310 MB.
    std::optional<MatBf16> text_embed;
    std::optional<MatBf16> audio_embed;

    [[nodiscard]] const std::uint16_t* text_row(std::size_t r) const {
        return text_embed->data->data() + r * text_embed->cols;
    }
    [[nodiscard]] const std::uint16_t* audio_row(std::size_t r) const {
        return audio_embed->data->data() + r * audio_embed->cols;
    }

    // Activation scratch, all sized for `max_padded` rows.
    id<MTLBuffer> x = nil;
    id<MTLBuffer> normed = nil;
    id<MTLBuffer> buf_q = nil;
    id<MTLBuffer> buf_k = nil;
    id<MTLBuffer> buf_v = nil;
    id<MTLBuffer> attn = nil;
    id<MTLBuffer> proj = nil;
    id<MTLBuffer> buf_gate = nil;
    id<MTLBuffer> buf_up = nil;
    id<MTLBuffer> logits = nil;

    std::size_t bytes = 0;

    [[nodiscard]] id<MTLBuffer> alloc(std::size_t n_bytes) {
        id<MTLBuffer> b = [device newBufferWithLength:n_bytes
                                              options:MTLResourceStorageModeShared];
        bytes += n_bytes;
        return b;
    }

    /// Upload a projection's BF16 weight, padding its row count to a tile
    /// multiple, and free the CPU copy.
    [[nodiscard]] id<MTLBuffer> upload_projection(Linear2& linear, std::size_t out_features,
                                                  std::size_t in_features) {
        const std::size_t rows = round_up(out_features, kTile);
        std::vector<std::uint16_t> padded(rows * in_features, 0);

        if (linear.bf16_weight) {
            const MatBf16& w = *linear.bf16_weight;
            std::memcpy(padded.data(), w.data->data(),
                        std::min(w.data->size(), out_features * in_features) * sizeof(std::uint16_t));
        } else {
            const Mat& w = linear.weight.data();
            for (std::size_t i = 0; i < out_features * in_features && i < w.data.size(); ++i) {
                padded[i] = f32_bits_to_bf16(w.data[i]);
            }
        }

        id<MTLBuffer> buf = alloc(padded.size() * sizeof(std::uint16_t));
        std::memcpy(buf.contents, padded.data(), padded.size() * sizeof(std::uint16_t));
        linear.clear_weight_data();
        return buf;
    }

    [[nodiscard]] id<MTLBuffer> upload_floats(const std::vector<float>& v) {
        id<MTLBuffer> buf = alloc(std::max<std::size_t>(v.size(), 1) * sizeof(float));
        if (!v.empty()) {
            std::memcpy(buf.contents, v.data(), v.size() * sizeof(float));
        }
        return buf;
    }
};

// =============================================================================
// Pipelines
// =============================================================================

namespace {

[[nodiscard]] id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                                        id<MTLLibrary> library,
                                                        const char* name, std::string& error) {
    if (!error.empty()) {
        return nil;
    }
    id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (fn == nil) {
        error = std::string("metal omnivoice: no kernel named '") + name + "'";
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (pso == nil) {
        error = std::string("metal omnivoice: pipeline '") + name +
                "' failed: " + err.localizedDescription.UTF8String;
    }
    return pso;
}

}  // namespace

// =============================================================================
// Construction
// =============================================================================

MetalOmniContext::MetalOmniContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalOmniContext::~MetalOmniContext() = default;

std::unique_ptr<MetalOmniContext> MetalOmniContext::create(OmniLm& lm, std::size_t max_tokens) {
    auto impl = std::make_unique<Impl>();
    impl->config = lm.config;
    impl->max_tokens = max_tokens;
    impl->max_padded = round_up(std::max<std::size_t>(max_tokens, 1), kTile);

    impl->device = MTLCreateSystemDefaultDevice();
    if (impl->device == nil) {
        std::fprintf(stderr, "[ Metal ] no GPU device\n");
        return nullptr;
    }
    impl->queue = [impl->device newCommandQueue];

    NSError* err = nil;
    id<MTLLibrary> library =
        [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kOmnivoiceMsl]
                                   options:nil
                                     error:&err];
    if (library == nil) {
        std::fprintf(stderr, "[ Metal ] shader compilation failed: %s\n",
                     err.localizedDescription.UTF8String);
        return nullptr;
    }

    std::string error;
    impl->gemm = make_pipeline(impl->device, library, "omni_gemm_bt", error);
    impl->rms_norm = make_pipeline(impl->device, library, "omni_rms_norm", error);
    impl->head_rms_norm = make_pipeline(impl->device, library, "omni_head_rms_norm", error);
    impl->rope = make_pipeline(impl->device, library, "omni_rope", error);
    impl->attention = make_pipeline(impl->device, library, "omni_attention", error);
    impl->silu_mul = make_pipeline(impl->device, library, "omni_silu_mul", error);
    impl->add = make_pipeline(impl->device, library, "omni_add", error);
    impl->zero = make_pipeline(impl->device, library, "omni_zero", error);
    if (!error.empty()) {
        std::fprintf(stderr, "[ Metal ] %s\n", error.c_str());
        return nullptr;
    }

    const Config6& c = impl->config;
    const std::size_t hidden = c.hidden_size;
    const std::size_t q_dim = c.num_attention_heads * c.head_dim;
    const std::size_t kv_dim = c.num_key_value_heads * c.head_dim;

    // -- weights -------------------------------------------------------------
    impl->layers.resize(lm.layers.size());
    for (std::size_t i = 0; i < lm.layers.size(); ++i) {
        OmniBlock& block = lm.layers[i];
        Impl::Layer& dst = impl->layers[i];

        dst.input_norm = impl->upload_floats(block.input_layernorm.gamma.data().data);
        dst.post_norm = impl->upload_floats(block.post_attention_layernorm.gamma.data().data);
        dst.q_norm = impl->upload_floats(block.self_attn.q_norm.gamma.data().data);
        dst.k_norm = impl->upload_floats(block.self_attn.k_norm.gamma.data().data);

        dst.q = impl->upload_projection(block.self_attn.q_proj, q_dim, hidden);
        dst.k = impl->upload_projection(block.self_attn.k_proj, kv_dim, hidden);
        dst.v = impl->upload_projection(block.self_attn.v_proj, kv_dim, hidden);
        dst.o = impl->upload_projection(block.self_attn.o_proj, hidden, q_dim);
        dst.gate = impl->upload_projection(block.mlp.gate_proj, c.intermediate_size, hidden);
        dst.up = impl->upload_projection(block.mlp.up_proj, c.intermediate_size, hidden);
        dst.down = impl->upload_projection(block.mlp.down_proj, hidden, c.intermediate_size);
    }

    if (!lm.text_embed || !lm.audio_embed) {
        std::fprintf(stderr, "[ Metal ] the model has no embedding tables loaded\n");
        return nullptr;
    }
    impl->text_embed = lm.text_embed;
    impl->audio_embed = lm.audio_embed;

    impl->final_norm = impl->upload_floats(lm.norm.gamma.data().data);
    impl->head_rows = round_up(c.audio_table_size(), kTile);
    impl->head = impl->upload_projection(lm.audio_head, c.audio_table_size(), hidden);
    impl->inv_freq = impl->upload_floats(lm.inv_freq_cache);

    // -- scratch -------------------------------------------------------------
    const std::size_t rows = impl->max_padded;
    impl->x = impl->alloc(rows * hidden * sizeof(float));
    impl->normed = impl->alloc(rows * hidden * sizeof(float));
    impl->proj = impl->alloc(rows * hidden * sizeof(float));
    impl->buf_q = impl->alloc(rows * q_dim * sizeof(float));
    impl->buf_k = impl->alloc(rows * kv_dim * sizeof(float));
    impl->buf_v = impl->alloc(rows * kv_dim * sizeof(float));
    impl->attn = impl->alloc(rows * q_dim * sizeof(float));
    impl->buf_gate = impl->alloc(rows * c.intermediate_size * sizeof(float));
    impl->buf_up = impl->alloc(rows * c.intermediate_size * sizeof(float));
    impl->logits = impl->alloc(rows * impl->head_rows * sizeof(float));

    return std::unique_ptr<MetalOmniContext>(new MetalOmniContext(std::move(impl)));
}

std::size_t MetalOmniContext::max_tokens() const { return impl_->max_tokens; }
std::size_t MetalOmniContext::buffer_bytes() const { return impl_->bytes; }

// =============================================================================
// Forward
// =============================================================================

namespace {

/// `C[Mp, N] = A[Mp, K] @ W[N, K]^T`, both dimensions already tile-aligned.
void encode_gemm(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                 id<MTLBuffer> a, id<MTLBuffer> w, id<MTLBuffer> c, std::size_t m_padded,
                 std::size_t k, std::size_t n_padded) {
    auto k32 = static_cast<std::uint32_t>(k);
    auto n32 = static_cast<std::uint32_t>(n_padded);
    [enc setComputePipelineState:pso];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:c offset:0 atIndex:2];
    [enc setBytes:&k32 length:sizeof(k32) atIndex:3];
    [enc setBytes:&n32 length:sizeof(n32) atIndex:4];
    // Four simdgroups of 32 threads cover the 64x64 tile.
    [enc dispatchThreadgroups:MTLSizeMake(n_padded / kTile, m_padded / kTile, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void encode_elementwise(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                        id<MTLBuffer> dst, id<MTLBuffer> src, std::size_t n) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:dst offset:0 atIndex:0];
    if (src != nil) {
        [enc setBuffer:src offset:0 atIndex:1];
    }
    const NSUInteger width = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

}  // namespace

Result<Mat> MetalOmniContext::forward(const std::vector<OmniToken>& tokens) const {
    const Impl& s = *impl_;
    const Config6& c = s.config;

    if (tokens.empty()) {
        return err("metal omnivoice: empty sequence");
    }
    if (tokens.size() > s.max_tokens) {
        return err("metal omnivoice: " + std::to_string(tokens.size()) +
                   " tokens exceeds the " + std::to_string(s.max_tokens) +
                   " this context was built for");
    }
    if (tokens.size() > kMaxAttentionLength) {
        return err("metal omnivoice: the attention kernel holds at most " +
                   std::to_string(kMaxAttentionLength) + " positions");
    }

    const std::size_t t = tokens.size();
    const std::size_t tp = round_up(t, kTile);
    const std::size_t hidden = c.hidden_size;
    const std::size_t q_dim = c.num_attention_heads * c.head_dim;
    const std::size_t kv_dim = c.num_key_value_heads * c.head_dim;

    // The embedding tables stay on the CPU: only `t` rows are read, and the
    // text table is 310 MB.
    {
        auto* dst = static_cast<float*>(s.x.contents);
        std::memset(dst, 0, tp * hidden * sizeof(float));
        for (std::size_t i = 0; i < t; ++i) {
            const OmniToken& token = tokens[i];
            float* row = dst + i * hidden;
            if (token.is_audio()) {
                if (token.audio.size() != c.num_audio_codebook) {
                    return err("metal omnivoice: position " + std::to_string(i) + " has " +
                               std::to_string(token.audio.size()) + " codebooks");
                }
                // A position's embedding is the sum across all eight codebooks,
                // so a fully masked one is still well defined.
                for (std::size_t cb = 0; cb < c.num_audio_codebook; ++cb) {
                    if (token.audio[cb] >= c.audio_vocab_size) {
                        return err("metal omnivoice: code " +
                                   std::to_string(token.audio[cb]) + " is out of range");
                    }
                    const std::size_t r = c.audio_row(cb, token.audio[cb]);
                    const std::uint16_t* src = s.audio_row(r);
                    for (std::size_t d = 0; d < hidden; ++d) {
                        std::uint32_t bits = static_cast<std::uint32_t>(src[d]) << 16;
                        float v = 0.0f;
                        std::memcpy(&v, &bits, sizeof(v));
                        row[d] += v;
                    }
                }
            } else {
                if (token.text_id >= c.text_vocab_size) {
                    return err("metal omnivoice: text id " + std::to_string(token.text_id) +
                               " is out of range");
                }
                const std::uint16_t* src = s.text_row(token.text_id);
                for (std::size_t d = 0; d < hidden; ++d) {
                    std::uint32_t bits = static_cast<std::uint32_t>(src[d]) << 16;
                    float v = 0.0f;
                    std::memcpy(&v, &bits, sizeof(v));
                    row[d] = v;
                }
            }
        }
    }

    id<MTLCommandBuffer> cb = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    const auto u32 = [](std::size_t v) { return static_cast<std::uint32_t>(v); };
    const std::uint32_t hidden32 = u32(hidden);
    const std::uint32_t head_dim32 = u32(c.head_dim);
    const std::uint32_t n_q32 = u32(c.num_attention_heads);
    const std::uint32_t n_kv32 = u32(c.num_key_value_heads);
    const std::uint32_t t32 = u32(t);
    const float eps = c.rms_norm_eps;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));

    const auto encode_norm = [&](id<MTLBuffer> src, id<MTLBuffer> w, id<MTLBuffer> dst) {
        [enc setComputePipelineState:s.rms_norm];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:w offset:0 atIndex:1];
        [enc setBuffer:dst offset:0 atIndex:2];
        [enc setBytes:&hidden32 length:sizeof(hidden32) atIndex:3];
        [enc setBytes:&eps length:sizeof(eps) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(tp, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    const auto encode_head_norm = [&](id<MTLBuffer> buf, id<MTLBuffer> w, std::size_t heads) {
        const std::uint32_t heads32 = u32(heads);
        [enc setComputePipelineState:s.head_rms_norm];
        [enc setBuffer:buf offset:0 atIndex:0];
        [enc setBuffer:w offset:0 atIndex:1];
        [enc setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:2];
        [enc setBytes:&heads32 length:sizeof(heads32) atIndex:3];
        [enc setBytes:&eps length:sizeof(eps) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(heads, t, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };

    const auto encode_rope = [&](id<MTLBuffer> buf, std::size_t heads) {
        const std::uint32_t heads32 = u32(heads);
        [enc setComputePipelineState:s.rope];
        [enc setBuffer:buf offset:0 atIndex:0];
        [enc setBuffer:s.inv_freq offset:0 atIndex:1];
        [enc setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:2];
        [enc setBytes:&heads32 length:sizeof(heads32) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(c.head_dim / 2, heads, t)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(c.head_dim / 2, 64), 1, 1)];
    };

    for (const Impl::Layer& layer : s.layers) {
        encode_norm(s.x, layer.input_norm, s.normed);

        encode_gemm(enc, s.gemm, s.normed, layer.q, s.buf_q, tp, hidden, q_dim);
        encode_gemm(enc, s.gemm, s.normed, layer.k, s.buf_k, tp, hidden, kv_dim);
        encode_gemm(enc, s.gemm, s.normed, layer.v, s.buf_v, tp, hidden, kv_dim);

        encode_head_norm(s.buf_q, layer.q_norm, c.num_attention_heads);
        encode_head_norm(s.buf_k, layer.k_norm, c.num_key_value_heads);
        encode_rope(s.buf_q, c.num_attention_heads);
        encode_rope(s.buf_k, c.num_key_value_heads);

        [enc setComputePipelineState:s.attention];
        [enc setBuffer:s.buf_q offset:0 atIndex:0];
        [enc setBuffer:s.buf_k offset:0 atIndex:1];
        [enc setBuffer:s.buf_v offset:0 atIndex:2];
        [enc setBuffer:s.attn offset:0 atIndex:3];
        [enc setBytes:&t32 length:sizeof(t32) atIndex:4];
        [enc setBytes:&n_q32 length:sizeof(n_q32) atIndex:5];
        [enc setBytes:&n_kv32 length:sizeof(n_kv32) atIndex:6];
        [enc setBytes:&head_dim32 length:sizeof(head_dim32) atIndex:7];
        [enc setBytes:&attn_scale length:sizeof(attn_scale) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(c.num_attention_heads, t, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        encode_gemm(enc, s.gemm, s.attn, layer.o, s.proj, tp, q_dim, hidden);
        encode_elementwise(enc, s.add, s.x, s.proj, tp * hidden);

        encode_norm(s.x, layer.post_norm, s.normed);
        encode_gemm(enc, s.gemm, s.normed, layer.gate, s.buf_gate, tp, hidden,
                    c.intermediate_size);
        encode_gemm(enc, s.gemm, s.normed, layer.up, s.buf_up, tp, hidden, c.intermediate_size);
        encode_elementwise(enc, s.silu_mul, s.buf_gate, s.buf_up, tp * c.intermediate_size);
        encode_gemm(enc, s.gemm, s.buf_gate, layer.down, s.proj, tp, c.intermediate_size, hidden);
        encode_elementwise(enc, s.add, s.x, s.proj, tp * hidden);
    }

    encode_norm(s.x, s.final_norm, s.normed);
    encode_gemm(enc, s.gemm, s.normed, s.head, s.logits, tp, hidden, s.head_rows);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error != nil) {
        return err(std::string("metal omnivoice: command buffer failed: ") +
                   cb.error.localizedDescription.UTF8String);
    }

    // The logits buffer is padded on both axes; the caller wants neither pad.
    const std::size_t table = c.audio_table_size();
    Mat out = Mat::zeros(t, table);
    const auto* src = static_cast<const float*>(s.logits.contents);
    for (std::size_t r = 0; r < t; ++r) {
        std::memcpy(out.row_mut(r).data(), src + r * s.head_rows, table * sizeof(float));
    }
    return out;
}

}  // namespace rt

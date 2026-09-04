#include "rt/metal_decode.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <optional>

#include "rt/decode_gemma3_msl.hpp"

namespace rt {

namespace {

/// Metal rejects zero-length buffers, so every allocation is at least this big.
constexpr std::size_t kMinBufferBytes = 16;
/// The batch path is sized for at most this many tokens per step.
constexpr std::size_t kMaxBatchTokens = 8;

[[noreturn]] void metal_fail(const std::string& what) {
    std::fprintf(stderr, "Metal: %s\n", what.c_str());
    std::abort();
}

id<MTLBuffer> alloc_buf(id<MTLDevice> device, std::size_t byte_len) {
    id<MTLBuffer> buf = [device newBufferWithLength:std::max(byte_len, kMinBufferBytes)
                                            options:MTLResourceStorageModeShared];
    if (buf == nil) {
        metal_fail("buffer allocation failed");
    }
    return buf;
}

id<MTLBuffer> upload_bytes(id<MTLDevice> device, const void* data, std::size_t byte_len) {
    id<MTLBuffer> buf = [device newBufferWithBytes:data
                                            length:std::max(byte_len, kMinBufferBytes)
                                           options:MTLResourceStorageModeShared];
    if (buf == nil) {
        metal_fail("buffer allocation failed");
    }
    return buf;
}

id<MTLBuffer> upload_f32(id<MTLDevice> device, const std::vector<float>& data) {
    return upload_bytes(device, data.data(), data.size() * 4);
}

id<MTLBuffer> upload_u16(id<MTLDevice> device, const std::vector<std::uint16_t>& data) {
    return upload_bytes(device, data.data(), data.size() * 2);
}

id<MTLLibrary> compile_library(id<MTLDevice> device, const char* source) {
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(source) options:nil error:&error];
    if (library == nil) {
        metal_fail(std::string("failed to compile MSL library: ") +
                   [[error localizedDescription] UTF8String]);
    }
    return library;
}

id<MTLComputePipelineState> pipeline_from_library(id<MTLDevice> device, id<MTLLibrary> library,
                                                  const char* func_name) {
    id<MTLFunction> func = [library newFunctionWithName:@(func_name)];
    if (func == nil) {
        metal_fail(std::string("function '") + func_name + "' not found");
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:func error:&error];
    if (pipeline == nil) {
        metal_fail(std::string("pipeline creation failed for ") + func_name);
    }
    return pipeline;
}

/// Push a 4-byte constant at `index`. Metal copies it inline, so no buffer is
/// allocated per dispatch.
template <typename T>
void set_const(id<MTLComputeCommandEncoder> enc, const T& value, NSUInteger index) {
    static_assert(sizeof(T) == 4, "decode kernels take 4-byte constants");
    [enc setBytes:&value length:4 atIndex:index];
}

/// Threadgroup count for a flat 1-D dispatch of `n` elements.
[[nodiscard]] std::size_t grid_1d(std::size_t n, std::size_t tg) { return (n + tg - 1) / tg; }

/// Everything the decode loop needs for one transformer layer.
struct LayerWeightBuffers {
    id<MTLBuffer> q_proj = nil;
    id<MTLBuffer> k_proj = nil;
    id<MTLBuffer> v_proj = nil;
    id<MTLBuffer> o_proj = nil;
    id<MTLBuffer> gate_proj = nil;
    id<MTLBuffer> up_proj = nil;
    id<MTLBuffer> down_proj = nil;
    // Whether each projection is BF16 (true) or Q4_K (false). The two need
    // different GEMV kernels, and dispatching the wrong one yields NaNs, so
    // the type is recorded per projection at upload time.
    bool q_proj_is_bf16 = false;
    bool k_proj_is_bf16 = false;
    bool v_proj_is_bf16 = false;
    bool o_proj_is_bf16 = false;
    bool gate_proj_is_bf16 = false;
    bool up_proj_is_bf16 = false;
    bool down_proj_is_bf16 = false;

    id<MTLBuffer> input_layernorm_gamma = nil;
    id<MTLBuffer> post_attn_layernorm_gamma = nil;
    id<MTLBuffer> pre_ffn_layernorm_gamma = nil;
    id<MTLBuffer> post_ffn_layernorm_gamma = nil;
    id<MTLBuffer> q_norm_gamma = nil;
    id<MTLBuffer> k_norm_gamma = nil;

    float rope_theta = 0.0f;
    float rope_freq_scale = 1.0f;
    std::optional<std::size_t> sliding_window;
};

/// Buffers and pipelines used only by the batch (speculative) path.
struct BatchState {
    id<MTLComputePipelineState> pipe_embed_bf16 = nil;
    id<MTLComputePipelineState> pipe_rms_norm = nil;
    id<MTLComputePipelineState> pipe_rms_norm_per_head = nil;
    id<MTLComputePipelineState> pipe_rope_neox = nil;
    id<MTLComputePipelineState> pipe_kv_append = nil;

    id<MTLBuffer> hidden = nil;
    id<MTLBuffer> hidden2 = nil;
    id<MTLBuffer> normed = nil;
    id<MTLBuffer> q = nil;
    id<MTLBuffer> k = nil;
    id<MTLBuffer> v = nil;
    id<MTLBuffer> q_normed = nil;
    id<MTLBuffer> k_normed = nil;
    id<MTLBuffer> q_roped = nil;
    id<MTLBuffer> k_roped = nil;
    id<MTLBuffer> attn_out = nil;
    id<MTLBuffer> o_proj_out = nil;
    id<MTLBuffer> gate = nil;
    id<MTLBuffer> up = nil;
    id<MTLBuffer> gate_act = nil;
    id<MTLBuffer> mlp_hidden = nil;
    id<MTLBuffer> down_out = nil;
    id<MTLBuffer> logits = nil;
    id<MTLBuffer> token_ids = nil;
    id<MTLBuffer> positions = nil;
};

/// Upload a projection and record which kernel reads it.
///
/// Q4_0 weights have no matching GPU kernel, so they are dequantized to BF16
/// on the way up.
std::pair<id<MTLBuffer>, bool> upload_linear_detect_type(id<MTLDevice> device,
                                                         const Linear2& linear) {
    if (linear.q4k_weight) {
        return {upload_bytes(device, linear.q4k_weight->blocks.data(),
                             linear.q4k_weight->blocks.size()),
                false};
    }
    if (linear.bf16_weight) {
        return {upload_u16(device, *linear.bf16_weight->data), true};
    }
    if (linear.q4_weight) {
        const Mat f32_mat = linear.q4_weight->dequantize();
        std::vector<std::uint16_t> bits(f32_mat.data.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            bits[i] = f32_to_bf16(f32_mat.data[i]);
        }
        return {upload_u16(device, bits), true};
    }
    return {upload_f32(device, linear.weight.data().data), false};
}

}  // namespace

// =============================================================================
// Impl
// =============================================================================

struct MetalDecodeContext::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> pipe_rms_norm = nil;
    id<MTLComputePipelineState> pipe_rms_norm_per_head = nil;
    id<MTLComputePipelineState> pipe_rope_neox = nil;
    id<MTLComputePipelineState> pipe_gelu_tanh = nil;
    id<MTLComputePipelineState> pipe_elem_mul = nil;
    id<MTLComputePipelineState> pipe_vec_add = nil;
    id<MTLComputePipelineState> pipe_attention_decode = nil;
    id<MTLComputePipelineState> pipe_kv_append = nil;
    id<MTLComputePipelineState> pipe_embed_bf16 = nil;
    id<MTLComputePipelineState> pipe_gemv_q4k = nil;
    id<MTLComputePipelineState> pipe_gemv_bf16 = nil;
    id<MTLComputePipelineState> pipe_gemv_q4k_batch = nil;
    id<MTLComputePipelineState> pipe_gemv_bf16_batch = nil;
    id<MTLComputePipelineState> pipe_f32_to_f16 = nil;

    std::optional<BatchState> batch;
    std::vector<LayerWeightBuffers> layer_weights;

    id<MTLBuffer> embed_buf = nil;
    /// For weight-tied models this is the same buffer as `embed_buf`, but it is
    /// tracked separately in case they diverge.
    id<MTLBuffer> lm_head_buf = nil;
    id<MTLBuffer> final_norm_gamma = nil;

    // Activation scratch, all single-token [1, ...] unless noted.
    id<MTLBuffer> buf_hidden = nil;
    id<MTLBuffer> buf_hidden2 = nil;
    id<MTLBuffer> buf_normed = nil;
    id<MTLBuffer> buf_q = nil;
    id<MTLBuffer> buf_k = nil;
    id<MTLBuffer> buf_v = nil;
    id<MTLBuffer> buf_q_normed = nil;
    id<MTLBuffer> buf_k_normed = nil;
    id<MTLBuffer> buf_q_roped = nil;
    id<MTLBuffer> buf_k_roped = nil;
    id<MTLBuffer> buf_attn_out = nil;
    id<MTLBuffer> buf_o_proj_out = nil;
    id<MTLBuffer> buf_gate = nil;
    id<MTLBuffer> buf_up = nil;
    id<MTLBuffer> buf_gate_act = nil;
    id<MTLBuffer> buf_mlp_hidden = nil;
    id<MTLBuffer> buf_down_out = nil;
    id<MTLBuffer> buf_logits = nil;

    /// KV cache, half precision.
    std::vector<id<MTLBuffer>> kv_k_bufs;
    std::vector<id<MTLBuffer>> kv_v_bufs;
    /// Staging for the f32-to-f16 sync; released once the sync is done.
    id<MTLBuffer> buf_kv_staging = nil;

    Config4 config;
    std::size_t max_seq_len = 0;
    std::size_t lm_head_vocab = 0;
    bool lm_head_is_bf16 = false;

    // -------------------------------------------------------------------------
    // Dispatch helpers -- each encodes one kernel into an open encoder
    // -------------------------------------------------------------------------

    void dispatch_rms_norm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> gamma,
                           id<MTLBuffer> out, std::size_t dim) const {
        [enc setComputePipelineState:pipe_rms_norm];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:gamma offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(dim), 3);
        set_const(enc, config.rms_norm_eps, 4);
        // One threadgroup; 256 threads loop over the row (10 elements each at
        // hidden_size = 2560).
        const std::size_t tg = std::min<std::size_t>(256, dim);
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    void dispatch_rms_norm_per_head(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                                    id<MTLBuffer> gamma, id<MTLBuffer> out, std::size_t n_heads,
                                    std::size_t head_dim) const {
        [enc setComputePipelineState:pipe_rms_norm_per_head];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:gamma offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(n_heads), 3);
        set_const(enc, static_cast<std::uint32_t>(head_dim), 4);
        set_const(enc, config.rms_norm_eps, 5);
        // One threadgroup per head, one SIMD group each.
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    void dispatch_rope(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> out,
                       std::size_t n_heads, std::size_t head_dim, float theta, float freq_scale,
                       std::size_t position) const {
        [enc setComputePipelineState:pipe_rope_neox];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        set_const(enc, static_cast<std::uint32_t>(n_heads), 2);
        set_const(enc, static_cast<std::uint32_t>(head_dim), 3);
        set_const(enc, theta, 4);
        set_const(enc, freq_scale, 5);
        set_const(enc, static_cast<std::uint32_t>(position), 6);
        // One threadgroup per head; each thread rotates one (i, i + half) pair.
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(head_dim / 2, 1, 1)];
    }

    /// Encode a flat elementwise kernel over `n` values with `n_inputs` inputs.
    void dispatch_elementwise(id<MTLComputeCommandEncoder> enc,
                              id<MTLComputePipelineState> pipeline,
                              std::initializer_list<id<MTLBuffer>> buffers, std::size_t n) const {
        [enc setComputePipelineState:pipeline];
        NSUInteger index = 0;
        for (id<MTLBuffer> buf : buffers) {
            [enc setBuffer:buf offset:0 atIndex:index++];
        }
        set_const(enc, static_cast<std::uint32_t>(n), index);
        constexpr std::size_t tg = 256;
        [enc dispatchThreadgroups:MTLSizeMake(grid_1d(n, tg), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    void dispatch_gelu_tanh(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> out,
                            std::size_t n) const {
        dispatch_elementwise(enc, pipe_gelu_tanh, {x, out}, n);
    }

    void dispatch_elem_mul(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> a, id<MTLBuffer> b,
                           id<MTLBuffer> out, std::size_t n) const {
        dispatch_elementwise(enc, pipe_elem_mul, {a, b, out}, n);
    }

    void dispatch_vec_add(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> a, id<MTLBuffer> b,
                          id<MTLBuffer> out, std::size_t n) const {
        dispatch_elementwise(enc, pipe_vec_add, {a, b, out}, n);
    }

    void dispatch_kv_append(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> k_new,
                            id<MTLBuffer> v_new, id<MTLBuffer> k_cache, id<MTLBuffer> v_cache,
                            std::size_t seq_len, std::size_t kv_dim) const {
        [enc setComputePipelineState:pipe_kv_append];
        [enc setBuffer:k_new offset:0 atIndex:0];
        [enc setBuffer:v_new offset:0 atIndex:1];
        [enc setBuffer:k_cache offset:0 atIndex:2];
        [enc setBuffer:v_cache offset:0 atIndex:3];
        set_const(enc, static_cast<std::uint32_t>(seq_len), 4);
        set_const(enc, static_cast<std::uint32_t>(kv_dim), 5);
        const std::size_t tg = std::min<std::size_t>(256, kv_dim);
        [enc dispatchThreadgroups:MTLSizeMake(grid_1d(kv_dim, tg), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    void dispatch_embed_lookup(id<MTLComputeCommandEncoder> enc, std::size_t token_id) const {
        [enc setComputePipelineState:pipe_embed_bf16];
        [enc setBuffer:embed_buf offset:0 atIndex:0];
        [enc setBuffer:buf_hidden offset:0 atIndex:1];
        set_const(enc, static_cast<std::uint32_t>(token_id), 2);
        set_const(enc, static_cast<std::uint32_t>(config.hidden_size), 3);
        // Gemma 3 scales the looked-up embedding by sqrt(hidden_size).
        set_const(enc, std::sqrt(static_cast<float>(config.hidden_size)), 4);
        constexpr std::size_t tg = 256;
        [enc dispatchThreadgroups:MTLSizeMake(grid_1d(config.hidden_size, tg), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    /// M=1 GEMV: `out[N] = act[K] @ W[N,K]^T`.
    void dispatch_gemv(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> act, id<MTLBuffer> weight,
                       id<MTLBuffer> out, std::size_t k, std::size_t n, bool is_bf16) const {
        [enc setComputePipelineState:(is_bf16 ? pipe_gemv_bf16 : pipe_gemv_q4k)];
        [enc setBuffer:act offset:0 atIndex:0];
        [enc setBuffer:weight offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(k), 3);
        set_const(enc, static_cast<std::uint32_t>(n), 4);
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    /// The same GEMV over M rows of activations.
    void dispatch_gemv_batch(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> act,
                             id<MTLBuffer> weight, id<MTLBuffer> out, std::size_t k, std::size_t n,
                             std::size_t m, bool is_bf16) const {
        [enc setComputePipelineState:(is_bf16 ? pipe_gemv_bf16_batch : pipe_gemv_q4k_batch)];
        [enc setBuffer:act offset:0 atIndex:0];
        [enc setBuffer:weight offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(k), 3);
        set_const(enc, static_cast<std::uint32_t>(n), 4);
        set_const(enc, static_cast<std::uint32_t>(m), 5);
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    /// Attention for one query row. `q_offset` and `out_offset` are byte offsets
    /// into the buffers, which lets the batch path reuse the single-token kernel
    /// per token.
    void dispatch_attention(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> q,
                            std::size_t q_offset, id<MTLBuffer> k_cache, id<MTLBuffer> v_cache,
                            id<MTLBuffer> out, std::size_t out_offset, std::size_t k_start,
                            std::size_t k_end) const {
        [enc setComputePipelineState:pipe_attention_decode];
        [enc setBuffer:q offset:q_offset atIndex:0];
        [enc setBuffer:k_cache offset:0 atIndex:1];
        [enc setBuffer:v_cache offset:0 atIndex:2];
        [enc setBuffer:out offset:out_offset atIndex:3];
        set_const(enc, static_cast<std::uint32_t>(k_start), 4);
        set_const(enc, static_cast<std::uint32_t>(k_end), 5);
        set_const(enc, static_cast<std::uint32_t>(config.num_attention_heads), 6);
        set_const(enc, static_cast<std::uint32_t>(config.num_key_value_heads), 7);
        set_const(enc, static_cast<std::uint32_t>(config.head_dim), 8);
        set_const(enc, 1.0f / std::sqrt(config.query_pre_attn_scalar), 9);
        set_const(enc, static_cast<std::uint32_t>(config.num_key_value_heads * config.head_dim),
                  10);
        // One threadgroup per query head.
        [enc dispatchThreadgroups:MTLSizeMake(config.num_attention_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    /// The cache window this layer attends over at `position`.
    [[nodiscard]] std::pair<std::size_t, std::size_t> attention_window(
        const LayerWeightBuffers& lw, std::size_t position) const {
        const std::size_t k_end = position + 1;
        const std::size_t k_start =
            lw.sliding_window ? (k_end > *lw.sliding_window ? k_end - *lw.sliding_window : 0) : 0;
        return {k_start, k_end};
    }

    /// Run the f32-to-f16 conversion kernel on its own command buffer.
    void dispatch_f32_to_f16(id<MTLBuffer> src, id<MTLBuffer> dst, std::size_t n) const {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipe_f32_to_f16];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            set_const(enc, static_cast<std::uint32_t>(n), 2);
            constexpr std::size_t tg = 256;
            [enc dispatchThreadgroups:MTLSizeMake(grid_1d(n, tg), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    // -------------------------------------------------------------------------
    // Graph encoding
    // -------------------------------------------------------------------------

    /// Encode one transformer layer's 21 dispatches.
    void encode_layer(id<MTLComputeCommandEncoder> enc, std::size_t layer_idx,
                      std::size_t position) const {
        const LayerWeightBuffers& lw = layer_weights[layer_idx];
        const std::size_t h = config.hidden_size;
        const std::size_t nq = config.num_attention_heads;
        const std::size_t nkv = config.num_key_value_heads;
        const std::size_t d = config.head_dim;
        const std::size_t inter = config.intermediate_size;
        const std::size_t kv_dim = nkv * d;

        dispatch_rms_norm(enc, buf_hidden, lw.input_layernorm_gamma, buf_normed, h);

        dispatch_gemv(enc, buf_normed, lw.q_proj, buf_q, h, nq * d, lw.q_proj_is_bf16);
        dispatch_gemv(enc, buf_normed, lw.k_proj, buf_k, h, nkv * d, lw.k_proj_is_bf16);
        dispatch_gemv(enc, buf_normed, lw.v_proj, buf_v, h, nkv * d, lw.v_proj_is_bf16);

        dispatch_rms_norm_per_head(enc, buf_q, lw.q_norm_gamma, buf_q_normed, nq, d);
        dispatch_rms_norm_per_head(enc, buf_k, lw.k_norm_gamma, buf_k_normed, nkv, d);

        dispatch_rope(enc, buf_q_normed, buf_q_roped, nq, d, lw.rope_theta, lw.rope_freq_scale,
                      position);
        dispatch_rope(enc, buf_k_normed, buf_k_roped, nkv, d, lw.rope_theta, lw.rope_freq_scale,
                      position);

        dispatch_kv_append(enc, buf_k_roped, buf_v, kv_k_bufs[layer_idx], kv_v_bufs[layer_idx],
                           position, kv_dim);

        const auto [k_start, k_end] = attention_window(lw, position);
        dispatch_attention(enc, buf_q_roped, 0, kv_k_bufs[layer_idx], kv_v_bufs[layer_idx],
                           buf_attn_out, 0, k_start, k_end);

        dispatch_gemv(enc, buf_attn_out, lw.o_proj, buf_o_proj_out, nq * d, h, lw.o_proj_is_bf16);
        dispatch_rms_norm(enc, buf_o_proj_out, lw.post_attn_layernorm_gamma, buf_normed, h);
        dispatch_vec_add(enc, buf_hidden, buf_normed, buf_hidden2, h);

        dispatch_rms_norm(enc, buf_hidden2, lw.pre_ffn_layernorm_gamma, buf_normed, h);
        dispatch_gemv(enc, buf_normed, lw.gate_proj, buf_gate, h, inter, lw.gate_proj_is_bf16);
        dispatch_gemv(enc, buf_normed, lw.up_proj, buf_up, h, inter, lw.up_proj_is_bf16);
        dispatch_gelu_tanh(enc, buf_gate, buf_gate_act, inter);
        dispatch_elem_mul(enc, buf_gate_act, buf_up, buf_mlp_hidden, inter);
        dispatch_gemv(enc, buf_mlp_hidden, lw.down_proj, buf_down_out, inter, h,
                      lw.down_proj_is_bf16);

        dispatch_rms_norm(enc, buf_down_out, lw.post_ffn_layernorm_gamma, buf_normed, h);
        // Leaves the result in buf_hidden, ready for the next layer.
        dispatch_vec_add(enc, buf_hidden2, buf_normed, buf_hidden, h);
    }

    // -------------------------------------------------------------------------
    // Batch variants
    // -------------------------------------------------------------------------

    void dispatch_rms_norm_batch(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                                 id<MTLBuffer> gamma, id<MTLBuffer> out, std::size_t dim,
                                 std::size_t m) const {
        [enc setComputePipelineState:batch->pipe_rms_norm];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:gamma offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(dim), 3);
        set_const(enc, config.rms_norm_eps, 4);
        set_const(enc, static_cast<std::uint32_t>(m), 5);
        const std::size_t tg = std::min<std::size_t>(256, dim);
        [enc dispatchThreadgroups:MTLSizeMake(m, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    void dispatch_rms_norm_per_head_batch(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                                          id<MTLBuffer> gamma, id<MTLBuffer> out,
                                          std::size_t n_heads, std::size_t head_dim,
                                          std::size_t m) const {
        [enc setComputePipelineState:batch->pipe_rms_norm_per_head];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:gamma offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(n_heads), 3);
        set_const(enc, static_cast<std::uint32_t>(head_dim), 4);
        set_const(enc, config.rms_norm_eps, 5);
        set_const(enc, static_cast<std::uint32_t>(m), 6);
        [enc dispatchThreadgroups:MTLSizeMake(m * n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    void dispatch_rope_batch(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> out,
                             std::size_t n_heads, std::size_t head_dim, float theta,
                             float freq_scale, std::size_t m) const {
        [enc setComputePipelineState:batch->pipe_rope_neox];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        set_const(enc, static_cast<std::uint32_t>(n_heads), 2);
        set_const(enc, static_cast<std::uint32_t>(head_dim), 3);
        set_const(enc, theta, 4);
        set_const(enc, freq_scale, 5);
        // Each row's absolute position comes from a buffer, not a constant.
        [enc setBuffer:batch->positions offset:0 atIndex:6];
        set_const(enc, static_cast<std::uint32_t>(m), 7);
        [enc dispatchThreadgroups:MTLSizeMake(m * n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(head_dim / 2, 1, 1)];
    }

    /// Encode one layer for M tokens at once.
    void encode_layer_batch(id<MTLComputeCommandEncoder> enc, std::size_t layer_idx,
                            std::size_t start_position, std::size_t m) const {
        const BatchState& b = *batch;
        const LayerWeightBuffers& lw = layer_weights[layer_idx];
        const std::size_t h = config.hidden_size;
        const std::size_t nq = config.num_attention_heads;
        const std::size_t nkv = config.num_key_value_heads;
        const std::size_t d = config.head_dim;
        const std::size_t inter = config.intermediate_size;
        const std::size_t kv_dim = nkv * d;

        dispatch_rms_norm_batch(enc, b.hidden, lw.input_layernorm_gamma, b.normed, h, m);

        dispatch_gemv_batch(enc, b.normed, lw.q_proj, b.q, h, nq * d, m, lw.q_proj_is_bf16);
        dispatch_gemv_batch(enc, b.normed, lw.k_proj, b.k, h, nkv * d, m, lw.k_proj_is_bf16);
        dispatch_gemv_batch(enc, b.normed, lw.v_proj, b.v, h, nkv * d, m, lw.v_proj_is_bf16);

        dispatch_rms_norm_per_head_batch(enc, b.q, lw.q_norm_gamma, b.q_normed, nq, d, m);
        dispatch_rms_norm_per_head_batch(enc, b.k, lw.k_norm_gamma, b.k_normed, nkv, d, m);

        dispatch_rope_batch(enc, b.q_normed, b.q_roped, nq, d, lw.rope_theta, lw.rope_freq_scale,
                            m);
        dispatch_rope_batch(enc, b.k_normed, b.k_roped, nkv, d, lw.rope_theta, lw.rope_freq_scale,
                            m);

        // Append all M rows starting at start_position.
        [enc setComputePipelineState:b.pipe_kv_append];
        [enc setBuffer:b.k_roped offset:0 atIndex:0];
        [enc setBuffer:b.v offset:0 atIndex:1];
        [enc setBuffer:kv_k_bufs[layer_idx] offset:0 atIndex:2];
        [enc setBuffer:kv_v_bufs[layer_idx] offset:0 atIndex:3];
        set_const(enc, static_cast<std::uint32_t>(start_position), 4);
        set_const(enc, static_cast<std::uint32_t>(kv_dim), 5);
        set_const(enc, static_cast<std::uint32_t>(m), 6);
        {
            constexpr std::size_t tg = 256;
            [enc dispatchThreadgroups:MTLSizeMake(grid_1d(m * kv_dim, tg), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // Attention stays per-token: token mi is causal up to start_position + mi,
        // so each row needs its own window. The single-token kernel is reused by
        // pointing it at the right offset in the batch buffers.
        for (std::size_t mi = 0; mi < m; ++mi) {
            const auto [k_start, k_end] = attention_window(lw, start_position + mi);
            const std::size_t offset = mi * nq * d * 4;
            dispatch_attention(enc, b.q_roped, offset, kv_k_bufs[layer_idx], kv_v_bufs[layer_idx],
                               b.attn_out, offset, k_start, k_end);
        }

        dispatch_gemv_batch(enc, b.attn_out, lw.o_proj, b.o_proj_out, nq * d, h, m,
                            lw.o_proj_is_bf16);
        dispatch_rms_norm_batch(enc, b.o_proj_out, lw.post_attn_layernorm_gamma, b.normed, h, m);
        // The elementwise kernels are shape-agnostic, so the single-token ones
        // work on M * n values unchanged.
        dispatch_vec_add(enc, b.hidden, b.normed, b.hidden2, m * h);

        dispatch_rms_norm_batch(enc, b.hidden2, lw.pre_ffn_layernorm_gamma, b.normed, h, m);
        dispatch_gemv_batch(enc, b.normed, lw.gate_proj, b.gate, h, inter, m, lw.gate_proj_is_bf16);
        dispatch_gemv_batch(enc, b.normed, lw.up_proj, b.up, h, inter, m, lw.up_proj_is_bf16);
        dispatch_gelu_tanh(enc, b.gate, b.gate_act, m * inter);
        dispatch_elem_mul(enc, b.gate_act, b.up, b.mlp_hidden, m * inter);
        dispatch_gemv_batch(enc, b.mlp_hidden, lw.down_proj, b.down_out, inter, h, m,
                            lw.down_proj_is_bf16);

        dispatch_rms_norm_batch(enc, b.down_out, lw.post_ffn_layernorm_gamma, b.normed, h, m);
        dispatch_vec_add(enc, b.hidden2, b.normed, b.hidden, m * h);
    }
};

// =============================================================================
// Construction
// =============================================================================

MetalDecodeContext::MetalDecodeContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalDecodeContext::~MetalDecodeContext() = default;

std::unique_ptr<MetalDecodeContext> MetalDecodeContext::create(Gemma3Model& model,
                                                               std::size_t draft_len) {
    const auto t_start = std::chrono::steady_clock::now();
    const auto ms_since = [](auto t) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - t)
            .count();
    };

    auto impl = std::make_unique<Impl>();
    const Config4& cfg = model.config;
    impl->config = cfg;

    impl->device = MTLCreateSystemDefaultDevice();
    if (impl->device == nil) {
        metal_fail("no GPU device found");
    }
    impl->queue = [impl->device newCommandQueue];
    if (impl->queue == nil) {
        metal_fail("command queue creation failed");
    }

    // One library compile for every pipeline: compiling per kernel took seconds.
    const auto t_msl = std::chrono::steady_clock::now();
    id<MTLLibrary> library = compile_library(impl->device, kDecodeGemma3Msl);
    const auto pipe = [&](const char* name) {
        return pipeline_from_library(impl->device, library, name);
    };
    impl->pipe_rms_norm = pipe("rms_norm_gemma3");
    impl->pipe_rms_norm_per_head = pipe("rms_norm_per_head");
    impl->pipe_rope_neox = pipe("rope_neox");
    impl->pipe_gelu_tanh = pipe("gelu_tanh_kernel");
    impl->pipe_elem_mul = pipe("elem_mul_kernel");
    impl->pipe_vec_add = pipe("vec_add_kernel");
    impl->pipe_attention_decode = pipe("attention_decode");
    impl->pipe_kv_append = pipe("kv_cache_append");
    impl->pipe_embed_bf16 = pipe("embed_bf16_lookup");
    impl->pipe_gemv_q4k = pipe("gemv_q4k_t");
    impl->pipe_gemv_bf16 = pipe("gemv_bf16_t");
    impl->pipe_gemv_q4k_batch = pipe("gemv_q4k_t_batch");
    impl->pipe_gemv_bf16_batch = pipe("gemv_bf16_t_batch");
    impl->pipe_f32_to_f16 = pipe("f32_to_f16_convert");
    std::fprintf(stderr, "[ Metal ]   MSL compile + pipelines: %lld ms\n", ms_since(t_msl));

    const auto t_upload = std::chrono::steady_clock::now();
    impl->layer_weights.reserve(cfg.num_hidden_layers);
    std::size_t total_upload_bytes = 0;

    for (Gemma3Block& layer : model.layers) {
        LayerWeightBuffers lw;
        // The reference is __strong to match the struct members' ownership;
        // a bare id<MTLBuffer>& would default to __autoreleasing.
        const auto take = [&](const Linear2& l, __strong id<MTLBuffer>& dst, bool& is_bf16) {
            auto [buf, bf16] = upload_linear_detect_type(impl->device, l);
            dst = buf;
            is_bf16 = bf16;
            total_upload_bytes += [buf length];
        };
        take(layer.self_attn.q_proj, lw.q_proj, lw.q_proj_is_bf16);
        take(layer.self_attn.k_proj, lw.k_proj, lw.k_proj_is_bf16);
        take(layer.self_attn.v_proj, lw.v_proj, lw.v_proj_is_bf16);
        take(layer.self_attn.o_proj, lw.o_proj, lw.o_proj_is_bf16);
        take(layer.mlp.gate_proj, lw.gate_proj, lw.gate_proj_is_bf16);
        take(layer.mlp.up_proj, lw.up_proj, lw.up_proj_is_bf16);
        take(layer.mlp.down_proj, lw.down_proj, lw.down_proj_is_bf16);

        // Free the CPU copy immediately, so peak memory never holds both.
        layer.self_attn.q_proj.clear_weight_data();
        layer.self_attn.k_proj.clear_weight_data();
        layer.self_attn.v_proj.clear_weight_data();
        layer.self_attn.o_proj.clear_weight_data();
        layer.mlp.gate_proj.clear_weight_data();
        layer.mlp.up_proj.clear_weight_data();
        layer.mlp.down_proj.clear_weight_data();

        // Norm gammas are a few KB each; not worth clearing.
        lw.input_layernorm_gamma =
            upload_f32(impl->device, layer.input_layernorm.gamma.data().data);
        lw.post_attn_layernorm_gamma =
            upload_f32(impl->device, layer.post_attention_layernorm.gamma.data().data);
        lw.pre_ffn_layernorm_gamma =
            upload_f32(impl->device, layer.pre_feedforward_layernorm.gamma.data().data);
        lw.post_ffn_layernorm_gamma =
            upload_f32(impl->device, layer.post_feedforward_layernorm.gamma.data().data);
        lw.q_norm_gamma = upload_f32(impl->device, layer.self_attn.q_norm.gamma.data().data);
        lw.k_norm_gamma = upload_f32(impl->device, layer.self_attn.k_norm.gamma.data().data);

        lw.rope_theta = layer.self_attn.rope_theta;
        lw.rope_freq_scale = layer.self_attn.rope_freq_scale;
        lw.sliding_window = layer.self_attn.sliding_window;

        impl->layer_weights.push_back(lw);
    }
    const auto layers_ms = ms_since(t_upload);

    const auto t_embed = std::chrono::steady_clock::now();
    if (model.embed_bf16) {
        impl->embed_buf = upload_u16(impl->device, *model.embed_bf16->data);
    } else {
        impl->embed_buf = upload_f32(impl->device, model.embed_tokens.data().data);
    }
    std::fprintf(stderr,
                 "[ Metal ]   layer weights: %lld ms (%.1f MB), embed: %lld ms (%.1f MB)\n",
                 layers_ms, static_cast<double>(total_upload_bytes) / 1e6, ms_since(t_embed),
                 static_cast<double>([impl->embed_buf length]) / 1e6);

    // lm_head is weight-tied to the embedding in Gemma 3, so when it holds the
    // same shared BF16 bits the GPU buffer is reused rather than uploaded twice.
    if (model.lm_head.bf16_weight) {
        const MatBf16& bf16 = *model.lm_head.bf16_weight;
        const bool tied = model.embed_bf16 && bf16.data == model.embed_bf16->data;
#if RT_FEATURE_Q4K_LM_HEAD
        const auto t_q = std::chrono::steady_clock::now();
        const Q4KMat q4k = Q4KMat::quantize_from_bf16(bf16);
        impl->lm_head_buf = upload_bytes(impl->device, q4k.blocks.data(), q4k.blocks.size());
        impl->lm_head_is_bf16 = false;
        std::fprintf(stderr,
                     "[ Metal ]   lm_head re-quantized BF16->Q4K: %lld ms (%zu rows x %zu cols, "
                     "%.1f MB -> %.1f MB)\n",
                     ms_since(t_q), bf16.rows, bf16.cols,
                     static_cast<double>(bf16.data->size() * 2) / 1e6,
                     static_cast<double>(q4k.blocks.size()) / 1e6);
        (void)tied;
#else
        impl->lm_head_buf = tied ? impl->embed_buf : upload_u16(impl->device, *bf16.data);
        impl->lm_head_is_bf16 = true;
#endif
        impl->lm_head_vocab = bf16.rows;
    } else if (model.lm_head.q4k_weight) {
        impl->lm_head_buf = upload_bytes(impl->device, model.lm_head.q4k_weight->blocks.data(),
                                         model.lm_head.q4k_weight->blocks.size());
        impl->lm_head_vocab = model.lm_head.q4k_weight->rows;
        impl->lm_head_is_bf16 = false;
    } else {
        const Mat& w = model.lm_head.weight.data();
        impl->lm_head_buf = upload_f32(impl->device, w.data);
        impl->lm_head_vocab = w.rows;
        impl->lm_head_is_bf16 = true;
    }

    // Release the CPU embedding and lm_head now the GPU holds copies. The pages
    // are marked reusable before the drop so macOS actually reclaims them.
    if (model.embed_bf16) {
        mark_pages_reusable(*model.embed_bf16->data);
    }
    model.embed_bf16.reset();
    model.lm_head.clear_weight_data();
    release_memory_to_os();

    impl->final_norm_gamma = upload_f32(impl->device, model.norm.gamma.data().data);

    const std::size_t h = cfg.hidden_size;
    const std::size_t nq = cfg.num_attention_heads;
    const std::size_t nkv = cfg.num_key_value_heads;
    const std::size_t d = cfg.head_dim;
    const std::size_t inter = cfg.intermediate_size;

    impl->buf_hidden = alloc_buf(impl->device, h * 4);
    impl->buf_hidden2 = alloc_buf(impl->device, h * 4);
    impl->buf_normed = alloc_buf(impl->device, h * 4);
    impl->buf_q = alloc_buf(impl->device, nq * d * 4);
    impl->buf_k = alloc_buf(impl->device, nkv * d * 4);
    impl->buf_v = alloc_buf(impl->device, nkv * d * 4);
    impl->buf_q_normed = alloc_buf(impl->device, nq * d * 4);
    impl->buf_k_normed = alloc_buf(impl->device, nkv * d * 4);
    impl->buf_q_roped = alloc_buf(impl->device, nq * d * 4);
    impl->buf_k_roped = alloc_buf(impl->device, nkv * d * 4);
    impl->buf_attn_out = alloc_buf(impl->device, nq * d * 4);
    impl->buf_o_proj_out = alloc_buf(impl->device, h * 4);
    impl->buf_gate = alloc_buf(impl->device, inter * 4);
    impl->buf_up = alloc_buf(impl->device, inter * 4);
    impl->buf_gate_act = alloc_buf(impl->device, inter * 4);
    impl->buf_mlp_hidden = alloc_buf(impl->device, inter * 4);
    impl->buf_down_out = alloc_buf(impl->device, h * 4);
    impl->buf_logits = alloc_buf(impl->device, impl->lm_head_vocab * 4);

    if (draft_len > 0) {
        constexpr std::size_t mb = kMaxBatchTokens;
        BatchState b;
        b.pipe_embed_bf16 = pipe("embed_bf16_lookup_batch");
        b.pipe_rms_norm = pipe("rms_norm_gemma3_batch");
        b.pipe_rms_norm_per_head = pipe("rms_norm_per_head_batch");
        b.pipe_rope_neox = pipe("rope_neox_batch");
        b.pipe_kv_append = pipe("kv_cache_append_batch");
        b.hidden = alloc_buf(impl->device, mb * h * 4);
        b.hidden2 = alloc_buf(impl->device, mb * h * 4);
        b.normed = alloc_buf(impl->device, mb * h * 4);
        b.q = alloc_buf(impl->device, mb * nq * d * 4);
        b.k = alloc_buf(impl->device, mb * nkv * d * 4);
        b.v = alloc_buf(impl->device, mb * nkv * d * 4);
        b.q_normed = alloc_buf(impl->device, mb * nq * d * 4);
        b.k_normed = alloc_buf(impl->device, mb * nkv * d * 4);
        b.q_roped = alloc_buf(impl->device, mb * nq * d * 4);
        b.k_roped = alloc_buf(impl->device, mb * nkv * d * 4);
        b.attn_out = alloc_buf(impl->device, mb * nq * d * 4);
        b.o_proj_out = alloc_buf(impl->device, mb * h * 4);
        b.gate = alloc_buf(impl->device, mb * inter * 4);
        b.up = alloc_buf(impl->device, mb * inter * 4);
        b.gate_act = alloc_buf(impl->device, mb * inter * 4);
        b.mlp_hidden = alloc_buf(impl->device, mb * inter * 4);
        b.down_out = alloc_buf(impl->device, mb * h * 4);
        b.logits = alloc_buf(impl->device, mb * impl->lm_head_vocab * 4);
        b.token_ids = alloc_buf(impl->device, mb * 4);
        b.positions = alloc_buf(impl->device, mb * 4);
        impl->batch = std::move(b);
    }

    // The KV cache is half precision: two bytes per element.
    impl->max_seq_len = std::min<std::size_t>(cfg.max_position_embeddings, 2048);
    const std::size_t kv_dim = nkv * d;
    impl->kv_k_bufs.reserve(cfg.num_hidden_layers);
    impl->kv_v_bufs.reserve(cfg.num_hidden_layers);
    for (std::size_t i = 0; i < cfg.num_hidden_layers; ++i) {
        impl->kv_k_bufs.push_back(alloc_buf(impl->device, impl->max_seq_len * kv_dim * 2));
        impl->kv_v_bufs.push_back(alloc_buf(impl->device, impl->max_seq_len * kv_dim * 2));
    }
    impl->buf_kv_staging = alloc_buf(impl->device, impl->max_seq_len * kv_dim * 4);

    std::fprintf(stderr, "[ Metal ] Decode context initialized in %lld ms (%zu layers)\n",
                 ms_since(t_start), cfg.num_hidden_layers);

    return std::unique_ptr<MetalDecodeContext>(new MetalDecodeContext(std::move(impl)));
}

std::size_t MetalDecodeContext::lm_head_vocab() const { return impl_->lm_head_vocab; }

void MetalDecodeContext::sync_kv_from_cpu(const Gemma3KvCache& cache) {
    if (impl_->buf_kv_staging == nil) {
        metal_fail("sync_kv_from_cpu: staging buffer already released");
    }
    const std::size_t kv_dim = impl_->config.num_key_value_heads * impl_->config.head_dim;

    for (std::size_t i = 0; i < cache.layers.size(); ++i) {
        const Gemma3LayerKvCache& lc = cache.layers[i];
        if (lc.seq_len == 0) {
            continue;
        }
        const std::size_t n_floats = lc.seq_len * kv_dim;

        std::memcpy([impl_->buf_kv_staging contents], lc.k.data.data(), n_floats * 4);
        impl_->dispatch_f32_to_f16(impl_->buf_kv_staging, impl_->kv_k_bufs[i], n_floats);

        std::memcpy([impl_->buf_kv_staging contents], lc.v.data.data(), n_floats * 4);
        impl_->dispatch_f32_to_f16(impl_->buf_kv_staging, impl_->kv_v_bufs[i], n_floats);
    }

    impl_->buf_kv_staging = nil;
}

std::vector<float> MetalDecodeContext::decode_step(std::size_t token_id, std::size_t position) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        impl_->dispatch_embed_lookup(enc, token_id);
        for (std::size_t layer_idx = 0; layer_idx < impl_->config.num_hidden_layers; ++layer_idx) {
            impl_->encode_layer(enc, layer_idx, position);
        }
        impl_->dispatch_rms_norm(enc, impl_->buf_hidden, impl_->final_norm_gamma,
                                 impl_->buf_normed, impl_->config.hidden_size);
        impl_->dispatch_gemv(enc, impl_->buf_normed, impl_->lm_head_buf, impl_->buf_logits,
                             impl_->config.hidden_size, impl_->lm_head_vocab,
                             impl_->lm_head_is_bf16);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        const auto* ptr = static_cast<const float*>([impl_->buf_logits contents]);
        return std::vector<float>(ptr, ptr + impl_->lm_head_vocab);
    }
}

std::vector<float> MetalDecodeContext::decode_step_batch(
    const std::vector<std::size_t>& token_ids, std::size_t start_position) {
    if (!impl_->batch) {
        metal_fail("decode_step_batch requires draft_len > 0");
    }
    const std::size_t m = token_ids.size();
    assert(m <= kMaxBatchTokens && "decode_step_batch: too many tokens");

    const BatchState& b = *impl_->batch;
    const std::size_t h = impl_->config.hidden_size;

    auto* tid_ptr = static_cast<std::uint32_t*>([b.token_ids contents]);
    auto* pos_ptr = static_cast<std::uint32_t*>([b.positions contents]);
    for (std::size_t i = 0; i < m; ++i) {
        tid_ptr[i] = static_cast<std::uint32_t>(token_ids[i]);
        pos_ptr[i] = static_cast<std::uint32_t>(start_position + i);
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:b.pipe_embed_bf16];
        [enc setBuffer:impl_->embed_buf offset:0 atIndex:0];
        [enc setBuffer:b.hidden offset:0 atIndex:1];
        [enc setBuffer:b.token_ids offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(h), 3);
        set_const(enc, std::sqrt(static_cast<float>(h)), 4);
        set_const(enc, static_cast<std::uint32_t>(m), 5);
        {
            constexpr std::size_t tg = 256;
            [enc dispatchThreadgroups:MTLSizeMake(grid_1d(m * h, tg), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        for (std::size_t layer_idx = 0; layer_idx < impl_->config.num_hidden_layers; ++layer_idx) {
            impl_->encode_layer_batch(enc, layer_idx, start_position, m);
        }

        impl_->dispatch_rms_norm_batch(enc, b.hidden, impl_->final_norm_gamma, b.normed, h, m);
        impl_->dispatch_gemv_batch(enc, b.normed, impl_->lm_head_buf, b.logits, h,
                                   impl_->lm_head_vocab, m, impl_->lm_head_is_bf16);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        const std::size_t n_logits = m * impl_->lm_head_vocab;
        const auto* ptr = static_cast<const float*>([b.logits contents]);
        return std::vector<float>(ptr, ptr + n_logits);
    }
}

void MetalDecodeContext::print_metal_memory() const {
    std::size_t weight_bytes = 0;
    for (const LayerWeightBuffers& lw : impl_->layer_weights) {
        for (id<MTLBuffer> buf : {lw.q_proj, lw.k_proj, lw.v_proj, lw.o_proj, lw.gate_proj,
                                  lw.up_proj, lw.down_proj}) {
            weight_bytes += [buf length];
        }
    }
    std::size_t kv_bytes = 0;
    for (std::size_t i = 0; i < impl_->kv_k_bufs.size(); ++i) {
        kv_bytes += [impl_->kv_k_bufs[i] length] + [impl_->kv_v_bufs[i] length];
    }
    const std::size_t embed_bytes = [impl_->embed_buf length];

    std::fprintf(stderr,
                 "[ Metal ] GPU buffers: weights %.2f GB, embed %.2f GB, KV cache %.2f GB\n",
                 static_cast<double>(weight_bytes) / 1e9, static_cast<double>(embed_bytes) / 1e9,
                 static_cast<double>(kv_bytes) / 1e9);
}

}  // namespace rt

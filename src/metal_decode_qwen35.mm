#include "rt/metal_decode_qwen35.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <variant>

#include "rt/decode_qwen35_msl.hpp"

namespace rt {

namespace {

/// Metal rejects zero-length buffers.
constexpr std::size_t kMinBufferBytes = 16;
/// KV cache depth. Longer contexts would need a larger allocation.
constexpr std::size_t kMaxSeqLen = 2048;

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

template <typename T>
void set_const(id<MTLComputeCommandEncoder> enc, const T& value, NSUInteger index) {
    static_assert(sizeof(T) == 4, "decode kernels take 4-byte constants");
    [enc setBytes:&value length:4 atIndex:index];
}

[[nodiscard]] std::size_t grid_1d(std::size_t n, std::size_t tg) { return (n + tg - 1) / tg; }

/// Upload a projection and record which GEMV kernel reads it.
///
/// Only BF16 and Q4_K have GPU kernels, so f32 and Q4_0 weights are converted
/// to BF16 on the way up. A zero-sized placeholder becomes a stub buffer, since
/// Metal will not allocate an empty one.
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

    const auto to_bf16_buffer = [device](const std::vector<float>& values) {
        std::vector<std::uint16_t> bits(values.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            bits[i] = f32_to_bf16(values[i]);
        }
        return upload_u16(device, bits);
    };

    if (linear.q4_weight) {
        return {to_bf16_buffer(linear.q4_weight->dequantize().data), true};
    }
    const Mat& w = linear.weight.data();
    if (w.numel() == 0) {
        return {alloc_buf(device, kMinBufferBytes), true};
    }
    return {to_bf16_buffer(w.data), true};
}

/// A projection's GPU buffer plus which kernel reads it.
struct WeightBuffer {
    id<MTLBuffer> buf = nil;
    bool is_bf16 = false;
};

WeightBuffer take_weight(id<MTLDevice> device, const Linear2& linear) {
    auto [buf, is_bf16] = upload_linear_detect_type(device, linear);
    return {buf, is_bf16};
}

/// The MLP and the two surrounding norms, which both layer kinds share.
struct SharedLayerWeights {
    WeightBuffer gate_proj;
    WeightBuffer up_proj;
    WeightBuffer down_proj;
    id<MTLBuffer> input_layernorm_gamma = nil;
    id<MTLBuffer> post_attn_layernorm_gamma = nil;
};

struct DeltaNetLayerWeights {
    WeightBuffer in_proj_qkv;
    WeightBuffer in_proj_z;
    WeightBuffer in_proj_a;
    WeightBuffer in_proj_b;
    WeightBuffer out_proj;
    /// All f32.
    id<MTLBuffer> conv1d_weight = nil;
    id<MTLBuffer> a_log = nil;
    id<MTLBuffer> dt_bias = nil;
    id<MTLBuffer> norm_weight = nil;
    SharedLayerWeights shared;
};

struct FullAttnLayerWeights {
    WeightBuffer q_proj;
    WeightBuffer k_proj;
    WeightBuffer v_proj;
    WeightBuffer o_proj;
    id<MTLBuffer> q_norm_gamma = nil;
    id<MTLBuffer> k_norm_gamma = nil;
    SharedLayerWeights shared;
};

using LayerWeightsQwen35 = std::variant<DeltaNetLayerWeights, FullAttnLayerWeights>;

}  // namespace

// =============================================================================
// Impl
// =============================================================================

struct MetalDecodeContextQwen35::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> pipe_rms_norm = nil;
    id<MTLComputePipelineState> pipe_rms_norm_per_head = nil;
    id<MTLComputePipelineState> pipe_embed_bf16 = nil;
    id<MTLComputePipelineState> pipe_vec_add = nil;
    id<MTLComputePipelineState> pipe_elem_mul = nil;
    id<MTLComputePipelineState> pipe_silu = nil;
    id<MTLComputePipelineState> pipe_scale = nil;
    id<MTLComputePipelineState> pipe_conv1d_silu = nil;
    id<MTLComputePipelineState> pipe_l2_normalize = nil;
    id<MTLComputePipelineState> pipe_dn_gates = nil;
    id<MTLComputePipelineState> pipe_dn_recurrent = nil;
    id<MTLComputePipelineState> pipe_gated_rms_norm = nil;
    id<MTLComputePipelineState> pipe_split_q_gate = nil;
    id<MTLComputePipelineState> pipe_sigmoid_gate = nil;
    id<MTLComputePipelineState> pipe_rope_partial = nil;
    id<MTLComputePipelineState> pipe_attention_decode = nil;
    id<MTLComputePipelineState> pipe_kv_append = nil;
    id<MTLComputePipelineState> pipe_gemv_bf16 = nil;
    id<MTLComputePipelineState> pipe_gemv_q4k = nil;
    id<MTLComputePipelineState> pipe_f32_to_f16 = nil;

    std::vector<LayerWeightsQwen35> layer_weights;

    id<MTLBuffer> embed_buf = nil;
    id<MTLBuffer> lm_head_buf = nil;
    bool lm_head_is_bf16 = false;
    id<MTLBuffer> final_norm_gamma = nil;

    // Activation scratch.
    id<MTLBuffer> buf_hidden = nil;
    id<MTLBuffer> buf_hidden2 = nil;
    id<MTLBuffer> buf_normed = nil;

    // DeltaNet scratch.
    id<MTLBuffer> buf_qkv = nil;
    id<MTLBuffer> buf_z = nil;
    id<MTLBuffer> buf_a = nil;
    id<MTLBuffer> buf_b = nil;
    id<MTLBuffer> buf_decay = nil;
    id<MTLBuffer> buf_beta = nil;
    id<MTLBuffer> buf_dn_output = nil;
    id<MTLBuffer> buf_dn_gated = nil;

    // Full-attention scratch.
    id<MTLBuffer> buf_qg = nil;
    id<MTLBuffer> buf_q = nil;
    id<MTLBuffer> buf_fa_gate = nil;
    id<MTLBuffer> buf_k = nil;
    id<MTLBuffer> buf_v = nil;
    id<MTLBuffer> buf_attn_out = nil;
    id<MTLBuffer> buf_sigmoid_out = nil;

    // MLP scratch.
    id<MTLBuffer> buf_mlp_gate = nil;
    id<MTLBuffer> buf_mlp_up = nil;
    id<MTLBuffer> buf_mlp_hidden = nil;
    id<MTLBuffer> buf_down_out = nil;

    id<MTLBuffer> buf_logits = nil;

    /// DeltaNet recurrent and conv1d state, persistent across decode steps.
    std::vector<id<MTLBuffer>> dn_state_bufs;
    std::vector<id<MTLBuffer>> dn_conv_state_bufs;
    /// Full-attention KV cache, half precision.
    std::vector<id<MTLBuffer>> fa_kv_k_bufs;
    std::vector<id<MTLBuffer>> fa_kv_v_bufs;

    ConfigQwen35 config;
    std::size_t lm_head_vocab = 0;

    // -------------------------------------------------------------------------
    // Dispatch helpers
    // -------------------------------------------------------------------------

    /// M=1 GEMV: `out[N] = act[K] @ W[N,K]^T`.
    void dispatch_gemv(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> act, const WeightBuffer& w,
                       id<MTLBuffer> out, std::size_t k, std::size_t n) const {
        [enc setComputePipelineState:(w.is_bf16 ? pipe_gemv_bf16 : pipe_gemv_q4k)];
        [enc setBuffer:act offset:0 atIndex:0];
        [enc setBuffer:w.buf offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(k), 3);
        set_const(enc, static_cast<std::uint32_t>(n), 4);
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    void dispatch_rms_norm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> gamma,
                           id<MTLBuffer> out, std::size_t dim) const {
        [enc setComputePipelineState:pipe_rms_norm];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:gamma offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(dim), 3);
        set_const(enc, config.rms_norm_eps, 4);
        const std::size_t tg = std::min<std::size_t>(256, dim);
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    /// Per-head RMSNorm, reading from `x` at `x_offset` bytes.
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
        [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    /// A flat elementwise kernel over `n` values.
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

    void dispatch_vec_add(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> a, id<MTLBuffer> b,
                          id<MTLBuffer> out, std::size_t n) const {
        dispatch_elementwise(enc, pipe_vec_add, {a, b, out}, n);
    }

    void dispatch_elem_mul(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> a, id<MTLBuffer> b,
                           id<MTLBuffer> out, std::size_t n) const {
        dispatch_elementwise(enc, pipe_elem_mul, {a, b, out}, n);
    }

    void dispatch_silu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> out,
                       std::size_t n) const {
        dispatch_elementwise(enc, pipe_silu, {x, out}, n);
    }

    void dispatch_sigmoid_gate(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                               id<MTLBuffer> gate, id<MTLBuffer> out, std::size_t n) const {
        dispatch_elementwise(enc, pipe_sigmoid_gate, {x, gate, out}, n);
    }

    /// The SwiGLU MLP and its trailing residual, identical in both layer kinds.
    void encode_mlp(id<MTLComputeCommandEncoder> enc, const SharedLayerWeights& sw) const {
        const std::size_t h = config.hidden_size;
        const std::size_t inter = config.intermediate_size;

        dispatch_gemv(enc, buf_normed, sw.gate_proj, buf_mlp_gate, h, inter);
        dispatch_gemv(enc, buf_normed, sw.up_proj, buf_mlp_up, h, inter);
        dispatch_silu(enc, buf_mlp_gate, buf_mlp_hidden, inter);
        dispatch_elem_mul(enc, buf_mlp_hidden, buf_mlp_up, buf_mlp_gate, inter);
        dispatch_gemv(enc, buf_mlp_gate, sw.down_proj, buf_down_out, inter, h);
        dispatch_vec_add(enc, buf_hidden2, buf_down_out, buf_hidden, h);
    }

    /// One Gated DeltaNet layer.
    void encode_deltanet_layer(id<MTLComputeCommandEncoder> enc, const DeltaNetLayerWeights& lw,
                               std::size_t dn_idx) const {
        const std::size_t h = config.hidden_size;
        const std::size_t qkv_dim = config.deltanet_qkv_dim();
        const std::size_t nk = config.linear_num_key_heads;
        const std::size_t nv = config.linear_num_value_heads;
        const std::size_t kd = config.linear_key_head_dim;
        const std::size_t vd = config.linear_value_head_dim;
        const std::size_t value_dim = nv * vd;
        const std::size_t key_dim = nk * kd;
        const std::size_t v_per_k = nv / nk;

        dispatch_rms_norm(enc, buf_hidden, lw.shared.input_layernorm_gamma, buf_normed, h);

        dispatch_gemv(enc, buf_normed, lw.in_proj_qkv, buf_qkv, h, qkv_dim);
        dispatch_gemv(enc, buf_normed, lw.in_proj_z, buf_z, h, value_dim);
        dispatch_gemv(enc, buf_normed, lw.in_proj_a, buf_a, h, nv);
        dispatch_gemv(enc, buf_normed, lw.in_proj_b, buf_b, h, nv);

        // Causal conv1d + SiLU, in place on buf_qkv, advancing the history.
        [enc setComputePipelineState:pipe_conv1d_silu];
        [enc setBuffer:buf_qkv offset:0 atIndex:0];
        [enc setBuffer:dn_conv_state_bufs[dn_idx] offset:0 atIndex:1];
        [enc setBuffer:lw.conv1d_weight offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(qkv_dim), 3);
        set_const(enc, static_cast<std::uint32_t>(config.linear_conv_kernel_dim), 4);
        {
            constexpr std::size_t tg = 256;
            [enc dispatchThreadgroups:MTLSizeMake(grid_1d(qkv_dim, tg), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // L2-normalize Q and K in place. The QKV buffer is laid out
        // [Q_all | K_all | V_all], so K starts key_dim floats in.
        const auto l2_normalize = [&](std::size_t byte_offset) {
            [enc setComputePipelineState:pipe_l2_normalize];
            [enc setBuffer:buf_qkv offset:byte_offset atIndex:0];
            set_const(enc, static_cast<std::uint32_t>(nk), 1);
            set_const(enc, static_cast<std::uint32_t>(kd), 2);
            [enc dispatchThreadgroups:MTLSizeMake(nk, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<std::size_t>(128, kd), 1, 1)];
        };
        l2_normalize(0);
        l2_normalize(key_dim * 4);

        // Scale Q by 1/sqrt(key_head_dim), in place.
        [enc setComputePipelineState:pipe_scale];
        [enc setBuffer:buf_qkv offset:0 atIndex:0];
        set_const(enc, 1.0f / std::sqrt(static_cast<float>(kd)), 1);
        set_const(enc, static_cast<std::uint32_t>(key_dim), 2);
        {
            constexpr std::size_t tg = 256;
            [enc dispatchThreadgroups:MTLSizeMake(grid_1d(key_dim, tg), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // Decay and write gates, one pair per value head.
        [enc setComputePipelineState:pipe_dn_gates];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:lw.a_log offset:0 atIndex:2];
        [enc setBuffer:lw.dt_bias offset:0 atIndex:3];
        [enc setBuffer:buf_decay offset:0 atIndex:4];
        [enc setBuffer:buf_beta offset:0 atIndex:5];
        set_const(enc, static_cast<std::uint32_t>(nv), 6);
        [enc dispatchThreadgroups:MTLSizeMake(grid_1d(nv, 32), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<std::size_t>(32, nv), 1, 1)];

        // The delta-rule state update. Q, K and V are three views into buf_qkv.
        [enc setComputePipelineState:pipe_dn_recurrent];
        [enc setBuffer:dn_state_bufs[dn_idx] offset:0 atIndex:0];
        [enc setBuffer:buf_qkv offset:0 atIndex:1];
        [enc setBuffer:buf_qkv offset:key_dim * 4 atIndex:2];
        [enc setBuffer:buf_qkv offset:key_dim * 2 * 4 atIndex:3];
        [enc setBuffer:buf_decay offset:0 atIndex:4];
        [enc setBuffer:buf_beta offset:0 atIndex:5];
        [enc setBuffer:buf_dn_output offset:0 atIndex:6];
        set_const(enc, static_cast<std::uint32_t>(kd), 7);
        set_const(enc, static_cast<std::uint32_t>(vd), 8);
        set_const(enc, static_cast<std::uint32_t>(v_per_k), 9);
        // One threadgroup per value head, one thread per value dimension.
        [enc dispatchThreadgroups:MTLSizeMake(nv, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(vd, 1, 1)];

        // Gated RMSNorm against SiLU(z).
        [enc setComputePipelineState:pipe_gated_rms_norm];
        [enc setBuffer:buf_dn_output offset:0 atIndex:0];
        [enc setBuffer:buf_z offset:0 atIndex:1];
        [enc setBuffer:lw.norm_weight offset:0 atIndex:2];
        [enc setBuffer:buf_dn_gated offset:0 atIndex:3];
        set_const(enc, static_cast<std::uint32_t>(nv), 4);
        set_const(enc, static_cast<std::uint32_t>(vd), 5);
        set_const(enc, 1e-6f, 6);
        [enc dispatchThreadgroups:MTLSizeMake(nv, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<std::size_t>(128, vd), 1, 1)];

        dispatch_gemv(enc, buf_dn_gated, lw.out_proj, buf_normed, value_dim, h);
        dispatch_vec_add(enc, buf_hidden, buf_normed, buf_hidden2, h);
        dispatch_rms_norm(enc, buf_hidden2, lw.shared.post_attn_layernorm_gamma, buf_normed, h);
        encode_mlp(enc, lw.shared);
    }

    /// One softmax-attention layer.
    void encode_fullattn_layer(id<MTLComputeCommandEncoder> enc, const FullAttnLayerWeights& lw,
                               std::size_t fa_idx, std::size_t position) const {
        const std::size_t h = config.hidden_size;
        const std::size_t nq = config.num_attention_heads;
        const std::size_t nkv = config.num_key_value_heads;
        const std::size_t d = config.head_dim;
        const std::size_t kv_dim = nkv * d;
        const std::size_t rope_dim = config.rope_dim();

        dispatch_rms_norm(enc, buf_hidden, lw.shared.input_layernorm_gamma, buf_normed, h);

        // q_proj emits query and gate together, so it is twice as wide.
        dispatch_gemv(enc, buf_normed, lw.q_proj, buf_qg, h, nq * d * 2);
        dispatch_gemv(enc, buf_normed, lw.k_proj, buf_k, h, kv_dim);
        dispatch_gemv(enc, buf_normed, lw.v_proj, buf_v, h, kv_dim);

        [enc setComputePipelineState:pipe_split_q_gate];
        [enc setBuffer:buf_qg offset:0 atIndex:0];
        [enc setBuffer:buf_q offset:0 atIndex:1];
        [enc setBuffer:buf_fa_gate offset:0 atIndex:2];
        set_const(enc, static_cast<std::uint32_t>(nq), 3);
        set_const(enc, static_cast<std::uint32_t>(d), 4);
        {
            constexpr std::size_t tg = 256;
            [enc dispatchThreadgroups:MTLSizeMake(grid_1d(nq * d, tg), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // The normalized Q and K go to scratch buffers that are free at this
        // point: buf_attn_out holds Q and buf_sigmoid_out holds K until the
        // attention kernel consumes them.
        dispatch_rms_norm_per_head(enc, buf_q, lw.q_norm_gamma, buf_attn_out, nq, d);
        dispatch_rms_norm_per_head(enc, buf_k, lw.k_norm_gamma, buf_sigmoid_out, nkv, d);

        // Partial RoPE, in place, over the first rope_dim of each head.
        const auto rope = [&](id<MTLBuffer> buf, std::size_t n_heads) {
            [enc setComputePipelineState:pipe_rope_partial];
            [enc setBuffer:buf offset:0 atIndex:0];
            set_const(enc, static_cast<std::uint32_t>(n_heads), 1);
            set_const(enc, static_cast<std::uint32_t>(d), 2);
            set_const(enc, static_cast<std::uint32_t>(rope_dim), 3);
            set_const(enc, config.rope_theta, 4);
            set_const(enc, static_cast<std::uint32_t>(position), 5);
            [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(rope_dim / 2, 1, 1)];
        };
        rope(buf_attn_out, nq);
        rope(buf_sigmoid_out, nkv);

        [enc setComputePipelineState:pipe_kv_append];
        [enc setBuffer:buf_sigmoid_out offset:0 atIndex:0];
        [enc setBuffer:buf_v offset:0 atIndex:1];
        [enc setBuffer:fa_kv_k_bufs[fa_idx] offset:0 atIndex:2];
        [enc setBuffer:fa_kv_v_bufs[fa_idx] offset:0 atIndex:3];
        set_const(enc, static_cast<std::uint32_t>(position), 4);
        set_const(enc, static_cast<std::uint32_t>(kv_dim), 5);
        {
            constexpr std::size_t tg = 256;
            [enc dispatchThreadgroups:MTLSizeMake(grid_1d(kv_dim, tg), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // Attend, writing into buf_q, which the split has already been read out of.
        [enc setComputePipelineState:pipe_attention_decode];
        [enc setBuffer:buf_attn_out offset:0 atIndex:0];
        [enc setBuffer:fa_kv_k_bufs[fa_idx] offset:0 atIndex:1];
        [enc setBuffer:fa_kv_v_bufs[fa_idx] offset:0 atIndex:2];
        [enc setBuffer:buf_q offset:0 atIndex:3];
        set_const(enc, static_cast<std::uint32_t>(nq), 4);
        set_const(enc, static_cast<std::uint32_t>(nkv), 5);
        set_const(enc, static_cast<std::uint32_t>(d), 6);
        set_const(enc, 1.0f / std::sqrt(static_cast<float>(d)), 7);
        set_const(enc, static_cast<std::uint32_t>(position + 1), 8);
        // Qwen 3.5's attention layers see the whole context.
        set_const(enc, std::uint32_t{0}, 9);
        [enc dispatchThreadgroups:MTLSizeMake(nq, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        dispatch_sigmoid_gate(enc, buf_q, buf_fa_gate, buf_sigmoid_out, nq * d);

        dispatch_gemv(enc, buf_sigmoid_out, lw.o_proj, buf_normed, nq * d, h);
        dispatch_vec_add(enc, buf_hidden, buf_normed, buf_hidden2, h);
        dispatch_rms_norm(enc, buf_hidden2, lw.shared.post_attn_layernorm_gamma, buf_normed, h);
        encode_mlp(enc, lw.shared);
    }

    /// Convert `n` f32 values into a half-precision buffer.
    void dispatch_f32_to_f16(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> src,
                             id<MTLBuffer> dst, std::size_t n) const {
        [enc setComputePipelineState:pipe_f32_to_f16];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        set_const(enc, static_cast<std::uint32_t>(n), 2);
        constexpr std::size_t tg = 256;
        [enc dispatchThreadgroups:MTLSizeMake(grid_1d(n, tg), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
};

// =============================================================================
// Construction
// =============================================================================

MetalDecodeContextQwen35::MetalDecodeContextQwen35(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MetalDecodeContextQwen35::~MetalDecodeContextQwen35() = default;

std::unique_ptr<MetalDecodeContextQwen35> MetalDecodeContextQwen35::create(
    const Qwen35Model& model) {
    auto impl = std::make_unique<Impl>();
    const ConfigQwen35& cfg = model.config;
    impl->config = cfg;

    impl->device = MTLCreateSystemDefaultDevice();
    if (impl->device == nil) {
        metal_fail("no GPU device found");
    }
    impl->queue = [impl->device newCommandQueue];
    if (impl->queue == nil) {
        metal_fail("command queue creation failed");
    }

    id<MTLLibrary> library = compile_library(impl->device, kDecodeQwen35Msl);
    const auto pipe = [&](const char* name) {
        return pipeline_from_library(impl->device, library, name);
    };
    impl->pipe_rms_norm = pipe("rms_norm_gemma3");
    impl->pipe_rms_norm_per_head = pipe("rms_norm_per_head");
    impl->pipe_embed_bf16 = pipe("embed_bf16_lookup");
    impl->pipe_vec_add = pipe("vec_add_kernel");
    impl->pipe_elem_mul = pipe("elem_mul_kernel");
    impl->pipe_silu = pipe("silu_kernel");
    impl->pipe_scale = pipe("scale_inplace");
    impl->pipe_conv1d_silu = pipe("conv1d_silu");
    impl->pipe_l2_normalize = pipe("l2_normalize_heads");
    impl->pipe_dn_gates = pipe("deltanet_compute_gates");
    impl->pipe_dn_recurrent = pipe("deltanet_recurrent");
    impl->pipe_gated_rms_norm = pipe("gated_rms_norm");
    impl->pipe_split_q_gate = pipe("split_q_gate");
    impl->pipe_sigmoid_gate = pipe("sigmoid_gate");
    impl->pipe_rope_partial = pipe("rope_neox_partial");
    impl->pipe_attention_decode = pipe("attention_decode");
    impl->pipe_kv_append = pipe("kv_cache_append");
    impl->pipe_gemv_bf16 = pipe("gemv_bf16_t");
    impl->pipe_gemv_q4k = pipe("gemv_q4k_t");
    impl->pipe_f32_to_f16 = pipe("f32_to_f16_convert");

    const auto upload_shared = [&](const Qwen35Mlp& mlp, const RmsNorm2& input_norm,
                                   const RmsNorm2& post_norm) {
        SharedLayerWeights sw;
        sw.gate_proj = take_weight(impl->device, mlp.gate_proj);
        sw.up_proj = take_weight(impl->device, mlp.up_proj);
        sw.down_proj = take_weight(impl->device, mlp.down_proj);
        sw.input_layernorm_gamma = upload_f32(impl->device, input_norm.gamma.data().data);
        sw.post_attn_layernorm_gamma = upload_f32(impl->device, post_norm.gamma.data().data);
        return sw;
    };

    std::size_t n_deltanet = 0;
    std::size_t n_fullattn = 0;
    impl->layer_weights.reserve(cfg.num_hidden_layers);

    for (const Qwen35Block& layer : model.layers) {
        SharedLayerWeights shared =
            upload_shared(layer.mlp, layer.input_layernorm, layer.post_attention_layernorm);

        if (const auto* dn = std::get_if<Qwen35DeltaNet>(&layer.token_mixer)) {
            DeltaNetLayerWeights lw;
            lw.in_proj_qkv = take_weight(impl->device, dn->in_proj_qkv);
            lw.in_proj_z = take_weight(impl->device, dn->in_proj_z);
            lw.in_proj_a = take_weight(impl->device, dn->in_proj_a);
            lw.in_proj_b = take_weight(impl->device, dn->in_proj_b);
            lw.out_proj = take_weight(impl->device, dn->out_proj);
            lw.conv1d_weight = upload_f32(impl->device, dn->conv1d_weight);
            lw.a_log = upload_f32(impl->device, dn->a_log);
            lw.dt_bias = upload_f32(impl->device, dn->dt_bias);
            lw.norm_weight = upload_f32(impl->device, dn->norm_weight);
            lw.shared = std::move(shared);
            impl->layer_weights.emplace_back(std::move(lw));
            ++n_deltanet;
        } else {
            const auto& fa = std::get<Qwen35FullAttention>(layer.token_mixer);
            FullAttnLayerWeights lw;
            lw.q_proj = take_weight(impl->device, fa.q_proj);
            lw.k_proj = take_weight(impl->device, fa.k_proj);
            lw.v_proj = take_weight(impl->device, fa.v_proj);
            lw.o_proj = take_weight(impl->device, fa.o_proj);
            lw.q_norm_gamma = upload_f32(impl->device, fa.q_norm.gamma.data().data);
            lw.k_norm_gamma = upload_f32(impl->device, fa.k_norm.gamma.data().data);
            lw.shared = std::move(shared);
            impl->layer_weights.emplace_back(std::move(lw));
            ++n_fullattn;
        }
    }

    if (model.embed_bf16) {
        impl->embed_buf = upload_u16(impl->device, *model.embed_bf16->data);
    } else {
        impl->embed_buf = upload_f32(impl->device, model.embed_tokens.data().data);
    }

    // lm_head is usually weight-tied, in which case it shares the embedding's
    // GPU buffer rather than being uploaded twice.
    if (model.lm_head.bf16_weight) {
        const MatBf16& bf16 = *model.lm_head.bf16_weight;
        const bool tied = model.embed_bf16 && bf16.data == model.embed_bf16->data;
        impl->lm_head_buf = tied ? impl->embed_buf : upload_u16(impl->device, *bf16.data);
        impl->lm_head_is_bf16 = true;
        impl->lm_head_vocab = bf16.rows;
    } else if (model.lm_head.q4k_weight) {
        impl->lm_head_buf = upload_bytes(impl->device, model.lm_head.q4k_weight->blocks.data(),
                                         model.lm_head.q4k_weight->blocks.size());
        impl->lm_head_is_bf16 = false;
        impl->lm_head_vocab = model.lm_head.q4k_weight->rows;
    } else {
        const auto [buf, is_bf16] = upload_linear_detect_type(impl->device, model.lm_head);
        impl->lm_head_buf = buf;
        impl->lm_head_is_bf16 = is_bf16;
        impl->lm_head_vocab = cfg.vocab_size;
    }

    impl->final_norm_gamma = upload_f32(impl->device, model.norm.gamma.data().data);

    const std::size_t h = cfg.hidden_size;
    const std::size_t nq = cfg.num_attention_heads;
    const std::size_t nkv = cfg.num_key_value_heads;
    const std::size_t d = cfg.head_dim;
    const std::size_t inter = cfg.intermediate_size;
    const std::size_t qkv_dim = cfg.deltanet_qkv_dim();
    const std::size_t nv = cfg.linear_num_value_heads;
    const std::size_t kd = cfg.linear_key_head_dim;
    const std::size_t vd = cfg.linear_value_head_dim;
    const std::size_t value_dim = nv * vd;
    const std::size_t kv_dim = nkv * d;

    impl->buf_hidden = alloc_buf(impl->device, h * 4);
    impl->buf_hidden2 = alloc_buf(impl->device, h * 4);
    impl->buf_normed = alloc_buf(impl->device, h * 4);

    impl->buf_qkv = alloc_buf(impl->device, qkv_dim * 4);
    impl->buf_z = alloc_buf(impl->device, value_dim * 4);
    impl->buf_a = alloc_buf(impl->device, nv * 4);
    impl->buf_b = alloc_buf(impl->device, nv * 4);
    impl->buf_decay = alloc_buf(impl->device, nv * 4);
    impl->buf_beta = alloc_buf(impl->device, nv * 4);
    impl->buf_dn_output = alloc_buf(impl->device, value_dim * 4);
    impl->buf_dn_gated = alloc_buf(impl->device, value_dim * 4);

    impl->buf_qg = alloc_buf(impl->device, nq * d * 2 * 4);
    impl->buf_q = alloc_buf(impl->device, nq * d * 4);
    impl->buf_fa_gate = alloc_buf(impl->device, nq * d * 4);
    impl->buf_k = alloc_buf(impl->device, kv_dim * 4);
    impl->buf_v = alloc_buf(impl->device, kv_dim * 4);
    impl->buf_attn_out = alloc_buf(impl->device, nq * d * 4);
    impl->buf_sigmoid_out = alloc_buf(impl->device, nq * d * 4);

    impl->buf_mlp_gate = alloc_buf(impl->device, inter * 4);
    impl->buf_mlp_up = alloc_buf(impl->device, inter * 4);
    impl->buf_mlp_hidden = alloc_buf(impl->device, inter * 4);
    impl->buf_down_out = alloc_buf(impl->device, h * 4);
    impl->buf_logits = alloc_buf(impl->device, impl->lm_head_vocab * 4);

    // Per-layer state, persistent across decode steps.
    const std::size_t hist = cfg.linear_conv_kernel_dim - 1;
    for (std::size_t i = 0; i < n_deltanet; ++i) {
        impl->dn_state_bufs.push_back(alloc_buf(impl->device, nv * kd * vd * 4));
        impl->dn_conv_state_bufs.push_back(alloc_buf(impl->device, qkv_dim * hist * 4));
    }
    for (std::size_t i = 0; i < n_fullattn; ++i) {
        impl->fa_kv_k_bufs.push_back(alloc_buf(impl->device, kMaxSeqLen * kv_dim * 2));
        impl->fa_kv_v_bufs.push_back(alloc_buf(impl->device, kMaxSeqLen * kv_dim * 2));
    }

    std::fprintf(stderr, "[ Metal ] Qwen3.5 decode context: %zu DeltaNet + %zu attention layers\n",
                 n_deltanet, n_fullattn);

    return std::unique_ptr<MetalDecodeContextQwen35>(
        new MetalDecodeContextQwen35(std::move(impl)));
}

std::size_t MetalDecodeContextQwen35::lm_head_vocab() const { return impl_->lm_head_vocab; }

std::vector<float> MetalDecodeContextQwen35::decode_step(std::size_t token_id,
                                                         std::size_t position) {
    const std::size_t h = impl_->config.hidden_size;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Unlike Gemma 3, Qwen 3.5 does not scale the looked-up embedding.
        [enc setComputePipelineState:impl_->pipe_embed_bf16];
        [enc setBuffer:impl_->embed_buf offset:0 atIndex:0];
        [enc setBuffer:impl_->buf_hidden offset:0 atIndex:1];
        set_const(enc, static_cast<std::uint32_t>(token_id), 2);
        set_const(enc, static_cast<std::uint32_t>(h), 3);
        {
            constexpr std::size_t tg = 256;
            [enc dispatchThreadgroups:MTLSizeMake(grid_1d(h, tg), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // The two layer kinds index their own state arrays, so each keeps its
        // own running counter.
        std::size_t dn_idx = 0;
        std::size_t fa_idx = 0;
        for (const LayerWeightsQwen35& lw : impl_->layer_weights) {
            if (const auto* dn = std::get_if<DeltaNetLayerWeights>(&lw)) {
                impl_->encode_deltanet_layer(enc, *dn, dn_idx++);
            } else {
                impl_->encode_fullattn_layer(enc, std::get<FullAttnLayerWeights>(lw), fa_idx++,
                                             position);
            }
        }

        impl_->dispatch_rms_norm(enc, impl_->buf_hidden, impl_->final_norm_gamma,
                                 impl_->buf_normed, h);
        impl_->dispatch_gemv(enc, impl_->buf_normed,
                             {impl_->lm_head_buf, impl_->lm_head_is_bf16}, impl_->buf_logits, h,
                             impl_->lm_head_vocab);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        const auto* ptr = static_cast<const float*>([impl_->buf_logits contents]);
        return std::vector<float>(ptr, ptr + impl_->lm_head_vocab);
    }
}

void MetalDecodeContextQwen35::sync_state_from_cpu(const Qwen35Cache& cache) {
    const std::size_t kv_dim = impl_->config.num_key_value_heads * impl_->config.head_dim;

    std::size_t dn_idx = 0;
    std::size_t fa_idx = 0;

    for (const LayerCache& layer_cache : cache.layers) {
        if (const auto* state = std::get_if<DeltaNetState>(&layer_cache)) {
            // The recurrent and conv states are f32 on both sides, so they copy
            // straight into the shared buffers.
            std::memcpy([impl_->dn_state_bufs[dn_idx] contents], state->state.data(),
                        state->state.size() * 4);
            std::memcpy([impl_->dn_conv_state_bufs[dn_idx] contents], state->conv_state.data(),
                        state->conv_state.size() * 4);
            ++dn_idx;
            continue;
        }

        const auto& kv = std::get<FullAttnKvCache>(layer_cache);
        if (kv.seq_len > 0) {
            // The GPU cache is half precision, so K and V go up as f32 and are
            // converted on device.
            const std::size_t n_floats = kv.seq_len * kv_dim;
            @autoreleasepool {
                id<MTLBuffer> staging_k = upload_bytes(impl_->device, kv.k.data.data(),
                                                       n_floats * 4);
                id<MTLBuffer> staging_v = upload_bytes(impl_->device, kv.v.data.data(),
                                                       n_floats * 4);

                id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                impl_->dispatch_f32_to_f16(enc, staging_k, impl_->fa_kv_k_bufs[fa_idx], n_floats);
                impl_->dispatch_f32_to_f16(enc, staging_v, impl_->fa_kv_v_bufs[fa_idx], n_floats);
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }
        }
        ++fa_idx;
    }
}

void MetalDecodeContextQwen35::print_memory_stats() const {
    std::size_t weight_bytes = 0;
    for (const LayerWeightsQwen35& lw : impl_->layer_weights) {
        const auto add = [&weight_bytes](const WeightBuffer& w) { weight_bytes += [w.buf length]; };
        if (const auto* dn = std::get_if<DeltaNetLayerWeights>(&lw)) {
            add(dn->in_proj_qkv);
            add(dn->in_proj_z);
            add(dn->in_proj_a);
            add(dn->in_proj_b);
            add(dn->out_proj);
            add(dn->shared.gate_proj);
            add(dn->shared.up_proj);
            add(dn->shared.down_proj);
        } else {
            const auto& fa = std::get<FullAttnLayerWeights>(lw);
            add(fa.q_proj);
            add(fa.k_proj);
            add(fa.v_proj);
            add(fa.o_proj);
            add(fa.shared.gate_proj);
            add(fa.shared.up_proj);
            add(fa.shared.down_proj);
        }
    }

    std::size_t state_bytes = 0;
    for (id<MTLBuffer> buf : impl_->dn_state_bufs) {
        state_bytes += [buf length];
    }
    for (id<MTLBuffer> buf : impl_->dn_conv_state_bufs) {
        state_bytes += [buf length];
    }
    std::size_t kv_bytes = 0;
    for (std::size_t i = 0; i < impl_->fa_kv_k_bufs.size(); ++i) {
        kv_bytes += [impl_->fa_kv_k_bufs[i] length] + [impl_->fa_kv_v_bufs[i] length];
    }

    std::fprintf(stderr,
                 "[ Metal ] GPU buffers: weights %.2f GB, embed %.2f GB, DeltaNet state %.2f GB, "
                 "KV cache %.2f GB\n",
                 static_cast<double>(weight_bytes) / 1e9,
                 static_cast<double>([impl_->embed_buf length]) / 1e9,
                 static_cast<double>(state_bytes) / 1e9, static_cast<double>(kv_bytes) / 1e9);
}

}  // namespace rt

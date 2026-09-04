#include "rt/metal_ops.hpp"

#import <Metal/Metal.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>

namespace rt {
namespace {

// =============================================================================
// MSL sources
// =============================================================================

// Tiled matmul with shared memory. TS must match the 16x16 threadgroup used at
// dispatch: each thread computes one output element, the K dimension is split
// into tiles, and every threadgroup loads one 16x16 block of A and B into fast
// threadgroup memory before accumulating. Out-of-range loads read 0, so partial
// tiles work when M, K or N are not multiples of TS.
constexpr const char* kMatmulMsl = R"(
#include <metal_stdlib>
using namespace metal;

#define TS 16u

// Single matrix multiply: C[M,N] = A[M,K] @ B[K,N]
kernel void matmul_tiled(
    device const float* A  [[ buffer(0) ]],
    device const float* B  [[ buffer(1) ]],
    device       float* C  [[ buffer(2) ]],
    constant     uint&  M  [[ buffer(3) ]],
    constant     uint&  K  [[ buffer(4) ]],
    constant     uint&  N  [[ buffer(5) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint2 tid  [[ thread_position_in_threadgroup ]])
{
    uint row = tgid.y * TS + tid.y;
    uint col = tgid.x * TS + tid.x;

    threadgroup float As[TS][TS];
    threadgroup float Bs[TS][TS];

    float acc = 0.0f;
    uint n_tiles = (K + TS - 1u) / TS;

    for (uint t = 0u; t < n_tiles; t++) {
        uint a_col = t * TS + tid.x;
        As[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        uint b_row = t * TS + tid.y;
        Bs[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0u; k < TS; k++)
            acc += As[tid.y][k] * Bs[k][tid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// Batched matrix multiply: C[b,M,N] = A[b,M,K] @ B[b,K,N].
//
// Every pair shares M, K and N, and the matrices are packed contiguously. The
// grid's Z dimension is the batch size, so tgid.z selects the slice.
kernel void matmul_tiled_batched(
    device const float* A  [[ buffer(0) ]],
    device const float* B  [[ buffer(1) ]],
    device       float* C  [[ buffer(2) ]],
    constant     uint&  M  [[ buffer(3) ]],
    constant     uint&  K  [[ buffer(4) ]],
    constant     uint&  N  [[ buffer(5) ]],
    uint3 tgid [[ threadgroup_position_in_grid ]],
    uint3 tid  [[ thread_position_in_threadgroup ]])
{
    uint b   = tgid.z;
    uint row = tgid.y * TS + tid.y;
    uint col = tgid.x * TS + tid.x;

    device const float* Ab = A + b * M * K;
    device const float* Bb = B + b * K * N;
    device       float* Cb = C + b * M * N;

    threadgroup float As[TS][TS];
    threadgroup float Bs[TS][TS];

    float acc = 0.0f;
    uint n_tiles = (K + TS - 1u) / TS;

    for (uint t = 0u; t < n_tiles; t++) {
        uint a_col = t * TS + tid.x;
        As[tid.y][tid.x] = (row < M && a_col < K) ? Ab[row * K + a_col] : 0.0f;

        uint b_row = t * TS + tid.y;
        Bs[tid.y][tid.x] = (b_row < K && col < N) ? Bb[b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0u; k < TS; k++)
            acc += As[tid.y][k] * Bs[k][tid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        Cb[row * N + col] = acc;
}
)";

// Q4_0 fused dequantize + GEMV, tuned for single-token decode.
//
// One threadgroup per output element, TG_K threads striding over K, collapsed
// by simd_sum. TG_K must equal the SIMD width (32 on every Apple GPU) so the
// reduction covers the whole threadgroup.
constexpr const char* kMatmulQ4Msl = R"(
#include <metal_stdlib>
using namespace metal;

#define Q4_BLOCK_SIZE 32u
#define TG_K          32u    // must equal simd_size (32 on all Apple GPUs)

kernel void matmul_q4_t(
    device const float* A       [[ buffer(0) ]],
    device const uchar* packed  [[ buffer(1) ]],
    device const float* scales  [[ buffer(2) ]],
    device       float* C       [[ buffer(3) ]],
    constant     uint&  M       [[ buffer(4) ]],
    constant     uint&  K       [[ buffer(5) ]],
    constant     uint&  N       [[ buffer(6) ]],
    uint3 tgid   [[ threadgroup_position_in_grid ]],
    uint  lid    [[ thread_index_in_threadgroup ]])
{
    uint j = tgid.x;   // output neuron (weight row)
    uint i = tgid.y;   // input row (0 for single-token decode)
    if (j >= N || i >= M) return;

    float acc = 0.0f;
    uint row_start = j * K;

    for (uint p = lid; p < K; p += TG_K) {
        uint  flat   = row_start + p;
        uchar byte_v = packed[flat >> 1u];
        uchar nibble = (flat & 1u) == 0u ? (byte_v & 0x0Fu) : ((byte_v >> 4u) & 0x0Fu);
        int   q      = (nibble >= 8u) ? (int(nibble) - 16) : int(nibble);
        float w      = float(q) * scales[flat / Q4_BLOCK_SIZE];
        acc += A[i * K + p] * w;
    }

    acc = simd_sum(acc);

    if (lid == 0)
        C[i * N + j] = acc;
}
)";

// Q4_K fused dequantize + GEMV.
//
// Each 256-element super-block occupies 144 bytes: f16 d, f16 dmin, 12 bytes of
// packed 6-bit scale/min pairs, then 128 bytes of nibbles. One threadgroup per
// output element strides over super-blocks and dequantizes on the fly.
constexpr const char* kGemvQ4kMsl = R"(
#include <metal_stdlib>
using namespace metal;

#define Q4K_BLOCK_BYTES 144u
#define Q4K_BLOCK_ELEMS 256u
#define TG_K 32u

// Extract the 6-bit scale and min for sub-block j (0..8) from the 12-byte array.
inline float2 scale_min(device const uchar* sc, uint j) {
    float sv, mv;
    if (j < 4u) {
        sv = float(sc[j] & 0x3Fu);
        mv = float(sc[j + 4u] & 0x3Fu);
    } else {
        sv = float((sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u));
        mv = float((sc[j + 4u] >> 4u)   | ((sc[j]      >> 6u) << 4u));
    }
    return float2(sv, mv);
}

kernel void gemv_q4k_t(
    device const float* A         [[ buffer(0) ]],
    device const uchar* blocks    [[ buffer(1) ]],
    device       float* C         [[ buffer(2) ]],
    constant     uint&  K         [[ buffer(3) ]],
    constant     uint&  N         [[ buffer(4) ]],
    uint  tgid_x [[ threadgroup_position_in_grid ]],
    uint  lid    [[ thread_index_in_threadgroup ]])
{
    uint j = tgid_x;
    if (j >= N) return;

    uint n_sb = K / Q4K_BLOCK_ELEMS;
    float acc = 0.0f;

    for (uint b = lid; b < n_sb; b += TG_K) {
        uint boff = (j * n_sb + b) * Q4K_BLOCK_BYTES;
        device const uchar* bp = blocks + boff;

        ushort d_bits    = ushort(bp[0]) | (ushort(bp[1]) << 8u);
        ushort dmin_bits = ushort(bp[2]) | (ushort(bp[3]) << 8u);
        float d    = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));

        device const uchar* sc = bp + 4u;
        device const uchar* qs = bp + 16u;
        uint a_base = b * Q4K_BLOCK_ELEMS;

        for (uint chunk = 0u; chunk < 4u; chunk++) {
            float2 sm1 = scale_min(sc, chunk * 2u);
            float2 sm2 = scale_min(sc, chunk * 2u + 1u);
            float scale1 = d * sm1.x;  float min1 = dmin * sm1.y;
            float scale2 = d * sm2.x;  float min2 = dmin * sm2.y;

            device const uchar* q = qs + chunk * 32u;
            uint a_off = a_base + chunk * 64u;

            float dot_lo = 0.0f, dot_hi = 0.0f;
            float sum_lo = 0.0f, sum_hi = 0.0f;

            for (uint l = 0u; l < 32u; l++) {
                float a_lo = A[a_off + l];
                float a_hi = A[a_off + 32u + l];
                dot_lo += float(q[l] & 0x0Fu) * a_lo;
                dot_hi += float(q[l] >> 4u)   * a_hi;
                sum_lo += a_lo;
                sum_hi += a_hi;
            }

            acc += scale1 * dot_lo - min1 * sum_lo
                 + scale2 * dot_hi - min2 * sum_hi;
        }
    }

    acc = simd_sum(acc);

    if (lid == 0u)
        C[j] = acc;
}
)";

// BF16 GEMV: the weights are the upper 16 bits of an f32, so widening is a
// shift rather than a conversion.
constexpr const char* kGemvBf16Msl = R"(
#include <metal_stdlib>
using namespace metal;

#define TG_K_BF16 32u

kernel void gemv_bf16_t(
    device const float*  A      [[ buffer(0) ]],
    device const ushort* W      [[ buffer(1) ]],
    device       float*  C      [[ buffer(2) ]],
    constant     uint&   K      [[ buffer(3) ]],
    constant     uint&   N      [[ buffer(4) ]],
    uint  tgid_x [[ threadgroup_position_in_grid ]],
    uint  lid    [[ thread_index_in_threadgroup ]])
{
    uint j = tgid_x;
    if (j >= N) return;

    float acc = 0.0f;
    uint row_start = j * K;

    for (uint p = lid; p < K; p += TG_K_BF16) {
        ushort bits = W[row_start + p];
        float w = as_type<float>(uint(bits) << 16u);
        acc += A[p] * w;
    }

    acc = simd_sum(acc);

    if (lid == 0u)
        C[j] = acc;
}
)";

// =============================================================================
// Context
// =============================================================================

/// The Metal objects a thread reuses across calls.
struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLComputePipelineState> pipeline_batched = nil;
    id<MTLComputePipelineState> pipeline_q4 = nil;
    id<MTLComputePipelineState> pipeline_q4k = nil;
    id<MTLComputePipelineState> pipeline_bf16 = nil;

    /// Persistent weight buffers, keyed by (data pointer, byte length): a
    /// weight matrix is uploaded once and reused for every later call.
    std::map<std::pair<std::uintptr_t, std::size_t>, id<MTLBuffer>> weight_cache;

    /// Scratch for the GEMV path, so a decode step allocates nothing.
    id<MTLBuffer> scratch_act = nil;
    id<MTLBuffer> scratch_out = nil;
    id<MTLBuffer> scratch_dims = nil;
    std::size_t scratch_act_cap = 0;  // in f32 elements
    std::size_t scratch_out_cap = 0;
};

[[noreturn]] void metal_fail(const std::string& what) {
    std::fprintf(stderr, "Metal: %s\n", what.c_str());
    std::abort();
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, const char* src,
                                          const char* name) {
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(src) options:nil error:&error];
    if (library == nil) {
        metal_fail(std::string("MSL compilation failed for '") + name + "': " +
                   [[error localizedDescription] UTF8String]);
    }
    id<MTLFunction> func = [library newFunctionWithName:@(name)];
    if (func == nil) {
        metal_fail(std::string("function '") + name + "' not found");
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:func error:&error];
    if (pipeline == nil) {
        metal_fail(std::string("could not create pipeline for '") + name + "'");
    }
    return pipeline;
}

MetalContext& context() {
    // One context per thread; creating the device is expensive, so it is built
    // once and kept for the thread's lifetime.
    thread_local std::unique_ptr<MetalContext> ctx;
    if (ctx) {
        return *ctx;
    }

    ctx = std::make_unique<MetalContext>();
    ctx->device = MTLCreateSystemDefaultDevice();
    if (ctx->device == nil) {
        metal_fail("no GPU device found");
    }
    ctx->queue = [ctx->device newCommandQueue];
    if (ctx->queue == nil) {
        metal_fail("could not create command queue");
    }

    ctx->pipeline = make_pipeline(ctx->device, kMatmulMsl, "matmul_tiled");
    ctx->pipeline_batched = make_pipeline(ctx->device, kMatmulMsl, "matmul_tiled_batched");
    ctx->pipeline_q4 = make_pipeline(ctx->device, kMatmulQ4Msl, "matmul_q4_t");
    ctx->pipeline_q4k = make_pipeline(ctx->device, kGemvQ4kMsl, "gemv_q4k_t");
    ctx->pipeline_bf16 = make_pipeline(ctx->device, kGemvBf16Msl, "gemv_bf16_t");

    // Sized for the Gemma 3 4B dimensions; both grow on demand.
    ctx->scratch_act_cap = 10240;   // largest K (down_proj input)
    ctx->scratch_out_cap = 262144;  // largest N (lm_head)
    ctx->scratch_act = [ctx->device newBufferWithLength:ctx->scratch_act_cap * 4
                                                options:MTLResourceStorageModeShared];
    ctx->scratch_out = [ctx->device newBufferWithLength:ctx->scratch_out_cap * 4
                                                options:MTLResourceStorageModeShared];
    // dims holds [K, N] as two u32s.
    ctx->scratch_dims = [ctx->device newBufferWithLength:8
                                                 options:MTLResourceStorageModeShared];
    return *ctx;
}

// =============================================================================
// Buffer helpers
// =============================================================================

id<MTLBuffer> upload(id<MTLDevice> device, const void* data, std::size_t byte_len) {
    id<MTLBuffer> buf = [device newBufferWithBytes:data
                                            length:byte_len
                                           options:MTLResourceStorageModeShared];
    if (buf == nil) {
        metal_fail("buffer allocation failed");
    }
    return buf;
}

id<MTLBuffer> upload_floats(id<MTLDevice> device, const std::vector<float>& data) {
    return upload(device, data.data(), data.size() * 4);
}

id<MTLBuffer> alloc_output(id<MTLDevice> device, std::size_t elems) {
    id<MTLBuffer> buf = [device newBufferWithLength:elems * 4
                                            options:MTLResourceStorageModeShared];
    if (buf == nil) {
        metal_fail("output buffer allocation failed");
    }
    return buf;
}

id<MTLBuffer> upload_u32(id<MTLDevice> device, std::uint32_t val) {
    return upload(device, &val, 4);
}

/// Read `elems` floats back out of a shared-memory buffer.
std::vector<float> read_back(id<MTLBuffer> buf, std::size_t elems) {
    const auto* ptr = static_cast<const float*>([buf contents]);
    return std::vector<float>(ptr, ptr + elems);
}

/// Grow a scratch buffer if it cannot hold `elems` floats.
///
/// The reference is `__strong` because that is the ownership the context's
/// members have; a plain `id<MTLBuffer>&` would default to `__autoreleasing`
/// and not bind to them.
void ensure_capacity(MetalContext& ctx, __strong id<MTLBuffer>& buf, std::size_t& cap,
                     std::size_t elems) {
    if (elems > cap) {
        buf = [ctx.device newBufferWithLength:elems * 4 options:MTLResourceStorageModeShared];
        cap = elems;
    }
}

/// Fetch the cached buffer for a weight blob, uploading it the first time.
id<MTLBuffer> cached_weights(MetalContext& ctx, const void* data, std::size_t byte_len) {
    const auto key = std::pair{reinterpret_cast<std::uintptr_t>(data), byte_len};
    const auto it = ctx.weight_cache.find(key);
    if (it != ctx.weight_cache.end()) {
        return it->second;
    }
    id<MTLBuffer> buf = upload(ctx.device, data, byte_len);
    ctx.weight_cache.emplace(key, buf);
    return buf;
}

/// Scalar `C = A @ B`, for problems too small to dispatch.
Mat cpu_matmul(const Mat& a, const Mat& b) {
    const std::size_t m = a.rows, k = a.cols, n = b.cols;
    Mat out = Mat::zeros(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const float a_ip = a.at(i, p);
            for (std::size_t j = 0; j < n; ++j) {
                out.at_mut(i, j) += a_ip * b.at(p, j);
            }
        }
    }
    return out;
}

}  // namespace

// =============================================================================
// Entry points
// =============================================================================

Mat metal_matmul(const Mat& a, const Mat& b) {
    const std::size_t m = a.rows, k = a.cols, n = b.cols;
    if (m * k * n < METAL_THRESHOLD) {
        return cpu_matmul(a, b);
    }

    @autoreleasepool {
        MetalContext& ctx = context();

        id<MTLBuffer> buf_a = upload_floats(ctx.device, a.data);
        id<MTLBuffer> buf_b = upload_floats(ctx.device, b.data);
        id<MTLBuffer> buf_c = alloc_output(ctx.device, m * n);
        id<MTLBuffer> buf_m = upload_u32(ctx.device, static_cast<std::uint32_t>(m));
        id<MTLBuffer> buf_k = upload_u32(ctx.device, static_cast<std::uint32_t>(k));
        id<MTLBuffer> buf_n = upload_u32(ctx.device, static_cast<std::uint32_t>(n));

        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx.pipeline];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_c offset:0 atIndex:2];
        [enc setBuffer:buf_m offset:0 atIndex:3];
        [enc setBuffer:buf_k offset:0 atIndex:4];
        [enc setBuffer:buf_n offset:0 atIndex:5];

        // The 16x16 threadgroup matches TS in the kernel; the grid covers M x N.
        const MTLSize tg = MTLSizeMake(16, 16, 1);
        const MTLSize grid = MTLSizeMake((n + 15) / 16, (m + 15) / 16, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        return Mat(read_back(buf_c, m * n), m, n);
    }
}

std::vector<Mat> metal_matmul_batched(const std::vector<std::pair<Mat, Mat>>& pairs) {
    if (pairs.empty()) {
        return {};
    }
    if (pairs.size() == 1) {
        return {metal_matmul(pairs[0].first, pairs[0].second)};
    }

    const std::size_t m = pairs[0].first.rows;
    const std::size_t k = pairs[0].first.cols;
    const std::size_t n = pairs[0].second.cols;

    for (const auto& pair : pairs) {
        assert(pair.first.rows == m && pair.first.cols == k &&
               "metal_matmul_batched: A shape mismatch");
        assert(pair.second.rows == k && pair.second.cols == n &&
               "metal_matmul_batched: B shape mismatch");
        (void)pair;
    }

    const std::size_t batch = pairs.size();
    if (m * k * n < METAL_THRESHOLD) {
        std::vector<Mat> out;
        out.reserve(batch);
        for (const auto& [a, b] : pairs) {
            out.push_back(metal_matmul(a, b));
        }
        return out;
    }

    @autoreleasepool {
        MetalContext& ctx = context();

        // Pack every A slice contiguously, then every B slice.
        std::vector<float> a_flat, b_flat;
        a_flat.reserve(batch * m * k);
        b_flat.reserve(batch * k * n);
        for (const auto& [a, b] : pairs) {
            a_flat.insert(a_flat.end(), a.data.begin(), a.data.end());
            b_flat.insert(b_flat.end(), b.data.begin(), b.data.end());
        }

        id<MTLBuffer> buf_a = upload_floats(ctx.device, a_flat);
        id<MTLBuffer> buf_b = upload_floats(ctx.device, b_flat);
        id<MTLBuffer> buf_c = alloc_output(ctx.device, batch * m * n);
        id<MTLBuffer> buf_m = upload_u32(ctx.device, static_cast<std::uint32_t>(m));
        id<MTLBuffer> buf_k = upload_u32(ctx.device, static_cast<std::uint32_t>(k));
        id<MTLBuffer> buf_n = upload_u32(ctx.device, static_cast<std::uint32_t>(n));

        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx.pipeline_batched];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_c offset:0 atIndex:2];
        [enc setBuffer:buf_m offset:0 atIndex:3];
        [enc setBuffer:buf_k offset:0 atIndex:4];
        [enc setBuffer:buf_n offset:0 atIndex:5];

        // XY covers M x N in 16x16 tiles; Z is the batch.
        const MTLSize tg = MTLSizeMake(16, 16, 1);
        const MTLSize grid = MTLSizeMake((n + 15) / 16, (m + 15) / 16, batch);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        const std::vector<float> flat = read_back(buf_c, batch * m * n);
        std::vector<Mat> out;
        out.reserve(batch);
        for (std::size_t b = 0; b < batch; ++b) {
            const std::size_t start = b * m * n;
            out.emplace_back(std::vector<float>(flat.begin() + static_cast<std::ptrdiff_t>(start),
                                                flat.begin() +
                                                    static_cast<std::ptrdiff_t>(start + m * n)),
                             m, n);
        }
        return out;
    }
}

Mat metal_matmul_q4_t(const Mat& a, const Q4Mat& q4) {
    const std::size_t m = a.rows, k = a.cols, n = q4.rows;
    assert(k == q4.cols && "metal_matmul_q4_t: a.cols != q4.cols");

    if (m * k * n < METAL_THRESHOLD) {
        return q4.matmul_q4_t(a);
    }

    @autoreleasepool {
        MetalContext& ctx = context();

        id<MTLBuffer> buf_a = upload_floats(ctx.device, a.data);
        id<MTLBuffer> buf_packed = upload(ctx.device, q4.packed.data(), q4.packed.size());
        id<MTLBuffer> buf_scales = upload_floats(ctx.device, q4.scales);
        id<MTLBuffer> buf_c = alloc_output(ctx.device, m * n);
        id<MTLBuffer> buf_m = upload_u32(ctx.device, static_cast<std::uint32_t>(m));
        id<MTLBuffer> buf_k = upload_u32(ctx.device, static_cast<std::uint32_t>(k));
        id<MTLBuffer> buf_n = upload_u32(ctx.device, static_cast<std::uint32_t>(n));

        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx.pipeline_q4];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_packed offset:0 atIndex:1];
        [enc setBuffer:buf_scales offset:0 atIndex:2];
        [enc setBuffer:buf_c offset:0 atIndex:3];
        [enc setBuffer:buf_m offset:0 atIndex:4];
        [enc setBuffer:buf_k offset:0 atIndex:5];
        [enc setBuffer:buf_n offset:0 atIndex:6];

        // One threadgroup per output element; 32 threads is one SIMD group.
        const MTLSize tg = MTLSizeMake(32, 1, 1);
        const MTLSize grid = MTLSizeMake(n, m, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        return Mat(read_back(buf_c, m * n), m, n);
    }
}

namespace {
/// Shared body of the two GEMV entry points: stage the activation and the
/// [K, N] dims into scratch, dispatch one threadgroup per output element
/// against `pipeline`, and read the result back.
Mat gemv_dispatch(const Mat& a, id<MTLComputePipelineState> pipeline,
                  const void* weights, std::size_t weight_bytes, std::size_t n) {
    const std::size_t k = a.cols;

    @autoreleasepool {
        MetalContext& ctx = context();

        id<MTLBuffer> buf_w = cached_weights(ctx, weights, weight_bytes);
        ensure_capacity(ctx, ctx.scratch_act, ctx.scratch_act_cap, k);
        ensure_capacity(ctx, ctx.scratch_out, ctx.scratch_out_cap, n);

        std::memcpy([ctx.scratch_act contents], a.data.data(), k * 4);
        auto* dims = static_cast<std::uint32_t*>([ctx.scratch_dims contents]);
        dims[0] = static_cast<std::uint32_t>(k);
        dims[1] = static_cast<std::uint32_t>(n);

        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:ctx.scratch_act offset:0 atIndex:0];
        [enc setBuffer:buf_w offset:0 atIndex:1];
        [enc setBuffer:ctx.scratch_out offset:0 atIndex:2];
        [enc setBuffer:ctx.scratch_dims offset:0 atIndex:3];
        [enc setBuffer:ctx.scratch_dims offset:4 atIndex:4];

        const MTLSize tg = MTLSizeMake(32, 1, 1);
        const MTLSize grid = MTLSizeMake(n, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        return Mat(read_back(ctx.scratch_out, n), 1, n);
    }
}
}  // namespace

Mat metal_gemv_q4k_t(const Mat& a, const Q4KMat& q4k) {
    assert(a.rows == 1 && "metal_gemv_q4k_t: only M=1 is supported");
    assert(a.cols == q4k.cols && "metal_gemv_q4k_t: a.cols != q4k.cols");
    return gemv_dispatch(a, context().pipeline_q4k, q4k.blocks.data(), q4k.blocks.size(),
                         q4k.rows);
}

Mat metal_gemv_bf16_t(const Mat& a, const MatBf16& bf16) {
    assert(a.rows == 1 && "metal_gemv_bf16_t: only M=1 is supported");
    assert(a.cols == bf16.cols && "metal_gemv_bf16_t: a.cols != bf16.cols");
    return gemv_dispatch(a, context().pipeline_bf16, bf16.data->data(), bf16.data->size() * 2,
                         bf16.rows);
}

}  // namespace rt

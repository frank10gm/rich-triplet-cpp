#include "rt/metal_flux.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "rt/flux_msl.hpp"
#include "rt/init_rng.hpp"

namespace rt {

namespace {

/// A GEMM tile is 64x64, so both the token count and every projection width
/// are rounded up to that.
constexpr std::size_t kTile = 64;

/// The attention kernel's threadgroup accumulator.
constexpr std::size_t kMaxHeadDim = 128;

[[nodiscard]] std::size_t round_up(std::size_t v, std::size_t to) {
    return (v + to - 1) / to * to;
}

[[nodiscard]] std::uint16_t f32_to_bf16_bits(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

}  // namespace

// =============================================================================
// Impl
// =============================================================================

struct MetalFluxContext::Impl {
    FluxConfig cfg;
    std::size_t max_tokens = 0;   // text + image, padded
    std::size_t max_img_tokens = 0;
    std::size_t max_txt_tokens = 0;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> dequant_q4k = nil;
    id<MTLComputePipelineState> gemm = nil;
    id<MTLComputePipelineState> add_bias = nil;
    id<MTLComputePipelineState> attention = nil;
    id<MTLComputePipelineState> layer_norm = nil;
    id<MTLComputePipelineState> head_rms_norm = nil;
    id<MTLComputePipelineState> modulate = nil;
    id<MTLComputePipelineState> gated_add = nil;
    id<MTLComputePipelineState> rope = nil;
    id<MTLComputePipelineState> gelu_tanh = nil;
    id<MTLComputePipelineState> silu = nil;
    id<MTLComputePipelineState> add = nil;
    id<MTLComputePipelineState> zero = nil;
    id<MTLComputePipelineState> slice_cols = nil;
    id<MTLComputePipelineState> paste_cols = nil;
    id<MTLComputePipelineState> copy_rows = nil;

    /// One uploaded projection: either packed Q4_K blocks or an already-wide
    /// bfloat weight, plus its bias.
    struct Weight {
        id<MTLBuffer> data = nil;   // Q4_K bytes, or bfloat when `packed` is false
        id<MTLBuffer> bias = nil;
        bool packed = false;
        std::size_t rows = 0;       // N, padded to kTile
        std::size_t cols = 0;       // K
        std::size_t n_blocks = 0;   // Q4_K super-blocks, when packed
    };

    Weight img_in, txt_in, time_1, time_2, vector_1, vector_2, guidance_1, guidance_2;
    Weight final_mod, final_linear;

    struct DoubleBlock {
        Weight img_mod, img_qkv, img_proj, img_mlp_in, img_mlp_out;
        Weight txt_mod, txt_qkv, txt_proj, txt_mlp_in, txt_mlp_out;
        id<MTLBuffer> img_q_norm = nil;
        id<MTLBuffer> img_k_norm = nil;
        id<MTLBuffer> txt_q_norm = nil;
        id<MTLBuffer> txt_k_norm = nil;
    };
    std::vector<DoubleBlock> doubles;

    struct SingleBlock {
        Weight modulation, linear1, linear2;
        id<MTLBuffer> q_norm = nil;
        id<MTLBuffer> k_norm = nil;
    };
    std::vector<SingleBlock> singles;

    /// Shared bfloat scratch, sized for the widest weight in the model.
    id<MTLBuffer> wide = nil;
    std::size_t wide_elems = 0;

    // Activations.
    id<MTLBuffer> img = nil;
    id<MTLBuffer> txt = nil;
    id<MTLBuffer> x = nil;
    id<MTLBuffer> tmp = nil;      // [max_tokens, hidden]
    id<MTLBuffer> qb = nil;
    id<MTLBuffer> kb = nil;
    id<MTLBuffer> vb = nil;
    id<MTLBuffer> attn = nil;
    id<MTLBuffer> mlp = nil;      // [max_tokens, mlp_hidden]
    id<MTLBuffer> fused = nil;    // [max_tokens, 3*hidden + mlp_hidden]
    id<MTLBuffer> joined = nil;   // [max_tokens, hidden + mlp_hidden]
    id<MTLBuffer> vec = nil;      // [kTile, hidden]
    id<MTLBuffer> vec_tmp = nil;
    /// A double block needs the image and text modulations alive at the same
    /// time: the gates are applied after a joint attention that sits between
    /// the two projections. Hence two buffers rather than one.
    id<MTLBuffer> mod_img = nil;  // [kTile, 6*hidden]
    id<MTLBuffer> mod_txt = nil;
    /// Staging for the three vectors that enter through a projection: the
    /// timestep embedding (256), the guidance embedding (256) and the pooled
    /// CLIP vector (768). Each needs its own buffer rather than one reused
    /// three times -- the host writes them all before the command buffer is
    /// submitted, so a shared buffer would hand every reader the last value
    /// written rather than the one encoded alongside it. Sized for the widest.
    id<MTLBuffer> emb_time = nil;
    id<MTLBuffer> emb_guidance = nil;
    id<MTLBuffer> emb_pooled = nil;
    std::size_t emb_width = 0;
    id<MTLBuffer> cos_tab = nil;
    id<MTLBuffer> sin_tab = nil;

    std::size_t device_bytes = 0;

    // -------------------------------------------------------------------------
    // Dispatch helpers
    // -------------------------------------------------------------------------

    void set_u32(id<MTLComputeCommandEncoder> enc, std::uint32_t v, NSUInteger index) const {
        [enc setBytes:&v length:sizeof(v) atIndex:index];
    }
    void set_f32(id<MTLComputeCommandEncoder> enc, float v, NSUInteger index) const {
        [enc setBytes:&v length:sizeof(v) atIndex:index];
    }

    /// Run `C[M,N] = A[M,K] @ W[N,K]^T`, widening the weight first if it is
    /// still packed.
    ///
    /// Only Q4_K weights take the dequantization step. Everything else went up
    /// as bfloat and is fed to the GEMM where it lies -- widening it again
    /// would reinterpret pairs of bfloat as single floats, which produces
    /// finite, plausibly-scaled garbage rather than anything that faults.
    void gemm_into(id<MTLComputeCommandEncoder> enc, const Weight& w, id<MTLBuffer> a,
                   id<MTLBuffer> c, std::size_t m) const {
        id<MTLBuffer> operand = w.data;
        if (w.packed) {
            [enc setComputePipelineState:dequant_q4k];
            [enc setBuffer:w.data offset:0 atIndex:0];
            [enc setBuffer:wide offset:0 atIndex:1];
            set_u32(enc, static_cast<std::uint32_t>(w.n_blocks), 2);
            [enc dispatchThreads:MTLSizeMake(w.n_blocks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            operand = wide;
        }

        [enc setComputePipelineState:gemm];
        [enc setBuffer:a offset:0 atIndex:0];
        [enc setBuffer:operand offset:0 atIndex:1];
        [enc setBuffer:c offset:0 atIndex:2];
        set_u32(enc, static_cast<std::uint32_t>(w.cols), 3);
        set_u32(enc, static_cast<std::uint32_t>(w.rows), 4);
        [enc dispatchThreadgroups:MTLSizeMake(w.rows / kTile, m / kTile, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];

        if (w.bias != nil) {
            [enc setComputePipelineState:add_bias];
            [enc setBuffer:c offset:0 atIndex:0];
            [enc setBuffer:w.bias offset:0 atIndex:1];
            set_u32(enc, static_cast<std::uint32_t>(w.rows), 2);
            [enc dispatchThreads:MTLSizeMake(w.rows, m, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
    }

    void run_layer_norm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> src, id<MTLBuffer> dst,
                        std::size_t rows, std::size_t width) const {
        [enc setComputePipelineState:layer_norm];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        set_u32(enc, static_cast<std::uint32_t>(width), 2);
        set_f32(enc, cfg.layer_norm_eps, 3);
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void run_modulate(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> target,
                      id<MTLBuffer> params, std::size_t rows, std::size_t shift_off,
                      std::size_t scale_off) const {
        [enc setComputePipelineState:modulate];
        [enc setBuffer:target offset:0 atIndex:0];
        [enc setBuffer:params offset:0 atIndex:1];
        set_u32(enc, static_cast<std::uint32_t>(cfg.hidden_size), 2);
        set_u32(enc, static_cast<std::uint32_t>(shift_off), 3);
        set_u32(enc, static_cast<std::uint32_t>(scale_off), 4);
        [enc dispatchThreads:MTLSizeMake(cfg.hidden_size, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    void run_gated_add(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> dst, id<MTLBuffer> src,
                       id<MTLBuffer> params, std::size_t rows, std::size_t gate_off) const {
        [enc setComputePipelineState:gated_add];
        [enc setBuffer:dst offset:0 atIndex:0];
        [enc setBuffer:src offset:0 atIndex:1];
        [enc setBuffer:params offset:0 atIndex:2];
        set_u32(enc, static_cast<std::uint32_t>(cfg.hidden_size), 3);
        set_u32(enc, static_cast<std::uint32_t>(gate_off), 4);
        [enc dispatchThreads:MTLSizeMake(cfg.hidden_size, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    /// Per-head RMSNorm on a column slice of a fused projection, so Q and K
    /// keep their own learned scales before the streams are concatenated.
    void run_qk_norm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> target, id<MTLBuffer> gain,
                     std::size_t row_stride, std::size_t col_offset, std::size_t rows) const {
        const std::size_t head_dim = cfg.head_dim();
        [enc setComputePipelineState:head_rms_norm];
        [enc setBuffer:target offset:0 atIndex:0];
        [enc setBuffer:gain offset:0 atIndex:1];
        set_u32(enc, static_cast<std::uint32_t>(row_stride), 2);
        set_u32(enc, static_cast<std::uint32_t>(col_offset), 3);
        set_u32(enc, static_cast<std::uint32_t>(head_dim), 4);
        set_f32(enc, cfg.qk_norm_eps, 5);
        [enc dispatchThreadgroups:MTLSizeMake(cfg.n_heads, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<std::size_t>(head_dim, 128), 1, 1)];
    }

    /// Rotate an assembled [n_tok, hidden] buffer. The tables are built for the
    /// concatenated stream, so this runs after the two halves are in place.
    void run_rope(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> target,
                  std::size_t rows) const {
        const std::size_t head_dim = cfg.head_dim();
        [enc setComputePipelineState:rope];
        [enc setBuffer:target offset:0 atIndex:0];
        [enc setBuffer:cos_tab offset:0 atIndex:1];
        [enc setBuffer:sin_tab offset:0 atIndex:2];
        set_u32(enc, static_cast<std::uint32_t>(cfg.n_heads), 3);
        set_u32(enc, static_cast<std::uint32_t>(head_dim), 4);
        [enc dispatchThreads:MTLSizeMake(head_dim / 2, cfg.n_heads, rows)
            threadsPerThreadgroup:MTLSizeMake(head_dim / 2, 1, 1)];
    }

    void run_attention(id<MTLComputeCommandEncoder> enc, std::size_t rows) const {
        const std::size_t head_dim = cfg.head_dim();
        [enc setComputePipelineState:attention];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:kb offset:0 atIndex:1];
        [enc setBuffer:vb offset:0 atIndex:2];
        [enc setBuffer:attn offset:0 atIndex:3];
        set_u32(enc, static_cast<std::uint32_t>(rows), 4);
        set_u32(enc, static_cast<std::uint32_t>(cfg.n_heads), 5);
        set_u32(enc, static_cast<std::uint32_t>(head_dim), 6);
        set_f32(enc, 1.0f / std::sqrt(static_cast<float>(head_dim)), 7);
        // One threadgroup per (head, block of 32 queries). Blocking the
        // queries is what keeps K and V from being re-streamed once per query.
        constexpr std::size_t kQueryBlock = 32;
        const std::size_t blocks = (rows + kQueryBlock - 1) / kQueryBlock;
        [enc dispatchThreadgroups:MTLSizeMake(cfg.n_heads, blocks, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void run_slice(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> src, id<MTLBuffer> dst,
                   std::size_t src_width, std::size_t dst_width, std::size_t from,
                   std::size_t src_row0, std::size_t dst_row0, std::size_t rows) const {
        if (rows == 0) {
            return;
        }
        [enc setComputePipelineState:slice_cols];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        set_u32(enc, static_cast<std::uint32_t>(src_width), 2);
        set_u32(enc, static_cast<std::uint32_t>(dst_width), 3);
        set_u32(enc, static_cast<std::uint32_t>(from), 4);
        set_u32(enc, static_cast<std::uint32_t>(src_row0), 5);
        set_u32(enc, static_cast<std::uint32_t>(dst_row0), 6);
        [enc dispatchThreads:MTLSizeMake(dst_width, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    void run_paste(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> src, id<MTLBuffer> dst,
                   std::size_t src_width, std::size_t dst_width, std::size_t at,
                   std::size_t rows) const {
        [enc setComputePipelineState:paste_cols];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        set_u32(enc, static_cast<std::uint32_t>(src_width), 2);
        set_u32(enc, static_cast<std::uint32_t>(dst_width), 3);
        set_u32(enc, static_cast<std::uint32_t>(at), 4);
        [enc dispatchThreads:MTLSizeMake(src_width, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    void run_copy_rows(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> src, id<MTLBuffer> dst,
                       std::size_t width, std::size_t src_from, std::size_t dst_from,
                       std::size_t rows) const {
        if (rows == 0) {
            return;
        }
        [enc setComputePipelineState:copy_rows];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        set_u32(enc, static_cast<std::uint32_t>(width), 2);
        set_u32(enc, static_cast<std::uint32_t>(src_from), 3);
        set_u32(enc, static_cast<std::uint32_t>(dst_from), 4);
        [enc dispatchThreads:MTLSizeMake(width, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    void run_elem(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
                  id<MTLBuffer> buf, std::size_t n) const {
        [enc setComputePipelineState:pipe];
        [enc setBuffer:buf offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void run_zero(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf, std::size_t n) const {
        run_elem(enc, zero, buf, n);
    }
};

// =============================================================================
// Construction
// =============================================================================

namespace {

[[nodiscard]] Result<id<MTLComputePipelineState>> make_pipeline(id<MTLDevice> device,
                                                                id<MTLLibrary> lib,
                                                                const char* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
    if (fn == nil) {
        return err(std::string("metal flux: shader has no kernel '") + name + "'");
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn
                                                                            error:&error];
    if (pipe == nil) {
        return err(std::string("metal flux: cannot build pipeline '") + name + "': " +
                   (error != nil ? error.localizedDescription.UTF8String : "unknown"));
    }
    return pipe;
}

/// Upload one projection, consuming it.
///
/// Q4_K weights go up packed and are widened per use. Everything else is
/// converted to bfloat once here, which is what the small projections want --
/// widening 256x3072 on every step to save 1.5 MB would be a poor trade.
[[nodiscard]] Result<MetalFluxContext::Impl::Weight> upload(id<MTLDevice> device, QLinear& src,
                                                            std::size_t& device_bytes,
                                                            std::size_t& wide_elems) {
    MetalFluxContext::Impl::Weight w;
    w.rows = src.out_features;
    w.cols = src.in_features;
    if (w.rows == 0 || w.cols == 0) {
        return w;  // an absent optional layer, such as schnell's guidance path
    }
    if (w.rows % kTile != 0) {
        return err("metal flux: projection width " + std::to_string(w.rows) +
                   " is not a multiple of " + std::to_string(kTile));
    }
    if (w.cols % 8 != 0) {
        return err("metal flux: contracted dimension " + std::to_string(w.cols) +
                   " is not a multiple of 8");
    }
    wide_elems = std::max(wide_elems, w.rows * w.cols);

    if (src.q4k.has_value()) {
        const std::vector<std::uint8_t>& blocks = src.q4k->blocks;
        w.packed = true;
        w.n_blocks = blocks.size() / 144;
        w.data = [device newBufferWithBytes:blocks.data()
                                     length:blocks.size()
                                    options:MTLResourceStorageModeShared];
        device_bytes += blocks.size();
    } else {
        std::vector<std::uint16_t> bits(w.rows * w.cols);
        if (src.bf16.has_value()) {
            bits = *src.bf16->data;
        } else {
            for (std::size_t i = 0; i < bits.size(); ++i) {
                bits[i] = f32_to_bf16_bits(src.f32.data[i]);
            }
        }
        w.data = [device newBufferWithBytes:bits.data()
                                     length:bits.size() * sizeof(std::uint16_t)
                                    options:MTLResourceStorageModeShared];
        device_bytes += bits.size() * sizeof(std::uint16_t);
    }

    if (!src.bias.empty()) {
        w.bias = [device newBufferWithBytes:src.bias.data()
                                     length:src.bias.size() * sizeof(float)
                                    options:MTLResourceStorageModeShared];
        device_bytes += src.bias.size() * sizeof(float);
    }

    // Freed as it goes up, so peak memory never holds both copies.
    src.free_weight();
    src.bias.clear();
    src.bias.shrink_to_fit();
    return w;
}

[[nodiscard]] id<MTLBuffer> upload_vec(id<MTLDevice> device, const std::vector<float>& v,
                                       std::size_t& device_bytes) {
    if (v.empty()) {
        return nil;
    }
    device_bytes += v.size() * sizeof(float);
    return [device newBufferWithBytes:v.data()
                               length:v.size() * sizeof(float)
                              options:MTLResourceStorageModeShared];
}

[[nodiscard]] id<MTLBuffer> alloc(id<MTLDevice> device, std::size_t elems, std::size_t elem_size,
                                  std::size_t& device_bytes) {
    device_bytes += elems * elem_size;
    return [device newBufferWithLength:elems * elem_size options:MTLResourceStorageModeShared];
}

}  // namespace

MetalFluxContext::MetalFluxContext() : impl_(std::make_unique<Impl>()) {}
MetalFluxContext::~MetalFluxContext() = default;

Result<std::unique_ptr<MetalFluxContext>> MetalFluxContext::create(FluxModel& model,
                                                                   std::size_t max_lat_h,
                                                                   std::size_t max_lat_w,
                                                                   std::size_t max_text_tokens) {
    const FluxConfig& cfg = model.cfg;
    if (cfg.head_dim() > kMaxHeadDim) {
        return err("metal flux: head_dim " + std::to_string(cfg.head_dim()) +
                   " exceeds the attention kernel's " + std::to_string(kMaxHeadDim));
    }
    // The attention kernel tiles both of its matmuls into 8x8 simdgroup
    // fragments along the head dimension.
    if (cfg.head_dim() % 8 != 0) {
        return err("metal flux: head_dim " + std::to_string(cfg.head_dim()) +
                   " is not a multiple of 8");
    }
    if (cfg.hidden_size % kTile != 0) {
        return err("metal flux: hidden_size must be a multiple of " + std::to_string(kTile));
    }
    if (max_lat_h % cfg.patch_size != 0 || max_lat_w % cfg.patch_size != 0) {
        return err("metal flux: latent bounds must be multiples of the patch size");
    }
    RT_TRY_VOID(flux_check_axes(cfg));

    auto ctx = std::unique_ptr<MetalFluxContext>(new MetalFluxContext());
    Impl& impl = *ctx->impl_;
    impl.cfg = cfg;

    impl.device = MTLCreateSystemDefaultDevice();
    if (impl.device == nil) {
        return err("metal flux: no Metal device");
    }
    impl.queue = [impl.device newCommandQueue];

    NSError* error = nil;
    id<MTLLibrary> lib = [impl.device newLibraryWithSource:@(kFluxMsl)
                                                   options:nil
                                                     error:&error];
    if (lib == nil) {
        return err(std::string("metal flux: shader compilation failed: ") +
                   (error != nil ? error.localizedDescription.UTF8String : "unknown"));
    }

    struct Bind {
        __strong id<MTLComputePipelineState>* slot;
        const char* name;
    };
    const Bind binds[] = {
        {&impl.dequant_q4k, "flux_dequant_q4k"}, {&impl.gemm, "flux_gemm_bt"},
        {&impl.add_bias, "flux_add_bias"},
        {&impl.attention, "flux_attention"},     {&impl.layer_norm, "flux_layer_norm"},
        {&impl.head_rms_norm, "flux_head_rms_norm"},
        {&impl.modulate, "flux_modulate"},       {&impl.gated_add, "flux_gated_add"},
        {&impl.rope, "flux_rope"},               {&impl.gelu_tanh, "flux_gelu_tanh"},
        {&impl.silu, "flux_silu"},               {&impl.add, "flux_add"},
        {&impl.zero, "flux_zero"},               {&impl.slice_cols, "flux_slice_cols"},
        {&impl.paste_cols, "flux_paste_cols"},
        {&impl.copy_rows, "flux_copy_rows"},
    };
    for (const Bind& b : binds) {
        RT_TRY(pipe, make_pipeline(impl.device, lib, b.name));
        *b.slot = pipe;
    }

    // --- Weights -------------------------------------------------------------
    std::size_t bytes = 0;
    std::size_t wide_elems = 0;

    struct Upload {
        QLinear* src;
        Impl::Weight* dst;
    };
    const Upload top[] = {
        {&model.img_in, &impl.img_in},         {&model.txt_in, &impl.txt_in},
        {&model.time_in_1, &impl.time_1},      {&model.time_in_2, &impl.time_2},
        {&model.vector_in_1, &impl.vector_1},  {&model.vector_in_2, &impl.vector_2},
        {&model.guidance_in_1, &impl.guidance_1},
        {&model.guidance_in_2, &impl.guidance_2},
        {&model.final_mod, &impl.final_mod},   {&model.final_linear, &impl.final_linear},
    };
    for (const Upload& u : top) {
        RT_TRY(w, upload(impl.device, *u.src, bytes, wide_elems));
        *u.dst = w;
    }

    impl.doubles.resize(model.double_blocks.size());
    for (std::size_t i = 0; i < model.double_blocks.size(); ++i) {
        FluxDoubleBlock& b = model.double_blocks[i];
        Impl::DoubleBlock& d = impl.doubles[i];
        const Upload parts[] = {
            {&b.img_mod, &d.img_mod},         {&b.img_qkv, &d.img_qkv},
            {&b.img_proj, &d.img_proj},       {&b.img_mlp_in, &d.img_mlp_in},
            {&b.img_mlp_out, &d.img_mlp_out}, {&b.txt_mod, &d.txt_mod},
            {&b.txt_qkv, &d.txt_qkv},         {&b.txt_proj, &d.txt_proj},
            {&b.txt_mlp_in, &d.txt_mlp_in},   {&b.txt_mlp_out, &d.txt_mlp_out},
        };
        for (const Upload& u : parts) {
            RT_TRY(w, upload(impl.device, *u.src, bytes, wide_elems));
            *u.dst = w;
        }
        d.img_q_norm = upload_vec(impl.device, b.img_norm.query_scale, bytes);
        d.img_k_norm = upload_vec(impl.device, b.img_norm.key_scale, bytes);
        d.txt_q_norm = upload_vec(impl.device, b.txt_norm.query_scale, bytes);
        d.txt_k_norm = upload_vec(impl.device, b.txt_norm.key_scale, bytes);
    }

    impl.singles.resize(model.single_blocks.size());
    for (std::size_t i = 0; i < model.single_blocks.size(); ++i) {
        FluxSingleBlock& b = model.single_blocks[i];
        Impl::SingleBlock& s = impl.singles[i];
        const Upload parts[] = {
            {&b.modulation, &s.modulation}, {&b.linear1, &s.linear1}, {&b.linear2, &s.linear2},
        };
        for (const Upload& u : parts) {
            RT_TRY(w, upload(impl.device, *u.src, bytes, wide_elems));
            *u.dst = w;
        }
        s.q_norm = upload_vec(impl.device, b.norm.query_scale, bytes);
        s.k_norm = upload_vec(impl.device, b.norm.key_scale, bytes);
    }

    // --- Scratch -------------------------------------------------------------
    const std::size_t hidden = cfg.hidden_size;
    const std::size_t mlp_hidden = cfg.mlp_hidden();
    const std::size_t img_tokens =
        (max_lat_h / cfg.patch_size) * (max_lat_w / cfg.patch_size);
    impl.max_img_tokens = img_tokens;
    impl.max_txt_tokens = max_text_tokens;
    impl.max_tokens = round_up(img_tokens + max_text_tokens, kTile);

    impl.wide_elems = wide_elems;
    impl.wide = alloc(impl.device, wide_elems, sizeof(std::uint16_t), bytes);

    const std::size_t rows = impl.max_tokens;
    impl.img = alloc(impl.device, rows * hidden, sizeof(float), bytes);
    impl.txt = alloc(impl.device, rows * hidden, sizeof(float), bytes);
    impl.x = alloc(impl.device, rows * hidden, sizeof(float), bytes);
    impl.tmp = alloc(impl.device, rows * hidden, sizeof(float), bytes);
    impl.qb = alloc(impl.device, rows * hidden, sizeof(float), bytes);
    impl.kb = alloc(impl.device, rows * hidden, sizeof(float), bytes);
    impl.vb = alloc(impl.device, rows * hidden, sizeof(float), bytes);
    impl.attn = alloc(impl.device, rows * hidden, sizeof(float), bytes);
    impl.mlp = alloc(impl.device, rows * mlp_hidden, sizeof(float), bytes);
    impl.fused = alloc(impl.device, rows * (3 * hidden + mlp_hidden), sizeof(float), bytes);
    impl.joined = alloc(impl.device, rows * (hidden + mlp_hidden), sizeof(float), bytes);
    impl.vec = alloc(impl.device, kTile * hidden, sizeof(float), bytes);
    impl.vec_tmp = alloc(impl.device, kTile * hidden, sizeof(float), bytes);
    impl.mod_img = alloc(impl.device, kTile * 6 * hidden, sizeof(float), bytes);
    impl.mod_txt = alloc(impl.device, kTile * 6 * hidden, sizeof(float), bytes);
    impl.emb_width = std::max<std::size_t>(256, cfg.pooled_dim);
    impl.emb_time = alloc(impl.device, kTile * impl.emb_width, sizeof(float), bytes);
    impl.emb_guidance = alloc(impl.device, kTile * impl.emb_width, sizeof(float), bytes);
    impl.emb_pooled = alloc(impl.device, kTile * impl.emb_width, sizeof(float), bytes);
    impl.cos_tab = alloc(impl.device, rows * (cfg.head_dim() / 2), sizeof(float), bytes);
    impl.sin_tab = alloc(impl.device, rows * (cfg.head_dim() / 2), sizeof(float), bytes);

    impl.device_bytes = bytes;
    return ctx;
}

std::size_t MetalFluxContext::device_bytes() const { return impl_->device_bytes; }
std::size_t MetalFluxContext::max_latent_tokens() const { return impl_->max_img_tokens; }

// =============================================================================
// Forward
// =============================================================================

Result<Mat> MetalFluxContext::forward(const Mat& latent, std::size_t lat_h, std::size_t lat_w,
                                      const Mat& context, std::span<const float> pooled,
                                      float timestep, float guidance) const {
    const Impl& s = *impl_;
    const FluxConfig& cfg = s.cfg;
    const std::size_t hidden = cfg.hidden_size;
    const std::size_t mlp_hidden = cfg.mlp_hidden();
    const std::size_t head_dim = cfg.head_dim();
    const std::size_t patch = cfg.patch_size;

    if (latent.cols != cfg.in_channels || latent.rows != lat_h * lat_w) {
        return err("metal flux: latent shape mismatch");
    }
    if (lat_h % patch != 0 || lat_w % patch != 0) {
        return err("metal flux: latent dimensions must be multiples of the patch size");
    }
    if (context.cols != cfg.context_dim) {
        return err("metal flux: context width mismatch");
    }
    if (pooled.size() != cfg.pooled_dim) {
        return err("metal flux: pooled width mismatch");
    }

    const std::size_t n_img = (lat_h / patch) * (lat_w / patch);
    const std::size_t n_txt = context.rows;
    const std::size_t n_tok = n_img + n_txt;
    if (n_img > s.max_img_tokens || n_txt > s.max_txt_tokens) {
        return err("metal flux: sequence exceeds the allocated scratch");
    }
    const std::size_t rows = round_up(n_tok, kTile);
    const std::size_t img_rows = round_up(n_img, kTile);
    const std::size_t txt_rows = round_up(n_txt, kTile);

    // --- Host-side inputs ----------------------------------------------------
    // The RoPE tables depend only on the image size, and the id layout is
    // simple enough that building them here costs less than a kernel would.
    {
        auto* cosp = static_cast<float*>(s.cos_tab.contents);
        auto* sinp = static_cast<float*>(s.sin_tab.contents);
        const std::size_t pairs = head_dim / 2;
        const std::size_t gw = lat_w / patch;
        for (std::size_t t = 0; t < rows; ++t) {
            // Text tokens sit at the origin on every axis, so their rotation is
            // the identity. Pad rows follow them and never reach attention.
            float pos_axis[3] = {0.0f, 0.0f, 0.0f};
            if (t >= n_txt && t < n_tok) {
                const std::size_t i = t - n_txt;
                pos_axis[1] = static_cast<float>(i / gw);
                pos_axis[2] = static_cast<float>(i % gw);
            }
            std::size_t p = 0;
            for (std::size_t a = 0; a < cfg.axes_dim.size(); ++a) {
                const std::size_t dim = cfg.axes_dim[a];
                for (std::size_t i = 0; i < dim / 2; ++i, ++p) {
                    const float exponent = static_cast<float>(2 * i) / static_cast<float>(dim);
                    const float angle = pos_axis[a] / std::pow(cfg.rope_theta, exponent);
                    cosp[t * pairs + p] = std::cos(angle);
                    sinp[t * pairs + p] = std::sin(angle);
                }
            }
        }
    }

    // The patchified latent and the T5 sequence go into `tmp` and `attn` as
    // staging, since both are wide enough to hold them and are dead here.
    const Mat patches = flux_patchify(latent, lat_h, lat_w, patch);
    {
        auto* dst = static_cast<float*>(s.tmp.contents);
        std::fill(dst, dst + img_rows * cfg.patch_dim(), 0.0f);
        std::copy(patches.data.begin(), patches.data.end(), dst);
    }
    {
        // The T5 sequence is 4096 wide, which is wider than the hidden size --
        // `mlp` is the only activation buffer that can hold it.
        auto* dst = static_cast<float*>(s.mlp.contents);
        std::fill(dst, dst + txt_rows * cfg.context_dim, 0.0f);
        std::copy(context.data.begin(), context.data.end(), dst);
    }
    {
        const std::vector<float> t_emb = flux_timestep_embedding(timestep, 256);
        auto* dst = static_cast<float*>(s.emb_time.contents);
        std::fill(dst, dst + kTile * s.emb_width, 0.0f);
        std::copy(t_emb.begin(), t_emb.end(), dst);
    }
    {
        auto* dst = static_cast<float*>(s.emb_pooled.contents);
        std::fill(dst, dst + kTile * s.emb_width, 0.0f);
        std::copy(pooled.begin(), pooled.end(), dst);
    }
    if (cfg.guidance_embed) {
        const std::vector<float> g_emb = flux_timestep_embedding(guidance, 256);
        auto* dst = static_cast<float*>(s.emb_guidance.contents);
        std::fill(dst, dst + kTile * s.emb_width, 0.0f);
        std::copy(g_emb.begin(), g_emb.end(), dst);
    }

    id<MTLCommandBuffer> cmd = [s.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // Pad rows are never read downstream, but zeroing them keeps a stray NaN
    // in fresh memory from turning into a NaN the debugger has to chase.
    s.run_zero(enc, s.img, rows * hidden);
    s.run_zero(enc, s.txt, rows * hidden);
    s.run_zero(enc, s.x, rows * hidden);

    // --- Conditioning vector -------------------------------------------------
    s.gemm_into(enc, s.time_1, s.emb_time, s.vec_tmp, kTile);
    s.run_elem(enc, s.silu, s.vec_tmp, kTile * hidden);
    s.gemm_into(enc, s.time_2, s.vec_tmp, s.vec, kTile);

    if (cfg.guidance_embed) {
        s.gemm_into(enc, s.guidance_1, s.emb_guidance, s.vec_tmp, kTile);
        s.run_elem(enc, s.silu, s.vec_tmp, kTile * hidden);
        s.gemm_into(enc, s.guidance_2, s.vec_tmp, s.mod_img, kTile);
        [enc setComputePipelineState:s.add];
        [enc setBuffer:s.vec offset:0 atIndex:0];
        [enc setBuffer:s.mod_img offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(hidden, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    s.gemm_into(enc, s.vector_1, s.emb_pooled, s.vec_tmp, kTile);
    s.run_elem(enc, s.silu, s.vec_tmp, kTile * hidden);
    s.gemm_into(enc, s.vector_2, s.vec_tmp, s.mod_txt, kTile);
    [enc setComputePipelineState:s.add];
    [enc setBuffer:s.vec offset:0 atIndex:0];
    [enc setBuffer:s.mod_txt offset:0 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(hidden, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

    // Every modulation projection reads `silu(vec)`, not `vec`.
    s.run_copy_rows(enc, s.vec, s.vec_tmp, hidden, 0, 0, kTile);
    s.run_elem(enc, s.silu, s.vec_tmp, kTile * hidden);

    // --- Streams -------------------------------------------------------------
    s.gemm_into(enc, s.img_in, s.tmp, s.img, img_rows);
    s.gemm_into(enc, s.txt_in, s.mlp, s.txt, txt_rows);

    const std::size_t qkv_width = 3 * hidden;

    // --- Double-stream blocks ------------------------------------------------
    for (const Impl::DoubleBlock& b : s.doubles) {
        s.gemm_into(enc, b.img_mod, s.vec_tmp, s.mod_img, kTile);
        s.gemm_into(enc, b.txt_mod, s.vec_tmp, s.mod_txt, kTile);

        // Image: norm, modulate, project, per-head norm on Q and K in place.
        s.run_layer_norm(enc, s.img, s.tmp, img_rows, hidden);
        s.run_modulate(enc, s.tmp, s.mod_img, img_rows, 0, hidden);
        s.gemm_into(enc, b.img_qkv, s.tmp, s.fused, img_rows);
        s.run_qk_norm(enc, s.fused, b.img_q_norm, qkv_width, 0, img_rows);
        s.run_qk_norm(enc, s.fused, b.img_k_norm, qkv_width, hidden, img_rows);

        // Text: the same, into a separate staging buffer.
        s.run_layer_norm(enc, s.txt, s.tmp, txt_rows, hidden);
        s.run_modulate(enc, s.tmp, s.mod_txt, txt_rows, 0, hidden);
        s.gemm_into(enc, b.txt_qkv, s.tmp, s.joined, txt_rows);
        s.run_qk_norm(enc, s.joined, b.txt_q_norm, qkv_width, 0, txt_rows);
        s.run_qk_norm(enc, s.joined, b.txt_k_norm, qkv_width, hidden, txt_rows);

        // Assemble [txt | img] -- the order the position ids were built in.
        s.run_slice(enc, s.joined, s.qb, qkv_width, hidden, 0, 0, 0, n_txt);
        s.run_slice(enc, s.joined, s.kb, qkv_width, hidden, hidden, 0, 0, n_txt);
        s.run_slice(enc, s.joined, s.vb, qkv_width, hidden, 2 * hidden, 0, 0, n_txt);
        s.run_slice(enc, s.fused, s.qb, qkv_width, hidden, 0, 0, n_txt, n_img);
        s.run_slice(enc, s.fused, s.kb, qkv_width, hidden, hidden, 0, n_txt, n_img);
        s.run_slice(enc, s.fused, s.vb, qkv_width, hidden, 2 * hidden, 0, n_txt, n_img);

        s.run_rope(enc, s.qb, n_tok);
        s.run_rope(enc, s.kb, n_tok);
        s.run_attention(enc, n_tok);

        // Image half of the attention, projected and gated back in.
        s.run_copy_rows(enc, s.attn, s.tmp, hidden, n_txt, 0, n_img);
        s.gemm_into(enc, b.img_proj, s.tmp, s.fused, img_rows);
        s.run_gated_add(enc, s.img, s.fused, s.mod_img, n_img, 2 * hidden);

        s.run_layer_norm(enc, s.img, s.tmp, img_rows, hidden);
        s.run_modulate(enc, s.tmp, s.mod_img, img_rows, 3 * hidden, 4 * hidden);
        s.gemm_into(enc, b.img_mlp_in, s.tmp, s.mlp, img_rows);
        s.run_elem(enc, s.gelu_tanh, s.mlp, img_rows * mlp_hidden);
        s.gemm_into(enc, b.img_mlp_out, s.mlp, s.fused, img_rows);
        s.run_gated_add(enc, s.img, s.fused, s.mod_img, n_img, 5 * hidden);

        // Text half, the same shape of work against its own parameters.
        s.run_copy_rows(enc, s.attn, s.tmp, hidden, 0, 0, n_txt);
        s.gemm_into(enc, b.txt_proj, s.tmp, s.joined, txt_rows);
        s.run_gated_add(enc, s.txt, s.joined, s.mod_txt, n_txt, 2 * hidden);

        s.run_layer_norm(enc, s.txt, s.tmp, txt_rows, hidden);
        s.run_modulate(enc, s.tmp, s.mod_txt, txt_rows, 3 * hidden, 4 * hidden);
        s.gemm_into(enc, b.txt_mlp_in, s.tmp, s.mlp, txt_rows);
        s.run_elem(enc, s.gelu_tanh, s.mlp, txt_rows * mlp_hidden);
        s.gemm_into(enc, b.txt_mlp_out, s.mlp, s.joined, txt_rows);
        s.run_gated_add(enc, s.txt, s.joined, s.mod_txt, n_txt, 5 * hidden);
    }

    // --- Single-stream blocks ------------------------------------------------
    s.run_copy_rows(enc, s.txt, s.x, hidden, 0, 0, n_txt);
    s.run_copy_rows(enc, s.img, s.x, hidden, 0, n_txt, n_img);

    const std::size_t fused_width = 3 * hidden + mlp_hidden;
    for (const Impl::SingleBlock& b : s.singles) {
        s.gemm_into(enc, b.modulation, s.vec_tmp, s.mod_img, kTile);

        s.run_layer_norm(enc, s.x, s.tmp, rows, hidden);
        s.run_modulate(enc, s.tmp, s.mod_img, rows, 0, hidden);

        // One projection produces QKV and the MLP's up-projection together.
        s.gemm_into(enc, b.linear1, s.tmp, s.fused, rows);
        s.run_qk_norm(enc, s.fused, b.q_norm, fused_width, 0, rows);
        s.run_qk_norm(enc, s.fused, b.k_norm, fused_width, hidden, rows);
        s.run_slice(enc, s.fused, s.qb, fused_width, hidden, 0, 0, 0, n_tok);
        s.run_slice(enc, s.fused, s.kb, fused_width, hidden, hidden, 0, 0, n_tok);
        s.run_slice(enc, s.fused, s.vb, fused_width, hidden, 2 * hidden, 0, 0, n_tok);
        s.run_slice(enc, s.fused, s.mlp, fused_width, mlp_hidden, 3 * hidden, 0, 0, rows);

        s.run_rope(enc, s.qb, n_tok);
        s.run_rope(enc, s.kb, n_tok);
        s.run_attention(enc, n_tok);
        s.run_elem(enc, s.gelu_tanh, s.mlp, rows * mlp_hidden);

        // Attention and MLP are joined before a single output projection,
        // rather than run in sequence.
        s.run_paste(enc, s.attn, s.joined, hidden, hidden + mlp_hidden, 0, rows);
        s.run_paste(enc, s.mlp, s.joined, mlp_hidden, hidden + mlp_hidden, hidden, rows);
        s.gemm_into(enc, b.linear2, s.joined, s.tmp, rows);
        s.run_gated_add(enc, s.x, s.tmp, s.mod_img, n_tok, 2 * hidden);
    }

    // --- Final layer ---------------------------------------------------------
    s.gemm_into(enc, s.final_mod, s.vec_tmp, s.mod_img, kTile);
    s.run_copy_rows(enc, s.x, s.tmp, hidden, n_txt, 0, n_img);
    s.run_layer_norm(enc, s.tmp, s.attn, img_rows, hidden);
    // Two values here, not three, and in the order shift then scale.
    s.run_modulate(enc, s.attn, s.mod_img, img_rows, 0, hidden);
    s.gemm_into(enc, s.final_linear, s.attn, s.fused, img_rows);

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.error != nil) {
        return err(std::string("metal flux: command buffer failed: ") +
                   cmd.error.localizedDescription.UTF8String);
    }

    const std::size_t patch_dim = cfg.patch_dim();
    Mat tokens = Mat::zeros(n_img, patch_dim);
    const auto* src = static_cast<const float*>(s.fused.contents);
    std::copy(src, src + n_img * patch_dim, tokens.data.begin());
    return flux_unpatchify(tokens, lat_h, lat_w, cfg.in_channels, patch);
}

// =============================================================================
// Sampling
// =============================================================================

Result<Mat> flux_sample_metal(const MetalFluxContext& ctx, const FluxConfig& cfg,
                              const Mat& context, std::span<const float> pooled,
                              const FluxSampleParams& params) {
    constexpr std::size_t kVaeFactor = 8;
    if (params.width % (kVaeFactor * cfg.patch_size) != 0 ||
        params.height % (kVaeFactor * cfg.patch_size) != 0) {
        return err("flux sample: width and height must be multiples of " +
                   std::to_string(kVaeFactor * cfg.patch_size));
    }
    if (params.steps == 0) {
        return err("flux sample: steps must be > 0");
    }

    const std::size_t lat_h = params.height / kVaeFactor;
    const std::size_t lat_w = params.width / kVaeFactor;

    InitRng rng(params.seed);
    Mat x = Mat::zeros(lat_h * lat_w, cfg.in_channels);
    for (float& v : x.data) {
        v = rng.next_normal();
    }

    const std::vector<float> sigmas = flux_schedule(params.steps, params.shift);
    for (std::size_t i = 0; i < params.steps; ++i) {
        RT_TRY(velocity,
               ctx.forward(x, lat_h, lat_w, context, pooled, sigmas[i], params.guidance));
        const float dt = sigmas[i + 1] - sigmas[i];
        for (std::size_t k = 0; k < x.data.size(); ++k) {
            x.data[k] += dt * velocity.data[k];
        }
    }
    return x;
}

}  // namespace rt

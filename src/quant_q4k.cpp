#include <algorithm>
#include <cmath>
#include <limits>

#include "rt/parallel.hpp"
#include "rt/quant.hpp"

#if RT_FEATURE_BLAS
#include <Accelerate/Accelerate.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rt {

namespace {
/// One Q4_K super-block covers 256 elements and occupies 144 bytes.
constexpr std::size_t Q4K_SUPERBLOCK = 256;
constexpr std::size_t Q4K_BLOCK_BYTES = 144;

/// Rust's `f32 as u8`: a *saturating* cast -- NaN maps to 0, out-of-range
/// values clamp to [0, 255], and the fractional part is truncated toward zero.
/// C++'s plain cast is UB outside the destination range, so the reference
/// semantics have to be spelled out.
[[nodiscard]] inline std::uint8_t sat_u8(float v) {
    if (!(v > 0.0f)) {  // false for NaN and for v <= 0
        return 0;
    }
    if (v >= 255.0f) {
        return 255;
    }
    return static_cast<std::uint8_t>(v);
}

[[nodiscard]] inline std::uint16_t read_u16_le(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) | static_cast<std::uint16_t>(p[1] << 8);
}

/// Pack 6-bit scale and min values into the 12-byte scales array.
/// `sv[0..8]` are 6-bit scale values, `mv[0..8]` are 6-bit min values.
void pack_scales(const std::uint8_t (&sv)[8], const std::uint8_t (&mv)[8],
                 std::uint8_t (&sc)[12]) {
    // Indices 0..3 hold the low 6 bits of sv/mv[0..3]; bits 4-5 of sv/mv[4..7]
    // ride along in the top two bits.
    for (std::size_t j = 0; j < 4; ++j) {
        sc[j] = static_cast<std::uint8_t>((sv[j] & 0x3f) | ((sv[j + 4] & 0x30) << 2));
        sc[j + 4] = static_cast<std::uint8_t>((mv[j] & 0x3f) | ((mv[j + 4] & 0x30) << 2));
    }
    // Indices 8..11: low nibble of sv[4..7], high nibble of mv[4..7].
    for (std::size_t j = 0; j < 4; ++j) {
        sc[8 + j] = static_cast<std::uint8_t>((sv[j + 4] & 0x0f) | ((mv[j + 4] & 0x0f) << 4));
    }
}
}  // namespace

void Q4KMat::scale_min(const std::uint8_t* sc, std::size_t j, float& scale, float& min) {
    std::uint8_t sv, mv;
    if (j < 4) {
        sv = sc[j] & 0x3f;
        mv = sc[j + 4] & 0x3f;
    } else {
        sv = static_cast<std::uint8_t>((sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4));
        mv = static_cast<std::uint8_t>((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    }
    scale = static_cast<float>(sv);
    min = static_cast<float>(mv);
}

void Q4KMat::quantize_block(std::span<const float> block_data, std::span<std::uint8_t> out) {
    // Step 1: per-sub-block scale and min (8 sub-blocks of 32 elements).
    float sub_scales[8] = {};
    float sub_mins[8] = {};
    for (std::size_t sb = 0; sb < 8; ++sb) {
        const float* sub = block_data.data() + sb * 32;
        float smin = std::numeric_limits<float>::infinity();
        float smax = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < 32; ++i) {
            smin = std::min(smin, sub[i]);
            smax = std::max(smax, sub[i]);
        }
        if (smin >= 0.0f) {
            sub_mins[sb] = 0.0f;
            sub_scales[sb] = smax > 0.0f ? smax / 15.0f : 0.0f;
        } else {
            sub_mins[sb] = smin;
            sub_scales[sb] = smax > smin ? (smax - smin) / 15.0f : 0.0f;
        }
    }

    // Step 2: super-block d and dmin.
    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (std::size_t j = 0; j < 8; ++j) {
        max_scale = std::max(max_scale, sub_scales[j]);
        max_min = std::max(max_min, -sub_mins[j]);
    }
    const float d = max_scale > 0.0f ? max_scale / 63.0f : 0.0f;
    const float dmin = max_min > 0.0f ? max_min / 63.0f : 0.0f;

    // Step 3: quantize the sub-block scales and mins to 6 bits.
    std::uint8_t sv[8] = {};
    std::uint8_t mv[8] = {};
    if (d > 0.0f) {
        const float inv_d = 1.0f / d;
        for (std::size_t j = 0; j < 8; ++j) {
            sv[j] = std::min<std::uint8_t>(63, sat_u8(sub_scales[j] * inv_d + 0.5f));
        }
    }
    if (dmin > 0.0f) {
        const float inv_dmin = 1.0f / dmin;
        for (std::size_t j = 0; j < 8; ++j) {
            mv[j] = std::min<std::uint8_t>(63, sat_u8(-sub_mins[j] * inv_dmin + 0.5f));
        }
    }

    // Step 4: store d and dmin as f16.
    const std::uint16_t d_f16 = f32_to_f16(d);
    const std::uint16_t dmin_f16 = f32_to_f16(dmin);
    out[0] = static_cast<std::uint8_t>(d_f16 & 0xff);
    out[1] = static_cast<std::uint8_t>(d_f16 >> 8);
    out[2] = static_cast<std::uint8_t>(dmin_f16 & 0xff);
    out[3] = static_cast<std::uint8_t>(dmin_f16 >> 8);

    // Step 5: pack the 6-bit scales/mins into the 12-byte array.
    std::uint8_t sc[12] = {};
    pack_scales(sv, mv, sc);
    for (std::size_t i = 0; i < 12; ++i) {
        out[4 + i] = sc[i];
    }

    // Step 6: quantize values into 4-bit nibbles, using the f16-rounded d/dmin
    // so encode and decode see the same scales.
    const float d_val = f16_to_f32(d_f16);
    const float dmin_val = f16_to_f32(dmin_f16);
    for (std::size_t chunk = 0; chunk < 4; ++chunk) {
        const float scale1 = d_val * static_cast<float>(sv[chunk * 2]);
        const float min1 = dmin_val * static_cast<float>(mv[chunk * 2]);
        const float scale2 = d_val * static_cast<float>(sv[chunk * 2 + 1]);
        const float min2 = dmin_val * static_cast<float>(mv[chunk * 2 + 1]);

        const float* lo_data = block_data.data() + chunk * 64;
        const float* hi_data = block_data.data() + chunk * 64 + 32;

        for (std::size_t l = 0; l < 32; ++l) {
            const std::uint8_t q_lo =
                scale1 > 0.0f
                    ? std::min<std::uint8_t>(15, sat_u8((lo_data[l] + min1) / scale1 + 0.5f))
                    : 0;
            const std::uint8_t q_hi =
                scale2 > 0.0f
                    ? std::min<std::uint8_t>(15, sat_u8((hi_data[l] + min2) / scale2 + 0.5f))
                    : 0;
            out[16 + chunk * 32 + l] = static_cast<std::uint8_t>(q_lo | (q_hi << 4));
        }
    }
}

namespace {
/// Shared driver for the three Q4_K constructors. `fill_row(row, buf)` writes
/// `cols` f32 values for the given source row; the caller decides where they
/// come from (f32 Mat, BF16 Mat, or a Q4_0 matrix).
template <typename FillRow>
Q4KMat build_q4k(std::size_t rows, std::size_t cols, FillRow&& fill_row) {
    const std::size_t n_blocks_per_row = cols / Q4K_SUPERBLOCK;
    Q4KMat q;
    q.rows = rows;
    q.cols = cols;
    q.blocks.assign(rows * n_blocks_per_row * Q4K_BLOCK_BYTES, 0);

    const std::size_t n_threads = hardware_threads();
    const std::size_t rows_per_thread = div_ceil(rows, n_threads);

    scoped_spawn(n_threads, [&](std::size_t tid) {
        const std::size_t row_start = tid * rows_per_thread;
        if (row_start >= rows) {
            return;
        }
        const std::size_t row_end = std::min(row_start + rows_per_thread, rows);
        // Each thread owns a disjoint band of rows, hence of blocks.
        std::vector<float> row_buf(cols);
        for (std::size_t row = row_start; row < row_end; ++row) {
            fill_row(row, std::span<float>(row_buf));
            for (std::size_t b = 0; b < n_blocks_per_row; ++b) {
                const std::size_t boff = (row * n_blocks_per_row + b) * Q4K_BLOCK_BYTES;
                Q4KMat::quantize_block(
                    std::span<const float>(row_buf).subspan(b * Q4K_SUPERBLOCK,
                                                            Q4K_SUPERBLOCK),
                    std::span<std::uint8_t>(q.blocks).subspan(boff, Q4K_BLOCK_BYTES));
            }
        }
    });

    return q;
}
}  // namespace

Q4KMat Q4KMat::quantize(const Mat& mat) {
    assert(mat.cols % Q4K_SUPERBLOCK == 0 &&
           "Q4KMat::quantize: cols must be a multiple of 256");
    return build_q4k(mat.rows, mat.cols, [&](std::size_t row, std::span<float> buf) {
        std::copy_n(mat.data.data() + row * mat.cols, mat.cols, buf.data());
    });
}

Q4KMat Q4KMat::quantize_from_bf16(const MatBf16& bf16) {
    assert(bf16.cols % Q4K_SUPERBLOCK == 0 &&
           "Q4KMat::quantize_from_bf16: cols must be a multiple of 256");
    const std::uint16_t* src_all = bf16.data->data();
    const std::size_t cols = bf16.cols;
    return build_q4k(bf16.rows, cols, [=](std::size_t row, std::span<float> buf) {
        const std::uint16_t* src = src_all + row * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            buf[c] = bf16_to_f32(src[c]);
        }
    });
}

Q4KMat Q4KMat::from_q4mat(const Q4Mat& q4) {
    assert(q4.cols % Q4K_SUPERBLOCK == 0 &&
           "Q4KMat::from_q4mat: cols must be a multiple of 256");
    return build_q4k(q4.rows, q4.cols, [&](std::size_t row, std::span<float> buf) {
        q4.dequantize_row_into(row, buf);
    });
}

#if defined(__aarch64__)
namespace {
/// NEON-accelerated Q4_K dequant for one 32-byte chunk (64 f32 outputs).
///
/// Processes 16 packed bytes at a time: extract low nibbles (AND 0x0F) and
/// high nibbles (SHR 4), widen u8 -> u16 -> u32 -> f32 in groups of 4, then
/// apply `w = scale * nibble - min`.
inline void dequant_chunk_neon(const std::uint8_t* q_ptr, float* lo_ptr, float* hi_ptr,
                               float scale1, float min1, float scale2, float min2) {
    const uint8x16_t mask_0f = vdupq_n_u8(0x0f);
    const float32x4_t s1 = vdupq_n_f32(scale1);
    const float32x4_t m1 = vdupq_n_f32(min1);
    const float32x4_t s2 = vdupq_n_f32(scale2);
    const float32x4_t m2 = vdupq_n_f32(min2);

    // Two passes of 16 bytes each cover all 32 bytes.
    for (std::size_t half = 0; half < 2; ++half) {
        const std::size_t off = half * 16;
        const uint8x16_t raw = vld1q_u8(q_ptr + off);
        const uint8x16_t lo_nib = vandq_u8(raw, mask_0f);
        const uint8x16_t hi_nib = vshrq_n_u8(raw, 4);

        // Low nibbles -> lo_ptr[off .. off+16]
        const uint16x8_t lo8a = vmovl_u8(vget_low_u8(lo_nib));
        const float32x4_t g0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo8a)));
        const float32x4_t g1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo8a)));
        const uint16x8_t lo8b = vmovl_u8(vget_high_u8(lo_nib));
        const float32x4_t g2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo8b)));
        const float32x4_t g3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo8b)));

        vst1q_f32(lo_ptr + off, vsubq_f32(vmulq_f32(s1, g0), m1));
        vst1q_f32(lo_ptr + off + 4, vsubq_f32(vmulq_f32(s1, g1), m1));
        vst1q_f32(lo_ptr + off + 8, vsubq_f32(vmulq_f32(s1, g2), m1));
        vst1q_f32(lo_ptr + off + 12, vsubq_f32(vmulq_f32(s1, g3), m1));

        // High nibbles -> hi_ptr[off .. off+16]
        const uint16x8_t hi8a = vmovl_u8(vget_low_u8(hi_nib));
        const float32x4_t h0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi8a)));
        const float32x4_t h1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi8a)));
        const uint16x8_t hi8b = vmovl_u8(vget_high_u8(hi_nib));
        const float32x4_t h2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi8b)));
        const float32x4_t h3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi8b)));

        vst1q_f32(hi_ptr + off, vsubq_f32(vmulq_f32(s2, h0), m2));
        vst1q_f32(hi_ptr + off + 4, vsubq_f32(vmulq_f32(s2, h1), m2));
        vst1q_f32(hi_ptr + off + 8, vsubq_f32(vmulq_f32(s2, h2), m2));
        vst1q_f32(hi_ptr + off + 12, vsubq_f32(vmulq_f32(s2, h3), m2));
    }
}
}  // namespace
#endif

void Q4KMat::dequantize_row_into(std::size_t row_idx, std::span<float> buf) const {
    const std::size_t k = cols;
    assert(k % Q4K_SUPERBLOCK == 0 && "cols must be a multiple of 256");
    const std::size_t n_blocks = k / Q4K_SUPERBLOCK;
    const std::size_t base_block = row_idx * n_blocks;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::uint8_t* bp = blocks.data() + (base_block + b) * Q4K_BLOCK_BYTES;
        const float d = f16_to_f32(read_u16_le(bp));
        const float dmin = f16_to_f32(read_u16_le(bp + 2));
        const std::uint8_t* sc = bp + 4;
        const std::uint8_t* qs = bp + 16;

        // 4 chunks of 64 elements; chunk c uses qs[c*32 .. (c+1)*32]:
        //   low  nibbles -> sub-block 2c
        //   high nibbles -> sub-block 2c+1
        float* out = buf.data() + b * Q4K_SUPERBLOCK;
        for (std::size_t chunk = 0; chunk < 4; ++chunk) {
            float sv1, mv1, sv2, mv2;
            scale_min(sc, chunk * 2, sv1, mv1);
            scale_min(sc, chunk * 2 + 1, sv2, mv2);
            const float scale1 = d * sv1, min1 = dmin * mv1;
            const float scale2 = d * sv2, min2 = dmin * mv2;
            const std::uint8_t* q = qs + chunk * 32;
            float* lo = out + chunk * 64;
            float* hi = lo + 32;

#if defined(__aarch64__)
            dequant_chunk_neon(q, lo, hi, scale1, min1, scale2, min2);
#else
            for (std::size_t l = 0; l < 32; ++l) {
                lo[l] = scale1 * static_cast<float>(q[l] & 0x0f) - min1;
                hi[l] = scale2 * static_cast<float>(q[l] >> 4) - min2;
            }
#endif
        }
    }
}

// =============================================================================
// Fused GEMV -- M=1 decode fast path
// =============================================================================

#if defined(__aarch64__)
namespace {
/// NEON-accelerated fused Q4_K dot product for one row.
///
/// For each 32-byte chunk (64 elements), processes 16 packed nibbles at a
/// time: extracts lo/hi nibbles into u8x16, widens to f32x4 groups, FMAs with
/// the activation vector, and separately accumulates sum(activation) for the
/// min-subtraction term.
float dot_row_neon(const std::uint8_t* blocks_ptr, const float* a_ptr,
                   std::size_t base_block, std::size_t n_blocks) {
    const uint8x16_t mask_0f = vdupq_n_u8(0x0f);
    float total = 0.0f;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::uint8_t* bp = blocks_ptr + (base_block + b) * Q4K_BLOCK_BYTES;
        const float d = f16_to_f32(read_u16_le(bp));
        const float dmin = f16_to_f32(read_u16_le(bp + 2));
        const std::uint8_t* sc_ptr = bp + 4;
        const std::uint8_t* qs_ptr = bp + 16;
        const std::size_t a_base = b * Q4K_SUPERBLOCK;

        for (std::size_t chunk = 0; chunk < 4; ++chunk) {
            float sv1, mv1, sv2, mv2;
            Q4KMat::scale_min(sc_ptr, chunk * 2, sv1, mv1);
            Q4KMat::scale_min(sc_ptr, chunk * 2 + 1, sv2, mv2);
            const float scale1 = d * sv1, min1 = dmin * mv1;
            const float scale2 = d * sv2, min2 = dmin * mv2;

            const std::uint8_t* q = qs_ptr + chunk * 32;
            const float* a_lo_ptr = a_ptr + a_base + chunk * 64;
            const float* a_hi_ptr = a_lo_ptr + 32;

            // Accumulators: dot(nibble, activation) and sum(activation).
            float32x4_t dot_lo = vdupq_n_f32(0.0f);
            float32x4_t dot_hi = vdupq_n_f32(0.0f);
            float32x4_t sum_lo = vdupq_n_f32(0.0f);
            float32x4_t sum_hi = vdupq_n_f32(0.0f);

            for (std::size_t half = 0; half < 2; ++half) {
                const std::size_t off = half * 16;
                const uint8x16_t raw = vld1q_u8(q + off);
                const uint8x16_t lo_nib = vandq_u8(raw, mask_0f);
                const uint8x16_t hi_nib = vshrq_n_u8(raw, 4);

                const uint16x8_t lo8a = vmovl_u8(vget_low_u8(lo_nib));
                const float32x4_t g0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo8a)));
                const float32x4_t g1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo8a)));
                const uint16x8_t lo8b = vmovl_u8(vget_high_u8(lo_nib));
                const float32x4_t g2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo8b)));
                const float32x4_t g3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo8b)));

                const float32x4_t a0 = vld1q_f32(a_lo_ptr + off);
                const float32x4_t a1 = vld1q_f32(a_lo_ptr + off + 4);
                const float32x4_t a2 = vld1q_f32(a_lo_ptr + off + 8);
                const float32x4_t a3 = vld1q_f32(a_lo_ptr + off + 12);

                dot_lo = vfmaq_f32(dot_lo, g0, a0);
                dot_lo = vfmaq_f32(dot_lo, g1, a1);
                dot_lo = vfmaq_f32(dot_lo, g2, a2);
                dot_lo = vfmaq_f32(dot_lo, g3, a3);

                sum_lo = vaddq_f32(sum_lo, a0);
                sum_lo = vaddq_f32(sum_lo, a1);
                sum_lo = vaddq_f32(sum_lo, a2);
                sum_lo = vaddq_f32(sum_lo, a3);

                const uint16x8_t hi8a = vmovl_u8(vget_low_u8(hi_nib));
                const float32x4_t h0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi8a)));
                const float32x4_t h1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi8a)));
                const uint16x8_t hi8b = vmovl_u8(vget_high_u8(hi_nib));
                const float32x4_t h2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi8b)));
                const float32x4_t h3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi8b)));

                const float32x4_t b0 = vld1q_f32(a_hi_ptr + off);
                const float32x4_t b1 = vld1q_f32(a_hi_ptr + off + 4);
                const float32x4_t b2 = vld1q_f32(a_hi_ptr + off + 8);
                const float32x4_t b3 = vld1q_f32(a_hi_ptr + off + 12);

                dot_hi = vfmaq_f32(dot_hi, h0, b0);
                dot_hi = vfmaq_f32(dot_hi, h1, b1);
                dot_hi = vfmaq_f32(dot_hi, h2, b2);
                dot_hi = vfmaq_f32(dot_hi, h3, b3);

                sum_hi = vaddq_f32(sum_hi, b0);
                sum_hi = vaddq_f32(sum_hi, b1);
                sum_hi = vaddq_f32(sum_hi, b2);
                sum_hi = vaddq_f32(sum_hi, b3);
            }

            // Reduce: scale*dot - min*sum for each sub-block.
            total += scale1 * vaddvq_f32(dot_lo) - min1 * vaddvq_f32(sum_lo) +
                     scale2 * vaddvq_f32(dot_hi) - min2 * vaddvq_f32(sum_hi);
        }
    }

    return total;
}

/// SDOT-accelerated Q4K x Q8 dot product for one row.
///
/// Uses the ARM SDOT instruction (4 x (4 x i8 -> i32) per lane) to skip the
/// u8 -> u16 -> u32 -> f32 widening chain that dominates `dot_row_neon`.
float dot_row_q8(const std::uint8_t* blocks_ptr, const Q8Activation& q8,
                 std::size_t base_block, std::size_t n_blocks) {
    const uint8x16_t mask_0f = vdupq_n_u8(0x0f);
    float total = 0.0f;

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::uint8_t* bp = blocks_ptr + (base_block + b) * Q4K_BLOCK_BYTES;
        const float d = f16_to_f32(read_u16_le(bp));
        const float dmin = f16_to_f32(read_u16_le(bp + 2));
        const std::uint8_t* sc_ptr = bp + 4;
        const std::uint8_t* qs_ptr = bp + 16;
        const std::size_t a_base = b * Q4K_SUPERBLOCK;  // offset into the q8 arrays

        for (std::size_t chunk = 0; chunk < 4; ++chunk) {
            float sv1, mv1, sv2, mv2;
            Q4KMat::scale_min(sc_ptr, chunk * 2, sv1, mv1);
            Q4KMat::scale_min(sc_ptr, chunk * 2 + 1, sv2, mv2);

            const std::uint8_t* q = qs_ptr + chunk * 32;
            const std::size_t q8_lo_off = a_base + chunk * 64;
            const std::size_t q8_hi_off = q8_lo_off + 32;
            const std::size_t q8_lo_blk = q8_lo_off / 32;  // scale/sum block index
            const std::size_t q8_hi_blk = q8_hi_off / 32;

            int32x4_t isum_lo = vdupq_n_s32(0);
            int32x4_t isum_hi = vdupq_n_s32(0);

            for (std::size_t half = 0; half < 2; ++half) {
                const std::size_t off = half * 16;
                const uint8x16_t raw = vld1q_u8(q + off);
                const int8x16_t lo_nib = vreinterpretq_s8_u8(vandq_u8(raw, mask_0f));
                const int8x16_t hi_nib = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));

                const int8x16_t a_lo = vld1q_s8(q8.q8.data() + q8_lo_off + off);
                const int8x16_t a_hi = vld1q_s8(q8.q8.data() + q8_hi_off + off);

                // isum += dot4(nibbles, q8_activations) per lane.
                isum_lo = vdotq_s32(isum_lo, lo_nib, a_lo);
                isum_hi = vdotq_s32(isum_hi, hi_nib, a_hi);
            }

            const float int_dot_lo = static_cast<float>(vaddvq_s32(isum_lo));
            const float int_dot_hi = static_cast<float>(vaddvq_s32(isum_hi));

            // acc += d * sv * q8_scale * int_dot - dmin * mv * sum_a
            total += d * sv1 * q8.scales[q8_lo_blk] * int_dot_lo -
                     dmin * mv1 * q8.sums[q8_lo_blk] +
                     d * sv2 * q8.scales[q8_hi_blk] * int_dot_hi -
                     dmin * mv2 * q8.sums[q8_hi_blk];
        }
    }

    return total;
}
}  // namespace
#endif  // __aarch64__

float Q4KMat::dot_row(std::size_t row_idx, std::span<const float> a) const {
    const std::size_t n_blocks = cols / Q4K_SUPERBLOCK;
    const std::size_t base_block = row_idx * n_blocks;

#if defined(__aarch64__)
    return dot_row_neon(blocks.data(), a.data(), base_block, n_blocks);
#else
    float acc = 0.0f;
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::uint8_t* bp = blocks.data() + (base_block + b) * Q4K_BLOCK_BYTES;
        const float d = f16_to_f32(read_u16_le(bp));
        const float dmin = f16_to_f32(read_u16_le(bp + 2));
        const std::uint8_t* sc = bp + 4;
        const std::uint8_t* qs = bp + 16;
        const std::size_t a_base = b * Q4K_SUPERBLOCK;

        for (std::size_t chunk = 0; chunk < 4; ++chunk) {
            float sv1, mv1, sv2, mv2;
            scale_min(sc, chunk * 2, sv1, mv1);
            scale_min(sc, chunk * 2 + 1, sv2, mv2);
            const float scale1 = d * sv1, min1 = dmin * mv1;
            const float scale2 = d * sv2, min2 = dmin * mv2;
            const std::uint8_t* q = qs + chunk * 32;
            const std::size_t a_off = a_base + chunk * 64;

            float dot_lo = 0.0f, dot_hi = 0.0f, sum_a_lo = 0.0f, sum_a_hi = 0.0f;
            for (std::size_t l = 0; l < 32; ++l) {
                const float a_lo = a[a_off + l];
                dot_lo += static_cast<float>(q[l] & 0x0f) * a_lo;
                sum_a_lo += a_lo;

                const float a_hi = a[a_off + 32 + l];
                dot_hi += static_cast<float>(q[l] >> 4) * a_hi;
                sum_a_hi += a_hi;
            }

            acc += scale1 * dot_lo - min1 * sum_a_lo + scale2 * dot_hi - min2 * sum_a_hi;
        }
    }
    return acc;
#endif
}

Q8Activation Q4KMat::quantize_activation_q8(std::span<const float> a) {
    const std::size_t k = a.size();
    const std::size_t n_blocks = div_ceil(k, 32);

    Q8Activation out;
    out.q8.assign(n_blocks * 32, 0);
    out.scales.assign(n_blocks, 0.0f);
    out.sums.assign(n_blocks, 0.0f);

    for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        const std::size_t start = blk * 32;
        const std::size_t end = std::min(start + 32, k);

        float max_abs = 0.0f;
        float sum = 0.0f;
        for (std::size_t i = start; i < end; ++i) {
            max_abs = std::max(max_abs, std::fabs(a[i]));
            sum += a[i];
        }
        out.sums[blk] = sum;

        if (max_abs == 0.0f) {
            out.scales[blk] = 0.0f;
            continue;
        }

        const float scale = max_abs / 127.0f;
        const float inv_scale = 1.0f / scale;
        out.scales[blk] = scale;

        for (std::size_t i = start; i < end; ++i) {
            out.q8[i] = static_cast<std::int8_t>(
                std::clamp(std::round(a[i] * inv_scale), -128.0f, 127.0f));
        }
    }

    return out;
}

Mat Q4KMat::gemv_mt(const Mat& a) const {
    const std::size_t n = rows;
    const std::size_t k = cols;
    assert(a.rows == 1 && a.cols == k);

    Mat out = Mat::zeros(1, n);

    const std::size_t n_threads = std::max<std::size_t>(1, std::min(hardware_threads(), n / 128));
    const std::size_t n_blocks = k / Q4K_SUPERBLOCK;

#if defined(__aarch64__)
    // Quantize the activation to Q8 once, shared across all threads.
    const Q8Activation q8 = quantize_activation_q8(std::span<const float>(a.data).first(k));
#else
    const std::span<const float> a_data{a.data.data(), k};
#endif

    float* out_ptr = out.data.data();
    const std::size_t chunk = div_ceil(n, n_threads);
    scoped_spawn(n_threads, [&](std::size_t tid) {
        const std::size_t j0 = tid * chunk;
        if (j0 >= n) {
            return;
        }
        const std::size_t j1 = std::min(j0 + chunk, n);
        // Each thread writes a disjoint range [j0, j1).
        for (std::size_t j = j0; j < j1; ++j) {
#if defined(__aarch64__)
            out_ptr[j] = dot_row_q8(blocks.data(), q8, j * n_blocks, n_blocks);
#else
            out_ptr[j] = dot_row(j, a_data);
#endif
        }
    });

    return out;
}

Mat Q4KMat::matmul_q4k_t(const Mat& a) const {
    const std::size_t m = a.rows, k = a.cols, n = rows;
    assert(k == cols && "matmul_q4k_t: a.cols != q4k.cols");

    // GEMV fast path for decode (M=1): fused NEON dot + multi-threading.
    if (m == 1) {
        return gemv_mt(a);
    }

    Mat out = Mat::zeros(m, n);
    std::vector<float> row_buf(k);
    for (std::size_t j = 0; j < n; ++j) {
        dequantize_row_into(j, row_buf);
        for (std::size_t i = 0; i < m; ++i) {
            float acc = 0.0f;
            for (std::size_t p = 0; p < k; ++p) {
                acc += a.data[i * k + p] * row_buf[p];
            }
            out.at_mut(i, j) = acc;
        }
    }
    return out;
}

Mat Q4KMat::matmul_q4k_t_blas(const Mat& a) const {
#if !RT_FEATURE_BLAS
    return matmul_q4k_t(a);
#else
    const std::size_t m = a.rows, k = a.cols, n = rows;
    assert(k == cols && "matmul_q4k_t_blas: a.cols != q4k.cols");

    // GEMV fast path for decode (M=1): fused NEON dot + multi-threading.
    if (m == 1) {
        return gemv_mt(a);
    }

    // Chunked SGEMM for both prefill and decode: split output neurons across
    // threads, each with its own scratch buffer. Accelerate's sgemm is
    // thread-safe and the output column bands are disjoint.
    const std::size_t n_threads = hardware_threads();
    const std::size_t chunk =
        std::min(n, std::max<std::size_t>(64, (8 * 1024 * 1024) / (k * 4)));
    Mat out = Mat::zeros(m, n);

    // Only parallelize when the matmul is large enough that thread-spawn
    // overhead (~200 us for 8 threads) is small next to the dequant work:
    // gate/up/down projections (N=10240) qualify, k/v (N=1024) do not.
    if (n_threads > 1 && n >= 4096) {
        float* out_ptr = out.data.data();
        const std::size_t rows_per_thread = div_ceil(n, n_threads);

        scoped_spawn(n_threads, [&](std::size_t tid) {
            const std::size_t j_start = tid * rows_per_thread;
            if (j_start >= n) {
                return;
            }
            const std::size_t j_end = std::min(j_start + rows_per_thread, n);
            const std::size_t range_n = j_end - j_start;
            const std::size_t local_chunk = std::min(chunk, range_n);
            std::vector<float> chunk_buf(local_chunk * k);

            for (std::size_t j0 = j_start; j0 < j_end;) {
                const std::size_t j1 = std::min(j0 + local_chunk, j_end);
                const std::size_t actual = j1 - j0;

                for (std::size_t ji = 0; ji < actual; ++ji) {
                    dequantize_row_into(j0 + ji,
                                        std::span<float>(chunk_buf).subspan(ji * k, k));
                }

                // Each thread's sgemm writes only its own column band.
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            static_cast<int>(m), static_cast<int>(actual),
                            static_cast<int>(k),
                            1.0f,
                            a.data.data(), static_cast<int>(k),
                            chunk_buf.data(), static_cast<int>(k),
                            0.0f,
                            out_ptr + j0, static_cast<int>(n));
                j0 = j1;
            }
        });
    } else {
        // Single-threaded fallback.
        std::vector<float> chunk_buf(chunk * k);
        for (std::size_t j0 = 0; j0 < n;) {
            const std::size_t j1 = std::min(j0 + chunk, n);
            const std::size_t actual = j1 - j0;
            for (std::size_t ji = 0; ji < actual; ++ji) {
                dequantize_row_into(j0 + ji, std::span<float>(chunk_buf).subspan(ji * k, k));
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        static_cast<int>(m), static_cast<int>(actual), static_cast<int>(k),
                        1.0f,
                        a.data.data(), static_cast<int>(k),
                        chunk_buf.data(), static_cast<int>(k),
                        0.0f,
                        out.data.data() + j0, static_cast<int>(n));
            j0 = j1;
        }
    }
    return out;
#endif
}

}  // namespace rt

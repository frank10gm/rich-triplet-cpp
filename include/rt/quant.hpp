#pragma once

// =============================================================================
// Weight quantization formats: Q4_0, Q8, and Q4_K
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "rt/mat.hpp"

namespace rt {

/// Block size for Q4_0 quantization.
inline constexpr std::size_t Q4_BLOCK_SIZE = 32;

/// Block size for Q8 quantization.
/// Larger blocks -> fewer scale values -> higher compression, lower accuracy.
inline constexpr std::size_t Q8_BLOCK_SIZE = 64;

/// A 4-bit quantized matrix.
///
/// Elements are stored as signed 4-bit integers, two per byte (packed nibbles).
/// Each block of `Q4_BLOCK_SIZE` elements has an associated f32 scale.
struct Q4Mat {
    std::size_t rows = 0;
    std::size_t cols = 0;

    /// Packed nibbles: ceil(rows*cols / 2) bytes.
    /// Element (r,c) at flat index k = r*cols+c:
    ///   k even -> low nibble of packed[k/2]
    ///   k odd  -> high nibble of packed[k/2]
    /// Each nibble is 4-bit two's complement: 0x0=0 ... 0x7=7, 0x9=-7 ... 0xF=-1.
    std::vector<std::uint8_t> packed;

    /// One scale per block: ceil(rows*cols / Q4_BLOCK_SIZE) values.
    std::vector<float> scales;

    /// Quantize a `Mat` to 4-bit.
    [[nodiscard]] static Q4Mat quantize(const Mat& mat);

    /// Recover an approximate `Mat` from the 4-bit representation.
    ///
    /// The quantization error is at most `0.5 * scale`, i.e. `absmax / 14`
    /// (~7% of the largest value in the block).
    [[nodiscard]] Mat dequantize() const;

    /// Fused `a [M,K] @ self^T [N,K] -> [M,N]`, no full dequantization.
    [[nodiscard]] Mat matmul_q4_t(const Mat& a) const;

    /// BLAS-accelerated `a [M,K] @ self^T [N,K] -> [M,N]`.
    ///
    /// * **Prefill (M > 4)** -- dequantize the whole weight matrix once, then a
    ///   single sgemm. sgemm's cache blocking across both M and N more than
    ///   pays for the N x K scratch.
    /// * **Decode (M <= 4)** -- dequantize CHUNK rows at a time (~8 MB scratch)
    ///   and issue one sgemm per chunk, cutting BLAS call overhead from N calls
    ///   to N/CHUNK.
    [[nodiscard]] Mat matmul_q4_t_blas(const Mat& a) const;

    /// Dequantize row `j` into `buf` (length >= cols).
    void dequantize_row_into(std::size_t j, std::span<float> buf) const;

    /// Memory usage in bytes (excluding struct overhead).
    [[nodiscard]] std::size_t size_bytes() const {
        return packed.size() + scales.size() * 4;
    }

    /// Compression ratio vs f32 storage.
    [[nodiscard]] float compression_ratio() const {
        return static_cast<float>(rows * cols * 4) / static_cast<float>(size_bytes());
    }
};

/// 8-bit symmetric block-quantized matrix.
///
/// Q8 uses signed 8-bit integers (range -127..127): 255 levels versus Q4's 15,
/// so quantization error is ~0.4% of the block max instead of ~7%. Memory is
/// 4x smaller than f32 (still 2x larger than Q4). Used where the error budget
/// is tight -- KV cache, activations, small models.
struct Q8Mat {
    std::size_t rows = 0;
    std::size_t cols = 0;
    /// One i8 per element, in row-major order.
    std::vector<std::int8_t> packed;
    /// One f32 scale per block of `Q8_BLOCK_SIZE` elements.
    std::vector<float> scales;

    [[nodiscard]] static Q8Mat quantize(const Mat& mat);

    /// Recover an approximate `Mat`. Error per element <= absmax/254.
    [[nodiscard]] Mat dequantize() const;

    /// Fused `a [M,K] @ self^T [N,K] -> [M,N]`, dequantizing B row by row.
    [[nodiscard]] Mat matmul_q8_t(const Mat& a) const;

    [[nodiscard]] std::size_t size_bytes() const {
        return packed.size() + scales.size() * 4;
    }

    [[nodiscard]] float compression_ratio() const {
        return static_cast<float>(rows * cols * 4) / static_cast<float>(size_bytes());
    }
};

/// Pre-quantized Q8 activation data for SDOT-accelerated Q4K dot products.
/// Activations are quantized to int8 per 32-element sub-block.
struct Q8Activation {
    /// Quantized activation values (length = K, padded to a multiple of 32).
    std::vector<std::int8_t> q8;
    /// Per-32-element scale: actual_value ~= q8[i] * scales[i/32].
    std::vector<float> scales;
    /// Per-32-element float sum of the original activations (for min subtraction).
    std::vector<float> sums;
};

/// Q4_K native packed weight matrix.
///
/// Stores weights in the raw GGUF Q4_K block layout (144 bytes per 256
/// elements):
///   bytes[0..2]    -- f16 `d`    (super-block scale for the 8 sub-block scales)
///   bytes[2..4]    -- f16 `dmin` (super-block scale for the 8 sub-block mins)
///   bytes[4..16]   -- 12 packed 6-bit scale/min pairs
///   bytes[16..144] -- 128 bytes of 4-bit quants (4 chunks x 32 bytes)
///
/// Dequantizing on the fly avoids keeping a BF16/f32 copy at rest, saving
/// ~3.5x RAM versus BF16 (144/256 ~= 0.56 bytes/elem vs 2 bytes/elem).
///
/// `rows` = out_features, `cols` = in_features, matching our Linear convention.
struct Q4KMat {
    std::size_t rows = 0;
    std::size_t cols = 0;
    /// Raw Q4_K block bytes: `n_blocks * 144` where
    /// `n_blocks = ceil(rows * cols / 256)`.
    std::vector<std::uint8_t> blocks;

    /// Quantize an f32 matrix to Q4_K. `cols` must be a multiple of 256.
    [[nodiscard]] static Q4KMat quantize(const Mat& mat);

    /// Quantize a BF16 matrix without materializing the full f32 matrix.
    ///
    /// Uses a per-row f32 scratch buffer (~10 KB for cols=2560) instead of
    /// rows x cols x 4 bytes (~2.7 GB for lm_head).
    [[nodiscard]] static Q4KMat quantize_from_bf16(const MatBf16& bf16);

    /// Convert a Q4_0 matrix to Q4_K, row by row with ~10 KB scratch/thread.
    [[nodiscard]] static Q4KMat from_q4mat(const Q4Mat& q4);

    /// Quantize a single 256-element block into 144 bytes of Q4_K.
    static void quantize_block(std::span<const float> block_data, std::span<std::uint8_t> out);

    /// Dequantize row `row_idx` into `buf` (length >= cols).
    /// `cols` must be a multiple of 256, so each row starts on a super-block
    /// boundary.
    void dequantize_row_into(std::size_t row_idx, std::span<float> buf) const;

    /// Scalar matmul: `a [M,K] @ self^T [N,K] -> [M,N]`.
    [[nodiscard]] Mat matmul_q4k_t(const Mat& a) const;

    /// BLAS-accelerated matmul with chunked SGEMM (~8 MB scratch).
    [[nodiscard]] Mat matmul_q4k_t_blas(const Mat& a) const;

    /// Fused Q4_K dot product of weight row `row_idx` against `a`.
    ///
    /// Uses the integer-accumulation trick: accumulate `nibble * activation`
    /// and `sum(activation)` separately, applying scale/min once per sub-block
    /// instead of once per element.
    [[nodiscard]] float dot_row(std::size_t row_idx, std::span<const float> a) const;

    /// Memory usage in bytes.
    [[nodiscard]] std::size_t size_bytes() const { return blocks.size(); }

    /// Extract the 6-bit scale and min for sub-block `j` (0..8) from the
    /// 12-byte scales array of a Q4_K super-block.
    static void scale_min(const std::uint8_t* sc, std::size_t j, float& scale, float& min);

    /// Quantize an f32 activation vector to Q8 (int8 per 32-element block).
    [[nodiscard]] static Q8Activation quantize_activation_q8(std::span<const float> a);

   private:
    /// Multi-threaded GEMV: `a[1,K] @ self^T[N,K] -> [1,N]`.
    [[nodiscard]] Mat gemv_mt(const Mat& a) const;
};

}  // namespace rt

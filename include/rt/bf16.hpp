#pragma once

// =============================================================================
// BF16 (bfloat16) and F16 (IEEE half) bit conversions
// =============================================================================
//
// ## What is BF16?
//
// BF16 (Brain Float 16) is a 16-bit floating-point format with:
//   - 1 sign bit
//   - 8 exponent bits  (same as f32 — same range: ~1e-38 to ~3e38)
//   - 7 mantissa bits  (vs 23 for f32 — lower precision)
//
// Because the exponent range equals f32, BF16 <-> f32 conversion is trivial:
//   f32_bits   = bf16_bits << 16          (zero-pad the lower 16 bits)
//   bf16_bits  = (f32_bits >> 16)         (truncate the lower 16 bits)
//
// ## Why use BF16?
//
// All major open-weight models (LLaMA, Gemma, Mistral, GPT-OSS) ship weights
// as BF16:
//   - Half the disk space and RAM of f32
//   - Full exponent range -> no clipping issues during fine-tuning
//   - Supported natively by NVIDIA Ampere / Hopper, Apple M3+, Google TPUs
//
// ## Precision loss
//
// BF16 has only 7 mantissa bits (2.3 decimal digits of precision, vs 7.2 for
// f32). This is acceptable for weights but not for accumulators (loss,
// gradients) -- those stay f32.

#include <bit>
#include <cmath>
#include <cstdint>

namespace rt {

/// Convert a BF16-packed uint16 to float.
///
/// Shift the 16 bits into the upper half of a uint32 (float's bit layout):
/// the sign and exponent fields are identical, mantissa is zero-padded.
[[nodiscard]] inline constexpr float bf16_to_f32(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

/// Convert a float to BF16 bits (round-to-nearest, ties-to-even).
///
/// For NaN, the special value is preserved with the mantissa bit forced on.
/// For finite values: shift right by 16, with rounding.
[[nodiscard]] inline std::uint16_t f32_to_bf16(float v) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(v);
    if (std::isnan(v)) {
        // Propagate NaN, ensure mantissa bit is set.
        return static_cast<std::uint16_t>(bits >> 16) | 0x0040u;
    }
    // Round-to-nearest-even: look at the truncated bits.
    const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
    const std::uint32_t rounded = bits + rounding_bias;  // wrapping add
    return static_cast<std::uint16_t>(rounded >> 16);
}

/// Convert IEEE 754 half-precision (f16) bits to float.
///
/// Q4_K super-block headers store `d` and `dmin` as f16, not bfloat16, so this
/// needs the full subnormal/Inf/NaN decode rather than a shift.
[[nodiscard]] inline float f16_to_f32(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exp = (bits >> 10) & 0x1fu;
    const std::uint32_t mant = bits & 0x3ffu;

    if (exp == 0) {
        if (mant == 0) {
            return std::bit_cast<float>(sign);  // +/- 0
        }
        // Subnormal f16 -> normalized f32: renormalize by shifting the
        // mantissa left until the implicit leading 1 appears.
        std::uint32_t m = mant;
        std::uint32_t e = 0;
        while ((m & 0x400u) == 0) {
            m <<= 1;
            ++e;
        }
        m &= 0x3ffu;
        const std::uint32_t f32_exp = 127u - 15u - e + 1u;
        return std::bit_cast<float>(sign | (f32_exp << 23) | (m << 13));
    }
    if (exp == 0x1f) {
        // Inf or NaN.
        return std::bit_cast<float>(sign | 0x7f800000u | (mant << 13));
    }
    const std::uint32_t f32_exp = exp - 15u + 127u;
    return std::bit_cast<float>(sign | (f32_exp << 23) | (mant << 13));
}

/// Convert a float to IEEE 754 half-precision (f16) bits.
///
/// Inverse of `f16_to_f32`. The mantissa is **truncated**, not rounded: this
/// matches the reference implementation bit for bit, and Q4_K block headers
/// are quantized through here, so changing the rounding would shift every
/// dequantized weight.
[[nodiscard]] inline std::uint16_t f32_to_f16(float v) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(v);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xffu);
    const std::uint32_t mant = bits & 0x7fffffu;

    if (exp == 0xff) {
        // Inf / NaN
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mant >> 13));
    }
    const std::int32_t new_exp = exp - 127 + 15;
    if (new_exp >= 0x1f) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);  // overflow -> Inf
    }
    if (new_exp <= 0) {
        // Denormal or zero.
        if (new_exp < -10) {
            return static_cast<std::uint16_t>(sign);  // too small
        }
        const std::uint32_t m = mant | 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(1 - new_exp) + 13u;
        return static_cast<std::uint16_t>(sign | (m >> shift));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(new_exp) << 10)
                                      | (mant >> 13));
}

}  // namespace rt

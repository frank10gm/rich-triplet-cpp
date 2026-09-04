#pragma once

// =============================================================================
// Metal GPU matrix multiplication
// =============================================================================
//
// Compiled only when RT_METAL is on. `Mat::matmul` and `NDArray::bmm` branch
// here ahead of the BLAS and scalar paths.
//
// ## Design
//
// One MetalContext is built lazily per thread: creating the system device costs
// about a millisecond, so it is paid once and the device, queue and pipelines
// live for the thread's lifetime. Buffers use StorageModeShared, which on
// Apple Silicon's unified memory needs no explicit synchronization. The MSL
// kernels are compiled from source at first use and cached.
//
// ## Kernels
//
// `matmul_tiled` walks 16x16 threadgroup tiles through shared memory, so each
// tile of A and B is read from global memory once and reused by every thread in
// its row and column. `matmul_tiled_batched` is the same algorithm with the
// batch mapped to the grid's Z axis, so all slices run concurrently.
//
// The quantized kernels give one threadgroup (32 threads, a single SIMD group)
// per output element and reduce with `simd_sum`, which keeps the GPU busy even
// at batch size 1.
//
// ## Threshold
//
// Below `METAL_THRESHOLD` multiply-adds, buffer allocation, command encoding
// and GPU wake-up cost more than the compute saves, so those calls run on the
// CPU instead.

#include <cstddef>
#include <utility>
#include <vector>

#include "rt/mat.hpp"
#include "rt/quant.hpp"

namespace rt {

/// Below this many multiply-adds, the dispatch overhead dominates.
inline constexpr std::size_t METAL_THRESHOLD = 32768;  // 32^3

/// `C = A @ B`, [M,K] x [K,N] -> [M,N]. Falls back to a scalar loop when the
/// problem is too small to be worth a dispatch.
[[nodiscard]] Mat metal_matmul(const Mat& a, const Mat& b);

/// Batched `C_i = A_i @ B_i`, all pairs sharing one shape, in a single
/// dispatch. Falls back to sequential single matmuls for an empty or
/// single-element batch, or when each slice is below the threshold.
[[nodiscard]] std::vector<Mat> metal_matmul_batched(
    const std::vector<std::pair<Mat, Mat>>& pairs);

/// `C = A @ dequant(Q4)^T`, with `q4` stored row-major as [N, K].
[[nodiscard]] Mat metal_matmul_q4_t(const Mat& a, const Q4Mat& q4);

/// Q4_K GEMV: `C[1,N] = A[1,K] @ dequant(Q4K[N,K])^T`.
///
/// Weight blocks are uploaded once and cached as persistent buffers; the
/// activation and output go through pre-allocated scratch, so a decode step
/// allocates nothing.
[[nodiscard]] Mat metal_gemv_q4k_t(const Mat& a, const Q4KMat& q4k);

/// BF16 GEMV: `C[1,N] = A[1,K] @ BF16[N,K]^T`.
[[nodiscard]] Mat metal_gemv_bf16_t(const Mat& a, const MatBf16& bf16);

}  // namespace rt

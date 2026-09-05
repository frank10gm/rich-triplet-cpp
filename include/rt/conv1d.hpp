#pragma once

// =============================================================================
// 1-D convolution primitives -- for neural audio codec decoders
// =============================================================================
//
// The transformer stack needs no convolution beyond Qwen 3.5's causal depthwise
// conv1d, which is specialised into `Qwen35DeltaNet`. Neural audio codecs
// (SNAC, DAC, Encodec) are built almost entirely out of convolutions, so they
// need the general forms: dilated, grouped, and transposed.
//
// ## Activation layout
//
// Activations are `Mat` in **time-major** order -- `[T, channels]`, one row per
// frame. PyTorch stores audio activations channel-major, `[channels, T]`. Time-
// major is what the rest of this project uses (a `Mat` row is always one token
// or one timestep), and it makes a kernel-size-1 convolution literally a
// matmul against the weight, so `Mat::matmul_bt` and BLAS apply unchanged.
//
// ## Weight layout
//
// Every weight is a `Mat` whose row-major bytes are **exactly** the PyTorch
// flat buffer, so loading is a wrap rather than a repack:
//
// | PyTorch parameter          | shape             | `Mat` passed here     |
// |----------------------------|-------------------|-----------------------|
// | `Conv1d.weight`            | [Cout, Cin, K]    | [Cout, Cin * K]       |
// | `Conv1d.weight` (k=1)      | [Cout, Cin, 1]    | [Cout, Cin]           |
// | `Conv1d.weight` (depthwise)| [C, 1, K]         | [C, K]                |
// | `ConvTranspose1d.weight`   | **[Cin, Cout, K]**| [Cin, Cout * K]       |
//
// Note the transposed convolution stores **input** channels first. That is not
// a typo carried over from `Conv1d`: it is PyTorch's actual layout, and it also
// decides which axis `weight_norm` normalizes over -- see
// `weight_norm_combine`.
//
// ## Output lengths
//
//   conv1d:           T_out = T + 2*padding - dilation * (K - 1)
//   conv_transpose1d: T_out = (T - 1) * stride - 2*padding + K + output_padding
//
// A codec decoder block picks `K = 2*stride`, `padding = ceil(stride/2)` and
// `output_padding = stride % 2`, which makes the second formula collapse to
// exactly `T * stride` for every stride, even or odd:
//
//   even s:  (T-1)s - 2(s/2)     + 2s + 0 = Ts - s - s     + 2s     = Ts
//   odd  s:  (T-1)s - 2((s+1)/2) + 2s + 1 = Ts - s - s - 1 + 2s + 1 = Ts
//
// That exact-multiple property is the cheapest end-to-end check a codec
// decoder has, so `conv_transpose1d` asserts it whenever those three
// parameters line up. Debugging a decoder by ear is miserable; a length
// assertion fires immediately.

#include <cstddef>
#include <span>
#include <vector>

#include "rt/mat.hpp"

namespace rt {

// =============================================================================
// Output lengths
// =============================================================================

/// Output length of a 1-D convolution. Returns 0 when the kernel does not fit,
/// which callers should treat as an error rather than an empty result.
[[nodiscard]] std::size_t conv1d_out_len(std::size_t t_in, std::size_t kernel,
                                         std::size_t dilation, std::size_t padding);

/// Output length of a 1-D transposed convolution.
[[nodiscard]] std::size_t conv_transpose1d_out_len(std::size_t t_in, std::size_t kernel,
                                                   std::size_t stride, std::size_t padding,
                                                   std::size_t output_padding);

// =============================================================================
// Convolutions
// =============================================================================

/// Kernel-size-1 convolution: `out[t] = weight @ x[t] + bias`.
///
/// This is a dense matmul, not a sliding window -- `x [T, Cin]` against
/// `weight [Cout, Cin]` is exactly `x @ weight^T`, so it goes straight to
/// `Mat::matmul_bt` and picks up BLAS. Codec decoders use k=1 convolutions
/// for every channel mix (the residual units' second conv, the noise block,
/// the quantizer's `out_proj`), so this is the hot one.
///
/// `bias` may be empty, meaning no bias.
[[nodiscard]] Mat conv1d_pointwise(const Mat& x, const Mat& weight, std::span<const float> bias);

/// Depthwise convolution -- `groups == channels`, so each channel is filtered
/// independently and no channel mixing happens.
///
/// `x` is [T, C], `weight` is [C, K], output is [T_out, C]. Zero padding.
///
/// `bias` may be empty, meaning no bias.
[[nodiscard]] Mat conv1d_depthwise(const Mat& x, const Mat& weight, std::span<const float> bias,
                                   std::size_t dilation, std::size_t padding);

/// Dense (`groups == 1`) convolution.
///
/// `x` is [T, Cin], `weight` is [Cout, Cin * K], output is [T_out, Cout].
///
/// Three paths, picked by shape:
///
///   * `kernel == 1` with no dilation or padding delegates to
///     `conv1d_pointwise`, which is a plain matmul.
///   * Small problems use a direct accumulation loop, which avoids the scratch
///     buffer entirely.
///   * Everything else goes through im2col: gather the `Cin * K` receptive
///     field of each output position into a row, then one `matmul_bt` against
///     the weight. That turns the convolution into a gemm and hands it to
///     BLAS.
///
/// The third path is not an optimisation so much as a requirement. A SNAC
/// decoder only needs a dense wide kernel for its final `[64 -> 1, k=7]`
/// projection, but OmniVoice's residual units are dense 7-taps at up to 512
/// channels running over a sequence that has already been upsampled toward
/// 96 000 samples -- tens of GFLOP per clip. The direct loop would take tens of
/// seconds where a gemm takes well under one.
///
/// im2col is materialised in row tiles rather than all at once, since the full
/// matrix would be `T_out * Cin * K` floats -- hundreds of megabytes at those
/// shapes.
///
/// `bias` may be empty, meaning no bias.
[[nodiscard]] Mat conv1d_dense(const Mat& x, const Mat& weight, std::size_t out_channels,
                               std::size_t kernel, std::span<const float> bias,
                               std::size_t dilation, std::size_t padding);

/// Transposed (fractionally-strided) convolution -- the upsampling operator.
///
/// `x` is [T, Cin], `weight` is [Cin, Cout * K] (PyTorch's input-channels-first
/// layout, flat), output is [T_out, Cout].
///
/// Implemented in **scatter** form:
///
///   out[t * stride - padding + k, co] += sum_ci x[t, ci] * w[ci, co, k]
///
/// which needs no kernel flip. The gather form -- reading the output position
/// and pulling from the input -- computes the same thing only with the kernel
/// reversed, and getting that backwards yields a time-reversed impulse
/// response: audio that sounds smeared rather than obviously broken. Scatter
/// keeps the indexing honest.
///
/// The `sum_ci` is hoisted into one gemm: `x [T, Cin] @ weight [Cin, Cout*K]`
/// gives every `(t, co, k)` product at once, and the scatter-add then places
/// column `co * K + k` of row `t` at output position `t * stride - padding + k`.
/// The weight needs no repacking because [Cin, Cout*K] row-major already *is*
/// the [Cin, Cout, K] flat buffer.
///
/// `bias` may be empty, meaning no bias. It is per **output** channel, so its
/// length is `out_channels`, not `x.cols`.
[[nodiscard]] Mat conv_transpose1d(const Mat& x, const Mat& weight, std::size_t out_channels,
                                   std::size_t kernel, std::span<const float> bias,
                                   std::size_t stride, std::size_t padding,
                                   std::size_t output_padding);

// =============================================================================
// Activations
// =============================================================================

/// Snake: `x + sin^2(alpha * x) / (alpha + 1e-9)`, with a learned per-channel
/// `alpha`.
///
/// A periodic activation. Unlike GELU or SiLU it does not saturate, which is
/// what lets a codec decoder reproduce the periodic structure of voiced speech
/// -- the inductive bias the function exists for.
///
/// The epsilon sits **inside** the reciprocal (`1 / (alpha + 1e-9)`, not
/// `1 / alpha` guarded afterwards), matching the reference; trained `alpha`
/// values pass close enough to zero for the difference to show.
///
/// `x` is [T, C] and `alpha` has length C. Do not substitute BigVGAN's
/// two-parameter `snake_beta` -- SNAC uses the single-alpha form.
[[nodiscard]] Mat snake1d(const Mat& x, std::span<const float> alpha);

/// In-place `snake1d`, for the long activation chains in a decoder where the
/// intermediate is dead immediately afterwards.
void snake1d_inplace(Mat& x, std::span<const float> alpha);

// =============================================================================
// Weight normalization
// =============================================================================

/// Reconstruct a weight-normalized parameter: `W = g * v / ||v||`.
///
/// `torch.nn.utils.parametrizations.weight_norm` stores the parameter as a
/// magnitude `g` (`parametrizations.weight.original0`) and a direction `v`
/// (`...original1`), and reconstructs `W = g * v / ||v||` with the norm taken
/// over every axis **except** axis 0 of the stored tensor.
///
/// Which axis that is depends on the layer, and this is a real trap:
///
/// | layer              | stored `v` shape  | `g` shape    | norm is per   |
/// |--------------------|-------------------|--------------|---------------|
/// | `Conv1d`           | [Cout, Cin, K]    | [Cout, 1, 1] | output channel|
/// | `ConvTranspose1d`  | [Cin, Cout, K]    | [Cin, 1, 1]  | **input** ch. |
///
/// So a transposed convolution normalizes per input channel while every other
/// convolution in the same model normalizes per output channel. Taking axis 0
/// of the *stored* tensor -- rather than assuming "output channels" -- gets
/// both right with no special case, which is why this takes a flat `v` and
/// derives the group count from `g.size()`.
///
/// Returns the reconstructed flat weight, same length and order as `v`.
[[nodiscard]] std::vector<float> weight_norm_combine(std::span<const float> g,
                                                     std::span<const float> v);

}  // namespace rt

#pragma once

// =============================================================================
// 2-D convolution primitives -- for image autoencoders
// =============================================================================
//
// `conv1d.hpp` exists because neural audio codecs are built out of
// convolutions. Image autoencoders are built out of the same operator one
// dimension up, so this is that file with a second spatial axis threaded
// through it. The three-path structure (pointwise fast path, direct loop,
// tiled im2col + gemm) is deliberately identical -- the reasoning that picked
// it there applies unchanged here, only more so, since the problems are
// larger.
//
// ## Activation layout
//
// Activations are `Mat` in **spatial-major** order -- `[H * W, channels]`, one
// row per pixel, row-major in the pixel index so that `p = y * W + x`. PyTorch
// stores images channel-major, `[channels, H, W]`. Spatial-major is the
// two-dimensional reading of the rule the rest of this project follows: a `Mat`
// row is always one token, one timestep, or -- here -- one pixel.
//
// The layout is not a matter of taste. It is what makes the rest cheap:
//
//   * A 1x1 convolution is literally `x @ weight^T`, so `Mat::matmul_bt` and
//     BLAS apply with no repacking. A VAE decoder is roughly half 1x1
//     convolutions once the residual shortcuts are counted.
//   * The mid-block's self-attention wants one row per spatial position, which
//     is what this already is -- the attention code written for transformers
//     applies to an image with no reshaping at all.
//   * A diffusion transformer's patchify step is a regrouping of rows, not a
//     transpose.
//
// It costs something in exactly one place: `group_norm` reduces over channels
// **and** space together, so its reduction is strided rather than contiguous.
// That is a handful of extra lines, paid once, against a repack on every
// convolution in the other layout.
//
// Because `Mat` is flat, `h` and `w` travel as explicit arguments. The
// invariant `x.rows == h * w` is asserted everywhere it is assumed.
//
// ## Weight layout
//
// As in `conv1d.hpp`, every weight is a `Mat` whose row-major bytes are
// **exactly** the PyTorch flat buffer, so loading is a wrap rather than a
// repack:
//
// | PyTorch parameter    | shape              | `Mat` passed here        |
// |----------------------|--------------------|--------------------------|
// | `Conv2d.weight`      | [Cout, Cin, KH, KW]| [Cout, Cin * KH * KW]    |
// | `Conv2d.weight` (1x1)| [Cout, Cin, 1, 1]  | [Cout, Cin]              |
//
// The innermost index is the kernel's x axis, then its y axis, then the input
// channel -- so column `ci * KH * KW + ky * KW + kx`. im2col gathers into that
// same order, which is why the gemm needs neither operand repacked.
//
// ## Output sizes
//
//   conv2d:  out = floor((in + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1
//
// applied independently per axis. A VAE decoder holds every 3x3 convolution at
// `padding = 1, stride = 1`, which makes that collapse to `out == in`; the
// size is asserted to be preserved whenever those parameters line up, because
// a decoder that silently shrinks by two pixels per layer still produces an
// image and the mistake shows up only as a crop.

#include <cstddef>
#include <span>
#include <vector>

#include "rt/mat.hpp"

namespace rt {

// =============================================================================
// Output sizes
// =============================================================================

/// Output extent of one axis of a 2-D convolution. Returns 0 when the kernel
/// does not fit, which callers should treat as an error rather than an empty
/// result.
[[nodiscard]] std::size_t conv2d_out_size(std::size_t in, std::size_t kernel,
                                          std::size_t dilation, std::size_t padding,
                                          std::size_t stride = 1);

// =============================================================================
// Convolutions
// =============================================================================

/// Kernel-size-1 convolution: `out[p] = weight @ x[p] + bias`.
///
/// `x` is [H*W, Cin], `weight` is [Cout, Cin], output is [H*W, Cout]. This is a
/// dense channel mix with no sliding window, so it goes straight to
/// `Mat::matmul_bt` and picks up BLAS.
///
/// `bias` may be empty, meaning no bias.
[[nodiscard]] Mat conv2d_pointwise(const Mat& x, const Mat& weight, std::span<const float> bias);

/// Dense (`groups == 1`) 2-D convolution with zero padding.
///
/// `x` is [h*w, Cin], `weight` is [Cout, Cin * kernel_h * kernel_w], output is
/// [out_h * out_w, Cout] where the extents come from `conv2d_out_size`.
///
/// Three paths, picked by shape, mirroring `conv1d_dense`:
///
///   * A 1x1 kernel with no dilation, padding or stride delegates to
///     `conv2d_pointwise`, which is a plain matmul.
///   * Small problems use a direct accumulation loop and no scratch buffer.
///   * Everything else goes through im2col: gather the `Cin * KH * KW`
///     receptive field of each output pixel into a row, then one `matmul_bt`
///     against the weight.
///
/// The tiling in the third path is load-bearing rather than an optimisation. A
/// VAE decoder's last level runs 128 channels of 3x3 over 1024x1024 pixels; the
/// full patch matrix would be `1048576 * 1152` floats, 4.8 GB. Tiles of a few
/// megabytes turn that into a sequence of cache-friendly gemms.
///
/// `bias` may be empty, meaning no bias.
[[nodiscard]] Mat conv2d_dense(const Mat& x, std::size_t h, std::size_t w, const Mat& weight,
                               std::size_t out_channels, std::size_t kernel_h,
                               std::size_t kernel_w, std::span<const float> bias,
                               std::size_t stride = 1, std::size_t padding = 0,
                               std::size_t dilation = 1);

// =============================================================================
// Resampling
// =============================================================================

/// Nearest-neighbour upsample by an integer factor in both axes.
///
/// `x` is [h*w, C], output is [(h*factor) * (w*factor), C].
///
/// This is the VAE decoder's only upsampling operator: it upsamples and *then*
/// convolves, rather than using a transposed convolution. Substituting a
/// transposed convolution with the same weights is a plausible-looking change
/// that produces checkerboard artefacts -- a regular high-frequency texture
/// that reads as heavy JPEG compression rather than as a bug.
///
/// The output is a pure repeat: each input pixel becomes a `factor x factor`
/// block. Interpolating instead softens every edge in the image by a fraction
/// of a pixel, which is invisible per layer and cumulative over four of them.
[[nodiscard]] Mat upsample_nearest2d(const Mat& x, std::size_t h, std::size_t w,
                                     std::size_t factor);

// =============================================================================
// Normalization and activations
// =============================================================================

/// GroupNorm over spatial-major activations.
///
/// `x` is [h*w, C]. The channels are split into `groups` contiguous blocks and
/// each block is normalized over its channels **and every spatial position**
/// jointly -- one mean and one variance per group, not per pixel.
///
/// That joint reduction is the whole content of the operator and the one thing
/// easy to get wrong. Normalizing each row independently is LayerNorm; it runs,
/// it is numerically well behaved, and it produces an image whose local
/// contrast is subtly flattened everywhere. There is no way to see it without a
/// reference, so the tests check it directly: a tensor that is constant within
/// each row but varies across rows must **not** normalize to zeros.
///
/// `weight` and `bias` are per channel and may be empty, meaning no affine.
///
/// `eps` defaults to 1e-6, which is what the diffusers VAE uses. The far more
/// common 1e-5 is close enough to look right and far enough to shift contrast
/// measurably once four levels have compounded it.
[[nodiscard]] Mat group_norm(const Mat& x, std::size_t groups, std::span<const float> weight,
                             std::span<const float> bias, float eps = 1e-6f);

/// In-place `group_norm`, for the long chains in a decoder where the
/// intermediate is dead immediately afterwards.
void group_norm_inplace(Mat& x, std::size_t groups, std::span<const float> weight,
                        std::span<const float> bias, float eps = 1e-6f);

/// `x * sigmoid(x)`, in place. The VAE decoder's activation throughout.
void silu_inplace(Mat& x);

/// `x * sigmoid(1.702 * x)`, in place.
///
/// CLIP's activation. It is a cheap approximation of GELU that predates the
/// tanh one, and CLIP-L was trained with it -- substituting either exact GELU
/// or the tanh approximation shifts the text embedding enough to change which
/// image a prompt produces.
void quick_gelu_inplace(Mat& x);

/// `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))`, in place.
///
/// The tanh approximation, which is what T5 v1.1 and FLUX both use.
void gelu_tanh_inplace(Mat& x);

}  // namespace rt

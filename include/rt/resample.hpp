#pragma once

// =============================================================================
// Sample rate conversion -- windowed-sinc polyphase
// =============================================================================
//
// The OmniVoice codec runs its acoustic path at 24 kHz and its semantic path at
// 16 kHz, off the same waveform. Something has to convert between them, and
// which resampler it is matters: the semantic model's features feed a
// quantizer, so a resampler with a different filter than the reference's puts
// the latents somewhere the codebooks were never fit.
//
// This is a port of `torchaudio.functional.resample` at its defaults --
// `lowpass_filter_width=6`, `rolloff=0.99`, `sinc_interp_hann`.
//
// ## The idea
//
// Converting L to M is band-limited interpolation: upsample by M, low-pass,
// downsample by L. Doing that literally would build a signal M times too long.
// The polyphase form never does: it notes that only every L-th sample of the
// filtered signal survives, so it precomputes M filter *phases* -- one per
// output position within a cycle -- and each output sample is one dot product
// against the input at stride L.
//
//   kernel[p][k] = sinc(base * (k/L - p/M)) * hann(...) * base/L
//   out[i*M + p] = sum_k padded[i*L + k] * kernel[p][k]
//
// with `L = orig/gcd`, `M = new/gcd`, so 24 kHz to 16 kHz is 3 to 2: two
// phases, a 23-tap kernel, and one multiply-add pass over the input.
//
// ## The parts that are conventions, not mathematics
//
// The window is `cos(t*pi/width/2)^2` -- a Hann window in the sinc's own
// argument, not in sample index. `rolloff` pulls the cutoff to 0.99 of Nyquist
// so the transition band has somewhere to live. The kernel is clamped to
// +-`lowpass_filter_width` zero crossings before windowing, which is what makes
// it finite. All three are torchaudio's defaults and all three change the
// output if altered, so they are exposed but should be left alone.

#include <cstddef>
#include <span>
#include <vector>

namespace rt {

/// The filter bank for one rate conversion: `phases` kernels of `taps` each.
///
/// Building it is the expensive part and depends only on the rates, so it is
/// separated from applying it.
struct ResampleKernel {
    /// `new_freq / gcd` -- the number of output samples per cycle.
    std::size_t phases = 1;
    /// `orig_freq / gcd` -- the input stride between cycles.
    std::size_t stride = 1;
    /// Taps per phase: `2 * width + stride`.
    std::size_t taps = 0;
    /// How far the kernel reaches back before its first tap.
    std::size_t width = 0;
    /// `phases * taps`, phase-major.
    std::vector<float> data;

    [[nodiscard]] std::span<const float> phase(std::size_t p) const {
        return std::span<const float>(data).subspan(p * taps, taps);
    }
};

/// Build the polyphase filter bank for `orig_freq` to `new_freq`.
[[nodiscard]] ResampleKernel make_resample_kernel(std::size_t orig_freq, std::size_t new_freq,
                                                  std::size_t lowpass_filter_width = 6,
                                                  double rolloff = 0.99);

/// Resample `input` from `orig_freq` to `new_freq`.
///
/// Returns exactly `ceil(new_freq * n / orig_freq)` samples. Equal rates return
/// the input untouched, which is not merely an optimisation -- it is what
/// torchaudio does, and passing 1:1 audio through the filter would soften it
/// for no reason.
[[nodiscard]] std::vector<float> resample(std::span<const float> input, std::size_t orig_freq,
                                          std::size_t new_freq);

/// Resample with a kernel built ahead of time, for callers converting many
/// clips between the same pair of rates.
[[nodiscard]] std::vector<float> resample_with(std::span<const float> input,
                                               const ResampleKernel& kernel);

}  // namespace rt

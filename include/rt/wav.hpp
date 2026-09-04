#pragma once

// =============================================================================
// WAV writer -- 16-bit PCM
// =============================================================================
//
// The last step of a text-to-speech pipeline. A decoder hands back f32 samples
// in [-1, 1] and something has to make a file out of them.
//
// RIFF is three chunks and no compression:
//
//   "RIFF" <u32 total-8> "WAVE"
//   "fmt " <u32 16> <u16 format=1> <u16 channels> <u32 rate>
//          <u32 byte-rate> <u16 block-align> <u16 bits=16>
//   "data" <u32 bytes> <interleaved little-endian i16 samples>
//
// Every length is 32-bit, which caps a file at 4 GiB -- about 24 hours of mono
// 24 kHz audio, so not a limit speech synthesis reaches.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "rt/result.hpp"

namespace rt {

/// Convert f32 samples to 16-bit PCM.
///
/// Scales by 32767 and clamps. A codec decoder ending in Tanh cannot exceed
/// [-1, 1], so the clamp is only insurance for callers that do not -- but
/// without it, out-of-range input wraps around into loud noise rather than
/// merely distorting, so it is worth the branch.
[[nodiscard]] std::vector<std::int16_t> f32_to_pcm16(std::span<const float> samples);

/// Write mono or interleaved multi-channel f32 samples as a 16-bit PCM WAV.
[[nodiscard]] Result<void> write_wav(const std::string& path, std::span<const float> samples,
                                     std::size_t sample_rate, std::size_t channels = 1);

/// Serialise a WAV to bytes, for tests and for callers that do their own I/O.
[[nodiscard]] std::vector<std::uint8_t> encode_wav(std::span<const float> samples,
                                                   std::size_t sample_rate,
                                                   std::size_t channels = 1);

// ---------------------------------------------------------------------------
// Signal checks
// ---------------------------------------------------------------------------

/// Cheap statistics over a waveform.
///
/// Debugging a synthesis pipeline by ear is miserable, because a wrong RoPE
/// scale, a wrong codebook stride and a wrong convolution padding all sound
/// like the same noise. These numbers separate "this is speech" from "this is
/// broken" without listening: speech at 24 kHz sits near an RMS of 0.03-0.2
/// with a DC offset close to zero, while a decoder fed bad latents saturates
/// its output Tanh and lands nearer 0.5 with most samples at the rails.
struct WaveStats {
    std::size_t n = 0;
    float peak = 0.0f;
    float rms = 0.0f;
    /// Mean sample value; a real recording is close to zero.
    float dc = 0.0f;
    /// Fraction of samples within 1e-3 of +-1, where Tanh has saturated.
    float clipped_fraction = 0.0f;
    /// Fraction of adjacent pairs that change sign -- a rough brightness
    /// proxy. Voiced speech runs well under 0.5; white noise sits near it.
    float zero_crossing_rate = 0.0f;

    /// True when every sample is finite and inside [-1, 1].
    [[nodiscard]] bool in_range() const;

    /// True when the waveform looks like speech rather than noise or silence.
    ///
    /// Deliberately loose -- it is a smoke test meant to catch a pipeline that
    /// is producing garbage, not a quality metric.
    [[nodiscard]] bool looks_like_speech() const;

    /// One-line summary for logs.
    [[nodiscard]] std::string describe() const;
};

[[nodiscard]] WaveStats wave_stats(std::span<const float> samples);

}  // namespace rt

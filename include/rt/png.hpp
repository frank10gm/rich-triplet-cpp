#pragma once

// =============================================================================
// PNG writer -- 8-bit RGB
// =============================================================================
//
// The last step of a text-to-image pipeline, and the counterpart of `wav.hpp`:
// a decoder hands back f32 pixels in [-1, 1] and something has to make a file
// out of them.
//
// PNG is a signature, a header chunk, one or more data chunks and an end
// chunk, each chunk length-prefixed and CRC-32 suffixed:
//
//   \x89 P N G \r \n \x1a \n
//   <u32 len> "IHDR" <u32 width> <u32 height> <u8 depth=8> <u8 colour=2>
//                    <u8 compression=0> <u8 filter=0> <u8 interlace=0> <u32 crc>
//   <u32 len> "IDAT" <zlib stream> <u32 crc>
//   <u32 0>   "IEND" <u32 crc>
//
// ## Why there is no compressor here
//
// The IDAT payload is a zlib stream, which normally means a DEFLATE encoder --
// Huffman tables, a match finder, the whole apparatus. But DEFLATE has a
// *stored* block mode that emits literal bytes with a five-byte header, and a
// zlib stream made entirely of stored blocks is completely valid. Every
// decoder reads it.
//
// So this file needs a zlib header, stored blocks, an Adler-32 over the
// uncompressed data and a CRC-32 per chunk, and no compression algorithm at
// all. A 1024x1024 RGB image lands at about 3 MB rather than the 1.5 MB a real
// encoder would manage. That is the right trade for a project whose point is
// that nothing is imported -- and the alternative, linking the system zlib,
// would be the first third-party dependency in the codebase.
//
// Each scanline is prefixed with filter type 0 (None), which is what makes the
// stored-block trick work: any other filter would need the reconstruction to
// be inverted at read time, which is fine, but choosing filters well is most
// of what a PNG encoder does and none of it is free.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "rt/mat.hpp"
#include "rt/result.hpp"

namespace rt {

/// Convert decoder output to 8-bit RGB.
///
/// `pixels` is [h*w, 3] in the spatial-major layout `conv2d.hpp` uses, holding
/// roughly [-1, 1]. The mapping is `x / 2 + 0.5`, then clamp, then scale by
/// 255 and round.
///
/// The order matters: clamping to [-1, 1] *before* the shift and clamping to
/// [0, 1] after are the same operation, but clamping to [0, 1] before the shift
/// crushes every negative value to mid-grey and flattens the shadows across the
/// whole image. It looks like a slightly hazy render rather than a bug.
[[nodiscard]] std::vector<std::uint8_t> f32_to_rgb8(const Mat& pixels);

/// Serialise 8-bit RGB samples as a PNG.
///
/// `rgb` is `h * w * 3` bytes, row-major.
[[nodiscard]] std::vector<std::uint8_t> encode_png(std::span<const std::uint8_t> rgb,
                                                   std::size_t width, std::size_t height);

/// Write a decoder's [h*w, 3] output to a PNG file.
[[nodiscard]] Result<void> write_png(const std::string& path, const Mat& pixels,
                                     std::size_t width, std::size_t height);

// ---------------------------------------------------------------------------
// Checksums -- exposed for testing
// ---------------------------------------------------------------------------

[[nodiscard]] std::uint32_t crc32(std::span<const std::uint8_t> bytes, std::uint32_t seed = 0);
[[nodiscard]] std::uint32_t adler32(std::span<const std::uint8_t> bytes);

// ---------------------------------------------------------------------------
// Image checks
// ---------------------------------------------------------------------------

/// Cheap statistics over a decoded image.
///
/// The image counterpart of `wave_stats`, and it exists for the same reason:
/// a wrong latent scale, a wrong GroupNorm epsilon and a wrong patch order all
/// produce something image-shaped, and looking at each one is slow. These
/// numbers separate "this is a picture" from "this is noise" without opening
/// the file.
struct ImageStats {
    std::size_t n = 0;
    float mean = 0.0f;
    float rms = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
    /// Fraction of samples outside [-1, 1] before clamping.
    float out_of_range_fraction = 0.0f;
    /// Mean absolute difference between horizontally adjacent pixels.
    ///
    /// The useful one. A photograph is locally smooth and lands well under
    /// 0.1; uniform noise sits near 0.5. A VAE fed a latent that was not
    /// rescaled produces something in between, and this catches it.
    float neighbour_delta = 0.0f;

    /// True when every value is finite.
    bool finite = true;

    /// True when the image looks like a picture rather than noise or a flat
    /// field. Deliberately loose -- a smoke test, not a quality metric.
    [[nodiscard]] bool looks_like_image() const;

    /// One-line summary for logs.
    [[nodiscard]] std::string describe() const;
};

[[nodiscard]] ImageStats image_stats(const Mat& pixels, std::size_t width, std::size_t height);

}  // namespace rt

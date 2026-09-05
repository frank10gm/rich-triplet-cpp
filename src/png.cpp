#include "rt/png.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "rt/mat.hpp"
#include "rt/result.hpp"

namespace rt {

namespace {

void push_be32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>(v >> 24));
    out.push_back(static_cast<std::uint8_t>(v >> 16));
    out.push_back(static_cast<std::uint8_t>(v >> 8));
    out.push_back(static_cast<std::uint8_t>(v));
}

/// Append a length-prefixed, CRC-suffixed chunk.
///
/// The CRC covers the type and the payload but **not** the length, which is
/// the one detail of the format that a reader will reject silently-looking
/// files over.
void push_chunk(std::vector<std::uint8_t>& out, const char (&type)[5],
                std::span<const std::uint8_t> payload) {
    push_be32(out, static_cast<std::uint32_t>(payload.size()));
    const std::size_t crc_start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), payload.begin(), payload.end());
    const std::uint32_t crc =
        crc32(std::span<const std::uint8_t>(out.data() + crc_start, out.size() - crc_start));
    push_be32(out, crc);
}

/// Wrap raw bytes in a zlib stream made of stored DEFLATE blocks.
///
/// Each block is `<u8 final> <u16 len> <u16 ~len> <len bytes>`, capped at
/// 65535 bytes, and the stream ends with an Adler-32 of everything stored.
[[nodiscard]] std::vector<std::uint8_t> zlib_stored(std::span<const std::uint8_t> raw) {
    std::vector<std::uint8_t> out;
    out.reserve(raw.size() + raw.size() / 65535 * 5 + 16);

    // CMF = deflate, 32 KiB window; FLG chosen so the pair is a multiple of 31.
    out.push_back(0x78);
    out.push_back(0x01);

    constexpr std::size_t kMaxBlock = 65535;
    std::size_t offset = 0;
    do {
        const std::size_t len = std::min(kMaxBlock, raw.size() - offset);
        const bool final = offset + len >= raw.size();
        out.push_back(final ? 1 : 0);
        out.push_back(static_cast<std::uint8_t>(len));
        out.push_back(static_cast<std::uint8_t>(len >> 8));
        const std::uint16_t nlen = static_cast<std::uint16_t>(~static_cast<std::uint16_t>(len));
        out.push_back(static_cast<std::uint8_t>(nlen));
        out.push_back(static_cast<std::uint8_t>(nlen >> 8));
        out.insert(out.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                   raw.begin() + static_cast<std::ptrdiff_t>(offset + len));
        offset += len;
    } while (offset < raw.size());

    push_be32(out, adler32(raw));
    return out;
}

}  // namespace

// =============================================================================
// Checksums
// =============================================================================

std::uint32_t crc32(std::span<const std::uint8_t> bytes, std::uint32_t seed) {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[n] = c;
        }
        return t;
    }();

    std::uint32_t c = seed ^ 0xFFFFFFFFu;
    for (const std::uint8_t b : bytes) {
        c = table[(c ^ b) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

std::uint32_t adler32(std::span<const std::uint8_t> bytes) {
    constexpr std::uint32_t kMod = 65521;
    std::uint32_t a = 1;
    std::uint32_t b = 0;
    // Reduce every 5552 bytes, the largest run that cannot overflow 32 bits.
    std::size_t i = 0;
    while (i < bytes.size()) {
        const std::size_t chunk = std::min<std::size_t>(5552, bytes.size() - i);
        for (std::size_t k = 0; k < chunk; ++k) {
            a += bytes[i + k];
            b += a;
        }
        a %= kMod;
        b %= kMod;
        i += chunk;
    }
    return (b << 16) | a;
}

// =============================================================================
// Encoding
// =============================================================================

std::vector<std::uint8_t> f32_to_rgb8(const Mat& pixels) {
    std::vector<std::uint8_t> out;
    out.reserve(pixels.rows * 3);
    for (std::size_t p = 0; p < pixels.rows; ++p) {
        for (std::size_t c = 0; c < 3; ++c) {
            // Shift into [0, 1] first, clamp second. Clamping the raw value to
            // [0, 1] instead would fold every negative pixel onto black rather
            // than onto mid-grey.
            const float v = pixels.at(p, c) * 0.5f + 0.5f;
            // NaN has to be caught by name: it compares false against both
            // bounds, so `std::clamp` passes it straight through and the cast
            // that follows is undefined. An infinity needs no special case --
            // clamping saturates it to the right end of the range.
            const float clamped = std::isnan(v) ? 0.0f : std::clamp(v, 0.0f, 1.0f);
            out.push_back(static_cast<std::uint8_t>(std::lround(clamped * 255.0f)));
        }
    }
    return out;
}

std::vector<std::uint8_t> encode_png(std::span<const std::uint8_t> rgb, std::size_t width,
                                     std::size_t height) {
    std::vector<std::uint8_t> out{0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};

    std::vector<std::uint8_t> ihdr;
    push_be32(ihdr, static_cast<std::uint32_t>(width));
    push_be32(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(2);  // colour type 2 = truecolour RGB
    ihdr.push_back(0);  // compression: deflate
    ihdr.push_back(0);  // filter method 0
    ihdr.push_back(0);  // no interlace
    push_chunk(out, "IHDR", ihdr);

    // Filter byte 0 (None) in front of every scanline.
    std::vector<std::uint8_t> raw;
    raw.reserve(height * (width * 3 + 1));
    for (std::size_t y = 0; y < height; ++y) {
        raw.push_back(0);
        const std::size_t base = y * width * 3;
        if (base + width * 3 <= rgb.size()) {
            raw.insert(raw.end(), rgb.begin() + static_cast<std::ptrdiff_t>(base),
                       rgb.begin() + static_cast<std::ptrdiff_t>(base + width * 3));
        } else {
            raw.insert(raw.end(), width * 3, 0);
        }
    }

    const std::vector<std::uint8_t> stream = zlib_stored(raw);
    push_chunk(out, "IDAT", stream);
    push_chunk(out, "IEND", {});
    return out;
}

Result<void> write_png(const std::string& path, const Mat& pixels, std::size_t width,
                       std::size_t height) {
    if (pixels.cols != 3) {
        return err("write_png: expected 3 channels, got " + std::to_string(pixels.cols));
    }
    if (pixels.rows != width * height) {
        return err("write_png: rows " + std::to_string(pixels.rows) + " != width*height " +
                   std::to_string(width * height));
    }

    const std::vector<std::uint8_t> rgb = f32_to_rgb8(pixels);
    const std::vector<std::uint8_t> bytes = encode_png(rgb, width, height);

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return err("write_png: cannot open " + path);
    }
    const std::size_t written = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    if (written != bytes.size()) {
        return err("write_png: short write to " + path);
    }
    return {};
}

// =============================================================================
// Image checks
// =============================================================================

bool ImageStats::looks_like_image() const {
    if (!finite || n == 0) {
        return false;
    }
    // Noise sits near 0.5; a flat field sits near 0. A picture is in between.
    return neighbour_delta > 0.0005f && neighbour_delta < 0.35f && rms > 0.01f &&
           out_of_range_fraction < 0.2f;
}

std::string ImageStats::describe() const {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "n=%zu mean=%.4f rms=%.4f range=[%.3f, %.3f] oor=%.3f neighbour=%.4f %s", n,
                  static_cast<double>(mean), static_cast<double>(rms), static_cast<double>(min),
                  static_cast<double>(max), static_cast<double>(out_of_range_fraction),
                  static_cast<double>(neighbour_delta),
                  looks_like_image() ? "(image)" : "(not image-like)");
    return buf;
}

ImageStats image_stats(const Mat& pixels, std::size_t width, std::size_t height) {
    ImageStats s;
    s.n = pixels.data.size();
    if (s.n == 0) {
        s.finite = true;
        return s;
    }

    double sum = 0.0;
    double sq = 0.0;
    std::size_t oor = 0;
    s.min = pixels.data[0];
    s.max = pixels.data[0];
    for (const float v : pixels.data) {
        if (!std::isfinite(v)) {
            s.finite = false;
            continue;
        }
        sum += v;
        sq += static_cast<double>(v) * v;
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
        if (v < -1.0f || v > 1.0f) {
            ++oor;
        }
    }
    const auto n = static_cast<double>(s.n);
    s.mean = static_cast<float>(sum / n);
    s.rms = static_cast<float>(std::sqrt(sq / n));
    s.out_of_range_fraction = static_cast<float>(static_cast<double>(oor) / n);

    // Horizontal neighbours only. The vertical direction says the same thing
    // and costs a second pass over a tensor that may be hundreds of megabytes.
    if (width >= 2 && height >= 1 && pixels.rows == width * height) {
        double delta = 0.0;
        std::size_t pairs = 0;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x + 1 < width; ++x) {
                const std::size_t a = y * width + x;
                for (std::size_t c = 0; c < pixels.cols; ++c) {
                    delta += std::fabs(pixels.at(a, c) - pixels.at(a + 1, c));
                    ++pairs;
                }
            }
        }
        if (pairs > 0) {
            s.neighbour_delta = static_cast<float>(delta / static_cast<double>(pairs));
        }
    }
    return s;
}

}  // namespace rt

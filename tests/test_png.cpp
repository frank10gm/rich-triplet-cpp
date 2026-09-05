#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rt/mat.hpp"
#include "rt/png.hpp"

using namespace rt;

namespace {

[[nodiscard]] std::uint32_t be32(std::span<const std::uint8_t> b, std::size_t at) {
    return (static_cast<std::uint32_t>(b[at]) << 24) | (static_cast<std::uint32_t>(b[at + 1]) << 16) |
           (static_cast<std::uint32_t>(b[at + 2]) << 8) | static_cast<std::uint32_t>(b[at + 3]);
}

struct Chunk {
    std::string type;
    std::vector<std::uint8_t> payload;
};

/// Walk the chunk list, checking every CRC as it goes.
[[nodiscard]] std::vector<Chunk> parse_png(std::span<const std::uint8_t> bytes) {
    static const std::uint8_t kSig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    REQUIRE(bytes.size() > 8);
    REQUIRE(std::equal(kSig, kSig + 8, bytes.begin()));

    std::vector<Chunk> chunks;
    std::size_t at = 8;
    while (at + 12 <= bytes.size()) {
        const std::uint32_t len = be32(bytes, at);
        REQUIRE(at + 12 + len <= bytes.size());
        Chunk c;
        c.type.assign(reinterpret_cast<const char*>(bytes.data() + at + 4), 4);
        c.payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(at + 8),
                         bytes.begin() + static_cast<std::ptrdiff_t>(at + 8 + len));
        const std::uint32_t stated = be32(bytes, at + 8 + len);
        const std::uint32_t actual =
            crc32(std::span<const std::uint8_t>(bytes.data() + at + 4, 4 + len));
        REQUIRE(stated == actual);
        chunks.push_back(std::move(c));
        at += 12 + len;
    }
    REQUIRE(at == bytes.size());
    return chunks;
}

/// Undo `zlib_stored`: the decoder side of the only DEFLATE mode this writes.
[[nodiscard]] std::vector<std::uint8_t> inflate_stored(std::span<const std::uint8_t> stream) {
    REQUIRE(stream.size() >= 6);
    REQUIRE(stream[0] == 0x78);
    // The CMF/FLG pair must be a multiple of 31 or a real decoder rejects it.
    REQUIRE((static_cast<unsigned>(stream[0]) * 256 + stream[1]) % 31 == 0);

    std::vector<std::uint8_t> out;
    std::size_t at = 2;
    bool final = false;
    while (!final) {
        REQUIRE(at + 5 <= stream.size());
        const std::uint8_t header = stream[at];
        REQUIRE((header & 0x06) == 0);  // stored block
        final = (header & 1) != 0;
        const std::size_t len =
            static_cast<std::size_t>(stream[at + 1]) | (static_cast<std::size_t>(stream[at + 2]) << 8);
        const std::size_t nlen =
            static_cast<std::size_t>(stream[at + 3]) | (static_cast<std::size_t>(stream[at + 4]) << 8);
        REQUIRE((len ^ 0xFFFF) == nlen);
        at += 5;
        REQUIRE(at + len <= stream.size());
        out.insert(out.end(), stream.begin() + static_cast<std::ptrdiff_t>(at),
                   stream.begin() + static_cast<std::ptrdiff_t>(at + len));
        at += len;
    }
    REQUIRE(at + 4 == stream.size());
    const std::uint32_t stated = be32(stream, at);
    REQUIRE(stated == adler32(out));
    return out;
}

}  // namespace

// =============================================================================
// Checksums
// =============================================================================

TEST_CASE("crc32 matches the published PNG test vectors", "[png]") {
    // The IEND chunk's CRC is fixed and widely quoted: 0xAE426082 over "IEND".
    const std::uint8_t iend[4] = {'I', 'E', 'N', 'D'};
    REQUIRE(crc32(std::span<const std::uint8_t>(iend, 4)) == 0xAE426082u);

    const std::string check = "123456789";
    REQUIRE(crc32(std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(check.data()), check.size())) == 0xCBF43926u);
}

TEST_CASE("adler32 matches its definition", "[png]") {
    REQUIRE(adler32({}) == 1u);
    const std::string wiki = "Wikipedia";
    REQUIRE(adler32(std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(wiki.data()), wiki.size())) == 0x11E60398u);
}

TEST_CASE("adler32 reduction survives more than one 5552-byte run", "[png]") {
    // The modular reduction happens per chunk; a run longer than one chunk is
    // the only thing that exercises the carry between them.
    std::vector<std::uint8_t> big(20000);
    for (std::size_t i = 0; i < big.size(); ++i) {
        big[i] = static_cast<std::uint8_t>(i * 7 + 3);
    }
    std::uint32_t a = 1;
    std::uint32_t b = 0;
    for (const std::uint8_t v : big) {
        a = (a + v) % 65521;
        b = (b + a) % 65521;
    }
    REQUIRE(adler32(big) == ((b << 16) | a));
}

// =============================================================================
// Pixel conversion
// =============================================================================

TEST_CASE("f32_to_rgb8 maps [-1, 1] onto the full byte range", "[png]") {
    Mat p = Mat::zeros(3, 3);
    p.row_mut(0)[0] = -1.0f;
    p.row_mut(0)[1] = 0.0f;
    p.row_mut(0)[2] = 1.0f;
    p.row_mut(1)[0] = -2.0f;  // clamps
    p.row_mut(1)[1] = 2.0f;   // clamps
    p.row_mut(1)[2] = 0.5f;
    p.row_mut(2)[0] = -0.5f;
    p.row_mut(2)[1] = 0.0f;
    p.row_mut(2)[2] = 0.0f;

    const std::vector<std::uint8_t> rgb = f32_to_rgb8(p);
    REQUIRE(rgb.size() == 9);
    REQUIRE(rgb[0] == 0);
    REQUIRE(rgb[1] == 128);
    REQUIRE(rgb[2] == 255);
    REQUIRE(rgb[3] == 0);
    REQUIRE(rgb[4] == 255);
    REQUIRE(rgb[5] == 191);
    REQUIRE(rgb[6] == 64);
}

TEST_CASE("f32_to_rgb8 shifts before it clamps", "[png]") {
    // A negative pixel must land in the lower half of the range, not on black.
    // Clamping to [0, 1] first would map every one of these to 0.
    Mat p = Mat::zeros(1, 3);
    p.row_mut(0)[0] = -0.25f;
    p.row_mut(0)[1] = -0.5f;
    p.row_mut(0)[2] = -0.75f;
    const std::vector<std::uint8_t> rgb = f32_to_rgb8(p);
    REQUIRE(rgb[0] == 96);
    REQUIRE(rgb[1] == 64);
    REQUIRE(rgb[2] == 32);
}

TEST_CASE("f32_to_rgb8 saturates infinities and blacks out NaN", "[png]") {
    Mat p = Mat::zeros(1, 3);
    p.row_mut(0)[0] = std::numeric_limits<float>::quiet_NaN();
    p.row_mut(0)[1] = std::numeric_limits<float>::infinity();
    p.row_mut(0)[2] = -std::numeric_limits<float>::infinity();
    const std::vector<std::uint8_t> rgb = f32_to_rgb8(p);
    REQUIRE(rgb[0] == 0);
    REQUIRE(rgb[1] == 255);
    REQUIRE(rgb[2] == 0);
}

// =============================================================================
// Container
// =============================================================================

TEST_CASE("encode_png produces a well-formed file", "[png]") {
    const std::size_t w = 5;
    const std::size_t h = 3;
    std::vector<std::uint8_t> rgb(w * h * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = static_cast<std::uint8_t>(i * 13 + 7);
    }

    const std::vector<std::uint8_t> png = encode_png(rgb, w, h);
    const std::vector<Chunk> chunks = parse_png(png);

    REQUIRE(chunks.size() == 3);
    REQUIRE(chunks[0].type == "IHDR");
    REQUIRE(chunks[1].type == "IDAT");
    REQUIRE(chunks[2].type == "IEND");
    REQUIRE(chunks[2].payload.empty());

    const std::vector<std::uint8_t>& ihdr = chunks[0].payload;
    REQUIRE(ihdr.size() == 13);
    REQUIRE(be32(ihdr, 0) == w);
    REQUIRE(be32(ihdr, 4) == h);
    REQUIRE(ihdr[8] == 8);   // bit depth
    REQUIRE(ihdr[9] == 2);   // truecolour
    REQUIRE(ihdr[10] == 0);  // deflate
    REQUIRE(ihdr[11] == 0);  // filter method
    REQUIRE(ihdr[12] == 0);  // no interlace
}

TEST_CASE("the IDAT stream inflates back to the original scanlines", "[png]") {
    const std::size_t w = 7;
    const std::size_t h = 4;
    std::vector<std::uint8_t> rgb(w * h * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = static_cast<std::uint8_t>(i * 31 + 11);
    }

    const std::vector<std::uint8_t> png = encode_png(rgb, w, h);
    const std::vector<Chunk> chunks = parse_png(png);
    const std::vector<std::uint8_t> raw = inflate_stored(chunks[1].payload);

    REQUIRE(raw.size() == h * (w * 3 + 1));
    for (std::size_t y = 0; y < h; ++y) {
        const std::size_t at = y * (w * 3 + 1);
        REQUIRE(raw[at] == 0);  // filter type None
        for (std::size_t i = 0; i < w * 3; ++i) {
            REQUIRE(raw[at + 1 + i] == rgb[y * w * 3 + i]);
        }
    }
}

TEST_CASE("an image larger than one stored block still round-trips", "[png]") {
    // 65535 bytes is the stored-block cap; this crosses it several times, which
    // is the only way to exercise the non-final block header.
    const std::size_t w = 300;
    const std::size_t h = 300;
    std::vector<std::uint8_t> rgb(w * h * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = static_cast<std::uint8_t>((i * 97 + i / 251) & 0xFF);
    }
    REQUIRE(h * (w * 3 + 1) > 4 * 65535);

    const std::vector<std::uint8_t> png = encode_png(rgb, w, h);
    const std::vector<Chunk> chunks = parse_png(png);
    const std::vector<std::uint8_t> raw = inflate_stored(chunks[1].payload);

    REQUIRE(raw.size() == h * (w * 3 + 1));
    for (std::size_t y = 0; y < h; ++y) {
        const std::size_t at = y * (w * 3 + 1);
        REQUIRE(raw[at] == 0);
        for (std::size_t i = 0; i < w * 3; ++i) {
            REQUIRE(raw[at + 1 + i] == rgb[y * w * 3 + i]);
        }
    }
}

TEST_CASE("a one-pixel image is still a valid PNG", "[png]") {
    const std::vector<std::uint8_t> rgb{10, 20, 30};
    const std::vector<std::uint8_t> png = encode_png(rgb, 1, 1);
    const std::vector<Chunk> chunks = parse_png(png);
    const std::vector<std::uint8_t> raw = inflate_stored(chunks[1].payload);
    REQUIRE(raw == std::vector<std::uint8_t>{0, 10, 20, 30});
}

// =============================================================================
// Image statistics
// =============================================================================

TEST_CASE("image_stats separates a picture from noise and from a flat field", "[png]") {
    const std::size_t w = 32;
    const std::size_t h = 32;

    // A smooth gradient with some structure: locally smooth, globally varied.
    Mat picture = Mat::zeros(w * h, 3);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            const float v = std::sin(static_cast<float>(x) * 0.2f) *
                            std::cos(static_cast<float>(y) * 0.15f) * 0.7f;
            for (std::size_t c = 0; c < 3; ++c) {
                picture.at_mut(y * w + x, c) = v;
            }
        }
    }
    const ImageStats ps = image_stats(picture, w, h);
    REQUIRE(ps.finite);
    REQUIRE(ps.looks_like_image());

    // Uncorrelated noise: neighbours differ by about a third of the range.
    Mat noise = Mat::zeros(w * h, 3);
    std::uint32_t state = 12345;
    for (float& v : noise.data) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>(state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    }
    const ImageStats ns = image_stats(noise, w, h);
    REQUIRE(ns.neighbour_delta > ps.neighbour_delta * 4.0f);
    REQUIRE_FALSE(ns.looks_like_image());

    // A flat field has no neighbour delta at all.
    const Mat flat = Mat::zeros(w * h, 3);
    REQUIRE_FALSE(image_stats(flat, w, h).looks_like_image());
}

TEST_CASE("image_stats reports non-finite values", "[png]") {
    Mat m = Mat::zeros(4, 3);
    m.at_mut(2, 1) = std::numeric_limits<float>::quiet_NaN();
    const ImageStats s = image_stats(m, 2, 2);
    REQUIRE_FALSE(s.finite);
    REQUIRE_FALSE(s.looks_like_image());
}

TEST_CASE("image_stats counts out-of-range samples", "[png]") {
    Mat m = Mat::zeros(4, 3);
    m.at_mut(0, 0) = 3.0f;
    m.at_mut(1, 1) = -5.0f;
    const ImageStats s = image_stats(m, 2, 2);
    REQUIRE(s.out_of_range_fraction > 0.16f);
    REQUIRE(s.out_of_range_fraction < 0.17f);
    REQUIRE(s.max > 2.9f);
    REQUIRE(s.min < -4.9f);
}

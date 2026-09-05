#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "rt/wav.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

namespace {

[[nodiscard]] std::string tag_at(const std::vector<std::uint8_t>& b, std::size_t off) {
    return std::string(reinterpret_cast<const char*>(b.data() + off), 4);
}

[[nodiscard]] std::uint32_t u32_at(const std::vector<std::uint8_t>& b, std::size_t off) {
    return static_cast<std::uint32_t>(b[off]) | (static_cast<std::uint32_t>(b[off + 1]) << 8) |
           (static_cast<std::uint32_t>(b[off + 2]) << 16) |
           (static_cast<std::uint32_t>(b[off + 3]) << 24);
}

[[nodiscard]] std::uint16_t u16_at(const std::vector<std::uint8_t>& b, std::size_t off) {
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(b[off]) |
                                      static_cast<std::uint16_t>(b[off + 1] << 8));
}

/// A 220 Hz sine at 24 kHz -- a stand-in for voiced speech.
[[nodiscard]] std::vector<float> tone(std::size_t n, float freq, float amplitude) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / 24000.0f;
        out[i] = amplitude * std::sin(2.0f * std::numbers::pi_v<float> * freq * t);
    }
    return out;
}

}  // namespace

// =============================================================================
// PCM conversion
// =============================================================================

TEST_CASE("f32_to_pcm16 scales by 32767", "[wav]") {
    const std::vector<float> in{0.0f, 1.0f, -1.0f, 0.5f};
    const std::vector<std::int16_t> pcm = f32_to_pcm16(in);
    REQUIRE(pcm.size() == 4);
    REQUIRE(pcm[0] == 0);
    REQUIRE(pcm[1] == 32767);
    REQUIRE(pcm[2] == -32767);
    REQUIRE(pcm[3] == 16384);  // lround(0.5 * 32767)
}

TEST_CASE("f32_to_pcm16 clamps instead of wrapping", "[wav]") {
    // Wrapping would turn a loud sample into the opposite rail, which is an
    // audible click rather than mild distortion.
    const std::vector<float> in{2.0f, -2.0f, 1e9f};
    const std::vector<std::int16_t> pcm = f32_to_pcm16(in);
    REQUIRE(pcm[0] == 32767);
    REQUIRE(pcm[1] == -32767);
    REQUIRE(pcm[2] == 32767);
}

TEST_CASE("f32_to_pcm16 maps non-finite samples to silence", "[wav]") {
    const std::vector<float> in{std::nanf(""), std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()};
    const std::vector<std::int16_t> pcm = f32_to_pcm16(in);
    REQUIRE(pcm[0] == 0);
    REQUIRE(pcm[1] == 32767);
    REQUIRE(pcm[2] == -32767);
}

// =============================================================================
// RIFF structure
// =============================================================================

TEST_CASE("encode_wav writes a valid 44-byte header", "[wav]") {
    const std::vector<float> samples(100, 0.0f);
    const std::vector<std::uint8_t> b = encode_wav(samples, 24000, 1);

    REQUIRE(b.size() == 44 + 200);
    REQUIRE(tag_at(b, 0) == "RIFF");
    REQUIRE(u32_at(b, 4) == 36 + 200);
    REQUIRE(tag_at(b, 8) == "WAVE");
    REQUIRE(tag_at(b, 12) == "fmt ");
    REQUIRE(u32_at(b, 16) == 16);
    REQUIRE(u16_at(b, 20) == 1);      // PCM
    REQUIRE(u16_at(b, 22) == 1);      // mono
    REQUIRE(u32_at(b, 24) == 24000);  // sample rate
    REQUIRE(u32_at(b, 28) == 48000);  // byte rate = 24000 * 2
    REQUIRE(u16_at(b, 32) == 2);      // block align
    REQUIRE(u16_at(b, 34) == 16);     // bits
    REQUIRE(tag_at(b, 36) == "data");
    REQUIRE(u32_at(b, 40) == 200);
}

TEST_CASE("encode_wav sets the byte rate from the channel count", "[wav]") {
    const std::vector<float> samples(8, 0.0f);
    const std::vector<std::uint8_t> b = encode_wav(samples, 44100, 2);
    REQUIRE(u16_at(b, 22) == 2);        // stereo
    REQUIRE(u16_at(b, 32) == 4);        // block align = 2ch * 2 bytes
    REQUIRE(u32_at(b, 28) == 176400);   // 44100 * 4
}

TEST_CASE("encode_wav stores samples little-endian after the header", "[wav]") {
    const std::vector<float> samples{1.0f, -1.0f};
    const std::vector<std::uint8_t> b = encode_wav(samples, 24000, 1);
    REQUIRE(b[44] == 0xff);
    REQUIRE(b[45] == 0x7f);  // 32767
    REQUIRE(b[46] == 0x01);
    REQUIRE(b[47] == 0x80);  // -32767
}

TEST_CASE("write_wav produces a file that reads back byte for byte", "[wav]") {
    const std::string path = "/tmp/rt_wav_roundtrip.wav";
    const std::vector<float> samples = tone(480, 220.0f, 0.25f);
    REQUIRE(write_wav(path, samples, 24000, 1).has_value());

    std::FILE* f = std::fopen(path.c_str(), "rb");
    REQUIRE(f != nullptr);
    std::vector<std::uint8_t> read_back(44 + 960);
    const std::size_t got = std::fread(read_back.data(), 1, read_back.size(), f);
    // Nothing should follow the data chunk.
    std::uint8_t extra = 0;
    const std::size_t trailing = std::fread(&extra, 1, 1, f);
    std::fclose(f);
    std::remove(path.c_str());

    REQUIRE(got == read_back.size());
    REQUIRE(trailing == 0);
    REQUIRE(read_back == encode_wav(samples, 24000, 1));
}

TEST_CASE("write_wav rejects impossible parameters", "[wav]") {
    const std::vector<float> samples(3, 0.0f);
    REQUIRE(!write_wav("/tmp/rt_wav_bad.wav", samples, 24000, 0).has_value());
    REQUIRE(!write_wav("/tmp/rt_wav_bad.wav", samples, 0, 1).has_value());
    // 3 samples cannot be split into 2 channels.
    REQUIRE(!write_wav("/tmp/rt_wav_bad.wav", samples, 24000, 2).has_value());
}

TEST_CASE("write_wav reports an unwritable path", "[wav]") {
    const std::vector<float> samples(4, 0.0f);
    const Result<void> r = write_wav("/nonexistent-dir-rt/out.wav", samples, 24000, 1);
    REQUIRE(!r.has_value());
    REQUIRE(r.error().find("cannot open") != std::string::npos);
}

// =============================================================================
// Signal checks
// =============================================================================

TEST_CASE("wave_stats measures peak, rms and dc", "[wav]") {
    // A full-scale sine has rms 1/sqrt(2) and no dc offset.
    const std::vector<float> samples = tone(24000, 100.0f, 1.0f);
    const WaveStats s = wave_stats(samples);
    REQUIRE(s.n == 24000);
    REQUIRE(s.peak > 0.99f);
    REQUIRE(std::fabs(s.rms - 0.7071f) < 0.01f);
    REQUIRE(std::fabs(s.dc) < 0.01f);
}

TEST_CASE("wave_stats counts zero crossings", "[wav]") {
    // A 1 kHz tone at 24 kHz crosses zero twice per cycle: 2000 crossings a
    // second out of 24000 samples.
    const std::vector<float> samples = tone(24000, 1000.0f, 0.5f);
    const WaveStats s = wave_stats(samples);
    REQUIRE(std::fabs(s.zero_crossing_rate - 2000.0f / 24000.0f) < 0.01f);
}

TEST_CASE("wave_stats flags a non-finite waveform", "[wav]") {
    std::vector<float> samples(100, 0.1f);
    samples[50] = std::nanf("");
    const WaveStats s = wave_stats(samples);
    REQUIRE(!s.in_range());
}

TEST_CASE("looks_like_speech accepts a speech-level tone", "[wav]") {
    const std::vector<float> samples = tone(24000, 220.0f, 0.15f);
    const WaveStats s = wave_stats(samples);
    INFO(s.describe());
    REQUIRE(s.looks_like_speech());
}

TEST_CASE("looks_like_speech rejects silence", "[wav]") {
    const std::vector<float> samples(24000, 0.0f);
    REQUIRE(!wave_stats(samples).looks_like_speech());
}

TEST_CASE("looks_like_speech rejects a saturated waveform", "[wav]") {
    // What a decoder fed bad latents produces: the output Tanh pinned to its
    // rails. This is the failure mode the check exists to catch.
    std::vector<float> samples(24000);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        samples[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }
    const WaveStats s = wave_stats(samples);
    INFO(s.describe());
    REQUIRE(!s.looks_like_speech());
    REQUIRE(s.clipped_fraction > 0.9f);
}

TEST_CASE("looks_like_speech rejects a dc offset", "[wav]") {
    std::vector<float> samples = tone(24000, 220.0f, 0.15f);
    for (float& v : samples) {
        v += 0.3f;
    }
    REQUIRE(!wave_stats(samples).looks_like_speech());
}

TEST_CASE("wave_stats handles an empty waveform", "[wav]") {
    const WaveStats s = wave_stats({});
    REQUIRE(s.n == 0);
    REQUIRE(!s.in_range());
    REQUIRE(!s.looks_like_speech());
}

// =============================================================================
// Reading
// =============================================================================

TEST_CASE("decode_wav round-trips what encode_wav writes", "[wav]") {
    const std::vector<float> samples = tone(2400, 220.0f, 0.5f);
    const std::vector<std::uint8_t> bytes = encode_wav(samples, 24000, 1);

    const Result<WavFile> got = decode_wav(bytes);
    REQUIRE(got.has_value());
    REQUIRE(got->sample_rate == 24000);
    REQUIRE(got->channels == 1);
    REQUIRE(got->samples.size() == samples.size());
    // 16-bit PCM quantizes to steps of 1/32767, so the round trip is lossy by
    // half a step and no more.
    for (std::size_t i = 0; i < samples.size(); ++i) {
        REQUIRE(std::fabs(got->samples[i] - samples[i]) < 1.0f / 32000.0f);
    }
}

TEST_CASE("decode_wav keeps stereo interleaved and averages it on request", "[wav]") {
    // Left is a constant +0.5, right a constant -0.25.
    std::vector<float> stereo;
    for (std::size_t i = 0; i < 100; ++i) {
        stereo.push_back(0.5f);
        stereo.push_back(-0.25f);
    }
    const Result<WavFile> got = decode_wav(encode_wav(stereo, 48000, 2));
    REQUIRE(got.has_value());
    REQUIRE(got->channels == 2);
    REQUIRE(got->frames() == 100);
    REQUIRE(got->samples.size() == 200);

    const std::vector<float> mono = got->mono();
    REQUIRE(mono.size() == 100);
    for (const float v : mono) {
        REQUIRE(approx(v, 0.125f));
    }
}

TEST_CASE("decode_wav skips chunks it does not know", "[wav]") {
    // Real encoders write LIST/INFO metadata between fmt and data. A reader
    // that assumes the 44-byte header this module writes would reject them.
    std::vector<std::uint8_t> bytes = encode_wav(tone(240, 220.0f, 0.3f), 24000, 1);
    const std::vector<std::uint8_t> junk{'L', 'I', 'S', 'T', 6, 0, 0, 0, 'I', 'N', 'F', 'O', 'x', 'y'};

    std::vector<std::uint8_t> spliced(bytes.begin(), bytes.begin() + 36);
    spliced.insert(spliced.end(), junk.begin(), junk.end());
    spliced.insert(spliced.end(), bytes.begin() + 36, bytes.end());
    // The RIFF size field covers everything after itself.
    const std::uint32_t riff = static_cast<std::uint32_t>(spliced.size() - 8);
    for (std::size_t i = 0; i < 4; ++i) {
        spliced[4 + i] = static_cast<std::uint8_t>((riff >> (8 * i)) & 0xff);
    }

    const Result<WavFile> got = decode_wav(spliced);
    REQUIRE(got.has_value());
    REQUIRE(got->samples.size() == 240);
}

TEST_CASE("decode_wav reads 8, 24 and 32-bit PCM and float", "[wav]") {
    // Build each format by hand around the same one-sample payload.
    const auto make = [](std::uint16_t format, std::uint16_t bits,
                         const std::vector<std::uint8_t>& payload) {
        std::vector<std::uint8_t> b;
        const auto tag = [&b](const char* t) {
            for (std::size_t i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(t[i]));
        };
        const auto u32 = [&b](std::uint32_t v) {
            for (std::size_t i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
        };
        const auto u16 = [&b](std::uint16_t v) {
            for (std::size_t i = 0; i < 2; ++i)
                b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
        };
        tag("RIFF");
        u32(static_cast<std::uint32_t>(36 + payload.size()));
        tag("WAVE");
        tag("fmt ");
        u32(16);
        u16(format);
        u16(1);
        u32(24000);
        u32(24000);
        u16(static_cast<std::uint16_t>(bits / 8));
        u16(bits);
        tag("data");
        u32(static_cast<std::uint32_t>(payload.size()));
        b.insert(b.end(), payload.begin(), payload.end());
        return b;
    };

    // 8-bit is unsigned and centred on 128, unlike every other integer depth.
    const Result<WavFile> u8 = decode_wav(make(1, 8, {192}));
    REQUIRE(u8.has_value());
    REQUIRE(approx(u8->samples[0], 0.5f));

    // 24-bit: 0x400000 is a quarter of full scale, and must sign-extend.
    const Result<WavFile> s24 = decode_wav(make(1, 24, {0x00, 0x00, 0x40}));
    REQUIRE(s24.has_value());
    REQUIRE(approx(s24->samples[0], 0.5f));
    const Result<WavFile> neg24 = decode_wav(make(1, 24, {0x00, 0x00, 0xC0}));
    REQUIRE(neg24.has_value());
    REQUIRE(approx(neg24->samples[0], -0.5f));

    const Result<WavFile> s32 = decode_wav(make(1, 32, {0x00, 0x00, 0x00, 0x40}));
    REQUIRE(s32.has_value());
    REQUIRE(approx(s32->samples[0], 0.5f));

    // Format 3 is IEEE float, already in [-1, 1].
    const Result<WavFile> f32 = decode_wav(make(3, 32, {0x00, 0x00, 0x00, 0xBF}));
    REQUIRE(f32.has_value());
    REQUIRE(approx(f32->samples[0], -0.5f));
}

TEST_CASE("decode_wav rejects what it cannot read", "[wav]") {
    REQUIRE_FALSE(decode_wav(std::vector<std::uint8_t>{}).has_value());
    REQUIRE_FALSE(decode_wav(std::vector<std::uint8_t>(64, 0)).has_value());

    // A valid header with a compressed format is a clear error rather than a
    // silent misread of the bytes.
    std::vector<std::uint8_t> bytes = encode_wav(tone(240, 220.0f, 0.3f), 24000, 1);
    bytes[20] = 2;  // WAVE_FORMAT_ADPCM
    REQUIRE_FALSE(decode_wav(bytes).has_value());
}

TEST_CASE("read_wav reports a missing file", "[wav]") {
    REQUIRE_FALSE(read_wav("/nonexistent/path/to/nothing.wav").has_value());
}

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

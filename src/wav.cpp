#include "rt/wav.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace rt {

namespace {

void push_u16(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xff));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
}

void push_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xff));
    }
}

void push_tag(std::vector<std::uint8_t>& out, const char (&tag)[5]) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::uint8_t>(tag[i]));
    }
}

}  // namespace

std::vector<std::int16_t> f32_to_pcm16(std::span<const float> samples) {
    std::vector<std::int16_t> pcm(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        // NaN carries no sign worth honouring, so it becomes silence.
        // Infinities do, so they clamp to the matching rail like any other
        // out-of-range sample.
        const float v = std::isnan(samples[i]) ? 0.0f : samples[i];
        const float scaled = std::clamp(v, -1.0f, 1.0f) * 32767.0f;
        pcm[i] = static_cast<std::int16_t>(std::lround(scaled));
    }
    return pcm;
}

std::vector<std::uint8_t> encode_wav(std::span<const float> samples, std::size_t sample_rate,
                                     std::size_t channels) {
    const std::vector<std::int16_t> pcm = f32_to_pcm16(samples);
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(pcm.size() * 2);
    const std::uint16_t bits = 16;
    const std::uint16_t block_align = static_cast<std::uint16_t>(channels * bits / 8);

    std::vector<std::uint8_t> out;
    out.reserve(44 + pcm.size() * 2);

    push_tag(out, "RIFF");
    push_u32(out, 36 + data_bytes);  // everything after this field
    push_tag(out, "WAVE");

    push_tag(out, "fmt ");
    push_u32(out, 16);  // PCM fmt chunk size
    push_u16(out, 1);   // format 1 == uncompressed PCM
    push_u16(out, static_cast<std::uint16_t>(channels));
    push_u32(out, static_cast<std::uint32_t>(sample_rate));
    push_u32(out, static_cast<std::uint32_t>(sample_rate * block_align));  // byte rate
    push_u16(out, block_align);
    push_u16(out, bits);

    push_tag(out, "data");
    push_u32(out, data_bytes);
    for (const std::int16_t s : pcm) {
        push_u16(out, static_cast<std::uint16_t>(s));
    }
    return out;
}

Result<void> write_wav(const std::string& path, std::span<const float> samples,
                       std::size_t sample_rate, std::size_t channels) {
    if (channels == 0) {
        return err("wav: channel count must be at least 1");
    }
    if (sample_rate == 0) {
        return err("wav: sample rate must be at least 1");
    }
    if (samples.size() % channels != 0) {
        return err("wav: " + std::to_string(samples.size()) + " samples do not divide into " +
                   std::to_string(channels) + " channels");
    }

    const std::vector<std::uint8_t> bytes = encode_wav(samples, sample_rate, channels);

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return err("wav: cannot open " + path + " for writing");
    }
    const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    std::fclose(f);
    if (!ok) {
        return err("wav: short write to " + path);
    }
    return {};
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] std::uint16_t read_u16(std::span<const std::uint8_t> b, std::size_t off) {
    return static_cast<std::uint16_t>(b[off]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(b[off + 1]) << 8);
}

[[nodiscard]] std::uint32_t read_u32(std::span<const std::uint8_t> b, std::size_t off) {
    return static_cast<std::uint32_t>(b[off]) | (static_cast<std::uint32_t>(b[off + 1]) << 8) |
           (static_cast<std::uint32_t>(b[off + 2]) << 16) |
           (static_cast<std::uint32_t>(b[off + 3]) << 24);
}

[[nodiscard]] bool tag_is(std::span<const std::uint8_t> b, std::size_t off, const char* tag) {
    for (std::size_t i = 0; i < 4; ++i) {
        if (b[off + i] != static_cast<std::uint8_t>(tag[i])) {
            return false;
        }
    }
    return true;
}

/// Reassemble a 32-bit float from its little-endian bytes.
[[nodiscard]] float bits_to_f32(std::uint32_t bits) {
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

}  // namespace

std::vector<float> WavFile::mono() const {
    if (channels <= 1) {
        return samples;
    }
    const std::size_t n = frames();
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (std::size_t c = 0; c < channels; ++c) {
            sum += samples[i * channels + c];
        }
        out[i] = sum / static_cast<float>(channels);
    }
    return out;
}

Result<WavFile> decode_wav(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < 12) {
        return err("wav: file is too short to be RIFF");
    }
    if (!tag_is(bytes, 0, "RIFF") || !tag_is(bytes, 8, "WAVE")) {
        return err("wav: not a RIFF/WAVE file");
    }

    std::uint16_t format = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t bits = 0;
    bool have_fmt = false;
    std::span<const std::uint8_t> data;
    bool have_data = false;

    // Walk the chunk list. Anything that is not fmt or data -- LIST, INFO,
    // fact, a trailing ID3 tag -- is skipped rather than rejected, since almost
    // no encoder writes the bare 44-byte header this module does.
    std::size_t off = 12;
    while (off + 8 <= bytes.size()) {
        const std::uint32_t size = read_u32(bytes, off + 4);
        const std::size_t body = off + 8;
        // A chunk claiming more than the file holds is truncation, not a
        // reason to lose what came before it.
        const std::size_t avail = std::min<std::size_t>(size, bytes.size() - body);

        if (tag_is(bytes, off, "fmt ") && avail >= 16) {
            format = read_u16(bytes, body);
            channels = read_u16(bytes, body + 2);
            sample_rate = read_u32(bytes, body + 4);
            bits = read_u16(bytes, body + 14);
            // WAVE_FORMAT_EXTENSIBLE hides the real format in the first two
            // bytes of its extension block; the rest of the GUID is fixed.
            if (format == 0xFFFE && avail >= 26) {
                format = read_u16(bytes, body + 24);
            }
            have_fmt = true;
        } else if (tag_is(bytes, off, "data")) {
            data = bytes.subspan(body, avail);
            have_data = true;
        }

        // Chunks are word-aligned: an odd size is followed by a pad byte.
        off = body + avail + (avail % 2);
    }

    if (!have_fmt) {
        return err("wav: no fmt chunk");
    }
    if (!have_data) {
        return err("wav: no data chunk");
    }
    if (channels == 0) {
        return err("wav: zero channels");
    }
    if (sample_rate == 0) {
        return err("wav: zero sample rate");
    }

    WavFile out;
    out.sample_rate = sample_rate;
    out.channels = channels;

    const auto scale_int = [](std::int64_t v, std::int64_t peak) {
        return static_cast<float>(static_cast<double>(v) / static_cast<double>(peak));
    };

    if (format == 1 && bits == 16) {
        out.samples.reserve(data.size() / 2);
        for (std::size_t i = 0; i + 1 < data.size(); i += 2) {
            out.samples.push_back(
                scale_int(static_cast<std::int16_t>(read_u16(data, i)), 32768));
        }
    } else if (format == 1 && bits == 8) {
        // 8-bit PCM is the odd one out: unsigned, centred on 128.
        out.samples.reserve(data.size());
        for (const std::uint8_t v : data) {
            out.samples.push_back((static_cast<float>(v) - 128.0f) / 128.0f);
        }
    } else if (format == 1 && bits == 24) {
        out.samples.reserve(data.size() / 3);
        for (std::size_t i = 0; i + 2 < data.size(); i += 3) {
            const std::uint32_t raw = static_cast<std::uint32_t>(data[i]) |
                                      (static_cast<std::uint32_t>(data[i + 1]) << 8) |
                                      (static_cast<std::uint32_t>(data[i + 2]) << 16);
            // Sign-extend from 24 bits.
            const std::int32_t v = static_cast<std::int32_t>(raw << 8) >> 8;
            out.samples.push_back(scale_int(v, 8388608));
        }
    } else if (format == 1 && bits == 32) {
        out.samples.reserve(data.size() / 4);
        for (std::size_t i = 0; i + 3 < data.size(); i += 4) {
            out.samples.push_back(
                scale_int(static_cast<std::int32_t>(read_u32(data, i)), 2147483648LL));
        }
    } else if (format == 3 && bits == 32) {
        out.samples.reserve(data.size() / 4);
        for (std::size_t i = 0; i + 3 < data.size(); i += 4) {
            out.samples.push_back(bits_to_f32(read_u32(data, i)));
        }
    } else if (format == 3 && bits == 64) {
        out.samples.reserve(data.size() / 8);
        for (std::size_t i = 0; i + 7 < data.size(); i += 8) {
            std::uint64_t raw = 0;
            for (std::size_t b = 0; b < 8; ++b) {
                raw |= static_cast<std::uint64_t>(data[i + b]) << (8 * b);
            }
            double v = 0.0;
            std::memcpy(&v, &raw, sizeof(v));
            out.samples.push_back(static_cast<float>(v));
        }
    } else {
        return err("wav: unsupported format " + std::to_string(format) + " at " +
                   std::to_string(bits) + " bits");
    }

    // A truncated final frame would leave the channels out of phase for every
    // consumer downstream, so drop it rather than carry it.
    out.samples.resize(out.frames() * channels);
    if (out.samples.empty()) {
        return err("wav: data chunk holds no whole frames");
    }
    return out;
}

Result<WavFile> read_wav(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return err("wav: cannot open " + path + " for reading");
    }
    std::vector<std::uint8_t> bytes;
    std::uint8_t buf[64 * 1024];
    while (const std::size_t n = std::fread(buf, 1, sizeof(buf), f)) {
        bytes.insert(bytes.end(), buf, buf + n);
    }
    const bool failed = std::ferror(f) != 0;
    std::fclose(f);
    if (failed) {
        return err("wav: read error on " + path);
    }
    return decode_wav(bytes);
}

// ---------------------------------------------------------------------------
// Signal checks
// ---------------------------------------------------------------------------

bool WaveStats::in_range() const { return n > 0 && peak <= 1.0f; }

bool WaveStats::looks_like_speech() const {
    if (!in_range()) {
        return false;
    }
    // Quiet enough to be silence, or loud enough that the Tanh is pinned.
    if (rms < 0.005f || rms > 0.45f) {
        return false;
    }
    // A real waveform swings both ways around zero.
    if (std::fabs(dc) > 0.05f) {
        return false;
    }
    // Saturation means the decoder was fed something it could not represent.
    if (clipped_fraction > 0.02f) {
        return false;
    }
    // Speech has voiced stretches; broadband noise crosses zero almost every
    // other sample.
    if (zero_crossing_rate > 0.45f) {
        return false;
    }
    return true;
}

std::string WaveStats::describe() const {
    const auto fmt = [](float v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.4f", static_cast<double>(v));
        return std::string(buf);
    };
    return "n=" + std::to_string(n) + " peak=" + fmt(peak) + " rms=" + fmt(rms) +
           " dc=" + fmt(dc) + " clipped=" + fmt(clipped_fraction) + " zcr=" + fmt(zero_crossing_rate);
}

WaveStats wave_stats(std::span<const float> samples) {
    WaveStats s;
    s.n = samples.size();
    if (samples.empty()) {
        return s;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    std::size_t clipped = 0;
    std::size_t crossings = 0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const float v = samples[i];
        if (!std::isfinite(v)) {
            s.peak = std::numeric_limits<float>::infinity();
            return s;
        }
        sum += v;
        sum_sq += static_cast<double>(v) * v;
        s.peak = std::max(s.peak, std::fabs(v));
        if (std::fabs(v) >= 1.0f - 1e-3f) {
            ++clipped;
        }
        if (i > 0 && ((samples[i - 1] < 0.0f) != (v < 0.0f))) {
            ++crossings;
        }
    }

    const auto count = static_cast<double>(samples.size());
    s.dc = static_cast<float>(sum / count);
    s.rms = static_cast<float>(std::sqrt(sum_sq / count));
    s.clipped_fraction = static_cast<float>(static_cast<double>(clipped) / count);
    s.zero_crossing_rate =
        samples.size() > 1
            ? static_cast<float>(static_cast<double>(crossings) / (count - 1.0))
            : 0.0f;
    return s;
}

}  // namespace rt

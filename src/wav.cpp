#include "rt/wav.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

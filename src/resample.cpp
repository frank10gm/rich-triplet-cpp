#include "rt/resample.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <numeric>

namespace rt {

ResampleKernel make_resample_kernel(std::size_t orig_freq, std::size_t new_freq,
                                    std::size_t lowpass_filter_width, double rolloff) {
    ResampleKernel k;
    if (orig_freq == 0 || new_freq == 0 || lowpass_filter_width == 0) {
        return k;
    }

    // Reduce the ratio first: 24 kHz to 16 kHz is 3 to 2, not 24000 to 16000,
    // and the kernel is sized off the reduced numbers.
    const std::size_t g = std::gcd(orig_freq, new_freq);
    const std::size_t l = orig_freq / g;
    const std::size_t m = new_freq / g;

    const double width_f = static_cast<double>(lowpass_filter_width);
    // The cutoff sits at the lower of the two Nyquists, pulled in by rolloff so
    // the filter's transition band has room below it.
    const double base = static_cast<double>(std::min(l, m)) * rolloff;
    const auto width =
        static_cast<std::size_t>(std::ceil(width_f * static_cast<double>(l) / base));

    k.phases = m;
    k.stride = l;
    k.width = width;
    k.taps = 2 * width + l;
    k.data.resize(k.phases * k.taps);

    const double scale = base / static_cast<double>(l);
    for (std::size_t p = 0; p < m; ++p) {
        // Phase p samples the filter offset by p/M of an output period.
        const double phase = -static_cast<double>(p) / static_cast<double>(m);
        for (std::size_t j = 0; j < k.taps; ++j) {
            const double idx = (static_cast<double>(j) - static_cast<double>(width)) /
                               static_cast<double>(l);
            // Clamping to +-width zero crossings is what truncates the sinc to
            // a finite kernel; the window then tapers what is left.
            double t = std::clamp((phase + idx) * base, -width_f, width_f);
            const double w = std::pow(std::cos(t * std::numbers::pi / width_f / 2.0), 2.0);
            t *= std::numbers::pi;
            const double sinc = t == 0.0 ? 1.0 : std::sin(t) / t;
            k.data[p * k.taps + j] = static_cast<float>(sinc * w * scale);
        }
    }
    return k;
}

std::vector<float> resample_with(std::span<const float> input, const ResampleKernel& kernel) {
    if (input.empty() || kernel.taps == 0) {
        return {};
    }

    const std::size_t n = input.size();
    const std::size_t l = kernel.stride;
    const std::size_t m = kernel.phases;

    // Zero-pad by `width` in front and `width + stride` behind, so the first
    // and last output samples see a full kernel. That is what makes the output
    // length exactly floor(n / stride) + 1 cycles.
    const std::size_t pad_front = kernel.width;
    const std::size_t cycles = n / l + 1;

    const std::size_t target =
        static_cast<std::size_t>((static_cast<std::uint64_t>(m) * n + l - 1) / l);
    std::vector<float> out(cycles * m, 0.0f);

    for (std::size_t c = 0; c < cycles; ++c) {
        const std::size_t start = c * l;
        for (std::size_t p = 0; p < m; ++p) {
            const std::span<const float> tap = kernel.phase(p);
            float acc = 0.0f;
            for (std::size_t j = 0; j < kernel.taps; ++j) {
                // The pad is conceptual: samples outside the input are zero, so
                // they are skipped rather than materialised.
                const std::size_t src = start + j;
                if (src < pad_front) {
                    continue;
                }
                const std::size_t i = src - pad_front;
                if (i >= n) {
                    break;
                }
                acc += input[i] * tap[j];
            }
            out[c * m + p] = acc;
        }
    }

    out.resize(std::min(target, out.size()));
    return out;
}

std::vector<float> resample(std::span<const float> input, std::size_t orig_freq,
                            std::size_t new_freq) {
    if (orig_freq == new_freq) {
        return {input.begin(), input.end()};
    }
    return resample_with(input, make_resample_kernel(orig_freq, new_freq));
}

}  // namespace rt

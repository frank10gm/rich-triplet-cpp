#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include "rt/resample.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

namespace {

/// A sine at `freq` sampled at `rate`.
[[nodiscard]] std::vector<float> sine(std::size_t n, float freq, float rate, float amp = 0.5f) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = amp * std::sin(2.0f * std::numbers::pi_v<float> * freq *
                                (static_cast<float>(i) / rate));
    }
    return out;
}

[[nodiscard]] float rms(std::span<const float> x) {
    double sum = 0.0;
    for (const float v : x) {
        sum += static_cast<double>(v) * v;
    }
    return x.empty() ? 0.0f : static_cast<float>(std::sqrt(sum / static_cast<double>(x.size())));
}

}  // namespace

// =============================================================================
// The kernel
// =============================================================================

TEST_CASE("the 24 kHz to 16 kHz kernel is two phases of 23 taps", "[resample]") {
    const ResampleKernel k = make_resample_kernel(24000, 16000);
    // 24000:16000 reduces to 3:2, so three input samples make two output ones.
    REQUIRE(k.stride == 3);
    REQUIRE(k.phases == 2);
    // width = ceil(6 * 3 / (2 * 0.99)) = 10, taps = 2 * width + stride.
    REQUIRE(k.width == 10);
    REQUIRE(k.taps == 23);
    REQUIRE(k.data.size() == 46);
}

TEST_CASE("the kernel matches the reference filter", "[resample]") {
    const ResampleKernel k = make_resample_kernel(24000, 16000);
    const std::span<const float> p0 = k.phase(0);
    const std::span<const float> p1 = k.phase(1);

    // Phase 0 lands exactly on an input sample, so its centre tap is the bare
    // scale factor base/stride = 1.98/3 and it is symmetric about it.
    REQUIRE(approx(p0[10], 0.66f));
    REQUIRE(approx(p0[9], 0.270691812f));
    REQUIRE(approx(p0[11], 0.270691812f));
    REQUIRE(approx(p0[12], -0.118959866f));
    // The window is zero at both ends, which is what makes the truncation
    // inaudible.
    REQUIRE(approx(p0[0], 0.0f));
    REQUIRE(approx(p0[22], 0.0f));

    // Phase 1 sits half an output period along, so it has no centre tap.
    REQUIRE(approx(p1[10], 0.00622774707f));
    REQUIRE(approx(p1[11], 0.543885589f));
}

TEST_CASE("every phase has unit DC gain", "[resample]") {
    // Each output sample is one phase's dot product, so each phase has to sum
    // to one on its own or a constant signal would come out scaled.
    for (const auto [from, to] : {std::pair{24000, 16000}, std::pair{16000, 24000},
                                  std::pair{44100, 16000}}) {
        const ResampleKernel k = make_resample_kernel(static_cast<std::size_t>(from),
                                                      static_cast<std::size_t>(to));
        for (std::size_t p = 0; p < k.phases; ++p) {
            float sum = 0.0f;
            for (const float v : k.phase(p)) {
                sum += v;
            }
            REQUIRE(std::fabs(sum - 1.0f) < 0.01f);
        }
    }
}

// =============================================================================
// Length
// =============================================================================

TEST_CASE("the output length is exactly ceil(new * n / orig)", "[resample]") {
    REQUIRE(resample(sine(1000, 440.0f, 24000.0f), 24000, 16000).size() == 667);
    REQUIRE(resample(sine(1000, 440.0f, 16000.0f), 16000, 24000).size() == 1500);
    REQUIRE(resample(sine(999, 440.0f, 48000.0f), 48000, 24000).size() == 500);
    // A ratio that does not reduce to anything small: 22050:16000 is 441:320.
    REQUIRE(resample(sine(1000, 440.0f, 22050.0f), 22050, 16000).size() == 726);
}

TEST_CASE("equal rates pass the samples through untouched", "[resample]") {
    const std::vector<float> x = sine(64, 440.0f, 24000.0f);
    const std::vector<float> y = resample(x, 24000, 24000);
    REQUIRE(y.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        // Bit-exact, not approximate: running 1:1 audio through the filter
        // would soften it for nothing.
        REQUIRE(y[i] == x[i]);
    }
}

TEST_CASE("an empty input resamples to nothing", "[resample]") {
    REQUIRE(resample({}, 24000, 16000).empty());
}

// =============================================================================
// Signal
// =============================================================================

TEST_CASE("a sine survives 24 kHz to 16 kHz", "[resample]") {
    // 440 Hz is far below either Nyquist, so the filter should leave it alone.
    const std::vector<float> x = sine(4800, 440.0f, 24000.0f);
    const std::vector<float> y = resample(x, 24000, 16000);
    REQUIRE(y.size() == 3200);

    const std::vector<float> want = sine(3200, 440.0f, 16000.0f);
    // Skip the kernel's reach at each end, where the zero padding shows.
    for (std::size_t i = 32; i + 32 < y.size(); ++i) {
        REQUIRE(std::fabs(y[i] - want[i]) < 2e-3f);
    }
    REQUIRE(std::fabs(rms(y) - rms(x)) < 1e-3f);
}

TEST_CASE("a constant stays constant", "[resample]") {
    const std::vector<float> x(3000, 0.25f);
    const std::vector<float> y = resample(x, 24000, 16000);
    for (std::size_t i = 32; i + 32 < y.size(); ++i) {
        REQUIRE(std::fabs(y[i] - 0.25f) < 1e-3f);
    }
}

TEST_CASE("upsampling is the same machinery", "[resample]") {
    const std::vector<float> x = sine(3200, 440.0f, 16000.0f);
    const std::vector<float> y = resample(x, 16000, 24000);
    REQUIRE(y.size() == 4800);

    const std::vector<float> want = sine(4800, 440.0f, 24000.0f);
    for (std::size_t i = 32; i + 32 < y.size(); ++i) {
        REQUIRE(std::fabs(y[i] - want[i]) < 2e-3f);
    }
}

TEST_CASE("content above the new Nyquist is filtered out", "[resample]") {
    // 10 kHz fits under 24 kHz's Nyquist but not under 16 kHz's, so the filter
    // has to remove it rather than let it fold back down as an audible alias.
    const std::vector<float> x = sine(4800, 10000.0f, 24000.0f, 0.5f);
    const std::vector<float> y = resample(x, 24000, 16000);
    const std::span<const float> interior(y.data() + 64, y.size() - 128);
    REQUIRE(rms(interior) < 0.05f);
}

TEST_CASE("down then up returns the signal", "[resample]") {
    const std::vector<float> x = sine(4800, 440.0f, 24000.0f);
    const std::vector<float> round = resample(resample(x, 24000, 16000), 16000, 24000);
    REQUIRE(round.size() == x.size());
    for (std::size_t i = 64; i + 64 < round.size(); ++i) {
        REQUIRE(std::fabs(round[i] - x[i]) < 5e-3f);
    }
}

TEST_CASE("a prebuilt kernel gives the same answer", "[resample]") {
    const std::vector<float> x = sine(600, 440.0f, 24000.0f);
    const ResampleKernel k = make_resample_kernel(24000, 16000);
    const std::vector<float> a = resample(x, 24000, 16000);
    const std::vector<float> b = resample_with(x, k);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(a[i] == b[i]);
    }
}

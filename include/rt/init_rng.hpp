#pragma once

// Deterministic LCG used for weight initialization. Shared by nn and nn2 so
// both engines produce identical weights from the same seed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace rt {

class InitRng {
   public:
    explicit InitRng(std::uint64_t seed) : state_(seed + 1) {}

    /// Next random float in (0, 1).
    [[nodiscard]] float next_f32() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        // Upper 32 bits: better quality than the low bits of an LCG.
        const auto bits = static_cast<std::uint32_t>(state_ >> 32);
        return static_cast<float>(bits) / static_cast<float>(UINT32_MAX);
    }

    /// Box-Muller: two uniform samples -> one standard normal sample.
    /// N(0,1) -- most values within [-3, 3], 68% within [-1, 1].
    [[nodiscard]] float next_normal() {
        const float u1 = std::max(next_f32(), 1e-7f);  // avoid log(0)
        const float u2 = next_f32();
        const float r = std::sqrt(-2.0f * std::log(u1));
        const float theta = 2.0f * std::numbers::pi_v<float> * u2;
        return r * std::cos(theta);
    }

    /// Sample `n` values from N(0, std).
    [[nodiscard]] std::vector<float> normal_vec(std::size_t n, float std_dev) {
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = next_normal() * std_dev;
        }
        return out;
    }

   private:
    std::uint64_t state_;
};

}  // namespace rt

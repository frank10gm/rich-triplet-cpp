#pragma once

// =============================================================================
// Metal full-graph decode engine (Qwen 3.5)
// =============================================================================
//
// The same idea as the Gemma 3 engine: one command buffer per decode step, so
// the per-dispatch cost is amortized rather than paid per GEMV.
//
// What differs is the layer mix. Qwen 3.5 alternates Gated DeltaNet layers,
// which carry a recurrent state and a conv1d history that persist in GPU
// buffers across steps, with softmax-attention layers that keep a half-precision
// KV cache. Each kind gets its own encode path.

#include <cstddef>
#include <memory>
#include <vector>

#include "rt/transformer_qwen35.hpp"

namespace rt {

class MetalDecodeContextQwen35 {
   public:
    /// Upload `model`'s weights and allocate the decode buffers.
    [[nodiscard]] static std::unique_ptr<MetalDecodeContextQwen35> create(
        const Qwen35Model& model);

    ~MetalDecodeContextQwen35();

    MetalDecodeContextQwen35(const MetalDecodeContextQwen35&) = delete;
    MetalDecodeContextQwen35& operator=(const MetalDecodeContextQwen35&) = delete;

    /// Copy the DeltaNet states and attention KV caches from a CPU prefill onto
    /// the GPU. Call once before the decode loop.
    void sync_state_from_cpu(const Qwen35Cache& cache);

    /// Run one decode step and return the logits.
    [[nodiscard]] std::vector<float> decode_step(std::size_t token_id, std::size_t position);

    /// Rows in the lm_head weight buffer, which can be smaller than the config's
    /// vocab_size when a GGUF embedding table is.
    [[nodiscard]] std::size_t lm_head_vocab() const;

    /// Print a breakdown of GPU buffer usage.
    void print_memory_stats() const;

   private:
    struct Impl;
    explicit MetalDecodeContextQwen35(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace rt

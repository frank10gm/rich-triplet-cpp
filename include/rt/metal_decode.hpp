#pragma once

// =============================================================================
// Metal full-graph decode engine (Gemma 3)
// =============================================================================
//
// Encodes an entire decode forward pass -- embedding, all layers, the final
// norm and lm_head -- into one Metal command buffer.
//
// The reason is dispatch overhead: at roughly 0.5 ms per command buffer,
// issuing one GEMV at a time is slower than CPU SDOT on Apple Silicon. Batching
// every dispatch into a single encoder amortizes that cost across the ~717
// dispatches a step needs (21 per layer, plus 3).
//
// Within one compute encoder on Apple Silicon, dispatches execute in order and
// each one's writes are visible to the next through the unified L2, so no
// barriers are needed between them.
//
// Per-dispatch constants -- dimensions, position, theta -- go through
// `setBytes`, which Metal copies inline, so no buffer is allocated per step.

#include <cstddef>
#include <memory>
#include <vector>

#include "rt/transformer4.hpp"

namespace rt {

/// Owns every GPU resource a decode loop needs: uploaded weights, activation
/// scratch, the KV cache, and the compiled pipelines.
///
/// Built once after the weights are loaded. The constructor frees each layer's
/// CPU-side weights as it uploads them, so peak memory never holds both copies.
class MetalDecodeContext {
   public:
    /// Upload `model`'s weights and allocate the decode buffers.
    ///
    /// `draft_len > 0` also allocates the batch path used by speculative
    /// decoding; leave it at 0 to skip those buffers entirely.
    [[nodiscard]] static std::unique_ptr<MetalDecodeContext> create(Gemma3Model& model,
                                                                    std::size_t draft_len);

    ~MetalDecodeContext();

    MetalDecodeContext(const MetalDecodeContext&) = delete;
    MetalDecodeContext& operator=(const MetalDecodeContext&) = delete;

    /// Copy a CPU KV cache into the GPU half-precision buffers, converting on
    /// device. Call once after CPU prefill, before the decode loop. The staging
    /// buffer is released afterwards.
    void sync_kv_from_cpu(const Gemma3KvCache& cache);

    /// Run one decode step and return the logits.
    [[nodiscard]] std::vector<float> decode_step(std::size_t token_id, std::size_t position);

    /// Run `token_ids.size()` tokens in one pass, returning logits row-major as
    /// `M * lm_head_vocab()` floats. Requires `draft_len > 0` at construction,
    /// and at most 8 tokens.
    [[nodiscard]] std::vector<float> decode_step_batch(
        const std::vector<std::size_t>& token_ids, std::size_t start_position);

    /// Rows in the lm_head weight buffer.
    ///
    /// This can be smaller than `config.vocab_size`: a GGUF embedding table has
    /// 262144 rows where the Gemma 3 4B config declares 262208. Dispatching
    /// against the config value would read past the buffer and return NaNs, so
    /// the actual row count is what drives the GEMV and the readback.
    [[nodiscard]] std::size_t lm_head_vocab() const;

    /// Print a breakdown of GPU buffer usage.
    void print_metal_memory() const;

   private:
    struct Impl;
    explicit MetalDecodeContext(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace rt

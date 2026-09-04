#pragma once

// =============================================================================
// Training for the scalar model
// =============================================================================
//
// The loop itself is short: sample a batch, forward to a loss, backward to
// gradients, clip, step the optimizer, zero the gradients, repeat. That is how
// every model in this repository is trained.
//
// At step 0 the loss sits around log(vocab_size) -- random guessing. It drops
// over the first few hundred steps and then plateaus, because language is
// stochastic and some uncertainty is irreducible.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rt/autograd.hpp"
#include "rt/dataset.hpp"
#include "rt/tokenizer.hpp"
#include "rt/transformer.hpp"

namespace rt {

// =============================================================================
// AdamW
// =============================================================================
//
// Adam keeps a running average of the gradient (first moment, the direction)
// and of its square (second moment, the magnitude), and steps by
// `lr * m_hat / (sqrt(v_hat) + eps)`. Both moments start at zero and are
// bias-corrected by `1 - beta^t` so the early steps are not damped.
//
// The "W" is decoupled weight decay: the weight is shrunk directly rather than
// through the gradient, which keeps the decay independent of the adaptive
// scaling.

class AdamW {
   public:
    float lr;
    /// Momentum for the first moment (gradient direction).
    float beta1 = 0.9f;
    /// Momentum for the second moment (gradient magnitude).
    float beta2 = 0.999f;
    /// Guards the division when the second moment is tiny.
    float eps = 1e-8f;
    /// L2 regularization strength.
    float weight_decay = 0.1f;
    /// Step counter, used for bias correction.
    std::uint32_t step_count = 0;

    AdamW(std::size_t n_params, float lr);

    /// One optimizer step. Call after `backward()` and before `zero_grad()`.
    /// `params` must be in the same order as when the optimizer was built.
    void step(const std::vector<Value>& params);

   private:
    /// First and second moment estimates, one per parameter.
    std::vector<float> m_;
    std::vector<float> v_;
    std::size_t n_params_ = 0;
};

// =============================================================================
// Training loop
// =============================================================================

struct TrainConfig {
    std::size_t max_steps = 500;
    std::size_t eval_interval = 50;
    float learning_rate = 3e-4f;
    /// Cap on the total gradient norm. A single bad batch can otherwise send
    /// the weights somewhere unrecoverable.
    float grad_clip = 1.0f;
};

/// Run the full training loop, returning the final training loss.
[[nodiscard]] float train(const Gpt& model, const CharTokenizer& tokenizer,
                          const TextDataset& train_data, const TextDataset& val_data,
                          const TrainConfig& cfg);

/// Average loss over `n_samples` random examples, forward only.
[[nodiscard]] float estimate_loss(const Gpt& model, const TextDataset& data,
                                  std::size_t n_samples);

// =============================================================================
// Generation
// =============================================================================
//
// The model predicts one token at a time and each prediction is fed back as
// input, so the context grows until it hits the window and starts sliding.
//
// Picking a token from the distribution: greedy argmax is deterministic and
// tends to repeat itself; temperature sharpens (T < 1) or flattens (T > 1) the
// distribution; top-k restricts sampling to the k most likely tokens so an
// unlikely one cannot slip through. This implements temperature plus top-k,
// which is what GPT-2 used.

/// Generate `max_new_tokens` continuations of `prompt`, printing as it goes,
/// and return the full decoded text.
[[nodiscard]] std::string generate(const Gpt& model, const CharTokenizer& tokenizer,
                                   const std::string& prompt, std::size_t max_new_tokens,
                                   float temperature, std::size_t top_k);

/// Sample from a distribution restricted to its k largest entries: threshold
/// at the k-th value, zero the rest, renormalize, then draw.
[[nodiscard]] std::size_t sample_top_k(const std::vector<float>& probs, std::size_t k);

/// One LCG draw in [0, 1), from a seed.
[[nodiscard]] float lcg_float(std::uint64_t seed);

}  // namespace rt

#pragma once

// =============================================================================
// Training for the tensor engine
// =============================================================================
//
// The same loop as `train.hpp`, but over TensorNode parameters, and with the
// pieces a real run needs: a learning-rate schedule, gradient accumulation,
// batching, label smoothing, early stopping and checkpointing.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "rt/dataset.hpp"
#include "rt/nn2.hpp"
#include "rt/tensor_node.hpp"

namespace rt {

// =============================================================================
// AdamW2
// =============================================================================

/// AdamW hyperparameters. Every field has a production-proven default;
/// override only what a run needs.
struct AdamW2Params {
    /// Step size. 1e-4 to 1e-3 for fine-tuning, 3e-4 for small pretraining.
    float lr = 3e-4f;
    /// First-moment decay (momentum). Standard: 0.9.
    float beta1 = 0.9f;
    /// Second-moment decay. Standard: 0.999.
    float beta2 = 0.999f;
    /// Numerical-stability constant.
    float eps = 1e-8f;
    /// L2 regularization. 0.1 at GPT scale, 0.01 for fine-tunes.
    float weight_decay = 0.1f;
};

class AdamW2 {
   public:
    float lr = 3e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.1f;
    std::uint32_t step_count = 0;

    /// Default hyperparameters, with the given learning rate.
    AdamW2(const std::vector<TensorNode>& params, float lr);

    /// Fully customized hyperparameters.
    AdamW2(const std::vector<TensorNode>& params, const AdamW2Params& hp);

    /// One optimizer step. Call after `backward()` and before `zero_grad()`.
    void step(const std::vector<TensorNode>& params);

    /// Update the learning rate, for use with `LrScheduler`.
    void set_lr(float new_lr) { lr = new_lr; }

   private:
    /// First and second moments, shaped like each parameter.
    std::vector<Mat> m_;
    std::vector<Mat> v_;
};

// =============================================================================
// LrScheduler -- cosine decay with linear warmup
// =============================================================================
//
// A fixed learning rate is either too high early, and training destabilizes, or
// too high late, and it never settles. The standard transformer schedule ramps
// linearly from 0 to lr_max over the warmup, which keeps the first steps from
// wrecking the random initialization, then decays along a cosine to lr_min,
// which avoids overshooting the minimum.

struct LrScheduler {
    /// Peak learning rate, reached at the end of warmup.
    float lr_max = 3e-4f;
    /// Floor at the end of the cosine decay. Usually lr_max / 10.
    float lr_min = 3e-5f;
    /// Warmup length, in steps.
    std::size_t warmup_steps = 0;
    /// Total training steps.
    std::size_t total_steps = 0;

    /// The learning rate at `step` (0-indexed).
    [[nodiscard]] float get(std::size_t step) const;
};

// =============================================================================
// Configuration
// =============================================================================

struct TrainConfig2 {
    std::size_t max_steps = 500;
    std::size_t eval_interval = 50;
    float learning_rate = 3e-4f;
    float grad_clip = 1.0f;
    /// Micro-steps to accumulate before an optimizer step. The effective batch
    /// is `batch_size * accumulate_steps`. 1 disables accumulation.
    std::size_t accumulate_steps = 1;
    /// Independent sequences per step. Each is run forward and backward and
    /// their gradients are averaged, which smooths the update. Memory grows
    /// linearly, since sequences are processed one at a time.
    std::size_t batch_size = 1;
    /// Label-smoothing epsilon. 0 is plain cross-entropy; 0.1 is typical.
    /// Replaces the one-hot target with
    /// `(1 - eps) * one_hot + eps / vocab_size`.
    float label_smoothing = 0.0f;
    /// Where to write a checkpoint once training finishes. Unset writes none.
    std::optional<std::string> checkpoint_path;
    /// Stop if validation loss has not improved for this many evaluations.
    /// 0 disables early stopping; 5 is typical.
    std::size_t early_stopping_patience = 0;
};

// =============================================================================
// Metrics
// =============================================================================

/// Fraction of positions where the argmax logit is the target.
[[nodiscard]] float token_accuracy(const Mat& logits, const std::vector<std::size_t>& targets);

/// Mean cross-entropy, optionally label-smoothed.
///
/// With `smoothing == 0` this is `-log(softmax(logits)[target])`. Otherwise the
/// target distribution becomes
/// `y[v] = (1 - eps) * one_hot(v == target) + eps / V`,
/// which keeps the model from driving any logit to negative infinity.
[[nodiscard]] float cross_entropy_smoothed(const Mat& logits,
                                           const std::vector<std::size_t>& targets,
                                           float smoothing);

/// Average loss over `n_samples` random examples, forward only.
[[nodiscard]] float estimate_loss2(const Trainable& model, const DataSource& data,
                                   std::size_t n_samples);

/// Sample from the top-k entries of a probability distribution.
[[nodiscard]] std::size_t sample_top_k2(const std::vector<float>& probs, std::size_t k);

// =============================================================================
// Training loop
// =============================================================================

/// Train any `Trainable` against any `DataSource`, returning the final training
/// loss. Works for both the GPT-2 and GPT-OSS models.
[[nodiscard]] float train2(const Trainable& model, const DataSource& train_data,
                           const DataSource& val_data, const TrainConfig2& cfg);

}  // namespace rt

#include "rt/train2.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>

namespace rt {

// =============================================================================
// AdamW2
// =============================================================================

AdamW2::AdamW2(const std::vector<TensorNode>& params, const AdamW2Params& hp)
    : lr(hp.lr),
      beta1(hp.beta1),
      beta2(hp.beta2),
      eps(hp.eps),
      weight_decay(hp.weight_decay) {
    m_.reserve(params.size());
    v_.reserve(params.size());
    for (const TensorNode& p : params) {
        m_.push_back(Mat::zeros(p.data().rows, p.data().cols));
        v_.push_back(Mat::zeros(p.data().rows, p.data().cols));
    }
}

AdamW2::AdamW2(const std::vector<TensorNode>& params, float lr)
    : AdamW2(params, [lr] {
          AdamW2Params hp;
          hp.lr = lr;
          return hp;
      }()) {}

void AdamW2::step(const std::vector<TensorNode>& params) {
    assert(params.size() == m_.size() && "parameter count changed");

    ++step_count;
    const auto t = static_cast<float>(step_count);
    const float bc1 = 1.0f - std::pow(beta1, t);
    const float bc2 = 1.0f - std::pow(beta2, t);

    for (std::size_t i = 0; i < params.size(); ++i) {
        const TensorNode& param = params[i];
        const Mat& w = param.data();
        const Mat& g = param.grad();

        // An all-zero gradient means this parameter was untouched -- an
        // embedding row no batch referenced, say -- so leave it alone.
        if (std::all_of(g.data.begin(), g.data.end(), [](float x) { return x == 0.0f; })) {
            continue;
        }

        // Decoupled weight decay: shrink the weight itself.
        const Mat w_decayed = w.scale(1.0f - lr * weight_decay);

        Mat m_new = Mat::from_fn(w.rows, w.cols, [&](std::size_t r, std::size_t c) {
            return beta1 * m_[i].at(r, c) + (1.0f - beta1) * g.at(r, c);
        });
        Mat v_new = Mat::from_fn(w.rows, w.cols, [&](std::size_t r, std::size_t c) {
            const float gv = g.at(r, c);
            return beta2 * v_[i].at(r, c) + (1.0f - beta2) * gv * gv;
        });

        Mat w_new = Mat::from_fn(w.rows, w.cols, [&](std::size_t r, std::size_t c) {
            const float m_hat = m_new.at(r, c) / bc1;
            const float v_hat = v_new.at(r, c) / bc2;
            return w_decayed.at(r, c) - lr * m_hat / (std::sqrt(v_hat) + eps);
        });

        m_[i] = std::move(m_new);
        v_[i] = std::move(v_new);
        param.set_data(std::move(w_new));
    }
}

// =============================================================================
// LrScheduler
// =============================================================================

float LrScheduler::get(std::size_t step) const {
    if (step < warmup_steps) {
        return lr_max * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }
    const float progress = static_cast<float>(step - warmup_steps) /
                           static_cast<float>(std::max<std::size_t>(total_steps - warmup_steps, 1));
    const float cosine = 0.5f * (1.0f + std::cos(std::numbers::pi_v<float> * progress));
    return lr_min + (lr_max - lr_min) * cosine;
}

// =============================================================================
// Metrics
// =============================================================================

float token_accuracy(const Mat& logits, const std::vector<std::size_t>& targets) {
    const std::size_t t = logits.rows;
    assert(t == targets.size() && "token_accuracy: logits rows != targets len");
    const std::size_t v = logits.cols;

    std::size_t correct = 0;
    for (std::size_t row = 0; row < t; ++row) {
        std::size_t pred = 0;
        for (std::size_t c = 1; c < v; ++c) {
            if (logits.at(row, pred) < logits.at(row, c)) {
                pred = c;
            }
        }
        correct += (pred == targets[row]) ? 1 : 0;
    }
    return static_cast<float>(correct) / static_cast<float>(t);
}

float cross_entropy_smoothed(const Mat& logits, const std::vector<std::size_t>& targets,
                             float smoothing) {
    const std::size_t t = logits.rows;
    const std::size_t v = logits.cols;
    float total = 0.0f;

    for (std::size_t row = 0; row < t; ++row) {
        // Numerically stable log-softmax.
        float max_l = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < v; ++c) {
            max_l = std::max(max_l, logits.at(row, c));
        }
        float sum_exp = 0.0f;
        for (std::size_t c = 0; c < v; ++c) {
            sum_exp += std::exp(logits.at(row, c) - max_l);
        }
        const float log_sum = std::log(sum_exp);
        const std::size_t tgt = targets[row];

        if (smoothing == 0.0f) {
            total -= logits.at(row, tgt) - max_l - log_sum;
        } else {
            // Every entry contributes eps/V; the target also gets (1 - eps).
            float base = 0.0f;
            for (std::size_t c = 0; c < v; ++c) {
                base += (smoothing / static_cast<float>(v)) *
                        (logits.at(row, c) - max_l - log_sum);
            }
            total -= base + (1.0f - smoothing) * (logits.at(row, tgt) - max_l - log_sum);
        }
    }
    return total / static_cast<float>(t);
}

float estimate_loss2(const Trainable& model, const DataSource& data, std::size_t n_samples) {
    float total = 0.0f;
    for (std::size_t i = 0; i < n_samples; ++i) {
        const auto [token_ids, target_ids] = data.sample(static_cast<std::uint64_t>(i) + 9999);
        model.zero_grad();
        total += model.loss_tokens(token_ids, target_ids).data().at(0, 0);
        model.zero_grad();
    }
    return total / static_cast<float>(n_samples);
}

std::size_t sample_top_k2(const std::vector<float>& probs, std::size_t k) {
    k = std::min(k, probs.size());
    std::vector<float> sorted = probs;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const float threshold = sorted[k - 1];

    std::vector<float> filtered(probs.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        filtered[i] = probs[i] >= threshold ? probs[i] : 0.0f;
        sum += filtered[i];
    }
    for (float& p : filtered) {
        p /= sum;
    }

    static std::atomic<std::uint64_t> counter{54321};
    const std::uint64_t seed = counter.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
    const float rand_val =
        static_cast<float>(static_cast<std::uint32_t>(s >> 32)) / static_cast<float>(UINT32_MAX);

    float cumulative = 0.0f;
    for (std::size_t i = 0; i < filtered.size(); ++i) {
        cumulative += filtered[i];
        if (rand_val <= cumulative) {
            return i;
        }
    }
    return filtered.size() - 1;
}

// =============================================================================
// Training loop
// =============================================================================

float train2(const Trainable& model, const DataSource& train_data, const DataSource& val_data,
             const TrainConfig2& cfg) {
    const std::vector<TensorNode> params = model.parameters();
    std::size_t n_params = 0;
    for (const TensorNode& p : params) {
        n_params += p.data().numel();
    }
    AdamW2 optimizer(params, cfg.learning_rate);

    // Warm up over the first 5% of steps, then decay to a tenth of the peak.
    const LrScheduler sched{cfg.learning_rate, cfg.learning_rate * 0.1f,
                            std::max<std::size_t>(cfg.max_steps / 20, 1), cfg.max_steps};

    const std::size_t batch_size = std::max<std::size_t>(cfg.batch_size, 1);
    const std::size_t accum = std::max<std::size_t>(cfg.accumulate_steps, 1);

    std::printf(
        "\n[tensor] Training: %zu parameters (in %zu tensors), %zu steps (batch=%zu, "
        "accumulate=%zu)\n",
        n_params, params.size(), cfg.max_steps, batch_size, accum);
    std::printf("%.*s\n", 65,
                "-----------------------------------------------------------------");

    float last_loss = std::numeric_limits<float>::infinity();
    float last_acc = 0.0f;
    float best_val_loss = std::numeric_limits<float>::infinity();
    std::size_t patience_counter = 0;

    for (std::size_t step = 0; step < cfg.max_steps; ++step) {
        optimizer.set_lr(sched.get(step));

        // Gradients are zeroed once per accumulation window, not per step.
        if (step % accum == 0) {
            model.zero_grad();
        }

        const std::vector<TokenPair> batch =
            train_data.sample_batch(static_cast<std::uint64_t>(step) + 1, batch_size);

        // loss_batch_tokens runs forward and backward per sequence, scaling
        // each upstream gradient by 1/B, and returns the mean loss as a leaf.
        const TensorNode loss = model.loss_batch_tokens(batch);

        const float loss_val = loss.data().at(0, 0);
        if (!std::isfinite(loss_val)) {
            std::fprintf(stderr, "[warn] step %zu: non-finite loss (%.4f), skipping\n", step,
                         static_cast<double>(loss_val));
            continue;
        }

        // Metrics come from the batch's first sequence.
        {
            const Mat logits = model.forward_tokens(batch[0].first).data();
            last_loss = cross_entropy_smoothed(logits, batch[0].second, cfg.label_smoothing);
            last_acc = token_accuracy(logits, batch[0].second);
        }

        // Weights only move at the end of an accumulation window.
        const bool is_update_step = (step + 1) % accum == 0 || step == cfg.max_steps - 1;
        if (!is_update_step) {
            continue;
        }

        const std::vector<TensorNode> params_now = model.parameters();
        float sum_sq = 0.0f;
        for (const TensorNode& p : params_now) {
            for (float x : p.grad().data) {
                sum_sq += x * x;
            }
        }
        const float grad_norm = std::sqrt(sum_sq);

        if (!std::isfinite(grad_norm)) {
            std::fprintf(stderr,
                         "[warn] step %zu: non-finite grad_norm (%.4f), skipping optimizer step\n",
                         step, static_cast<double>(grad_norm));
            model.zero_grad();
            continue;
        }

        if (grad_norm > cfg.grad_clip) {
            const float scale = cfg.grad_clip / grad_norm;
            for (const TensorNode& p : params_now) {
                p.set_grad(p.grad().scale(scale));
            }
        }

        optimizer.step(params_now);

        const std::size_t display_step = step / accum;
        const std::size_t display_total = cfg.max_steps / accum;
        if (display_step % std::max<std::size_t>(cfg.eval_interval / accum, 1) == 0 ||
            step == cfg.max_steps - 1) {
            const float val_loss = estimate_loss2(model, val_data, 5);
            std::printf(
                "step %4zu/%zu | lr: %.2e | train_loss: %.4f | val_loss: %.4f | acc: %.3f | "
                "grad_norm: %.4f\n",
                display_step, display_total, static_cast<double>(optimizer.lr),
                static_cast<double>(last_loss), static_cast<double>(val_loss),
                static_cast<double>(last_acc), static_cast<double>(grad_norm));

            if (cfg.early_stopping_patience > 0) {
                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    patience_counter = 0;
                } else if (++patience_counter >= cfg.early_stopping_patience) {
                    std::printf("[early stop] val loss has not improved for %zu evals, stopping.\n",
                                patience_counter);
                    break;
                }
            }
        }
    }

    std::printf("%.*s\n", 65,
                "-----------------------------------------------------------------");

    if (cfg.checkpoint_path) {
        const std::vector<TensorNode> params_final = model.parameters();
        std::vector<std::pair<std::string, TensorNode>> named;
        named.reserve(params_final.size());
        for (std::size_t i = 0; i < params_final.size(); ++i) {
            named.emplace_back("param_" + std::to_string(i), params_final[i]);
        }
        if (const auto saved = save_checkpoint(*cfg.checkpoint_path, named)) {
            std::printf("[checkpoint] Saved %zu tensors to %s\n", named.size(),
                        cfg.checkpoint_path->c_str());
        } else {
            std::fprintf(stderr, "[checkpoint] Failed to save %s: %s\n",
                         cfg.checkpoint_path->c_str(), saved.error().c_str());
        }
    }

    return last_loss;
}

}  // namespace rt

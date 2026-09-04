#include "rt/train.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

#include "rt/nn.hpp"

namespace rt {

// =============================================================================
// AdamW
// =============================================================================

AdamW::AdamW(std::size_t n_params, float lr)
    : lr(lr), m_(n_params, 0.0f), v_(n_params, 0.0f), n_params_(n_params) {}

void AdamW::step(const std::vector<Value>& params) {
    assert(params.size() == n_params_ && "parameter count changed");

    ++step_count;
    const auto t = static_cast<float>(step_count);

    // Both moments start at zero, so early estimates are biased toward zero;
    // dividing by (1 - beta^t) corrects for that and fades as t grows.
    const float bc1 = 1.0f - std::pow(beta1, t);
    const float bc2 = 1.0f - std::pow(beta2, t);

    for (std::size_t i = 0; i < params.size(); ++i) {
        const Value& param = params[i];
        const float g = param.grad();

        // Untouched parameters (an embedding row no batch used) stay put.
        if (g == 0.0f) {
            continue;
        }

        // Decoupled weight decay: shrink the weight itself, not the gradient.
        param.set_val(param.val() * (1.0f - lr * weight_decay));

        m_[i] = beta1 * m_[i] + (1.0f - beta1) * g;
        v_[i] = beta2 * v_[i] + (1.0f - beta2) * g * g;

        const float m_hat = m_[i] / bc1;
        const float v_hat = v_[i] / bc2;

        param.set_val(param.val() - lr * m_hat / (std::sqrt(v_hat) + eps));
    }
}

// =============================================================================
// Training loop
// =============================================================================

namespace {
/// Convert a batch row of float-encoded ids back into token ids.
[[nodiscard]] std::vector<std::size_t> to_ids(const Tensor& t) {
    std::vector<std::size_t> ids(t.data.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<std::size_t>(t.data[i]);
    }
    return ids;
}
}  // namespace

float estimate_loss(const Gpt& model, const TextDataset& data, std::size_t n_samples) {
    float total = 0.0f;
    for (std::size_t i = 0; i < n_samples; ++i) {
        const auto [inp_t, tgt_t] = data.random_batch(1, static_cast<std::uint64_t>(i) + 9999);
        model.zero_grad();
        total += model.loss(to_ids(inp_t), to_ids(tgt_t)).val();
    }
    return total / static_cast<float>(n_samples);
}

float train(const Gpt& model, const CharTokenizer& tokenizer, const TextDataset& train_data,
            const TextDataset& val_data, const TrainConfig& cfg) {
    (void)tokenizer;

    const std::vector<Value> params = model.parameters();
    AdamW optimizer(params.size(), cfg.learning_rate);

    std::printf("\nTraining: %zu parameters, %zu steps\n", params.size(), cfg.max_steps);
    std::printf("%.*s\n", 55, "-------------------------------------------------------");

    float last_loss = std::numeric_limits<float>::infinity();

    for (std::size_t step = 0; step < cfg.max_steps; ++step) {
        // Seeding by step keeps every step on a different example while
        // leaving the whole run reproducible.
        const auto [inp_t, tgt_t] =
            train_data.random_batch(1, static_cast<std::uint64_t>(step) + 1);

        model.zero_grad();

        const Value loss = model.loss(to_ids(inp_t), to_ids(tgt_t));
        last_loss = loss.val();
        loss.backward();

        // Clip on the total gradient norm across every parameter.
        const std::vector<Value> params_now = model.parameters();
        float sum_sq = 0.0f;
        for (const Value& p : params_now) {
            sum_sq += p.grad() * p.grad();
        }
        const float grad_norm = std::sqrt(sum_sq);

        if (grad_norm > cfg.grad_clip) {
            const float scale = cfg.grad_clip / grad_norm;
            for (const Value& p : params_now) {
                p.set_grad(p.grad() * scale);
            }
        }

        optimizer.step(params_now);

        if (step % cfg.eval_interval == 0 || step == cfg.max_steps - 1) {
            const float val_loss = estimate_loss(model, val_data, 5);
            std::printf("step %4zu/%zu | train_loss: %.4f | val_loss: %.4f | grad_norm: %.4f\n",
                        step, cfg.max_steps, static_cast<double>(last_loss),
                        static_cast<double>(val_loss), static_cast<double>(grad_norm));
        }
    }

    std::printf("%.*s\n", 55, "-------------------------------------------------------");
    return last_loss;
}

// =============================================================================
// Generation
// =============================================================================

float lcg_float(std::uint64_t seed) {
    const std::uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<float>(static_cast<std::uint32_t>(s >> 32)) /
           static_cast<float>(UINT32_MAX);
}

std::size_t sample_top_k(const std::vector<float>& probs, std::size_t k) {
    k = std::min(k, probs.size());

    // Threshold at the k-th largest probability, then keep everything at or
    // above it.
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

    // A process-wide counter drives the LCG, so successive calls differ
    // without the caller tracking RNG state.
    static std::atomic<std::uint64_t> sample_counter{12345};
    const float rand_val = lcg_float(sample_counter.fetch_add(1, std::memory_order_relaxed));

    float cumulative = 0.0f;
    for (std::size_t i = 0; i < filtered.size(); ++i) {
        cumulative += filtered[i];
        if (rand_val <= cumulative) {
            return i;
        }
    }
    return filtered.size() - 1;
}

std::string generate(const Gpt& model, const CharTokenizer& tokenizer, const std::string& prompt,
                     std::size_t max_new_tokens, float temperature, std::size_t top_k) {
    const std::size_t ctx_len = model.config.context_length;

    std::vector<std::size_t> token_ids;
    for (std::uint32_t id : tokenizer.encode(prompt)) {
        token_ids.push_back(id);
    }

    std::printf("%s", prompt.c_str());

    for (std::size_t n = 0; n < max_new_tokens; ++n) {
        // Slide the window once the context outgrows it.
        const std::vector<std::size_t> context =
            token_ids.size() > ctx_len
                ? std::vector<std::size_t>(token_ids.end() - static_cast<std::ptrdiff_t>(ctx_len),
                                           token_ids.end())
                : token_ids;

        model.zero_grad();
        const auto logits = model.forward(context);
        const std::vector<Value>& last_logits = logits.back();

        // Temperature: below 1 sharpens the distribution, above 1 flattens it.
        std::vector<Value> scaled;
        scaled.reserve(last_logits.size());
        for (const Value& v : last_logits) {
            scaled.emplace_back(v.val() / temperature);
        }

        std::vector<float> probs_vals;
        probs_vals.reserve(scaled.size());
        for (const Value& v : softmax(scaled)) {
            probs_vals.push_back(v.val());
        }

        const std::size_t next_token = sample_top_k(probs_vals, top_k);
        std::printf("%s", tokenizer.decode({static_cast<std::uint32_t>(next_token)}).c_str());
        token_ids.push_back(next_token);
    }

    std::printf("\n");

    std::vector<std::uint32_t> all(token_ids.size());
    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        all[i] = static_cast<std::uint32_t>(token_ids[i]);
    }
    return tokenizer.decode(all);
}

}  // namespace rt

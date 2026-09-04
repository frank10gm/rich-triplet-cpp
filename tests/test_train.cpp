#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <set>

#include "rt/train.hpp"
#include "rt/train2.hpp"
#include "rt/transformer2.hpp"

using namespace rt;

namespace {

Config tiny_cfg() {
    Config c;
    c.vocab_size = 12;
    c.context_length = 8;
    c.d_model = 8;
    c.n_layers = 1;
    c.n_heads = 2;
    return c;
}

bool approx(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) < tol; }

}  // namespace

// -----------------------------------------------------------------------------
// AdamW (scalar)
// -----------------------------------------------------------------------------

TEST_CASE("adamw moves a weight against its gradient", "[train]") {
    const Value w(1.0f);
    AdamW opt(1, 0.1f);
    opt.weight_decay = 0.0f;  // isolate the Adam term

    w.set_grad(1.0f);
    opt.step({w});

    // With bias correction, the first step is almost exactly -lr.
    REQUIRE(opt.step_count == 1);
    REQUIRE(w.val() < 1.0f);
    REQUIRE(approx(w.val(), 1.0f - 0.1f, 1e-3f));
}

TEST_CASE("adamw skips parameters with zero gradient", "[train]") {
    const Value w(2.0f);
    AdamW opt(1, 0.1f);
    w.set_grad(0.0f);
    opt.step({w});
    // Untouched, including no weight decay.
    REQUIRE(w.val() == 2.0f);
}

TEST_CASE("adamw applies decoupled weight decay", "[train]") {
    const Value w(1.0f);
    AdamW opt(1, 0.1f);
    opt.weight_decay = 0.5f;
    w.set_grad(1e-12f);  // non-zero, so the parameter is not skipped
    opt.step({w});
    // The decay factor (1 - lr*wd) = 0.95 dominates the negligible Adam step.
    REQUIRE(w.val() < 0.96f);
}

TEST_CASE("scalar training reduces the loss", "[train]") {
    // A repetitive corpus is easy enough that a handful of steps must help.
    std::string text;
    for (int i = 0; i < 30; ++i) {
        text += "abcabcabc";
    }
    const CharTokenizer tok = CharTokenizer::from_text(text);

    Config cfg = tiny_cfg();
    cfg.vocab_size = tok.vocab_size();
    cfg.context_length = 8;
    InitRng rng(1);
    const Gpt model(cfg, rng);

    const TextDataset data = TextDataset::from_text(text, tok, 8);

    const float before = estimate_loss(model, data, 3);

    TrainConfig tcfg;
    tcfg.max_steps = 20;
    tcfg.eval_interval = 1000;  // suppress logging
    tcfg.learning_rate = 0.05f;
    const float after = train(model, tok, data, data, tcfg);

    INFO("loss " << before << " -> " << after);
    REQUIRE(std::isfinite(after));
    REQUIRE(after < before);
}

TEST_CASE("scalar generate produces the requested length", "[train]") {
    const std::string text = "abcabcabcabcabcabc";
    const CharTokenizer tok = CharTokenizer::from_text(text);

    Config cfg = tiny_cfg();
    cfg.vocab_size = tok.vocab_size();
    InitRng rng(2);
    const Gpt model(cfg, rng);

    const std::string out = generate(model, tok, "abc", 5, 1.0f, 2);
    // The prompt plus five new characters.
    REQUIRE(out.size() == 8);
    REQUIRE(out.starts_with("abc"));
}

TEST_CASE("sample_top_k restricts to the top entries", "[train]") {
    // Only index 2 survives a top-1 filter.
    const std::vector<float> probs{0.1f, 0.2f, 0.7f};
    for (int i = 0; i < 10; ++i) {
        REQUIRE(sample_top_k(probs, 1) == 2);
    }
    // With k = 3 every index is reachable across draws.
    std::set<std::size_t> seen;
    for (int i = 0; i < 200; ++i) {
        seen.insert(sample_top_k(probs, 3));
    }
    REQUIRE(seen.size() > 1);

    // k larger than the distribution is clamped.
    REQUIRE(sample_top_k(probs, 99) < 3);
}

TEST_CASE("lcg_float stays in range and varies", "[train]") {
    std::set<float> seen;
    for (std::uint64_t s = 0; s < 50; ++s) {
        const float v = lcg_float(s);
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
        seen.insert(v);
    }
    REQUIRE(seen.size() > 40);
}

// -----------------------------------------------------------------------------
// AdamW2 and the tensor training loop
// -----------------------------------------------------------------------------

TEST_CASE("adamw2 moves weights against their gradients", "[train2]") {
    const TensorNode w = TensorNode::leaf(Mat::ones(2, 3));
    AdamW2 opt({w}, 0.1f);
    opt.weight_decay = 0.0f;

    w.set_grad(Mat::ones(2, 3));
    opt.step({w});

    REQUIRE(opt.step_count == 1);
    for (float v : w.data().data) {
        REQUIRE(approx(v, 1.0f - 0.1f, 1e-3f));
    }
}

TEST_CASE("adamw2 skips an all-zero gradient", "[train2]") {
    const TensorNode w = TensorNode::leaf(Mat::ones(2, 2).scale(3.0f));
    AdamW2 opt({w}, 0.1f);
    opt.step({w});  // gradient starts at zero
    for (float v : w.data().data) {
        REQUIRE(v == 3.0f);
    }
}

TEST_CASE("lr scheduler warms up then decays", "[train2]") {
    const LrScheduler sched{1.0f, 0.1f, 10, 100};

    // Linear warmup, reaching the peak on the last warmup step.
    REQUIRE(approx(sched.get(0), 0.1f));
    REQUIRE(approx(sched.get(4), 0.5f));
    REQUIRE(approx(sched.get(9), 1.0f));

    // Cosine decay afterwards, monotonically down to lr_min.
    REQUIRE(approx(sched.get(10), 1.0f, 1e-3f));
    REQUIRE(sched.get(50) < sched.get(20));
    REQUIRE(approx(sched.get(99), 0.1f, 1e-2f));
}

TEST_CASE("token_accuracy counts argmax matches", "[train2]") {
    // Row 0's argmax is 1, row 1's is 0.
    const Mat logits({0.1f, 0.9f, 0.0f, 5.0f, 1.0f, 2.0f}, 2, 3);
    REQUIRE(approx(token_accuracy(logits, {1, 0}), 1.0f));
    REQUIRE(approx(token_accuracy(logits, {1, 2}), 0.5f));
    REQUIRE(approx(token_accuracy(logits, {0, 2}), 0.0f));
}

TEST_CASE("cross_entropy_smoothed", "[train2]") {
    const Mat logits({0.0f, 0.0f, 0.0f, 0.0f}, 1, 4);

    // A uniform distribution over 4 tokens gives log(4).
    REQUIRE(approx(cross_entropy_smoothed(logits, {0}, 0.0f), std::log(4.0f), 1e-4f));
    // With a uniform distribution, smoothing changes nothing.
    REQUIRE(approx(cross_entropy_smoothed(logits, {0}, 0.1f), std::log(4.0f), 1e-4f));

    // A confident, correct prediction has a low loss; smoothing raises it,
    // because the smoothed target still puts mass on the other tokens.
    const Mat sharp({10.0f, 0.0f, 0.0f, 0.0f}, 1, 4);
    const float plain = cross_entropy_smoothed(sharp, {0}, 0.0f);
    const float smoothed = cross_entropy_smoothed(sharp, {0}, 0.1f);
    REQUIRE(plain < 0.01f);
    REQUIRE(smoothed > plain);
}

TEST_CASE("tensor training reduces the loss", "[train2]") {
    std::string text;
    for (int i = 0; i < 30; ++i) {
        text += "abcabcabc";
    }
    const CharTokenizer tok = CharTokenizer::from_text(text);

    Config cfg = tiny_cfg();
    cfg.vocab_size = tok.vocab_size();
    InitRng rng(3);
    const Gpt2 model(cfg, rng);

    const TextDataset data = TextDataset::from_text(text, tok, 8);

    const float before = estimate_loss2(model, data, 3);

    TrainConfig2 tcfg;
    tcfg.max_steps = 20;
    tcfg.eval_interval = 1000;
    tcfg.learning_rate = 0.05f;
    const float after = train2(model, data, data, tcfg);

    INFO("loss " << before << " -> " << after);
    REQUIRE(std::isfinite(after));
    REQUIRE(after < before);
}

TEST_CASE("tensor training honours batching and accumulation", "[train2]") {
    std::string text;
    for (int i = 0; i < 30; ++i) {
        text += "abcabcabc";
    }
    const CharTokenizer tok = CharTokenizer::from_text(text);

    Config cfg = tiny_cfg();
    cfg.vocab_size = tok.vocab_size();
    InitRng rng(4);
    const Gpt2 model(cfg, rng);
    const TextDataset data = TextDataset::from_text(text, tok, 8);

    TrainConfig2 tcfg;
    tcfg.max_steps = 6;
    tcfg.eval_interval = 1000;
    tcfg.batch_size = 2;
    tcfg.accumulate_steps = 2;
    tcfg.label_smoothing = 0.1f;
    const float after = train2(model, data, data, tcfg);
    REQUIRE(std::isfinite(after));
}

TEST_CASE("tensor training writes a checkpoint", "[train2]") {
    const std::string path = "/tmp/rt_train2_ckpt.bin";
    std::string text;
    for (int i = 0; i < 20; ++i) {
        text += "abcabc";
    }
    const CharTokenizer tok = CharTokenizer::from_text(text);

    Config cfg = tiny_cfg();
    cfg.vocab_size = tok.vocab_size();
    InitRng rng(5);
    const Gpt2 model(cfg, rng);
    const TextDataset data = TextDataset::from_text(text, tok, 8);

    TrainConfig2 tcfg;
    tcfg.max_steps = 2;
    tcfg.eval_interval = 1000;
    tcfg.checkpoint_path = path;
    (void)train2(model, data, data, tcfg);

    const auto loaded = load_checkpoint(path);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->size() == model.parameters().size());
    REQUIRE((*loaded)[0].first == "param_0");
    std::remove(path.c_str());
}

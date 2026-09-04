#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "rt/autograd.hpp"
#include "rt/nn.hpp"
#include "rt/tensor.hpp"

using namespace rt;
using Shape = std::vector<std::size_t>;

namespace {
bool approx(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) < tol; }
}  // namespace

// -----------------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------------

TEST_CASE("tensor strides and indexing", "[tensor]") {
    REQUIRE(Tensor::compute_strides({2, 3}) == Shape{3, 1});
    REQUIRE(Tensor::compute_strides({2, 3, 4}) == Shape{12, 4, 1});
    REQUIRE(Tensor::compute_strides({5}) == Shape{1});

    Tensor t = Tensor::zeros({3, 4});
    REQUIRE(t.flat_index({1, 2}) == 6);
    t.set({1, 2}, 9.0f);
    REQUIRE(t.get({1, 2}) == 9.0f);
    REQUIRE(t.numel() == 12);
    REQUIRE(t.ndim() == 2);
}

TEST_CASE("tensor constructors", "[tensor]") {
    REQUIRE(Tensor::ones({2, 2}).sum_all() == 4.0f);
    REQUIRE(Tensor::zeros({2, 2}).sum_all() == 0.0f);
    const Tensor r = Tensor::arange(5);
    REQUIRE(r.shape == Shape{5});
    REQUIRE(r.data == std::vector<float>{0, 1, 2, 3, 4});
}

TEST_CASE("tensor elementwise ops", "[tensor]") {
    const Tensor a({1, 2, 3, 4}, {2, 2});
    const Tensor b({10, 20, 30, 40}, {2, 2});
    REQUIRE(a.add(b).data == std::vector<float>{11, 22, 33, 44});
    REQUIRE(b.sub(a).data == std::vector<float>{9, 18, 27, 36});
    REQUIRE(a.mul(b).data == std::vector<float>{10, 40, 90, 160});
    REQUIRE(a.scale(2.0f).data == std::vector<float>{2, 4, 6, 8});
    REQUIRE(a.map([](float x) { return x * x; }).data == std::vector<float>{1, 4, 9, 16});
}

TEST_CASE("tensor reductions over the last dimension", "[tensor]") {
    const Tensor t({1, 2, 3, 40, 50, 60}, {2, 3});

    const Tensor sums = t.sum_last_dim();
    REQUIRE(sums.shape == Shape{2});
    REQUIRE(sums.data == std::vector<float>{6, 150});

    const Tensor maxes = t.max_last_dim();
    REQUIRE(maxes.shape == Shape{2});
    REQUIRE(maxes.data == std::vector<float>{3, 60});

    REQUIRE(t.sum_all() == 156.0f);

    // Reducing a 1-D tensor collapses to shape [1] rather than an empty shape.
    const Tensor flat({1, 2, 3}, {3});
    REQUIRE(flat.sum_last_dim().shape == Shape{1});
    REQUIRE(flat.sum_last_dim().data == std::vector<float>{6});
}

TEST_CASE("tensor matmul and transpose", "[tensor]") {
    // [1 2 3; 4 5 6] @ [1 2; 3 4; 5 6] = [22 28; 49 64]
    const Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor b({1, 2, 3, 4, 5, 6}, {3, 2});
    const Tensor c = a.matmul(b);
    REQUIRE(c.shape == Shape{2, 2});
    REQUIRE(c.data == std::vector<float>{22, 28, 49, 64});

    const Tensor at = a.transpose();
    REQUIRE(at.shape == Shape{3, 2});
    REQUIRE(at.get({0, 1}) == a.get({1, 0}));
    REQUIRE(at.transpose().data == a.data);
}

TEST_CASE("tensor reshape keeps the data", "[tensor]") {
    const Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor r = t.reshape({3, 2});
    REQUIRE(r.shape == Shape{3, 2});
    REQUIRE(r.strides == Shape{2, 1});
    REQUIRE(r.data == t.data);
}

TEST_CASE("tensor softmax rows sum to one and stay stable", "[tensor]") {
    const Tensor t({1, 2, 3, 1, 2, 3}, {2, 3});
    const Tensor s = softmax(t);
    REQUIRE(s.shape == t.shape);
    for (std::size_t r = 0; r < 2; ++r) {
        float sum = 0.0f;
        for (std::size_t c = 0; c < 3; ++c) {
            sum += s.get({r, c});
        }
        REQUIRE(approx(sum, 1.0f));
    }
    // Larger logits get more probability.
    REQUIRE(s.get({0, 2}) > s.get({0, 1}));
    REQUIRE(s.get({0, 1}) > s.get({0, 0}));

    // Subtracting the row max keeps huge logits finite.
    const Tensor big({1000.0f, 1001.0f, 1002.0f}, {3});
    const Tensor sb = softmax(big);
    for (float v : sb.data) {
        REQUIRE(std::isfinite(v));
    }
    REQUIRE(approx(sb.sum_all(), 1.0f));
}

// -----------------------------------------------------------------------------
// Value (scalar autograd)
// -----------------------------------------------------------------------------

TEST_CASE("value add and mul gradients", "[autograd]") {
    // f = x * y + x, so df/dx = y + 1 and df/dy = x.
    const Value x(3.0f);
    const Value y(4.0f);
    const Value f = x.mul(y).add(x);
    REQUIRE(f.val() == 15.0f);

    f.backward();
    REQUIRE(approx(x.grad(), 5.0f));
    REQUIRE(approx(y.grad(), 3.0f));
}

TEST_CASE("value reuses a shared node once", "[autograd]") {
    // f = x * x, so df/dx = 2x. The node appears twice but is visited once.
    const Value x(5.0f);
    const Value f = x.mul(x);
    f.backward();
    REQUIRE(approx(x.grad(), 10.0f));
}

TEST_CASE("value pow, exp, ln", "[autograd]") {
    const Value x(2.0f);
    const Value p = x.pow(3.0f);
    REQUIRE(approx(p.val(), 8.0f));
    p.backward();
    REQUIRE(approx(x.grad(), 12.0f));  // 3 * x^2

    const Value e_in(1.0f);
    const Value e = e_in.exp();
    REQUIRE(approx(e.val(), std::exp(1.0f)));
    e.backward();
    REQUIRE(approx(e_in.grad(), std::exp(1.0f)));  // d/dx exp = exp

    const Value l_in(4.0f);
    const Value l = l_in.ln();
    REQUIRE(approx(l.val(), std::log(4.0f)));
    l.backward();
    REQUIRE(approx(l_in.grad(), 0.25f));  // 1/x
}

TEST_CASE("value relu gates the gradient", "[autograd]") {
    const Value pos(2.0f);
    const Value out_pos = pos.relu();
    REQUIRE(out_pos.val() == 2.0f);
    out_pos.backward();
    REQUIRE(pos.grad() == 1.0f);

    const Value neg(-2.0f);
    const Value out_neg = neg.relu();
    REQUIRE(out_neg.val() == 0.0f);
    out_neg.backward();
    REQUIRE(neg.grad() == 0.0f);
}

TEST_CASE("value tanh", "[autograd]") {
    const Value x(0.5f);
    const Value t = x.tanh();
    REQUIRE(approx(t.val(), std::tanh(0.5f)));
    t.backward();
    const float th = std::tanh(0.5f);
    REQUIRE(approx(x.grad(), 1.0f - th * th));
}

TEST_CASE("value neg, sub and div", "[autograd]") {
    const Value a(6.0f);
    const Value b(3.0f);
    REQUIRE(approx(a.neg().val(), -6.0f));
    REQUIRE(approx(a.sub(b).val(), 3.0f));

    const Value q = a.div(b);
    REQUIRE(approx(q.val(), 2.0f));
    q.backward();
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    REQUIRE(approx(a.grad(), 1.0f / 3.0f));
    REQUIRE(approx(b.grad(), -6.0f / 9.0f));
}

TEST_CASE("value gradients match a numerical estimate", "[autograd]") {
    // f(x) = tanh(x * 2 + 1), checked against a central difference.
    const auto f = [](float xv) { return std::tanh(xv * 2.0f + 1.0f); };
    const float x0 = 0.3f;
    constexpr float h = 1e-3f;
    const float numerical = (f(x0 + h) - f(x0 - h)) / (2.0f * h);

    const Value x(x0);
    const Value out = x.mul(Value(2.0f)).add(Value(1.0f)).tanh();
    out.backward();
    REQUIRE(approx(x.grad(), numerical, 1e-3f));
}

TEST_CASE("value zero_grad, set_val, set_grad and label", "[autograd]") {
    const Value x = Value::with_label(1.0f, "w1");
    x.set_grad(5.0f);
    REQUIRE(x.grad() == 5.0f);
    x.zero_grad();
    REQUIRE(x.grad() == 0.0f);
    x.set_val(2.5f);
    REQUIRE(x.val() == 2.5f);
    REQUIRE(x.label("renamed").debug_string().find("renamed") != std::string::npos);
}

// -----------------------------------------------------------------------------
// Scalar layers
// -----------------------------------------------------------------------------

TEST_CASE("linear shape, bias init and parameters", "[nn]") {
    InitRng rng(0);
    const Linear layer(3, 2, rng);
    REQUIRE(layer.weight.size() == 2);
    REQUIRE(layer.weight[0].size() == 3);
    for (const Value& b : layer.bias) {
        REQUIRE(b.val() == 0.0f);
    }
    // 2*3 weights + 2 biases
    REQUIRE(layer.parameters().size() == 8);

    const std::vector<Value> out = layer.forward({Value(1.0f), Value(2.0f), Value(3.0f)});
    REQUIRE(out.size() == 2);
    for (const Value& o : out) {
        REQUIRE(std::isfinite(o.val()));
    }
}

TEST_CASE("linear forward matches a hand-computed dot product", "[nn]") {
    InitRng rng(1);
    Linear layer(2, 1, rng);
    layer.weight[0][0].set_val(2.0f);
    layer.weight[0][1].set_val(-3.0f);
    layer.bias[0].set_val(0.5f);

    const std::vector<Value> out = layer.forward({Value(4.0f), Value(1.0f)});
    REQUIRE(approx(out[0].val(), 2.0f * 4.0f + -3.0f * 1.0f + 0.5f));

    out[0].backward();
    // d(out)/d(w0) = input[0]
    REQUIRE(approx(layer.weight[0][0].grad(), 4.0f));
    REQUIRE(approx(layer.bias[0].grad(), 1.0f));
}

TEST_CASE("scalar relu and gelu", "[nn]") {
    const std::vector<Value> xs{Value(-1.0f), Value(0.0f), Value(2.0f)};

    const std::vector<Value> r = relu(xs);
    REQUIRE(r[0].val() == 0.0f);
    REQUIRE(r[2].val() == 2.0f);

    const std::vector<Value> g = gelu(xs);
    // GELU(x) = x * sigmoid(1.702 x)
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const float x = xs[i].val();
        const float want = x / (1.0f + std::exp(-1.702f * x));
        INFO("gelu at " << x);
        REQUIRE(approx(g[i].val(), want));
    }
    // GELU is smooth: it passes a little negative mass through, unlike ReLU.
    REQUIRE(g[0].val() < 0.0f);
}

TEST_CASE("scalar softmax", "[nn]") {
    const std::vector<Value> xs{Value(1.0f), Value(2.0f), Value(3.0f)};
    const std::vector<Value> s = softmax(xs);

    float sum = 0.0f;
    for (const Value& v : s) {
        sum += v.val();
        REQUIRE(v.val() > 0.0f);
    }
    REQUIRE(approx(sum, 1.0f));
    REQUIRE(s[2].val() > s[1].val());

    // Huge logits stay finite thanks to the max shift.
    const std::vector<Value> big{Value(1000.0f), Value(1001.0f)};
    for (const Value& v : softmax(big)) {
        REQUIRE(std::isfinite(v.val()));
    }
}

TEST_CASE("layernorm normalizes and exposes its parameters", "[nn]") {
    const LayerNorm ln(4);
    REQUIRE(ln.parameters().size() == 8);  // gamma + beta
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(ln.gamma[i].val() == 1.0f);
        REQUIRE(ln.beta[i].val() == 0.0f);
    }

    const std::vector<Value> xs{Value(1.0f), Value(2.0f), Value(3.0f), Value(4.0f)};
    const std::vector<Value> out = ln.forward(xs);

    float mean = 0.0f;
    for (const Value& v : out) {
        mean += v.val();
    }
    mean /= 4.0f;
    REQUIRE(approx(mean, 0.0f, 1e-4f));

    float var = 0.0f;
    for (const Value& v : out) {
        var += (v.val() - mean) * (v.val() - mean);
    }
    REQUIRE(approx(std::sqrt(var / 4.0f), 1.0f, 1e-3f));
}

TEST_CASE("scalar mlp shape and backward", "[nn]") {
    InitRng rng(3);
    const Mlp mlp(2, rng);
    // fc1 is 2 -> 8, fc2 is 8 -> 2: (8*2 + 8) + (2*8 + 2) parameters.
    REQUIRE(mlp.parameters().size() == (8 * 2 + 8) + (2 * 8 + 2));

    const std::vector<Value> out = mlp.forward({Value(0.5f), Value(-0.3f)});
    REQUIRE(out.size() == 2);

    Value loss = out[0];
    for (std::size_t i = 1; i < out.size(); ++i) {
        loss = loss.add(out[i]);
    }
    loss.backward();

    bool any_nonzero = false;
    for (const Value& p : mlp.parameters()) {
        REQUIRE(std::isfinite(p.grad()));
        any_nonzero = any_nonzero || std::fabs(p.grad()) > 1e-9f;
    }
    REQUIRE(any_nonzero);

    mlp.zero_grad();
    for (const Value& p : mlp.parameters()) {
        REQUIRE(p.grad() == 0.0f);
    }
}

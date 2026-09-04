#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numeric>

#include "rt/tensor_node.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;
using rt::testing::numerical_grad;

namespace {

/// Sum of a node's forward values -- the scalar "loss" the gradient checks use.
float sum_of(const TensorNode& n) {
    return n.data().sum();
}

/// Seed `out` with a gradient of ones and fire its backward rule once.
void backward_from_ones(const TensorNode& out) {
    out.seed_grad_ones();
    out.call_backward_fn();
}

/// Assert every element of the analytical gradient matches the numerical one.
void require_grad_matches(const Mat& analytical, const Mat& numerical, const char* what,
                          float tol = 1e-3f) {
    REQUIRE(analytical.rows == numerical.rows);
    REQUIRE(analytical.cols == numerical.cols);
    for (std::size_t r = 0; r < analytical.rows; ++r) {
        for (std::size_t c = 0; c < analytical.cols; ++c) {
            INFO(what << "[" << r << "," << c << "]: analytical=" << analytical.at(r, c)
                      << " numerical=" << numerical.at(r, c));
            REQUIRE(std::fabs(analytical.at(r, c) - numerical.at(r, c)) < tol);
        }
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// matmul / add / add_bias
// -----------------------------------------------------------------------------

TEST_CASE("matmul gradient w.r.t. A", "[tensor_node]") {
    const Mat a_data({1., 2., 3., 4., 5., 6.}, 2, 3);
    const Mat b_data({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 3, 2);

    const Mat num = numerical_grad(
        [&](const Mat& a) {
            return sum_of(TensorNode::leaf(a).matmul(TensorNode::leaf(b_data)));
        },
        a_data);

    const TensorNode a = TensorNode::leaf(a_data);
    backward_from_ones(a.matmul(TensorNode::leaf(b_data)));
    require_grad_matches(a.grad(), num, "dA");
}

TEST_CASE("matmul gradient w.r.t. B", "[tensor_node]") {
    const Mat a_data({1., 2., 3., 4., 5., 6.}, 2, 3);
    const Mat b_data({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 3, 2);

    const Mat num = numerical_grad(
        [&](const Mat& b) {
            return sum_of(TensorNode::leaf(a_data).matmul(TensorNode::leaf(b)));
        },
        b_data);

    const TensorNode b = TensorNode::leaf(b_data);
    backward_from_ones(TensorNode::leaf(a_data).matmul(b));
    require_grad_matches(b.grad(), num, "dB");
}

TEST_CASE("add passes the gradient to both inputs", "[tensor_node]") {
    const TensorNode a = TensorNode::leaf(Mat({1., 2., 3., 4.}, 2, 2));
    const TensorNode b = TensorNode::leaf(Mat({5., 6., 7., 8.}, 2, 2));
    const TensorNode c = a.add(b);
    REQUIRE(c.data().at(1, 1) == 12.0f);

    backward_from_ones(c);
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(a.grad().data[i] == 1.0f);
        REQUIRE(b.grad().data[i] == 1.0f);
    }
}

TEST_CASE("add_bias gradient", "[tensor_node]") {
    const Mat a_data({1., 2., 3., 4., 5., 6.}, 3, 2);
    const Mat b_data({0.5f, -0.5f}, 1, 2);

    const Mat num = numerical_grad(
        [&](const Mat& b) {
            return sum_of(TensorNode::leaf(a_data).add_bias(TensorNode::leaf(b)));
        },
        b_data);

    const TensorNode b = TensorNode::leaf(b_data);
    backward_from_ones(TensorNode::leaf(a_data).add_bias(b));
    require_grad_matches(b.grad(), num, "d_bias");
}

// -----------------------------------------------------------------------------
// Activations
// -----------------------------------------------------------------------------

TEST_CASE("gelu gradient", "[tensor_node]") {
    const Mat x_data({-1.0f, 0.0f, 0.5f, 2.0f}, 1, 4);
    const Mat num =
        numerical_grad([](const Mat& x) { return sum_of(TensorNode::leaf(x).gelu()); }, x_data);

    const TensorNode x = TensorNode::leaf(x_data);
    backward_from_ones(x.gelu());
    require_grad_matches(x.grad(), num, "GELU grad");
}

TEST_CASE("gelu_tanh gradient", "[tensor_node]") {
    const Mat x_data({-1.5f, -0.2f, 0.0f, 0.7f, 2.5f}, 1, 5);
    const Mat num = numerical_grad(
        [](const Mat& x) { return sum_of(TensorNode::leaf(x).gelu_tanh()); }, x_data);

    const TensorNode x = TensorNode::leaf(x_data);
    backward_from_ones(x.gelu_tanh());
    require_grad_matches(x.grad(), num, "gelu_tanh grad");
}

TEST_CASE("silu gradient", "[tensor_node]") {
    const Mat x_data({-2.0f, -0.5f, 0.0f, 0.5f, 3.0f}, 1, 5);
    const Mat num =
        numerical_grad([](const Mat& x) { return sum_of(TensorNode::leaf(x).silu()); }, x_data);

    const TensorNode x = TensorNode::leaf(x_data);
    backward_from_ones(x.silu());
    require_grad_matches(x.grad(), num, "SiLU grad");
}

TEST_CASE("clamp forward and gradient masking", "[tensor_node]") {
    const TensorNode x = TensorNode::leaf(Mat({-3.0f, -0.5f, 0.5f, 4.0f}, 1, 4));
    const TensorNode c = x.clamp(-1.0f, 1.0f);
    REQUIRE(c.data().at(0, 0) == -1.0f);
    REQUIRE(c.data().at(0, 1) == -0.5f);
    REQUIRE(c.data().at(0, 3) == 1.0f);

    backward_from_ones(c);
    // Gradient flows only where the input was strictly inside the range.
    REQUIRE(x.grad().at(0, 0) == 0.0f);
    REQUIRE(x.grad().at(0, 1) == 1.0f);
    REQUIRE(x.grad().at(0, 2) == 1.0f);
    REQUIRE(x.grad().at(0, 3) == 0.0f);
}

TEST_CASE("mul_elem_node gradient", "[tensor_node]") {
    const Mat a_data({1.0f, -2.0f, 3.0f, 0.5f}, 2, 2);
    const Mat b_data({0.25f, 4.0f, -1.0f, 2.0f}, 2, 2);

    const Mat num_a = numerical_grad(
        [&](const Mat& a) {
            return sum_of(TensorNode::leaf(a).mul_elem_node(TensorNode::leaf(b_data)));
        },
        a_data);

    const TensorNode a = TensorNode::leaf(a_data);
    const TensorNode b = TensorNode::leaf(b_data);
    backward_from_ones(a.mul_elem_node(b));
    require_grad_matches(a.grad(), num_a, "dA");
    // dB = dC * A, with dC all ones.
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(approx(b.grad().data[i], a_data.data[i]));
    }
}

// -----------------------------------------------------------------------------
// softmax
// -----------------------------------------------------------------------------

TEST_CASE("softmax rows are probability distributions", "[tensor_node]") {
    const TensorNode s = TensorNode::leaf(Mat({1., 2., 3., 4., 5., 6.}, 2, 3)).softmax();
    for (std::size_t r = 0; r < 2; ++r) {
        float row_sum = 0.0f;
        for (std::size_t c = 0; c < 3; ++c) {
            row_sum += s.data().at(r, c);
        }
        INFO("row " << r << " sum = " << row_sum);
        REQUIRE(approx(row_sum, 1.0f));
    }
}

TEST_CASE("softmax gradient", "[tensor_node]") {
    const Mat x_data({1.0f, 2.0f, 0.5f}, 1, 3);

    // Weighted loss so the gradient is non-trivial: sum_i s[i] * (i + 1).
    const Mat num = numerical_grad(
        [](const Mat& x) {
            const Mat s = TensorNode::leaf(x).softmax().data();
            float acc = 0.0f;
            for (std::size_t i = 0; i < s.data.size(); ++i) {
                acc += s.data[i] * static_cast<float>(i + 1);
            }
            return acc;
        },
        x_data);

    const TensorNode x = TensorNode::leaf(x_data);
    const TensorNode s = x.softmax();
    s.set_grad(Mat({1.0f, 2.0f, 3.0f}, 1, 3));  // dLoss/dS[i] = i + 1
    s.call_backward_fn();
    require_grad_matches(x.grad(), num, "softmax grad");
}

// -----------------------------------------------------------------------------
// Normalization
// -----------------------------------------------------------------------------

TEST_CASE("layer_norm output has zero mean", "[tensor_node]") {
    const TensorNode out = TensorNode::leaf(Mat({1., 2., 3., 4.}, 1, 4))
                               .layer_norm(TensorNode::leaf(Mat::ones(1, 4)),
                                           TensorNode::leaf(Mat::zeros(1, 4)));
    const float mean = out.data().sum() / 4.0f;
    INFO("LN output mean = " << mean);
    REQUIRE(std::fabs(mean) < 1e-5f);
}

TEST_CASE("layer_norm gradients", "[tensor_node]") {
    const Mat x_data({0.5f, -0.3f, 1.2f, -0.8f}, 1, 4);
    const Mat g_data({1.0f, 0.8f, 1.2f, 0.9f}, 1, 4);
    const Mat b_data = Mat::zeros(1, 4);

    const Mat num_x = numerical_grad(
        [&](const Mat& x) {
            return sum_of(TensorNode::leaf(x).layer_norm(TensorNode::leaf(g_data),
                                                         TensorNode::leaf(b_data)));
        },
        x_data);
    const Mat num_g = numerical_grad(
        [&](const Mat& g) {
            return sum_of(TensorNode::leaf(x_data).layer_norm(TensorNode::leaf(g),
                                                              TensorNode::leaf(b_data)));
        },
        g_data);

    const TensorNode x = TensorNode::leaf(x_data);
    const TensorNode g = TensorNode::leaf(g_data);
    const TensorNode b = TensorNode::leaf(b_data);
    backward_from_ones(x.layer_norm(g, b));

    require_grad_matches(x.grad(), num_x, "LN dX");
    require_grad_matches(g.grad(), num_g, "LN dGamma");
    // dBeta = sum over rows of ones = 1 per column.
    for (std::size_t c = 0; c < 4; ++c) {
        REQUIRE(approx(b.grad().at(0, c), 1.0f));
    }
}

TEST_CASE("rms_norm gradients", "[tensor_node]") {
    const Mat x_data = Mat::from_fn(2, 4, [](std::size_t r, std::size_t c) {
        return (static_cast<float>(r) + 1.0f) * (static_cast<float>(c) * 0.3f - 0.4f);
    });
    const Mat g_data({1.0f, 0.8f, 1.2f, 0.9f}, 1, 4);
    constexpr float eps = 1e-6f;

    const Mat num_x = numerical_grad(
        [&](const Mat& x) {
            return sum_of(TensorNode::leaf(x).rms_norm(TensorNode::leaf(g_data), eps));
        },
        x_data);
    const Mat num_g = numerical_grad(
        [&](const Mat& g) {
            return sum_of(TensorNode::leaf(x_data).rms_norm(TensorNode::leaf(g), eps));
        },
        g_data);

    const TensorNode x = TensorNode::leaf(x_data);
    const TensorNode g = TensorNode::leaf(g_data);
    backward_from_ones(x.rms_norm(g, eps));

    require_grad_matches(x.grad(), num_x, "RMS dX");
    require_grad_matches(g.grad(), num_g, "RMS dGamma");
}

TEST_CASE("rms_norm_gemma3 scales by (1 + gamma)", "[tensor_node]") {
    // With gamma = 0 the Gemma 3 form must equal a plain RMS normalization.
    const Mat x_data({1.0f, 2.0f, 3.0f, 4.0f}, 1, 4);
    constexpr float eps = 1e-6f;

    const TensorNode plain =
        TensorNode::leaf(x_data).rms_norm(TensorNode::leaf(Mat::ones(1, 4)), eps);
    const TensorNode gemma =
        TensorNode::leaf(x_data).rms_norm_gemma3(TensorNode::leaf(Mat::zeros(1, 4)), eps);
    for (std::size_t c = 0; c < 4; ++c) {
        REQUIRE(approx(gemma.data().at(0, c), plain.data().at(0, c)));
    }
}

TEST_CASE("rms_norm_gemma3 gradients", "[tensor_node]") {
    const Mat x_data = Mat::from_fn(2, 4, [](std::size_t r, std::size_t c) {
        return (static_cast<float>(r) + 1.0f) * (static_cast<float>(c) * 0.25f - 0.3f);
    });
    const Mat g_data({0.1f, -0.2f, 0.3f, 0.05f}, 1, 4);
    constexpr float eps = 1e-6f;

    const Mat num_x = numerical_grad(
        [&](const Mat& x) {
            return sum_of(TensorNode::leaf(x).rms_norm_gemma3(TensorNode::leaf(g_data), eps));
        },
        x_data);
    const Mat num_g = numerical_grad(
        [&](const Mat& g) {
            return sum_of(TensorNode::leaf(x_data).rms_norm_gemma3(TensorNode::leaf(g), eps));
        },
        g_data);

    const TensorNode x = TensorNode::leaf(x_data);
    const TensorNode g = TensorNode::leaf(g_data);
    backward_from_ones(x.rms_norm_gemma3(g, eps));

    require_grad_matches(x.grad(), num_x, "Gemma3 RMS dX");
    require_grad_matches(g.grad(), num_g, "Gemma3 RMS dGamma");
}

// -----------------------------------------------------------------------------
// RoPE
// -----------------------------------------------------------------------------

TEST_CASE("rope_apply is norm-preserving", "[tensor_node]") {
    // RoPE is an orthogonal rotation, so each row's L2 norm is unchanged.
    const Mat x_data = Mat::from_fn(4, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.1f + 0.1f;
    });
    const TensorNode out = TensorNode::leaf(x_data).rope_apply(0, 10000.0f);
    for (std::size_t r = 0; r < 4; ++r) {
        float before = 0.0f, after = 0.0f;
        for (std::size_t c = 0; c < 8; ++c) {
            before += x_data.at(r, c) * x_data.at(r, c);
            after += out.data().at(r, c) * out.data().at(r, c);
        }
        INFO("row " << r);
        REQUIRE(std::fabs(std::sqrt(before) - std::sqrt(after)) < 1e-4f);
    }
}

TEST_CASE("rope_apply gradient", "[tensor_node]") {
    // Trig in the central difference accumulates more error, hence the looser
    // tolerance than the other gradient checks.
    constexpr float tol = 5e-3f;
    const Mat x_data = Mat::from_fn(3, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.1f + 0.1f;
    });

    const Mat num = numerical_grad(
        [](const Mat& x) { return sum_of(TensorNode::leaf(x).rope_apply(0, 10000.0f)); }, x_data);

    const TensorNode x = TensorNode::leaf(x_data);
    backward_from_ones(x.rope_apply(0, 10000.0f));
    require_grad_matches(x.grad(), num, "RoPE dX", tol);
}

TEST_CASE("rope_apply_yarn gradient", "[tensor_node]") {
    constexpr float tol = 5e-3f;
    const Mat x_data = Mat::from_fn(3, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.1f + 0.1f;
    });
    const auto apply = [](const Mat& x) {
        return TensorNode::leaf(x).rope_apply_yarn(0, 150000.0f, 4096, 131072, 32.0f, 1.0f);
    };

    const Mat num = numerical_grad([&](const Mat& x) { return sum_of(apply(x)); }, x_data);

    const TensorNode x = TensorNode::leaf(x_data);
    backward_from_ones(
        x.rope_apply_yarn(0, 150000.0f, 4096, 131072, 32.0f, 1.0f));
    require_grad_matches(x.grad(), num, "YaRN RoPE dX", tol);
}

// -----------------------------------------------------------------------------
// Attention
// -----------------------------------------------------------------------------

TEST_CASE("causal_attention output shape", "[tensor_node]") {
    const std::size_t t = 4, d = 8;
    const TensorNode out =
        TensorNode::causal_attention(TensorNode::leaf(Mat::zeros(t, d)),
                                     TensorNode::leaf(Mat::zeros(t, d)),
                                     TensorNode::leaf(Mat::zeros(t, d)), d);
    REQUIRE(out.data().rows == t);
    REQUIRE(out.data().cols == d);
}

TEST_CASE("causal_attention first row attends only to itself", "[tensor_node]") {
    // With the causal mask, output row 0 must equal V row 0.
    const std::size_t t = 4, d = 4;
    const Mat q = Mat::from_fn(t, d, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c) * 0.1f;
    });
    const Mat v = Mat::from_fn(t, d, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c) * 0.07f + 1.0f;
    });
    const TensorNode out = TensorNode::causal_attention(
        TensorNode::leaf(q), TensorNode::leaf(q), TensorNode::leaf(v), d);
    for (std::size_t c = 0; c < d; ++c) {
        REQUIRE(approx(out.data().at(0, c), v.at(0, c)));
    }
}

TEST_CASE("causal_attention gradient w.r.t. V", "[tensor_node]") {
    const std::size_t t = 3, d = 4;
    const Mat q_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * d + c) * 0.1f;
    });
    const Mat k_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * d + c) * 0.05f;
    });
    const Mat v_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * d + c) * 0.07f;
    });

    const Mat num = numerical_grad(
        [&](const Mat& v) {
            return sum_of(TensorNode::causal_attention(TensorNode::leaf(q_data),
                                                       TensorNode::leaf(k_data),
                                                       TensorNode::leaf(v), d));
        },
        v_data);

    const TensorNode v = TensorNode::leaf(v_data);
    backward_from_ones(TensorNode::causal_attention(TensorNode::leaf(q_data),
                                                    TensorNode::leaf(k_data), v, d));
    require_grad_matches(v.grad(), num, "attn dV");
}

TEST_CASE("gqa_attention gradient w.r.t. Q", "[tensor_node]") {
    const std::size_t t = 3, n_q = 4, n_kv = 2, dh = 4;
    const Mat q_data = Mat::from_fn(t, n_q * dh, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * (n_q * dh) + c) * 0.05f + 0.1f;
    });
    const Mat k_data = Mat::from_fn(t, n_kv * dh, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * (n_kv * dh) + c) * 0.03f + 0.05f;
    });
    const Mat v_data = Mat::from_fn(t, n_kv * dh, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * (n_kv * dh) + c) * 0.04f + 0.02f;
    });

    const Mat num = numerical_grad(
        [&](const Mat& q) {
            return sum_of(TensorNode::gqa_attention(TensorNode::leaf(q), TensorNode::leaf(k_data),
                                                    TensorNode::leaf(v_data), n_q, n_kv, dh));
        },
        q_data);

    const TensorNode q = TensorNode::leaf(q_data);
    backward_from_ones(TensorNode::gqa_attention(q, TensorNode::leaf(k_data),
                                                 TensorNode::leaf(v_data), n_q, n_kv, dh));
    require_grad_matches(q.grad(), num, "GQA dQ");
}

TEST_CASE("batched_gqa_attention matches the per-head loop", "[tensor_node]") {
    const std::size_t t = 5, n_q = 4, n_kv = 2, dh = 4;
    const Mat q_data = Mat::from_fn(t, n_q * dh, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * (n_q * dh) + c) * 0.05f - 0.3f;
    });
    const Mat k_data = Mat::from_fn(t, n_kv * dh, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * (n_kv * dh) + c) * 0.03f + 0.05f;
    });
    const Mat v_data = Mat::from_fn(t, n_kv * dh, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * (n_kv * dh) + c) * 0.04f + 0.02f;
    });

    const TensorNode looped =
        TensorNode::gqa_attention(TensorNode::leaf(q_data), TensorNode::leaf(k_data),
                                  TensorNode::leaf(v_data), n_q, n_kv, dh);
    const TensorNode batched =
        TensorNode::batched_gqa_attention(TensorNode::leaf(q_data), TensorNode::leaf(k_data),
                                          TensorNode::leaf(v_data), n_q, n_kv, dh);

    REQUIRE(batched.data().rows == looped.data().rows);
    REQUIRE(batched.data().cols == looped.data().cols);
    for (std::size_t i = 0; i < looped.data().data.size(); ++i) {
        INFO("element " << i);
        REQUIRE(std::fabs(batched.data().data[i] - looped.data().data[i]) < 1e-4f);
    }

    // And their gradients agree too.
    const TensorNode q1 = TensorNode::leaf(q_data);
    const TensorNode q2 = TensorNode::leaf(q_data);
    backward_from_ones(TensorNode::gqa_attention(q1, TensorNode::leaf(k_data),
                                                 TensorNode::leaf(v_data), n_q, n_kv, dh));
    backward_from_ones(TensorNode::batched_gqa_attention(q2, TensorNode::leaf(k_data),
                                                         TensorNode::leaf(v_data), n_q, n_kv, dh));
    require_grad_matches(q2.grad(), q1.grad(), "batched vs looped dQ");
}

// -----------------------------------------------------------------------------
// Flash Attention
// -----------------------------------------------------------------------------

TEST_CASE("flash_attention output shape", "[tensor_node]") {
    const std::size_t t = 5, d = 8;
    const auto fill = [&](float k) {
        return Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
            return static_cast<float>(r * d + c) * k;
        });
    };
    const TensorNode out =
        TensorNode::flash_attention(TensorNode::leaf(fill(0.1f)), TensorNode::leaf(fill(0.05f)),
                                    TensorNode::leaf(fill(0.07f)), d);
    REQUIRE(out.data().rows == t);
    REQUIRE(out.data().cols == d);
}

TEST_CASE("flash_attention matches causal_attention", "[tensor_node]") {
    const std::size_t t = 8, d = 16;
    const Mat q_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * d + c) * 0.1f - 0.5f;
    });
    const Mat k_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * d + c) * 0.05f + 0.1f;
    });
    const Mat v_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * d + c) * 0.07f - 0.2f;
    });

    const TensorNode flash = TensorNode::flash_attention(
        TensorNode::leaf(q_data), TensorNode::leaf(k_data), TensorNode::leaf(v_data), d);
    const TensorNode causal = TensorNode::causal_attention(
        TensorNode::leaf(q_data), TensorNode::leaf(k_data), TensorNode::leaf(v_data), d);

    for (std::size_t r = 0; r < t; ++r) {
        for (std::size_t c = 0; c < d; ++c) {
            INFO("flash vs causal [" << r << "," << c << "]");
            REQUIRE(std::fabs(flash.data().at(r, c) - causal.data().at(r, c)) < 1e-4f);
        }
    }
}

TEST_CASE("flash_attention matches causal_attention past the tile size", "[tensor_node]") {
    // T > BLOCK_R (64) exercises the multi-tile path in both loops.
    const std::size_t t = 70, d = 8;
    const Mat q_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * d + c) * 0.01f);
    });
    const Mat k_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return std::cos(static_cast<float>(r * d + c) * 0.013f);
    });
    const Mat v_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return std::sin(static_cast<float>(r * d + c) * 0.007f) * 0.5f;
    });

    const TensorNode flash = TensorNode::flash_attention(
        TensorNode::leaf(q_data), TensorNode::leaf(k_data), TensorNode::leaf(v_data), d);
    const TensorNode causal = TensorNode::causal_attention(
        TensorNode::leaf(q_data), TensorNode::leaf(k_data), TensorNode::leaf(v_data), d);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < flash.data().data.size(); ++i) {
        max_err = std::max(max_err, std::fabs(flash.data().data[i] - causal.data().data[i]));
    }
    INFO("flash vs causal max error " << max_err);
    REQUIRE(max_err < 1e-4f);
}

TEST_CASE("flash_attention first token attends only to itself", "[tensor_node]") {
    const std::size_t t = 6, d = 4;
    const Mat q = Mat::from_fn(t, d, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c) * 0.1f;
    });
    const Mat v = Mat::from_fn(t, d, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c) * 0.07f + 1.0f;
    });
    const TensorNode out = TensorNode::flash_attention(TensorNode::leaf(q), TensorNode::leaf(q),
                                                       TensorNode::leaf(v), d);
    for (std::size_t c = 0; c < d; ++c) {
        REQUIRE(approx(out.data().at(0, c), v.at(0, c)));
    }
    for (float x : out.data().data) {
        REQUIRE(std::isfinite(x));
    }
}

TEST_CASE("flash_attention gradients", "[tensor_node]") {
    const std::size_t t = 5, d = 4;
    const Mat q_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * d + c) * 0.05f - 0.2f;
    });
    const Mat k_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * d + c) * 0.03f + 0.1f;
    });
    const Mat v_data = Mat::from_fn(t, d, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * d + c) * 0.04f;
    });

    const auto run = [&](const Mat& q, const Mat& k, const Mat& v) {
        return sum_of(TensorNode::flash_attention(TensorNode::leaf(q), TensorNode::leaf(k),
                                                  TensorNode::leaf(v), d));
    };

    const Mat num_q = numerical_grad([&](const Mat& q) { return run(q, k_data, v_data); }, q_data);
    const Mat num_k = numerical_grad([&](const Mat& k) { return run(q_data, k, v_data); }, k_data);
    const Mat num_v = numerical_grad([&](const Mat& v) { return run(q_data, k_data, v); }, v_data);

    const TensorNode q = TensorNode::leaf(q_data);
    const TensorNode k = TensorNode::leaf(k_data);
    const TensorNode v = TensorNode::leaf(v_data);
    backward_from_ones(TensorNode::flash_attention(q, k, v, d));

    require_grad_matches(q.grad(), num_q, "flash dQ");
    require_grad_matches(k.grad(), num_k, "flash dK");
    require_grad_matches(v.grad(), num_v, "flash dV");
}

// -----------------------------------------------------------------------------
// Graph traversal
// -----------------------------------------------------------------------------

TEST_CASE("backward walks the whole graph", "[tensor_node]") {
    const TensorNode a = TensorNode::leaf(Mat({1., 2., 3., 4.}, 2, 2));
    const TensorNode b = TensorNode::leaf(Mat({0.5f, 0.5f, 0.5f, 0.5f}, 2, 2));
    const TensorNode c = a.matmul(b);
    const TensorNode g = c.gelu();

    // Reduce to a scalar so backward() has a 1x1 seed, wiring the reduction's
    // backward by hand: d(sum)/dg is all ones.
    const TensorNode loss = TensorNode::leaf(Mat({g.data().sum()}, 1, 1));
    const std::size_t gr = g.data().rows, gc = g.data().cols;
    loss.set_backward([g, gr, gc] { g.grad_add(Mat::ones(gr, gc)); }, {g});

    loss.backward();

    bool any_nonzero = false;
    for (float x : a.grad().data) {
        REQUIRE(std::isfinite(x));
        any_nonzero = any_nonzero || std::fabs(x) > 1e-6f;
    }
    REQUIRE(any_nonzero);
}

TEST_CASE("backward visits a shared node only once", "[tensor_node]") {
    // x feeds two branches that are summed. Its gradient must be the sum of the
    // two contributions, not a double-count of one traversal.
    const TensorNode x = TensorNode::leaf(Mat({1.0f, 2.0f}, 1, 2));
    const TensorNode y = x.add(x);  // y = 2x
    const TensorNode loss = TensorNode::leaf(Mat({y.data().sum()}, 1, 1));
    loss.set_backward([y] { y.grad_add(Mat::ones(1, 2)); }, {y});
    loss.backward();

    // dy/dx = 2 on each side, so dx = 2.
    REQUIRE(approx(x.grad().at(0, 0), 2.0f));
    REQUIRE(approx(x.grad().at(0, 1), 2.0f));
}

TEST_CASE("zero_grad and set_grad", "[tensor_node]") {
    const TensorNode x = TensorNode::leaf(Mat({1.0f, 2.0f}, 1, 2));
    x.set_grad(Mat({5.0f, 6.0f}, 1, 2));
    REQUIRE(x.grad().at(0, 1) == 6.0f);
    x.grad_add(Mat({1.0f, 1.0f}, 1, 2));
    REQUIRE(x.grad().at(0, 1) == 7.0f);
    x.zero_grad();
    REQUIRE(x.grad().at(0, 0) == 0.0f);
    REQUIRE(x.grad().rows == 1);
    REQUIRE(x.grad().cols == 2);
}

TEST_CASE("free_graph empties intermediates but keeps leaves", "[tensor_node]") {
    const TensorNode a = TensorNode::leaf(Mat({1., 2., 3., 4.}, 2, 2));
    const TensorNode b = TensorNode::leaf(Mat({0.5f, 0.5f, 0.5f, 0.5f}, 2, 2));
    const TensorNode c = a.matmul(b).gelu();
    REQUIRE(c.data().numel() == 4);

    c.free_graph();

    // Intermediates are emptied...
    REQUIRE(c.data().rows == 0);
    REQUIRE(c.data().cols == 0);
    // ...but the parameters survive.
    REQUIRE(a.data().numel() == 4);
    REQUIRE(b.data().numel() == 4);
}

// -----------------------------------------------------------------------------
// Checkpoints
// -----------------------------------------------------------------------------

TEST_CASE("checkpoint save / load / restore round-trip", "[tensor_node]") {
    const std::string path = "/tmp/rt_ckpt_test.bin";

    const TensorNode w = TensorNode::leaf(Mat({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 2, 3));
    const TensorNode bias = TensorNode::leaf(Mat({0.5f, -0.5f}, 1, 2));

    REQUIRE(save_checkpoint(path, {{"w", w}, {"bias", bias}}).has_value());

    const auto loaded = load_checkpoint(path);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->size() == 2);
    REQUIRE((*loaded)[0].first == "w");
    REQUIRE((*loaded)[0].second.rows == 2);
    REQUIRE((*loaded)[0].second.cols == 3);
    REQUIRE((*loaded)[0].second.data == w.data().data);
    REQUIRE((*loaded)[1].first == "bias");
    REQUIRE((*loaded)[1].second.data == bias.data().data);

    // Restore into fresh, differently-valued parameters.
    const TensorNode w2 = TensorNode::leaf(Mat::zeros(2, 3));
    const TensorNode bias2 = TensorNode::leaf(Mat::zeros(1, 2));
    REQUIRE(restore_checkpoint(path, {w2, bias2}).has_value());
    REQUIRE(w2.data().data == w.data().data);
    REQUIRE(bias2.data().data == bias.data().data);

    std::remove(path.c_str());
}

TEST_CASE("checkpoint restore reports shape and count mismatches", "[tensor_node]") {
    const std::string path = "/tmp/rt_ckpt_bad.bin";
    const TensorNode w = TensorNode::leaf(Mat({1.0f, 2.0f}, 1, 2));
    REQUIRE(save_checkpoint(path, {{"w", w}}).has_value());

    const auto wrong_count = restore_checkpoint(path, {w, w});
    REQUIRE_FALSE(wrong_count.has_value());
    REQUIRE(wrong_count.error().find("1 tensors but model has 2") != std::string::npos);

    const auto wrong_shape = restore_checkpoint(path, {TensorNode::leaf(Mat::zeros(2, 1))});
    REQUIRE_FALSE(wrong_shape.has_value());
    REQUIRE(wrong_shape.error().find("does not match parameter shape") != std::string::npos);

    REQUIRE_FALSE(load_checkpoint("/definitely/not/here.bin").has_value());

    std::remove(path.c_str());
}

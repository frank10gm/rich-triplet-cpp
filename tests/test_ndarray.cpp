#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include "rt/ndarray.hpp"

using namespace rt;
using Shape = std::vector<std::size_t>;
using Index = std::vector<std::size_t>;

namespace {
/// Collect every multi-index a shape produces, in iteration order.
std::vector<Index> all_indices(const Shape& shape) {
    std::vector<Index> out;
    for (IndexIter it(shape); it.valid(); it.next()) {
        out.push_back(it.index());
    }
    return out;
}
}  // namespace

// -----------------------------------------------------------------------------
// c_strides
// -----------------------------------------------------------------------------

TEST_CASE("c_strides", "[ndarray]") {
    REQUIRE(c_strides({5}) == Shape{1});
    REQUIRE(c_strides({3, 4}) == Shape{4, 1});
    REQUIRE(c_strides({2, 3, 4}) == Shape{12, 4, 1});
    REQUIRE(c_strides({2, 3, 4, 5}) == Shape{60, 20, 5, 1});
    REQUIRE(c_strides({}).empty());
}

// -----------------------------------------------------------------------------
// IndexIter
// -----------------------------------------------------------------------------

TEST_CASE("IndexIter walks C-order", "[ndarray]") {
    REQUIRE(all_indices({2, 3}) ==
            std::vector<Index>{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}});
    REQUIRE(all_indices({4}) == std::vector<Index>{{0}, {1}, {2}, {3}});
    // A scalar shape yields exactly one (empty) index.
    REQUIRE(all_indices({}) == std::vector<Index>{{}});
    // Any zero dimension yields nothing.
    REQUIRE(all_indices({0, 3}).empty());
    REQUIRE(all_indices({2, 3, 4}).size() == 24);
}

// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------

TEST_CASE("zeros and ones", "[ndarray]") {
    const NDArray nd = NDArray::zeros({2, 3, 4});
    REQUIRE(nd.shape == Shape{2, 3, 4});
    REQUIRE(nd.strides == Shape{12, 4, 1});
    REQUIRE(nd.numel() == 24);
    for (IndexIter it(nd.shape); it.valid(); it.next()) {
        REQUIRE(nd.at(it.index()) == 0.0f);
    }

    const NDArray on = NDArray::ones({2, 3});
    for (IndexIter it(on.shape); it.valid(); it.next()) {
        REQUIRE(on.at(it.index()) == 1.0f);
    }
}

TEST_CASE("from_fn and from_vec", "[ndarray]") {
    const NDArray nd = NDArray::from_fn(
        {2, 3}, [](const Index& idx) { return static_cast<float>(idx[0] * 3 + idx[1]); });
    REQUIRE(nd.at({0, 0}) == 0.0f);
    REQUIRE(nd.at({1, 2}) == 5.0f);
    REQUIRE(nd.at({0, 2}) == 2.0f);

    const NDArray v = NDArray::from_vec({1, 2, 3, 4, 5, 6}, {2, 3});
    REQUIRE(v.at({0, 0}) == 1.0f);
    REQUIRE(v.at({1, 2}) == 6.0f);
}

// -----------------------------------------------------------------------------
// Indexing
// -----------------------------------------------------------------------------

TEST_CASE("flat_index", "[ndarray]") {
    REQUIRE(NDArray::zeros({3, 4}).flat_index({1, 2}) == 6);       // 1*4 + 2
    REQUIRE(NDArray::zeros({2, 3, 4}).flat_index({1, 2, 3}) == 23);  // 12 + 8 + 3
}

TEST_CASE("at_mut writes through", "[ndarray]") {
    NDArray nd = NDArray::zeros({3, 4});
    nd.at_mut({2, 3}) = 99.0f;
    REQUIRE(nd.at({2, 3}) == 99.0f);
    REQUIRE(nd.at({0, 0}) == 0.0f);
}

TEST_CASE("at_mut copies on write when storage is shared", "[ndarray]") {
    // A view shares storage; mutating it must not disturb the original.
    const NDArray base = NDArray::from_fn({2, 3}, [](const Index& i) {
        return static_cast<float>(i[0] * 3 + i[1]);
    });
    NDArray view = base.reshape({6});
    view.at_mut({0}) = -1.0f;
    REQUIRE(view.at({0}) == -1.0f);
    REQUIRE(base.at({0, 0}) == 0.0f);
}

// -----------------------------------------------------------------------------
// Mat interop
// -----------------------------------------------------------------------------

TEST_CASE("Mat interop", "[ndarray]") {
    const Mat m = Mat::from_fn(3, 4, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c);
    });
    const NDArray nd = NDArray::from_mat(m);
    REQUIRE(nd.shape == Shape{3, 4});
    REQUIRE(nd.strides == Shape{4, 1});
    REQUIRE(nd.offset == 0);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            REQUIRE(nd.at({r, c}) == m.at(r, c));
        }
    }

    const Mat m2 = nd.into_mat();
    REQUIRE(m2.rows == m.rows);
    REQUIRE(m2.cols == m.cols);
    REQUIRE(m2.data == m.data);
}

// -----------------------------------------------------------------------------
// Contiguity and views
// -----------------------------------------------------------------------------

TEST_CASE("contiguous preserves values", "[ndarray]") {
    const NDArray nd = NDArray::from_fn({3, 4}, [](const Index& i) {
        return static_cast<float>(i[0] * 4 + i[1]);
    });
    REQUIRE(nd.is_contiguous());

    const NDArray p = nd.permute({1, 0});
    REQUIRE_FALSE(p.is_contiguous());

    const NDArray cc = p.contiguous();
    REQUIRE(cc.is_contiguous());
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            REQUIRE(cc.at({i, j}) == nd.at({j, i}));
        }
    }
}

TEST_CASE("reshape", "[ndarray]") {
    const NDArray r = NDArray::zeros({2, 3, 4}).reshape({6, 4});
    REQUIRE(r.shape == Shape{6, 4});
    REQUIRE(r.strides == Shape{4, 1});

    const NDArray flat =
        NDArray::from_fn({24}, [](const Index& i) { return static_cast<float>(i[0]); });
    REQUIRE(flat.reshape({2, 3, 4}).at({1, 2, 3}) == 23.0f);

    const NDArray six =
        NDArray::from_fn({6}, [](const Index& i) { return static_cast<float>(i[0]); });
    REQUIRE(six.reshape({2, 3}).at({1, 2}) == 5.0f);
}

TEST_CASE("reshape after permute makes a contiguous copy first", "[ndarray]") {
    const NDArray nd = NDArray::from_fn({3, 4}, [](const Index& i) {
        return static_cast<float>(i[0] * 4 + i[1]);
    });
    const NDArray r = nd.permute({1, 0}).reshape({12});
    REQUIRE(r.shape == Shape{12});
    // Flat order follows the permuted view: p[0,0]=nd[0,0], p[0,1]=nd[1,0], ...
    REQUIRE(r.at({0}) == 0.0f);
    REQUIRE(r.at({1}) == 4.0f);
    REQUIRE(r.at({2}) == 8.0f);
}

TEST_CASE("permute", "[ndarray]") {
    REQUIRE(NDArray::zeros({3, 4}).permute({1, 0}).shape == Shape{4, 3});
    REQUIRE(NDArray::zeros({2, 3, 4}).permute({2, 0, 1}).shape == Shape{4, 2, 3});

    const NDArray nd = NDArray::zeros({2, 3, 4});
    const NDArray id = nd.permute({0, 1, 2});
    REQUIRE(id.shape == nd.shape);
    REQUIRE(id.strides == nd.strides);

    const NDArray vals = NDArray::from_fn({3, 4}, [](const Index& i) {
        return static_cast<float>(i[0] * 4 + i[1]);
    });
    const NDArray t = vals.permute({1, 0});
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            REQUIRE(t.at({c, r}) == vals.at({r, c}));
        }
    }
}

TEST_CASE("slice", "[ndarray]") {
    const NDArray nd = NDArray::from_fn({4, 3, 2}, [](const Index& i) {
        return static_cast<float>(i[0] * 6 + i[1] * 2 + i[2]);
    });
    const NDArray s = nd.slice(0, 2);
    REQUIRE(s.shape == Shape{3, 2});
    REQUIRE(s.at({1, 0}) == nd.at({2, 1, 0}));
    REQUIRE(s.at({2, 1}) == nd.at({2, 2, 1}));

    const NDArray mid =
        NDArray::from_fn({2, 5, 3}, [](const Index& i) { return static_cast<float>(i[1]); })
            .slice(1, 3);
    REQUIRE(mid.shape == Shape{2, 3});
    for (IndexIter it(mid.shape); it.valid(); it.next()) {
        REQUIRE(mid.at(it.index()) == 3.0f);
    }
}

TEST_CASE("expand_dims", "[ndarray]") {
    REQUIRE(NDArray::zeros({3, 4}).expand_dims(0).shape == Shape{1, 3, 4});
    REQUIRE(NDArray::zeros({3, 4}).expand_dims(1).shape == Shape{3, 1, 4});
    REQUIRE(NDArray::zeros({3, 4}).expand_dims(2).shape == Shape{3, 4, 1});

    const NDArray e =
        NDArray::from_fn({3}, [](const Index& i) { return static_cast<float>(i[0]); })
            .expand_dims(0);
    REQUIRE(e.at({0, 0}) == 0.0f);
    REQUIRE(e.at({0, 2}) == 2.0f);
}

// -----------------------------------------------------------------------------
// Elementwise ops
// -----------------------------------------------------------------------------

TEST_CASE("elementwise ops", "[ndarray]") {
    const NDArray c = NDArray::ones({2, 3}).add(NDArray::ones({2, 3}).scale(2.0f));
    for (IndexIter it(c.shape); it.valid(); it.next()) {
        REQUIRE(c.at(it.index()) == 3.0f);
    }

    const NDArray p = NDArray::ones({2, 3}).scale(3.0f).mul(NDArray::ones({2, 3}).scale(4.0f));
    REQUIRE(p.at({0, 0}) == 12.0f);
    REQUIRE(p.at({1, 2}) == 12.0f);

    const NDArray a =
        NDArray::from_fn({3}, [](const Index& i) { return static_cast<float>(i[0]); });
    REQUIRE(a.scale(2.0f).at({2}) == 4.0f);
    REQUIRE(a.add_scalar(10.0f).at({0}) == 10.0f);
    REQUIRE(a.add_scalar(10.0f).at({2}) == 12.0f);
    REQUIRE(a.map([](float x) { return x * x; }).at({2}) == 4.0f);
}

// -----------------------------------------------------------------------------
// Reductions
// -----------------------------------------------------------------------------

TEST_CASE("reduce_sum", "[ndarray]") {
    const NDArray r0 = NDArray::ones({3, 4}).reduce_sum(0);
    REQUIRE(r0.shape == Shape{4});
    for (std::size_t c = 0; c < 4; ++c) {
        REQUIRE(std::fabs(r0.at({c}) - 3.0f) < 1e-6f);
    }

    const NDArray r1 = NDArray::ones({3, 4}).reduce_sum(1);
    REQUIRE(r1.shape == Shape{3});
    for (std::size_t r = 0; r < 3; ++r) {
        REQUIRE(std::fabs(r1.at({r}) - 4.0f) < 1e-6f);
    }

    const NDArray mid = NDArray::ones({2, 3, 4}).reduce_sum(1);
    REQUIRE(mid.shape == Shape{2, 4});
    for (IndexIter it(mid.shape); it.valid(); it.next()) {
        REQUIRE(std::fabs(mid.at(it.index()) - 3.0f) < 1e-6f);
    }

    // nd[i,j] = i*3 + j; summing over j gives 9i + 3.
    const NDArray vals = NDArray::from_fn({4, 3}, [](const Index& i) {
        return static_cast<float>(i[0] * 3 + i[1]);
    });
    const NDArray sums = vals.reduce_sum(1);
    for (std::size_t i = 0; i < 4; ++i) {
        INFO("row " << i);
        REQUIRE(std::fabs(sums.at({i}) - static_cast<float>(9 * i + 3)) < 1e-5f);
    }
}

// -----------------------------------------------------------------------------
// softmax
// -----------------------------------------------------------------------------

TEST_CASE("softmax rows sum to one", "[ndarray]") {
    const NDArray s = NDArray::from_fn({3, 4},
                                       [](const Index& i) {
                                           return static_cast<float>(i[0] * 4 + i[1]);
                                       })
                          .softmax(1);
    for (std::size_t r = 0; r < 3; ++r) {
        float total = 0.0f;
        for (std::size_t c = 0; c < 4; ++c) {
            total += s.at({r, c});
        }
        INFO("row " << r << " sum = " << total);
        REQUIRE(std::fabs(total - 1.0f) < 1e-6f);
    }
}

TEST_CASE("softmax of uniform input is uniform", "[ndarray]") {
    const NDArray s = NDArray::ones({2, 4}).softmax(1);
    for (IndexIter it(s.shape); it.valid(); it.next()) {
        REQUIRE(std::fabs(s.at(it.index()) - 0.25f) < 1e-6f);
    }
}

TEST_CASE("softmax over axis 0", "[ndarray]") {
    const NDArray s = NDArray::ones({4, 3}).softmax(0);
    for (std::size_t c = 0; c < 3; ++c) {
        float col_sum = 0.0f;
        for (std::size_t r = 0; r < 4; ++r) {
            col_sum += s.at({r, c});
        }
        REQUIRE(std::fabs(col_sum - 1.0f) < 1e-6f);
    }
}

TEST_CASE("softmax is stable for large values", "[ndarray]") {
    // Subtracting the row max must prevent overflow / NaN.
    const NDArray s =
        NDArray::from_fn({1, 4}, [](const Index& i) { return static_cast<float>(i[1]) * 1000.0f; })
            .softmax(1);
    REQUIRE(s.at({0, 3}) > 0.999f);
    REQUIRE(std::isfinite(s.at({0, 0})));
}

TEST_CASE("softmax over the last axis of a 3-D array", "[ndarray]") {
    const NDArray s = NDArray::ones({2, 3, 8}).softmax(2);
    REQUIRE(s.shape == Shape{2, 3, 8});
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            float total = 0.0f;
            for (std::size_t k = 0; k < 8; ++k) {
                total += s.at({i, j, k});
            }
            REQUIRE(std::fabs(total - 1.0f) < 1e-6f);
        }
    }
}

// -----------------------------------------------------------------------------
// bmm
// -----------------------------------------------------------------------------

TEST_CASE("bmm 2-D", "[ndarray]") {
    REQUIRE(NDArray::zeros({3, 4}).bmm(NDArray::zeros({4, 5})).shape == Shape{3, 5});

    const NDArray c = NDArray::ones({3, 4}).bmm(NDArray::ones({4, 5}));
    for (IndexIter it(c.shape); it.valid(); it.next()) {
        REQUIRE(std::fabs(c.at(it.index()) - 4.0f) < 1e-5f);
    }
}

TEST_CASE("bmm 3-D", "[ndarray]") {
    REQUIRE(NDArray::zeros({2, 3, 4}).bmm(NDArray::zeros({2, 4, 5})).shape == Shape{2, 3, 5});

    const std::size_t k = 4;
    const NDArray c = NDArray::ones({2, 3, k}).bmm(NDArray::ones({2, k, 5}));
    REQUIRE(c.shape == Shape{2, 3, 5});
    for (IndexIter it(c.shape); it.valid(); it.next()) {
        REQUIRE(std::fabs(c.at(it.index()) - static_cast<float>(k)) < 1e-5f);
    }
}

TEST_CASE("bmm 4-D attention shape", "[ndarray]") {
    const std::size_t b = 2, h = 4, t = 8, d = 16;
    REQUIRE(NDArray::zeros({b, h, t, d}).bmm(NDArray::zeros({b, h, d, t})).shape ==
            Shape{b, h, t, t});

    const std::size_t b2 = 1, h2 = 2, t2 = 4, d2 = 8;
    const NDArray scores = NDArray::ones({b2, h2, t2, d2})
                               .scale(0.1f)
                               .bmm(NDArray::ones({b2, h2, d2, t2}).scale(0.1f));
    const float expected = static_cast<float>(d2) * 0.01f;
    for (IndexIter it(scores.shape); it.valid(); it.next()) {
        REQUIRE(std::fabs(scores.at(it.index()) - expected) < 1e-4f);
    }
}

TEST_CASE("bmm matches Mat::matmul", "[ndarray]") {
    const Mat m = Mat::from_fn(3, 4, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c);
    });
    const Mat n = Mat::from_fn(4, 5, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 5 + c);
    });
    const Mat c_mat = m.matmul(n);
    const NDArray c_nd = NDArray::from_mat(m).bmm(NDArray::from_mat(n));

    REQUIRE(c_nd.shape == Shape{3, 5});
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 5; ++c) {
            INFO("mismatch at [" << r << "," << c << "]");
            REQUIRE(std::fabs(c_nd.at({r, c}) - c_mat.at(r, c)) < 1e-4f);
        }
    }
}

TEST_CASE("attention head bmm pattern", "[ndarray]") {
    // What batched attention does: scores = Q.bmm(K.permute([0,1,3,2])).
    const std::size_t b = 1, h = 4, t = 8, d = 16;
    const NDArray q =
        NDArray::from_fn({b, h, t, d}, [](const Index& i) { return static_cast<float>(i[3]) * 0.01f; });
    const NDArray k =
        NDArray::from_fn({b, h, t, d}, [](const Index& i) { return static_cast<float>(i[3]) * 0.01f; });

    const NDArray k_t = k.permute({0, 1, 3, 2});
    REQUIRE(k_t.shape == Shape{b, h, d, t});

    const NDArray scores = q.bmm(k_t);
    REQUIRE(scores.shape == Shape{b, h, t, t});

    // score[b,h,i,j] = sum_d (d*0.01)^2
    float expected = 0.0f;
    for (std::size_t dv = 0; dv < d; ++dv) {
        expected += (static_cast<float>(dv) * 0.01f) * (static_cast<float>(dv) * 0.01f);
    }
    for (IndexIter it(scores.shape); it.valid(); it.next()) {
        REQUIRE(std::fabs(scores.at(it.index()) - expected) < 1e-4f);
    }
}

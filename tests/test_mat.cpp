#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "rt/mat.hpp"
#include "test_helpers.hpp"

using namespace rt;
using rt::testing::approx;

TEST_CASE("matmul shape", "[mat]") {
    const Mat c = Mat::zeros(3, 4).matmul(Mat::zeros(4, 5));
    REQUIRE(c.rows == 3);
    REQUIRE(c.cols == 5);
}

TEST_CASE("matmul values", "[mat]") {
    // [1 2; 3 4] @ [5; 6] = [1*5+2*6; 3*5+4*6] = [17; 39]
    const Mat a({1., 2., 3., 4.}, 2, 2);
    const Mat b({5., 6.}, 2, 1);
    const Mat c = a.matmul(b);
    REQUIRE(approx(c.at(0, 0), 17.0f));
    REQUIRE(approx(c.at(1, 0), 39.0f));
}

TEST_CASE("matmul matches explicit triple loop", "[mat]") {
    // With RT_BLAS on this exercises cblas_sgemm; otherwise the scalar path.
    // Either way the result must match the reference.
    const Mat a = Mat::from_fn(8, 16, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 16 + c) * 0.01f - 0.5f;
    });
    const Mat b = Mat::from_fn(16, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.02f - 0.3f;
    });
    const Mat result = a.matmul(b);

    const std::size_t m = 8, k = 16, n = 8;
    Mat expected = Mat::zeros(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            for (std::size_t j = 0; j < n; ++j) {
                expected.at_mut(i, j) += a.at(i, p) * b.at(p, j);
            }
        }
    }
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            INFO("matmul[" << r << "," << c << "]");
            REQUIRE(std::fabs(result.at(r, c) - expected.at(r, c)) < 1e-4f);
        }
    }
}

TEST_CASE("transpose", "[mat]") {
    const Mat a({1., 2., 3., 4., 5., 6.}, 2, 3);
    const Mat at = a.transpose();
    REQUIRE(at.rows == 3);
    REQUIRE(at.cols == 2);
    REQUIRE(approx(at.at(0, 0), 1.0f));
    REQUIRE(approx(at.at(1, 0), 2.0f));
    REQUIRE(approx(at.at(0, 1), 4.0f));
}

TEST_CASE("matmul_bt matches matmul of transpose", "[mat]") {
    const Mat a = Mat::from_fn(5, 7, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 7 + c) * 0.03f - 0.4f;
    });
    const Mat b = Mat::from_fn(6, 7, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 7 + c) * 0.02f - 0.1f;
    });
    const Mat got = a.matmul_bt(b);
    const Mat want = a.matmul(b.transpose());
    REQUIRE(got.rows == want.rows);
    REQUIRE(got.cols == want.cols);
    for (std::size_t i = 0; i < got.data.size(); ++i) {
        REQUIRE(std::fabs(got.data[i] - want.data[i]) < 1e-4f);
    }
}

TEST_CASE("matmul_parallel matches sequential", "[mat]") {
    const std::size_t m = 32, k = 64, n = 48;
    const Mat a = Mat::from_fn(m, k, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * k + c) * 0.01f - 0.5f;
    });
    const Mat b = Mat::from_fn(k, n, [&](std::size_t r, std::size_t c) {
        return static_cast<float>(r * n + c) * 0.02f - 0.3f;
    });

    // Compare parallel vs single-thread directly rather than via matmul()
    // dispatch, which may route elsewhere with different fp rounding.
    const Mat seq = a.matmul_parallel(b, 1);
    const Mat par = a.matmul_parallel(b, 4);

    REQUIRE(par.rows == seq.rows);
    REQUIRE(par.cols == seq.cols);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            INFO("par[" << r << "," << c << "]=" << par.at(r, c) << " seq=" << seq.at(r, c));
            REQUIRE(std::fabs(par.at(r, c) - seq.at(r, c)) < 1e-4f);
        }
    }
}

TEST_CASE("matmul_parallel single thread matches matmul", "[mat]") {
    const Mat a = Mat::from_fn(3, 4, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r + c);
    });
    const Mat b = Mat::from_fn(4, 2, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 2 + c);
    });
    const Mat seq = a.matmul(b);
    const Mat par = a.matmul_parallel(b, 1);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            REQUIRE(std::fabs(par.at(r, c) - seq.at(r, c)) < 1e-5f);
        }
    }
}

TEST_CASE("matmul_parallel auto thread count", "[mat]") {
    const Mat a = Mat::from_fn(16, 8, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 8 + c) * 0.1f;
    });
    const Mat b = Mat::from_fn(8, 16, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 16 + c) * 0.1f;
    });
    const Mat seq = a.matmul(b);
    const Mat par = a.matmul_parallel(b, 0);  // 0 = auto-detect thread count
    for (std::size_t r = 0; r < 16; ++r) {
        for (std::size_t c = 0; c < 16; ++c) {
            INFO("auto-thread par[" << r << "," << c << "]");
            REQUIRE(std::fabs(par.at(r, c) - seq.at(r, c)) < 1e-3f);
        }
    }
}

TEST_CASE("elementwise ops and reductions", "[mat]") {
    const Mat a = Mat::from_fn(3, 4, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c);
    });
    const Mat b = Mat::ones(3, 4);

    REQUIRE(approx(a.add(b).at(1, 2), a.at(1, 2) + 1.0f));
    REQUIRE(approx(a.mul_elem(b).at(2, 3), a.at(2, 3)));
    REQUIRE(approx(a.scale(2.0f).at(1, 1), a.at(1, 1) * 2.0f));
    REQUIRE(approx(a.sum(), 66.0f));  // 0..11
    REQUIRE(approx(a.map([](float x) { return x * x; }).at(1, 0), 16.0f));

    Mat acc = Mat::zeros(3, 4);
    acc.add_assign(a);
    REQUIRE(approx(acc.at(2, 2), a.at(2, 2)));

    const Mat sr = a.sum_rows();
    REQUIRE(sr.rows == 1);
    REQUIRE(sr.cols == 4);
    REQUIRE(approx(sr.at(0, 0), 0.0f + 4.0f + 8.0f));

    const Mat rm = a.row_mean();
    REQUIRE(rm.rows == 3);
    REQUIRE(rm.cols == 1);
    REQUIRE(approx(rm.at(0, 0), (0.f + 1.f + 2.f + 3.f) / 4.0f));

    REQUIRE(approx(Mat({3.0f, 4.0f}, 1, 2).norm(), 5.0f));
}

TEST_CASE("broadcast helpers", "[mat]") {
    const Mat a = Mat::from_fn(3, 4, [](std::size_t r, std::size_t c) {
        return static_cast<float>(r * 4 + c);
    });
    const Mat col = Mat({10.f, 20.f, 30.f}, 3, 1);
    const Mat row = Mat({1.f, 2.f, 3.f, 4.f}, 1, 4);

    REQUIRE(approx(a.add_col_broadcast(col).at(1, 2), a.at(1, 2) + 20.0f));
    REQUIRE(approx(a.sub_col_broadcast(col).at(2, 0), a.at(2, 0) - 30.0f));
    REQUIRE(approx(a.mul_col_broadcast(col).at(0, 3), a.at(0, 3) * 10.0f));
    REQUIRE(approx(a.mul_row_broadcast(row).at(1, 3), a.at(1, 3) * 4.0f));
}

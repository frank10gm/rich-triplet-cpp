#pragma once

// Threading helpers mirroring Rust's `std::thread::scope` / `available_parallelism`.
//
// `scoped_spawn` collects `std::jthread`s in a vector; their destructor joins,
// so the function cannot return until every worker has finished -- the same
// guarantee `thread::scope` gives, which is what makes the disjoint-slice
// aliasing in the GEMV/GEMM kernels sound.

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace rt {

/// Number of logical CPUs, or 1 if the platform will not say.
[[nodiscard]] inline std::size_t hardware_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1u : static_cast<std::size_t>(n);
}

/// Ceiling division.
[[nodiscard]] inline constexpr std::size_t div_ceil(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

/// Run `body(thread_id)` on `n_threads` threads and join them all before
/// returning. Runs inline when `n_threads <= 1` to avoid spawn overhead.
template <typename F>
void scoped_spawn(std::size_t n_threads, F&& body) {
    if (n_threads <= 1) {
        body(std::size_t{0});
        return;
    }
    std::vector<std::jthread> workers;
    workers.reserve(n_threads);
    for (std::size_t tid = 0; tid < n_threads; ++tid) {
        workers.emplace_back([&body, tid] { body(tid); });
    }
    // ~jthread joins.
}

}  // namespace rt

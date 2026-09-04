#pragma once

// `Result<T>` is the port of the Rust code's `Result<T, String>` / `io::Result<T>`:
// a value or an error message, with no exceptions on the error path.

#include <expected>
#include <string>
#include <utility>

namespace rt {

template <typename T>
using Result = std::expected<T, std::string>;

/// Build an error result. Mirrors `Err(format!(...))` at the call sites.
[[nodiscard]] inline std::unexpected<std::string> err(std::string message) {
    return std::unexpected<std::string>(std::move(message));
}

/// Propagate an error out of `expr` (a `Result<U>`) from a function returning
/// `Result<T>`. Mirrors Rust's `?` operator.
#define RT_TRY(var, expr)                       \
    auto var##_res_ = (expr);                   \
    if (!var##_res_) {                          \
        return rt::err(var##_res_.error());     \
    }                                           \
    auto& var = *var##_res_

/// Like `RT_TRY` but for a `Result<void>`, which has no value to bind.
#define RT_TRY_VOID(expr)                              \
    if (auto rt_void_res_ = (expr); !rt_void_res_) {   \
        return rt::err(rt_void_res_.error());          \
    }                                                  \
    do {                                               \
    } while (0)

}  // namespace rt

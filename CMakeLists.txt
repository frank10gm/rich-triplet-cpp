cmake_minimum_required(VERSION 3.24)
project(rich-triplet-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Feature options — mirror of the Rust crate's cargo features.
# ---------------------------------------------------------------------------
option(RT_BLAS            "Use Apple Accelerate / CBLAS for sgemm" ON)
option(RT_METAL           "Enable Metal GPU backend"               OFF)
option(RT_PARALLEL        "Enable threaded fallback matmul"        OFF)
option(RT_WEIGHT_LOADING  "Enable safetensors weight loading"      ON)
option(RT_MMAP_LOADING    "Enable mmap-based weight loading"       ON)
option(RT_Q4K_LM_HEAD     "Quantize lm_head to Q4_K"               OFF)
option(RT_BUILD_TESTS     "Build the test suite"                   ON)

# ---------------------------------------------------------------------------
# Numerics: the Rust build never enables fast-math, and NEON kernels are
# hand-written with a fixed accumulation order. Letting the compiler contract
# a*b+c into an FMA (or reassociate) makes results diverge from the reference
# implementation, so both are disabled globally.
# ---------------------------------------------------------------------------
add_library(rt_flags INTERFACE)
target_compile_options(rt_flags INTERFACE
  -ffp-contract=off
  -fno-fast-math
  -Wall -Wextra
  -Wno-unused-parameter
)
target_compile_options(rt_flags INTERFACE
  $<$<CONFIG:Release>:-O3>
)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(rt_core STATIC)
target_include_directories(rt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rt_core PUBLIC rt_flags)

if(RT_BLAS)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(rt_core PUBLIC ${ACCELERATE_FRAMEWORK})
  # ACCELERATE_NEW_LAPACK selects the non-deprecated CBLAS headers; ILP64 is
  # left off so the interface stays 32-bit `int`, matching the Rust cblas crate.
  target_compile_definitions(rt_core PUBLIC RT_FEATURE_BLAS=1 ACCELERATE_NEW_LAPACK=1)
endif()

if(RT_METAL)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  target_link_libraries(rt_core PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  target_compile_definitions(rt_core PUBLIC RT_FEATURE_METAL=1)
endif()

if(RT_PARALLEL)
  target_compile_definitions(rt_core PUBLIC RT_FEATURE_PARALLEL=1)
endif()
if(RT_WEIGHT_LOADING)
  target_compile_definitions(rt_core PUBLIC RT_FEATURE_WEIGHT_LOADING=1)
endif()
if(RT_MMAP_LOADING)
  target_compile_definitions(rt_core PUBLIC RT_FEATURE_MMAP_LOADING=1)
endif()
if(RT_Q4K_LM_HEAD)
  target_compile_definitions(rt_core PUBLIC RT_FEATURE_Q4K_LM_HEAD=1)
endif()

find_package(Threads REQUIRED)
target_link_libraries(rt_core PUBLIC Threads::Threads)

add_subdirectory(src)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(rich-triplet src/main.cpp)
target_link_libraries(rich-triplet PRIVATE rt_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(RT_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
  )
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(CTest)
  include(Catch)
  add_subdirectory(tests)
endif()

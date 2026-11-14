cmake_minimum_required(VERSION 3.24)
project(goldbach_l0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_library(LEVEL_ZERO_LOADER NAMES ze_loader REQUIRED)
find_package(Threads REQUIRED)

# Quality gates for this repo's own code (verifier + host tests; the companion
# tools keep their own flags). CI builds with GB_WERROR=ON under gcc and clang
# and again with GB_SANITIZE=address,undefined.
option(GB_WERROR "Treat compiler warnings as errors" OFF)
set(GB_SANITIZE "" CACHE STRING "Sanitizers for the verifier and host tests, e.g. address,undefined")
set(GB_WARNINGS -Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion
                -Wformat=2 -Wnon-virtual-dtor -Woverloaded-virtual -Wnull-dereference
                -Wimplicit-fallthrough)
function(gb_quality target)
    target_compile_options(${target} PRIVATE ${GB_WARNINGS})
    if(GB_WERROR)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(GB_SANITIZE)
        target_compile_options(${target} PRIVATE -fsanitize=${GB_SANITIZE} -fno-omit-frame-pointer
                                                 -fno-sanitize-recover=all)
        target_link_options(${target} PRIVATE -fsanitize=${GB_SANITIZE})
    endif()
endfunction()

# The verifier itself — the only target a standalone clone needs. It runs the
# precompiled kernels/goldbach_bmg.spv shipped in the repo (rebuildable from
# kernels/goldbach.cl with ocloc via the kernels target / build.sh).
add_executable(goldbach_l0
    src/main.cpp
    src/l0_context.cpp
    src/primes.cpp
    src/verifier.cpp
    src/sparse.cpp
)
target_include_directories(goldbach_l0 PRIVATE include /usr/include/level_zero)
target_link_libraries(goldbach_l0 PRIVATE ${LEVEL_ZERO_LOADER} Threads::Threads)
target_compile_options(goldbach_l0 PRIVATE -O3 -g)
gb_quality(goldbach_l0)

# GPU-free tests (ctest): host math vs brute force, host <-> kernel constant
# mirrors, CLI contract. `ctest --test-dir build --output-on-failure`.
enable_testing()
add_executable(host_tests tests/host_tests.cpp src/primes.cpp)
target_include_directories(host_tests PRIVATE include)
target_link_libraries(host_tests PRIVATE Threads::Threads)
target_compile_options(host_tests PRIVATE -O2 -g)
target_compile_definitions(host_tests PRIVATE GB_TEST_DATA="${CMAKE_SOURCE_DIR}/tests/data")
gb_quality(host_tests)
add_test(NAME host_tests COMMAND host_tests)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(NAME check_constants COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/check_constants.py)
    add_test(NAME campaign_queue COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/test_campaign_queue.py)
    add_test(NAME production_env COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/test_production_env.py)
endif()
add_test(NAME cli_tests COMMAND bash ${CMAKE_SOURCE_DIR}/tests/cli_tests.sh $<TARGET_FILE:goldbach_l0>)
# Exact Goldbach statistics (v2.2): stats_ref is the independent CPU
# reference. GPU-free: `stats_known` checks p(n) against T. Oliveira e
# Silva's published S(p) table, `stats_tools` the summary / dashboard
# scripts on synthetic files. Local only (label gpu, CI runs -LE gpu):
# `gpu_validate` — GPU statistics vs stats_ref, end to end in every mode,
# bucket verification, kernel validators (`ctest -L gpu`).
add_executable(stats_ref tests/stats_ref.cpp src/primes.cpp)
target_include_directories(stats_ref PRIVATE include)
target_link_libraries(stats_ref PRIVATE Threads::Threads)
target_compile_options(stats_ref PRIVATE -O3)
gb_quality(stats_ref)
add_test(NAME stats_known COMMAND stats_ref --known)
if(Python3_FOUND)
    add_test(NAME stats_tools COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/test_stats_tools.py)
    # published windows of an independent verifier (tests/known_windows.py):
    # the reference table and the comparison logic; gpu_validate runs them
    add_test(NAME known_windows COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/known_windows.py selftest)
endif()
add_test(NAME gpu_validate COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_validate.sh ${CMAKE_BINARY_DIR})
set_tests_properties(gpu_validate PROPERTIES LABELS gpu TIMEOUT 3600)

# ---------------------------------------------------------------------------
# SPIR-V kernel build: kernels/goldbach.cl -> kernels/goldbach_bmg.spv
# Uses Intel's ocloc (ships with the compute-runtime / Level Zero packages).
# ---------------------------------------------------------------------------
find_program(OCLOC_EXE ocloc)

if(OCLOC_EXE)
    add_custom_command(
        OUTPUT ${CMAKE_SOURCE_DIR}/kernels/goldbach_bmg.spv
        COMMAND ${OCLOC_EXE} compile -file ${CMAKE_SOURCE_DIR}/kernels/goldbach.cl
                -device bmg
                -output ${CMAKE_SOURCE_DIR}/kernels/goldbach
        DEPENDS ${CMAKE_SOURCE_DIR}/kernels/goldbach.cl
        COMMENT "Compiling goldbach.cl -> goldbach_bmg.spv (ocloc, Battlemage)"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${CMAKE_SOURCE_DIR}/kernels/sparse_bmg.spv
        COMMAND ${OCLOC_EXE} compile -file ${CMAKE_SOURCE_DIR}/kernels/sparse.cl
                -device bmg
                -output ${CMAKE_SOURCE_DIR}/kernels/sparse
        DEPENDS ${CMAKE_SOURCE_DIR}/kernels/sparse.cl
        COMMENT "Compiling sparse.cl -> sparse_bmg.spv (ocloc, Battlemage)"
        VERBATIM
    )
    # the same source for the mod-30030 geometry (include/sparse.hpp scm_geo)
    add_custom_command(
        OUTPUT ${CMAKE_SOURCE_DIR}/kernels/sparse30030_bmg.spv
        COMMAND ${OCLOC_EXE} compile -file ${CMAKE_SOURCE_DIR}/kernels/sparse.cl
                -device bmg
                -output ${CMAKE_SOURCE_DIR}/kernels/sparse30030
                -options "-DSCM_M=30030u -DSCM_K=12u -DSCM_RES_LIST=2377u,7979u,9731u,11267u,13799u,14611u,18427u,18649u,21467u,21523u,22823u,23911u -DA3_SPAN=8"
        DEPENDS ${CMAKE_SOURCE_DIR}/kernels/sparse.cl
        COMMENT "Compiling sparse.cl (M = 30030) -> sparse30030_bmg.spv"
        VERBATIM
    )
    add_custom_target(kernels ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/kernels/goldbach_bmg.spv
                ${CMAKE_SOURCE_DIR}/kernels/sparse_bmg.spv
                ${CMAKE_SOURCE_DIR}/kernels/sparse30030_bmg.spv)
else()
    # Keep the target (the validators below depend on it) and fall back to the
    # shipped SPIR-V binary.
    add_custom_target(kernels ALL
        COMMENT "kernels: ocloc not found, using shipped kernels/goldbach_bmg.spv")
    message(STATUS "ocloc not found — using the shipped kernels/goldbach_bmg.spv "
        "(to rebuild kernels: ocloc compile -file kernels/goldbach.cl -device bmg -output kernels/goldbach)")
endif()

# ---------------------------------------------------------------------------
# Companion tooling (benchmark, tuner, bit-exact validators, experiment
# instrumentation) lives in the separate goldbach-tools repo, exposed here
# through the tools/ symlink. Without the sibling repo only goldbach_l0 is
# built.
# ---------------------------------------------------------------------------
if(EXISTS ${CMAKE_SOURCE_DIR}/tools/sweep.py)
    add_executable(goldbach_bench
        tools/benchmark.cpp
        src/l0_context.cpp
        src/primes.cpp
        src/verifier.cpp
    )

    add_executable(goldbach_tuner
        tools/tuner.cpp
    )

    target_include_directories(goldbach_bench PRIVATE include /usr/include/level_zero)
    target_link_libraries(goldbach_bench PRIVATE ${LEVEL_ZERO_LOADER} Threads::Threads)
    target_compile_options(goldbach_bench PRIVATE -O3)

    target_include_directories(goldbach_tuner PRIVATE include)
    target_compile_options(goldbach_tuner PRIVATE -O3)

    # Bit-exact GPU-vs-CPU validation of the sieve and check kernels in isolation.
    # These must be rebuilt with the current config.hpp — stale copies report
    # false failures when constants change.
    add_executable(debug_sieve tools/debug_sieve.cpp src/l0_context.cpp src/primes.cpp)
    add_executable(debug_check tools/debug_check.cpp src/l0_context.cpp src/primes.cpp)
    add_executable(debug_check_hi tools/debug_check_hi.cpp src/l0_context.cpp src/primes.cpp)
    add_executable(debug_fb tools/debug_fb.cpp src/l0_context.cpp src/primes.cpp)
    add_executable(debug_map tools/debug_map.cpp src/l0_context.cpp src/primes.cpp)
    # Sparse class map (v2.0): full-segment map vs CPU sieve, checker vs CPU
    # reference (skipped with an older goldbach-tools checkout).
    if(EXISTS ${CMAKE_SOURCE_DIR}/tools/debug_sparse.cpp)
        add_executable(debug_sparse tools/debug_sparse.cpp src/sparse.cpp src/l0_context.cpp src/primes.cpp)
        target_include_directories(debug_sparse PRIVATE include /usr/include/level_zero)
        target_link_libraries(debug_sparse PRIVATE ${LEVEL_ZERO_LOADER} Threads::Threads)
        target_compile_options(debug_sparse PRIVATE -O3)
        add_dependencies(debug_sparse kernels)
    endif()
    foreach(t debug_sieve debug_check debug_check_hi debug_fb debug_map)
        target_include_directories(${t} PRIVATE include /usr/include/level_zero)
        target_link_libraries(${t} PRIVATE ${LEVEL_ZERO_LOADER} Threads::Threads)
        target_compile_options(${t} PRIVATE -O3)
        add_dependencies(${t} kernels)
    endforeach()
else()
    message(STATUS "tools/ not found (companion goldbach-tools repo) — building goldbach_l0 only")
endif()

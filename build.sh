#!/usr/bin/env bash
# Build goldbach-l0 (Intel Arc B580, Level Zero): configure + build with CMake.
#
# Dependencies (Ubuntu 24.04 names): level-zero-dev, intel-level-zero-gpu,
# cmake >= 3.24, a C++20 compiler; intel-ocloc optional. With ocloc on PATH
# CMake recompiles kernels/*_bmg.spv whenever a kernel source changes;
# without it the shipped modules are used.
set -euo pipefail
cd "$(dirname "$0")"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo
echo "done. run from the project root so the default --spv path resolves:"
echo "  ./build/goldbach_l0 --limit 100000000 --verbose"

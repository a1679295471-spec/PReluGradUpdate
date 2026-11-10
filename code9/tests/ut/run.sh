#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/tests/ut/build"

# Keep Release as the default: a Debug build makes googletest emit libgtestd.a,
# which then does not match the `gtest` link target on some source trees.
cmake -S "$ROOT/tests/ut" -B "$BUILD" \
    -DGTEST_SOURCE_DIR="${GTEST_SOURCE_DIR:-}" \
    -DENABLE_INNER_KERNEL_UT="${ENABLE_INNER_KERNEL_UT:-ON}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$BUILD" --parallel "${CANNJUDGE_COMPILE_JOBS:-4}"

"$BUILD/op_host/strided_slice_grad_op_host_ut"
"$BUILD/op_kernel/strided_slice_grad_op_kernel_ut"
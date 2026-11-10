#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/tests/ut/build"

cmake -S "$ROOT/tests/ut" -B "$BUILD" \
    -DGTEST_SOURCE_DIR="${GTEST_SOURCE_DIR:-}" \
    -DENABLE_INNER_KERNEL_UT="${ENABLE_INNER_KERNEL_UT:-ON}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$BUILD" --parallel "${CANNJUDGE_COMPILE_JOBS:-4}"

"$BUILD/op_host/quant_update_scatter_op_host_ut"
"$BUILD/op_kernel/quant_update_scatter_op_kernel_ut"

if [ -x "$BUILD/op_api/quant_update_scatter_op_api_ut" ]; then
    export ASCEND_CUSTOM_OPP_PATH="$ROOT/build/packages/vendors/quant_update_scatter_custom${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    "$BUILD/op_api/quant_update_scatter_op_api_ut"
fi

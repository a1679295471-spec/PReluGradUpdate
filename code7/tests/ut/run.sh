#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/tests/ut/build"
cmake -S "$ROOT/tests/ut" -B "$BUILD" -DGTEST_SOURCE_DIR="${GTEST_SOURCE_DIR:-}" -DENABLE_INNER_KERNEL_UT="${ENABLE_INNER_KERNEL_UT:-OFF}"
cmake --build "$BUILD" --parallel "${CANNJUDGE_COMPILE_JOBS:-4}"
"$BUILD/op_host/p_relu_grad_update_op_host_ut"
# op_kernel UT 自带多 shape / 多 dtype 的 CPU golden 校验，不再依赖 samples 输出文件。
"$BUILD/op_kernel/p_relu_grad_update_op_kernel_ut"
if [ -x "$BUILD/op_api/p_relu_grad_update_op_api_ut" ]; then
    export ASCEND_CUSTOM_OPP_PATH="$ROOT/build/packages/vendors/p_relu_grad_update_custom${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    "$BUILD/op_api/p_relu_grad_update_op_api_ut"
fi

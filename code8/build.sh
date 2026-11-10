#!/bin/bash
set -e

export BASE_PATH=$(
  cd "$(dirname "$0")"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"

CORE_NUMS=4

usage() {
  echo "Build script for QuantUpdateScatter operator"
  echo "Usage: bash build.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -h, --help              Print this help message"
  echo "  -j[n]                   Compile thread nums, default is ${CORE_NUMS}, eg: -j8"
  echo "  --make_clean            Clean build artifacts"
  echo "  -u, --ut                Run host tiling UT + kernel CPU UT"
  echo "  -e, --example           Build, install and run the first NPU sample"
  echo ""
  echo "Environment:"
  echo "  ASCEND_HOME_PATH        Required. CANN 9.0.0 is the target."
  echo "  ASCEND_COMPUTE_UNIT     Optional, default ascend910b"
  echo "  GTEST_SOURCE_DIR        GoogleTest source tree for -u"
  echo "  NLOHMANN_JSON_INCLUDE_DIR  Directory containing nlohmann/json.hpp for -u"
}

check_env() {
  if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Run: source /path/to/cann/set_env.sh"
    exit 1
  fi
  if [ ! -d "${ASCEND_HOME_PATH}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH does not exist: ${ASCEND_HOME_PATH}"
    exit 1
  fi
  local ver_file="${ASCEND_HOME_PATH}/version.cfg"
  if [ -f "${ver_file}" ]; then
    local ver
    ver="$(head -n 1 "${ver_file}")"
    echo "[INFO] CANN: ${ver} (${ASCEND_HOME_PATH})"
    case "${ver}" in
      *8.5*)
        echo "[WARN] This project targets CANN 9.0.0."
        echo "[WARN] CANN 8.5 headers are not API-compatible with this kernel."
        ;;
    esac
  else
    echo "[INFO] CANN home: ${ASCEND_HOME_PATH}"
  fi
  case "${ASCEND_HOME_PATH}" in
    *8.5*)
      echo "[WARN] ASCEND_HOME_PATH looks like a CANN 8.5 installation."
      echo "[WARN] Set it to the CANN 9.0.0 directory before building."
      ;;
  esac
}

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_PATH:?}"/*
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    echo "Cleaning build_out directory..."
    rm -rf "${BUILD_OUT_PATH:?}"/*
  fi
}

THREAD_NUM=${CORE_NUMS}
COMPUTE_UNIT="${ASCEND_COMPUTE_UNIT:-ascend910b}"
ENABLE_CLEAN=FALSE
RUN_UT=FALSE
RUN_EXAMPLE=FALSE

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -j*)
      THREAD_NUM="${1:2}"
      if [ -z "$THREAD_NUM" ]; then
        THREAD_NUM=${CORE_NUMS}
      fi
      shift
      ;;
    -u|--ut)
      RUN_UT=true
      shift
      ;;
    -e|--example)
      RUN_EXAMPLE=true
      shift
      ;;
    --make_clean)
      ENABLE_CLEAN=TRUE
      shift
      ;;
    -*)
      echo "[ERROR] Invalid option: $1"
      usage
      exit 1
      ;;
    *)
      echo "[ERROR] Unexpected argument: $1"
      usage
      exit 1
      ;;
  esac
done

check_env

if [ "$ENABLE_CLEAN" = "TRUE" ]; then
  clean_build
  clean_build_out
  exit 0
fi

if [ "$RUN_UT" = true ]; then
  echo "[INFO] Running UT tests..."
  export CANNJUDGE_COMPILE_JOBS="$THREAD_NUM"
  cd "${BASE_PATH}/tests/ut"
  if ! bash ./run.sh; then
    echo "[ERROR] UT tests failed"
    exit 1
  fi
  echo "[INFO] UT tests passed"
  exit 0
fi

export CANNJUDGE_COMPILE_JOBS="$THREAD_NUM"
# Do not rely on find_package(ASC)'s default search order: the machine may
# contain an older CANN under /usr/local/Ascend.
CMAKE_ARGS="-DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT"
CMAKE_ARGS="$CMAKE_ARGS -DCANNJUDGE_COMPILE_JOBS=$THREAD_NUM"
CMAKE_ARGS="$CMAKE_ARGS -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}"
CMAKE_ARGS="$CMAKE_ARGS -DASC_DIR=${ASCEND_HOME_PATH}/lib64/cmake"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_POLICY_VERSION_MINIMUM=3.5"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release"

if [ ! -d "${BUILD_PATH}" ]; then
  mkdir -p "${BUILD_PATH}"
fi

[ -f "${BUILD_PATH}/CMakeCache.txt" ] && rm -f "${BUILD_PATH}/CMakeCache.txt"

echo "----------------------------------------------------------------"
SOURCE_FINGERPRINT=$(
  { printf '%s\n' "$ASCEND_HOME_PATH";
    for d in op_host op_kernel op_api; do
      if [ -d "$BASE_PATH/$d" ]; then find "$BASE_PATH/$d" -type f -print0; fi
    done | sort -z | xargs -0 sha256sum;
    sha256sum "$BASE_PATH/CMakeLists.txt";
  } | sha256sum
)
if [ -L "$BUILD_PATH" ]; then
  echo "[ERROR] build must not be a symlink"
  exit 2
fi
if [ ! -f "$BUILD_PATH/.source-fingerprint" ] || \
   [ "$(cat "$BUILD_PATH/.source-fingerprint")" != "$SOURCE_FINGERPRINT" ]; then
  rm -rf "$BUILD_PATH"
  mkdir -p "$BUILD_PATH"
  printf '%s' "$SOURCE_FINGERPRINT" > "$BUILD_PATH/.source-fingerprint"
fi

echo "[INFO] Configuring project..."
echo "[INFO] CMAKE_ARGS: ${CMAKE_ARGS}"
cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..

echo "----------------------------------------------------------------"
echo "[INFO] Building project with ${THREAD_NUM} threads..."
for target in all binary package install; do
  cmake --build . --target "$target" --parallel "$THREAD_NUM"
done

KERNEL_O=$(find "${BUILD_PATH}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}" -name "*.o" 2>/dev/null | head -1)
if [ -z "$KERNEL_O" ]; then
  echo "[ERROR] Kernel binary not found"
  exit 1
fi

PKG_PATH=$(ls "${BUILD_PATH}"/custom_opp_*.run 2>/dev/null | head -n 1)
if [ -z "$PKG_PATH" ] || [ ! -f "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
  echo "[ERROR] Package not found or empty"
  exit 1
fi

echo "----------------------------------------------------------------"
echo "[INFO] Build completed successfully!"
echo "[INFO] Kernel binary: ${KERNEL_O}"
echo "[INFO] Package: ${PKG_PATH}"

if [ "$RUN_EXAMPLE" = true ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] Running examples..."
  cd "${BASE_PATH}/examples"
  if ! bash ./run.sh; then
    echo "[ERROR] Example execution failed"
    exit 1
  fi
  echo "[INFO] Example completed successfully!"
fi

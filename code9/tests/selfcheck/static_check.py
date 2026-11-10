#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""Static checks for the code9 StridedSliceGrad project.

These checks only look at the sources; they are not a substitute for a real
CANN 9.0.0 build.  They catch the API patterns that broke code7/code8:
4-argument DataCopyExtParams, Adds(tensor, tensor), bf16 scalar casts, and
hard-coded CANN version paths.
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAILS = []


def check(name, ok, detail=""):
    if ok:
        print("[PASS] " + name)
    else:
        print("[FAIL] " + name + ((": " + detail) if detail else ""))
        FAILS.append(name)


def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def brace_args(text, start):
    depth = 0
    args = 1
    for index in range(start, len(text)):
        ch = text[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return args
        elif ch == "," and depth == 1:
            args += 1
    return -1


def main():
    required = [
        "op_host/strided_slice_grad_def.cpp",
        "op_host/strided_slice_grad_tiling.cpp",
        "op_host/strided_slice_grad_infershape.cpp",
        "op_kernel/strided_slice_grad.h",
        "op_kernel/strided_slice_grad.cpp",
        "op_kernel/strided_slice_grad_tiling_data.h",
        "op_kernel/strided_slice_grad_tiling_key.h",
        "examples/test_aclnn_strided_slice_grad.cpp",
        "tests/ut/CMakeLists.txt",
        "tests/ut/run.sh",
        "tests/ut/op_host/test_strided_slice_grad_tiling.cpp",
        "tests/ut/op_kernel/test_strided_slice_grad.cpp",
        "samples/sample.py",
        "samples/reference.py",
        "build.sh",
        "README.md",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    check("project structure complete", not missing, ", ".join(missing))

    kernel = read(ROOT / "op_kernel/strided_slice_grad.h")
    kernel_entry = read(ROOT / "op_kernel/strided_slice_grad.cpp")
    tiling = read(ROOT / "op_host/strided_slice_grad_tiling.cpp")
    infershape = read(ROOT / "op_host/strided_slice_grad_infershape.cpp")
    build = read(ROOT / "build.sh")
    run = read(ROOT / "tests/ut/run.sh")
    defsrc = read(ROOT / "op_host/strided_slice_grad_def.cpp")

    counts = []
    for match in re.finditer(r"DataCopyExtParams\s+\w+\s*\{", kernel):
        start = kernel.index("{", match.start())
        counts.append(brace_args(kernel, start))
    check("DataCopyExtParams uses 5 fields (CANN 9.0)", bool(counts) and all(c == 5 for c in counts),
          "arg counts: %s" % counts)

    check("no Adds() in kernel", "Adds(" not in kernel and "Adds(" not in kernel_entry)
    check("no Cast() in kernel (bf16 stays raw)", "Cast(" not in kernel and "Cast(" not in kernel_entry)
    check("kernel writes with DataCopyPad", "DataCopyPad(" in kernel)

    kernel_lines = [l for l in kernel.splitlines() if l.strip()]
    check("kernel avoids std::vector/std::string", not any("std::" in l for l in kernel_lines))

    check("build.sh requires ASCEND_HOME_PATH", "ASCEND_HOME_PATH is not set" in build)
    check("build.sh passes ASCEND_CANN_PACKAGE_PATH",
          "-DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}" in build)
    check("build.sh passes ASC_DIR", "-DASC_DIR=${ASCEND_HOME_PATH}/lib64/cmake" in build)
    check("build.sh passes CMAKE_POLICY_VERSION_MINIMUM", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" in build)
    check("build.sh supports -e", "-e|--example" in build)
    check("build.sh supports -u", "-u|--ut" in build)
    check("run.sh passes CMAKE_POLICY_VERSION_MINIMUM", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" in run)
    check("run.sh defaults to Release", 'CMAKE_BUILD_TYPE:-Release' in run)
    check("UT does not default to Debug", "set(CMAKE_BUILD_TYPE Debug" not in read(ROOT / "tests/ut/CMakeLists.txt"))

    scripts = "\n".join(read(p) for p in [
        ROOT / "build.sh",
        ROOT / "tests/ut/run.sh",
        ROOT / "tests/ut/CMakeLists.txt",
        ROOT / "tests/ut/op_host/CMakeLists.txt",
        ROOT / "tests/ut/op_kernel/CMakeLists.txt",
        ROOT / "tests/ut/cmake/BuildGoogleTest.cmake",
    ])
    hardcoded = re.findall(r"/usr/local/Ascend/(?:cann|ascend-toolkit)-?[0-9.]*", scripts)
    check("no hard-coded CANN version path", not hardcoded, str(hardcoded))

    modes = re.findall(r"#define SSG_TPL_MODE_\w+\s+(\d+)", read(ROOT / "op_kernel/strided_slice_grad_tiling_key.h"))
    dispatches = len(re.findall(r"if constexpr \(schMode ==", kernel_entry))
    check("6 tiling keys", len(modes) == 6, str(modes))
    check("6 kernel dispatch branches", dispatches == 6, str(dispatches))

    combos = len(re.findall(r"ge::DT_(?:FLOAT16|FLOAT|BF16|INT32|INT64)", defsrc))
    check("dtype profiles cover idx x value", combos >= 12, str(combos))

    data_h = read(ROOT / "op_kernel/strided_slice_grad_tiling_data.h")
    field_types = set(re.findall(r"^\s*(?:constexpr\s+)?([A-Za-z_0-9]+)\s+\w+\s*(?:\[[^\]]*\])?\s*(?:=|;)", data_h, re.M))
    allowed = {"int32_t", "int64_t", "struct", "constexpr"}
    check("tiling data is POD", field_types.issubset(allowed | {"SSG_MAX_RANK", "SSG_MAX_SPEC"}),
          str(sorted(field_types)))

    check("infershape reads the shape tensor", "GetInputTensor" in infershape)
    check("shape input is ValueDepend", "ValueDepend(REQUIRED)" in defsrc)
    check("tiling limits rank", "SSG_MAX_RANK" in tiling and "SSG_MAX_SPEC" in tiling)
    check("tiling sets tiling key and block dim", "SetTilingKey" in tiling and "SetBlockDim" in tiling)

    print("")
    if FAILS:
        print("static_check FAILED: %d issue(s)" % len(FAILS))
        return 1
    print("static_check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
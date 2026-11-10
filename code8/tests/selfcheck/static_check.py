#!/usr/bin/env python3
"""Offline static checks for the QuantUpdateScatter project.

This does not replace `bash build.sh -e` or `bash build.sh -u` on a machine
with CANN 9.0.0.  It only checks project completeness and the API patterns
that caused build failures in earlier operator exercises.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = [
    "op_host/quant_update_scatter_def.cpp",
    "op_host/quant_update_scatter_infershape.cpp",
    "op_host/quant_update_scatter_tiling.cpp",
    "op_kernel/quant_update_scatter.cpp",
    "op_kernel/quant_update_scatter.h",
    "op_kernel/quant_update_scatter_tiling_data.h",
    "op_kernel/quant_update_scatter_tiling_key.h",
    "examples/test_aclnn_quant_update_scatter.cpp",
    "examples/run.sh",
    "samples/case.json",
    "samples/reference.py",
    "samples/sample.py",
    "tests/ut/CMakeLists.txt",
    "tests/ut/run.sh",
    "tests/ut/cmake/BuildGoogleTest.cmake",
    "tests/ut/op_host/test_quant_update_scatter_tiling.cpp",
    "tests/ut/op_kernel/test_quant_update_scatter.cpp",
    "build.sh",
    "CMakeLists.txt",
]


def fail(message):
    print("FAIL: " + message)
    return False


def check_layout():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    for directory in ("op_host", "op_kernel", "examples", "tests", "samples"):
        if not (ROOT / directory).is_dir():
            missing.append(directory + "/")
    if missing:
        return fail("missing entries: " + ", ".join(missing))
    print("PASS layout: op_host/op_kernel/examples/tests/samples complete")
    return True


def check_no_todos():
    bad = []
    for directory in ("op_host", "op_kernel"):
        for path in (ROOT / directory).glob("*"):
            if path.is_file() and "TODO" in path.read_text(encoding="utf-8"):
                bad.append(str(path.relative_to(ROOT)))
    if bad:
        return fail("TODO remains in " + ", ".join(bad))
    print("PASS: no TODO placeholders in op_host/op_kernel")
    return True


def check_data_copy_params():
    text = (ROOT / "op_kernel/quant_update_scatter.h").read_text(encoding="utf-8")
    pattern = re.compile(r"DataCopyExtParams\s+\w+\s*\{([^}]*)\}", re.S)
    calls = pattern.findall(text)
    if not calls:
        return fail("no DataCopyExtParams found")
    for index, body in enumerate(calls, 1):
        fields = [item.strip() for item in body.split(",") if item.strip()]
        if len(fields) != 5:
            return fail(f"DataCopyExtParams #{index} has {len(fields)} fields, expected 5")
    print(f"PASS: all {len(calls)} DataCopyExtParams use the CANN 9.0.0 5-field form")
    return True


def check_tensor_arithmetic():
    text = (ROOT / "op_kernel/quant_update_scatter.h").read_text(encoding="utf-8")
    if "Add(value, value, zpF32Buf.Get<float>(), count);" not in text:
        return fail("tensor + tensor zero-point add does not use Add(dst, src0, src1, n)")
    compute = text[text.index("ComputeUpdateRow"):text.index("ProcessRankN")]
    if re.search(r"\bAdds\s*\(", compute):
        return fail("Adds() used inside the tensor arithmetic path")
    print("PASS: tensor + tensor uses Add; Adds is not used for two LocalTensors")
    return True


def check_bf16_casts():
    text = (ROOT / "op_kernel/quant_update_scatter.h").read_text(encoding="utf-8")
    if "scalesGm.GetValue" in text or "zpGm.GetValue" in text or "updatesGm.GetValue" in text:
        return fail("bf16 input is read through the scalar unit")
    if text.count("RoundMode::CAST_NONE") < 3:
        return fail("bf16 -> fp32 vector casts are missing")
    if "Cast(outHalf, value, RoundMode::CAST_RINT, count);" not in text:
        return fail("fp32 -> half RINT cast is missing")
    if "Cast(outLocal, outHalf, RoundMode::CAST_RINT, count);" not in text:
        return fail("half -> int8 RINT cast is missing")
    if "Cast(outLocal, value, RoundMode::CAST_RINT, count);" in text:
        return fail("direct fp32 -> int8 Cast is not supported by CANN 9.0.0")
    print("PASS: bf16/fp16 inputs are widened by vector Cast; fp32 -> half -> int8 uses CAST_RINT")
    return True


def check_build_script():
    text = (ROOT / "build.sh").read_text(encoding="utf-8")
    required = [
        "-DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}",
        "-DASC_DIR=${ASCEND_HOME_PATH}/lib64/cmake",
        "-e, --example",
        "-u, --ut",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        return fail("build.sh missing: " + ", ".join(missing))
    old_cann = "cann-" + "8.5.0"
    if old_cann in text:
        return fail("build.sh hard-codes a CANN version path")
    print("PASS: build.sh passes explicit 9.0 CANN paths and supports -e/-u")
    return True


def check_no_hardcoded_cann_in_tests():
    bad = []
    for path in (ROOT / "tests").rglob("*"):
        if path.is_file() and path.suffix in (".txt", ".cmake", ".sh", ".cpp", ".h"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if ("cann-" + "8.5.0") in text:
                bad.append(str(path.relative_to(ROOT)))
    if bad:
        return fail("hard-coded CANN path in " + ", ".join(bad))
    print("PASS: UT CMake uses ASCEND_HOME_PATH only, with no version fallback")
    return True


def check_test_coverage_markers():
    text = (ROOT / "tests/ut/op_kernel/test_quant_update_scatter.cpp").read_text(encoding="utf-8")
    markers = ["fp16_", "bf16_", "rank2", "no_zp", "reciprocal", "wide_row", "empty_updates"]
    missing = [marker for marker in markers if marker not in text]
    if missing:
        return fail("kernel UT missing scenarios: " + ", ".join(missing))
    print("PASS: kernel UT covers fp16/bf16, rank2, no-zp, reciprocal, wide rows, empty updates")
    return True


def main():
    checks = [
        check_layout,
        check_no_todos,
        check_data_copy_params,
        check_tensor_arithmetic,
        check_bf16_casts,
        check_build_script,
        check_no_hardcoded_cann_in_tests,
        check_test_coverage_markers,
    ]
    ok = True
    for check in checks:
        ok = check() and ok
    print("STATIC CHECK PASSED" if ok else "STATIC CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

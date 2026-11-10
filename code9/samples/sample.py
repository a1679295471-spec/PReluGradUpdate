#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""Generate and verify StridedSliceGrad sample data.

The first case is the NPU example point (fp32, rank 4).  Every other case is
written under ``cases/<name>`` so the same coverage can be inspected or reused
by other harnesses; they exercise negative strides, clipping, every mask and
int64 index tensors.
"""

import argparse
from pathlib import Path

import numpy as np

import reference

ROOT = Path(__file__).resolve().parent

# (name, y_shape, begin, end, strides, masks, dtype, idx dtype)
CASES = [
    ("sample", [2, 4, 5, 6], [0, 1, 0, 0], [2, 4, 5, 6], [1, 1, 1, 1],
     (0, 0, 0, 0, 0), np.dtype("float32"), np.dtype("int32")),
    ("rank2_partial_spec", [8, 13], [2], [7], [2],
     (0, 0, 0, 0, 0), np.dtype("float32"), np.dtype("int32")),
    ("negative_strides", [3, 4, 5], [2, -1, 4], [-4, -5, -6], [-1, -2, -1],
     (0, 0, 0, 0, 0), np.dtype("float32"), np.dtype("int32")),
    ("ellipsis_middle", [2, 3, 4, 5], [0, 1, 0], [2, 3, 4], [1, 1, 1],
     (0, 0, 1 << 1, 0, 0), np.dtype("float32"), np.dtype("int32")),
    ("new_axis", [3, 4], [1, 0, 0], [3, 1, 2], [1, 1, 1],
     (0, 0, 0, 1 << 1, 0), np.dtype("float32"), np.dtype("int32")),
    ("shrink_axis", [4, 5, 6], [0, 2, 0], [4, 3, 6], [2, 1, 3],
     (0, 0, 0, 0, 1 << 1), np.dtype("float32"), np.dtype("int32")),
    ("begin_end_mask", [3, 4, 5], [1, 1, 1], [2, 2, 2], [1, 1, 1],
     (0x5, 0x2, 0, 0, 0), np.dtype("float32"), np.dtype("int32")),
    ("clipped_bounds", [3, 4], [-10, 1], [100, 4], [1, 1],
     (0, 0, 0, 0, 0), np.dtype("float32"), np.dtype("int32")),
    ("fp16_basic", [2, 4, 5, 6], [0, 1, 0, 0], [2, 4, 5, 6], [1, 1, 1, 1],
     (0, 0, 0, 0, 0), np.dtype("float16"), np.dtype("int32")),
    ("int64_index", [5, 6], [0, 1], [5, 6], [1, 2],
     (0, 0, 0, 0, 0), np.dtype("float32"), np.dtype("int64")),
]


def case_dirs(directory, name):
    base = directory if name == "sample" else directory / "cases" / name
    for sub in ("input", "output", "golden"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def generate(directory):
    directory = Path(directory)
    rng = np.random.default_rng(20241109)
    for index, case in enumerate(CASES):
        name, y_shape, begin, end, strides, masks, dtype, idx_dtype = case
        base = case_dirs(directory, name)
        total_y = int(np.prod(y_shape, dtype=np.int64))
        info = reference.plan(y_shape, begin, end, strides, masks[0], masks[1], masks[2],
                              masks[3], masks[4])
        dy_shape = [int(v) for v in info["dy_shape"]]
        dy = rng.uniform(-10.0, 10.0, size=dy_shape).astype(dtype)
        golden = reference.impl(np.asarray(y_shape, dtype=idx_dtype),
                                np.asarray(begin, dtype=idx_dtype),
                                np.asarray(end, dtype=idx_dtype),
                                np.asarray(strides, dtype=idx_dtype),
                                dy,
                                begin_mask=masks[0], end_mask=masks[1],
                                ellipsis_mask=masks[2], new_axis_mask=masks[3],
                                shrink_axis_mask=masks[4])
        if list(golden.shape) != [int(v) for v in y_shape]:
            raise ValueError("reference shape mismatch for " + name)

        (base / "input" / "shape.bin").write_bytes(np.asarray(y_shape, dtype=idx_dtype).tobytes())
        (base / "input" / "begin.bin").write_bytes(np.asarray(begin, dtype=idx_dtype).tobytes())
        (base / "input" / "end.bin").write_bytes(np.asarray(end, dtype=idx_dtype).tobytes())
        (base / "input" / "strides.bin").write_bytes(np.asarray(strides, dtype=idx_dtype).tobytes())
        (base / "input" / "dy.bin").write_bytes(dy.tobytes())
        (base / "golden" / "y.bin").write_bytes(golden.tobytes())
        for stale in ("output/y.bin",):
            (base / stale).unlink(missing_ok=True)
        print("generated %-20s y=%s dy=%s bytes=%d" % (name, y_shape, list(dy.shape), total_y * dtype.itemsize))
    print("Sample data generated.")


def verify(directory):
    directory = Path(directory)
    name, y_shape, _b, _e, _s, _m, dtype, _i = CASES[0]
    size = int(np.prod(y_shape, dtype=np.int64)) * dtype.itemsize
    out_file = directory / "output" / "y.bin"
    golden_file = directory / "golden" / "y.bin"
    if not out_file.exists() or not golden_file.exists():
        print("VERIFY ERROR: missing %s or %s" % (out_file, golden_file))
        return 1
    raw = out_file.read_bytes()
    golden = golden_file.read_bytes()
    if len(raw) != size or len(golden) != size:
        print("VERIFY ERROR: byte size mismatch, got %d/%d expect %d" % (len(raw), len(golden), size))
        return 1

    got = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    want = np.frombuffer(golden, dtype=dtype).astype(np.float64)
    tol = 1e-4 if dtype == np.dtype("float32") else 1e-3
    bad = np.flatnonzero(np.abs(got - want) > tol)
    for index in bad[:5]:
        print("mismatch idx=%d got=%r expect=%r" % (index, got[index], want[index]))
    ok = bad.size == 0
    print(("PASS" if ok else "Wrong Answer") + ": y.bin, mismatched=%d/%d" % (bad.size, got.size))
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "verify"))
    parser.add_argument("--directory", type=Path, default=ROOT / "build-npu")
    args = parser.parse_args()
    try:
        if args.action == "generate":
            generate(args.directory)
        else:
            raise SystemExit(verify(args.directory))
    except Exception as error:
        print("SAMPLE ERROR: " + str(error))
        raise SystemExit(2)
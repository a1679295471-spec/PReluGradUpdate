#!/usr/bin/env python3
"""Offline differential check for the QuantUpdateScatter schedule.

This script models the kernel's p/row/column-tile loops and compares the
result with samples/reference.py.  It is useful on a machine without CANN,
but it is NOT a replacement for compiling and running `build.sh -e/-u`.
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "samples"))
import reference  # noqa: E402

UB_SIZE = 196608
RESERVED_UB_BYTES = 16384
CORE_NUM = 16
QUEUE_AND_TEMP_BYTES = 22


def bf16_round(array):
    x = np.asarray(array, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return (rounded << np.uint32(16)).view(np.float32)


def quantize_input(array, profile):
    if profile == "fp16":
        return np.asarray(array, dtype=np.float16).astype(np.float32)
    return bf16_round(array)


def make_tiling(var_shape, updates_shape, profile):
    rank = len(updates_shape)
    axis = rank - 2
    var_dim0 = var_shape[0]
    var_axis = var_shape[axis]
    upd_dim0 = updates_shape[0]
    upd_axis = updates_shape[axis]
    outer = math.prod(updates_shape[1:axis]) if axis > 1 else 1
    row_len = updates_shape[-1]
    type_size_t = 2
    type_size_s = 4 if profile == "fp16" else 2
    type_size_z = 4 if profile == "fp16" else 2
    bytes_per_elem = 2 * type_size_t + 2 * type_size_s + 2 * type_size_z + QUEUE_AND_TEMP_BYTES
    max_col = max(1, (UB_SIZE - RESERVED_UB_BYTES) // bytes_per_elem)
    col_tile = min(row_len, max_col) if row_len > 0 else 1
    p_per_core = (var_dim0 + CORE_NUM - 1) // CORE_NUM if var_dim0 > 0 else 0
    block_dim = max(1, min(CORE_NUM, (var_dim0 + p_per_core - 1) // p_per_core)) if p_per_core else 1
    return {
        "rank": rank,
        "axis": axis,
        "var_dim0": var_dim0,
        "var_axis": var_axis,
        "upd_dim0": upd_dim0,
        "upd_axis": upd_axis,
        "outer": outer,
        "row_len": row_len,
        "var_stride": math.prod(var_shape[1:]),
        "upd_stride": math.prod(updates_shape[1:]),
        "col_tile": col_tile,
        "p_per_core": p_per_core,
        "block_dim": block_dim,
    }


def quant_scalar(update, scale, zero, reciprocal):
    u = np.float32(update)
    s = np.float32(scale)
    if reciprocal:
        s = np.float32(1.0) / s
    value = np.float32(u / s) + np.float32(zero)
    value = np.rint(value)
    value = np.float32(min(value, np.float32(127.0)))
    value = np.float32(max(value, np.float32(-128.0)))
    return int(value)


def simulate(var, indices, updates, scales, zeros, reciprocal, profile):
    var = np.asarray(var, dtype=np.int8)
    updates = np.asarray(updates, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    zeros = np.asarray(zeros, dtype=np.float32)
    out = var.copy()
    coverage = np.zeros(var.size, dtype=np.int16)
    var_flat = var.reshape(-1)
    out_flat = out.reshape(-1)
    upd_flat = updates.reshape(-1)
    info = make_tiling(var.shape, updates.shape, profile)

    if info["rank"] == 2:
        for block in range(info["block_dim"]):
            start = block * info["p_per_core"]
            end = min(info["var_dim0"], start + info["p_per_core"])
            for col in range(0, info["row_len"], info["col_tile"]):
                n = min(info["col_tile"], info["row_len"] - col)
                for row in range(start, end):
                    src_row = -1
                    for p in range(len(indices) - 1, -1, -1):
                        idx = int(indices[p])
                        if row >= idx and row < idx + info["upd_axis"]:
                            src_row = row - idx
                            break
                    dst = row * info["row_len"] + col
                    if src_row >= 0:
                        for c in range(n):
                            src = src_row * info["row_len"] + col + c
                            out_flat[dst + c] = quant_scalar(
                                upd_flat[src], scales[col + c], zeros[col + c], reciprocal
                            )
                    else:
                        out_flat[dst:dst + n] = var_flat[dst:dst + n]
                    coverage[dst:dst + n] += 1
        return out, coverage

    for block in range(info["block_dim"]):
        start = block * info["p_per_core"]
        end = min(info["var_dim0"], start + info["p_per_core"])
        for p in range(start, end):
            has_update = p < info["upd_dim0"] and p < len(indices)
            idx = int(indices[p]) if has_update else 0
            for col in range(0, info["row_len"], info["col_tile"]):
                n = min(info["col_tile"], info["row_len"] - col)
                for j in range(info["outer"]):
                    for row in range(info["var_axis"]):
                        k = row - idx
                        dst = p * info["var_stride"] + (j * info["var_axis"] + row) * info["row_len"] + col
                        if has_update and 0 <= k < info["upd_axis"]:
                            src_base = p * info["upd_stride"] + (j * info["upd_axis"] + k) * info["row_len"] + col
                            for c in range(n):
                                out_flat[dst + c] = quant_scalar(
                                    upd_flat[src_base + c], scales[col + c], zeros[col + c], reciprocal
                                )
                        else:
                            out_flat[dst:dst + n] = var_flat[dst:dst + n]
                        coverage[dst:dst + n] += 1
    return out, coverage


def run_case(var_shape, updates_shape, profile, has_zp, reciprocal, seed):
    rng = np.random.default_rng(seed)
    rank = len(var_shape)
    axis = rank - 2
    row_len = updates_shape[-1]
    var = rng.integers(-128, 128, size=var_shape, dtype=np.int8)
    if rank == 2:
        upd_dim0 = updates_shape[0]
        indices = rng.integers(0, var_shape[0] - upd_dim0 + 1, size=upd_dim0)
    else:
        indices = rng.integers(0, var_shape[axis] - updates_shape[axis] + 1,
                               size=updates_shape[0])
    updates = quantize_input(rng.uniform(-20.0, 20.0, size=updates_shape), profile)
    scales = quantize_input(rng.choice(np.array([0.25, 0.5, 1.0, 2.0, 4.0]), size=row_len),
                            profile)
    if has_zp:
        zeros = quantize_input(rng.integers(-8, 9, size=row_len).astype(np.float32), profile)
    else:
        zeros = np.zeros(row_len, dtype=np.float32)

    expected = reference.impl(
        var, indices, updates, scales, zeros if has_zp else None,
        reduce="update", axis=-2, quant_axis=-1, reciprocal_scale=reciprocal,
    )
    got, coverage = simulate(var, indices, updates, scales, zeros, reciprocal, profile)
    if not np.all(coverage == 1):
        raise AssertionError(f"coverage mismatch: {(coverage != 1).sum()} elements")
    bad = np.argwhere(got != expected).reshape(-1)
    if bad.size:
        idx = int(bad[0])
        raise AssertionError(
            f"value mismatch at {idx}: got={got.reshape(-1)[idx]} "
            f"expected={expected.reshape(-1)[idx]}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=180)
    args = parser.parse_args()
    random.seed(20241109)

    fixed = [
        ((1, 2, 32), (1, 1, 32), "fp16", True, False),
        ((3, 5, 17), (3, 2, 17), "fp16", True, False),
        ((2, 3, 4, 7), (2, 3, 2, 7), "fp16", True, False),
        ((2, 3, 2, 4, 9), (2, 3, 2, 2, 9), "fp16", True, False),
        ((2, 4, 9), (2, 1, 9), "fp16", False, False),
        ((1, 3, 17), (1, 1, 17), "fp16", True, True),
        ((1, 3, 9000), (1, 1, 9000), "fp16", True, False),
        ((8, 13), (4, 13), "fp16", True, False),
        ((2, 3, 16), (0, 1, 16), "fp16", False, False),
        ((3, 4, 16), (1, 1, 16), "fp16", False, False),
        ((2, 3, 32), (2, 1, 32), "bf16", True, False),
        ((2, 3, 4, 6), (2, 3, 2, 6), "bf16", True, False),
        ((2, 2, 3, 5, 10), (2, 2, 3, 2, 10), "bf16", True, False),
        ((3, 5, 13), (3, 2, 13), "bf16", False, False),
        ((1, 4, 11), (1, 2, 11), "bf16", True, True),
        ((1, 2, 7000), (1, 1, 7000), "bf16", True, False),
        ((7, 10), (3, 10), "bf16", True, False),
    ]
    for index, (var_shape, upd_shape, profile, has_zp, reciprocal) in enumerate(fixed):
        run_case(var_shape, upd_shape, profile, has_zp, reciprocal, 1000 + index)
    print(f"PASS fixed cases: {len(fixed)}")

    for index in range(args.random):
        rank = random.choice([2, 2, 3, 3, 4, 4, 5])
        profile = random.choice(["fp16", "bf16"])
        has_zp = random.random() < 0.7
        reciprocal = random.random() < 0.2
        row_len = random.choice([1, 7, 13, 32, 127, 1000, 4099])
        if rank == 2:
            upd_dim0 = random.randint(1, 8)
            var_dim0 = upd_dim0 + random.randint(0, 8)
            var_shape = (var_dim0, row_len)
            upd_shape = (upd_dim0, row_len)
        else:
            var_dim0 = random.randint(0 if random.random() < 0.05 else 1, 12)
            upd_axis = random.randint(1, 6)
            var_axis = upd_axis + random.randint(0, 5)
            outer = tuple(random.randint(1, 4) for _ in range(rank - 3))
            var_shape = (var_dim0,) + outer + (var_axis, row_len)
            upd_shape = (var_dim0,) + outer + (upd_axis, row_len)
        run_case(var_shape, upd_shape, profile, has_zp, reciprocal, 10000 + index)
    print(f"PASS random differential: {args.random} cases")
    print("FUZZ CHECK PASSED (schedule model vs samples/reference.py; no CANN compile)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

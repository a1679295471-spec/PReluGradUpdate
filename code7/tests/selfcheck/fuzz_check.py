#!/usr/bin/env python3
"""Randomised differential check of the PReluGradUpdate scheduling design.

This mirrors, in Python, exactly what op_host/p_relu_grad_update_tiling.cpp
computes and how op_kernel/p_relu_grad_update.h walks blocks / tiles.  It then
compares against samples/reference.py (the TBE-aligned CPU reference) with the
same dtype quantisation the kernel performs (fp32 compute + RNE cast).

bf16 is emulated with bit-level RNE so that the script only needs numpy (no
ml_dtypes) and no CANN/NPU:

    python tests/selfcheck/fuzz_check.py

It is a design-level check, not a replacement for `bash build.sh -e/-u`.
"""

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "samples"))
import reference  # noqa: E402

UB_SIZE = 262144
CORE_NUM = 64
RESERVED_UB = 8192
FLOAT_TEMP_BYTES = 20
DOUBLE_BUFFER_FACTOR = 8

TOL = {"float32": 1e-4, "float16": 1e-3, "bfloat16": 4e-3}


def ceildiv(a, b):
    return -(-a // b)


def ceilalign(a, b):
    return ceildiv(a, b) * b


def flooralign(a, b):
    return a // b * b


def make_tiling(total, channels, inner, type_size, ub_size=UB_SIZE, cores=CORE_NUM):
    """Mirror of PReluGradUpdateTilingFunc in op_host."""
    align = 32 // type_size
    bytes_per_elem = DOUBLE_BUFFER_FACTOR * type_size + FLOAT_TEMP_BYTES
    max_tile_base = max(align, flooralign((ub_size - RESERVED_UB) // bytes_per_elem, align))
    period = channels * inner
    pattern_size = period // math.gcd(period, align) * align
    pattern_bytes = pattern_size * 4
    use_pattern = False
    max_tile_pattern = 0
    if pattern_bytes + RESERVED_UB < ub_size:
        max_tile_pattern = flooralign(
            (ub_size - RESERVED_UB - pattern_bytes) // bytes_per_elem, align
        )
        use_pattern = max_tile_pattern >= pattern_size
    if use_pattern:
        ub_factor = max(pattern_size, flooralign(max_tile_pattern, pattern_size))
    else:
        ub_factor = max(1, min(max_tile_base, inner))
    if total > 0:
        per_core = ceildiv(total, cores)
        block_factor = ceilalign(per_core, period) if use_pattern else per_core
        block_dim = max(1, min(ceildiv(total, block_factor), cores))
    else:
        block_factor = 0
        block_dim = 1
    return {
        "total": total,
        "blockFactor": block_factor,
        "ubFactor": ub_factor,
        "innerSize": inner,
        "channelNum": channels,
        "patternSize": pattern_size if use_pattern else 0,
        "usePattern": 1 if use_pattern else 0,
        "blockDim": block_dim,
    }


def quantize(a, dtype):
    """Quantise a float32 array exactly like the kernel's output cast."""
    a = np.asarray(a, dtype=np.float32)
    if dtype == "float32":
        return a
    if dtype == "float16":
        return a.astype(np.float16).astype(np.float32)
    bits = a.view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1
    out = ((bits + 0x7FFF + lsb) >> 16).astype(np.uint32) << 16
    return out.view(np.float32)


def run_case(shape, dtype, seed, ub_size=UB_SIZE):
    shape = tuple(int(v) for v in shape)
    total = math.prod(shape)
    channels = shape[1]
    inner = math.prod(shape[2:]) if len(shape) > 2 else 1
    type_size = 4 if dtype == "float32" else 2

    rng = np.random.default_rng(seed)
    g_in = quantize(rng.uniform(-5.0, 5.0, shape), dtype)
    f_in = quantize(rng.uniform(-5.0, 5.0, shape), dtype)
    w_in = quantize(rng.uniform(0.01, 0.5, (channels,)), dtype)

    if total == 0:
        # op_host keeps every divisor non-zero for empty tensors and the
        # kernel exits before touching GM; only the shape contract matters.
        got_dx, got_update = reference.impl(g_in, f_in, w_in)
        assert got_dx.size == 0 and got_update.size == 0
        return 0

    info = make_tiling(total, channels, inner, type_size, ub_size)

    g = g_in.reshape(-1)
    f = f_in.reshape(-1)
    w = w_in
    dx = np.zeros(total, dtype=np.float32)
    update = np.zeros(total, dtype=np.float32)
    coverage = np.zeros(total, dtype=np.int32)

    period = channels * inner
    if info["usePattern"]:
        pattern = w[(np.arange(info["patternSize"]) // inner) % channels]
        for block in range(info["blockDim"]):
            start = block * info["blockFactor"]
            end = min(total, start + info["blockFactor"])
            offset = start
            while offset < end:
                count = min(info["ubFactor"], end - offset)
                # The kernel always replays pattern from index 0, so every
                # tile must start on a period boundary.
                assert offset % period == 0, (shape, dtype, offset, period)
                wv = np.tile(pattern, ceildiv(count, info["patternSize"]))[:count]
                ft = f[offset:offset + count]
                gt = g[offset:offset + count]
                p = (ft > 0.0).astype(np.float32)
                dx[offset:offset + count] = gt * (p + (1.0 - p) * wv)
                update[offset:offset + count] = ft * gt - p * ft * gt
                coverage[offset:offset + count] += 1
                offset += count
    else:
        for block in range(info["blockDim"]):
            start = block * info["blockFactor"]
            end = min(total, start + info["blockFactor"])
            offset = start
            while offset < end:
                channel = (offset // inner) % channels
                channel_end = offset + inner - (offset % inner)
                segment_end = min(end, channel_end)
                assert segment_end - offset <= inner
                while offset < segment_end:
                    count = min(info["ubFactor"], segment_end - offset)
                    ft = f[offset:offset + count]
                    gt = g[offset:offset + count]
                    p = (ft > 0.0).astype(np.float32)
                    dx[offset:offset + count] = gt * (p + (1.0 - p) * w[channel])
                    update[offset:offset + count] = ft * gt - p * ft * gt
                    coverage[offset:offset + count] += 1
                    offset += count

    assert np.all(coverage == 1), ("coverage", shape, dtype, int((coverage != 1).sum()))

    got_dx = quantize(dx, dtype).reshape(shape)
    got_update = quantize(update, dtype).reshape(shape)
    exp_dx, exp_update = reference.impl(g_in, f_in, w_in)
    tol = TOL[dtype]
    bad = 0
    for got, exp, name in ((got_dx, exp_dx, "dx"), (got_update, exp_update, "update")):
        close = np.isclose(
            got.astype(np.float64), exp.astype(np.float64), rtol=tol, atol=tol
        )
        n = int((~close).sum())
        if n:
            print(f"FAIL {name} shape={shape} dtype={dtype}: {n}/{got.size} mismatches")
            idx = np.argwhere(~close).reshape(-1)[:3]
            for i in idx:
                print("   idx", int(i), "got", float(got.reshape(-1)[i]),
                      "exp", float(exp.reshape(-1)[i]))
            bad += n
    return bad


def main():
    random.seed(20241109)
    cases = []

    fixed = [
        (2, 3), (1, 1), (4, 6, 8), (1, 1, 1, 1), (2, 3, 5, 7), (7, 5),
        (3, 5, 1), (1, 8, 64, 64), (1, 16, 56, 56), (2, 3, 4, 5),
        (5, 7, 3), (9, 4, 6, 2), (1, 1, 4096), (1, 64, 1), (1, 33, 1, 3),
        (2, 7, 9, 11), (3, 2, 1, 1, 5), (0, 3), (3, 0, 4),
    ]
    for shape in fixed:
        for dtype in ("float32", "float16", "bfloat16"):
            cases.append((shape, dtype, UB_SIZE))

    # Random shapes; occasional large inner / large channel dims force the
    # channel scheduling mode, the rest exercise the periodic mode.
    for _ in range(240):
        rank = random.choice([2, 2, 3, 3, 4, 4, 4, 5, 6])
        dims = [random.randint(1, 4)]
        dims.append(random.randint(1, 33))
        for _ in range(rank - 2):
            dims.append(random.randint(1, 9))
        if random.random() < 0.15:
            dims[-1] = random.choice([1001, 3001, 8191])
        if random.random() < 0.10:
            dims[1] = random.choice([257, 1024, 4096])
        shape = tuple(dims)
        if math.prod(shape) > 400000:
            continue
        dtype = random.choice(["float32", "float16", "bfloat16"])
        cases.append((shape, dtype, UB_SIZE))

    # Small UB budget: forces channel mode for ordinary 4-D shapes.
    for shape in ((1, 8, 8, 8), (2, 5, 7, 3), (1, 4, 16, 16), (3, 2, 37)):
        for dtype in ("float32", "float16"):
            cases.append((shape, dtype, 16384))

    pattern_cases = 0
    channel_cases = 0
    for shape, dtype, ub in cases:
        total = math.prod(shape)
        if total == 0:
            continue
        info = make_tiling(
            total, shape[1], math.prod(shape[2:]) if len(shape) > 2 else 1,
            4 if dtype == "float32" else 2, ub,
        )
        if info["usePattern"]:
            pattern_cases += 1
        else:
            channel_cases += 1
    assert channel_cases >= 20, channel_cases

    bad = 0
    for i, (shape, dtype, ub) in enumerate(cases):
        bad += run_case(shape, dtype, seed=1000 + i, ub_size=ub)
        if bad:
            break
    if bad:
        print("FUZZ CHECK FAILED")
        return 1
    print(f"PASS fuzz differential: {len(cases)} cases "
          f"(pattern={pattern_cases}, channel={channel_cases}), "
          f"fp32/fp16/bf16, rank 2-6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
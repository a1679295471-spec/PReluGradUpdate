#!/usr/bin/env python3
"""No-CANN static / algorithm self-check for the PReluGradUpdate project.

This script does not replace `bash build.sh -e` or `-u`; it verifies the
project layout, the host tiling expectations in the UT table, and simulates
the two kernel scheduling modes against the reference formula. It only uses
the Python standard library, so it also runs on Windows.
"""

import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESERVED_UB = 8192
FLOAT_TEMP_BYTES = 20
DOUBLE_BUFFER_FACTOR = 8
UB_SIZE = 262144
CORE_NUM = 64

REQUIRED = [
    "op_host/p_relu_grad_update_def.cpp",
    "op_host/p_relu_grad_update_infershape.cpp",
    "op_host/p_relu_grad_update_tiling.cpp",
    "op_kernel/p_relu_grad_update.cpp",
    "op_kernel/p_relu_grad_update.h",
    "op_kernel/p_relu_grad_update_tiling_data.h",
    "op_kernel/p_relu_grad_update_tiling_key.h",
    "examples/test_aclnn_p_relu_grad_update.cpp",
    "examples/run.sh",
    "samples/case.json",
    "samples/reference.py",
    "samples/sample.py",
    "tests/ut/op_host/test_p_relu_grad_update_tiling.cpp",
    "tests/ut/op_kernel/test_p_relu_grad_update.cpp",
    "build.sh",
    "CMakeLists.txt",
]


def ceildiv(a, b):
    return -(-a // b)


def ceilalign(a, b):
    return ceildiv(a, b) * b


def flooralign(a, b):
    return a // b * b


def make_tiling(total, channels, inner, type_size, ub_size=UB_SIZE, cores=CORE_NUM):
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
    ub_factor = flooralign(max_tile_pattern, pattern_size) if use_pattern else min(max_tile_base, inner)
    ub_factor = max(1, ub_factor)
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


def check_layout():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    for directory in ("op_host", "op_kernel", "examples", "tests", "samples"):
        if not (ROOT / directory).is_dir():
            missing.append(directory + "/")
    if missing:
        print("FAIL layout, missing:", ", ".join(missing))
        return False
    print("PASS layout: op_host/op_kernel/examples/tests/samples complete")
    return True


def check_tiling_cases():
    path = ROOT / "tests/ut/op_host/test_p_relu_grad_update_tiling.cpp"
    text = path.read_text(encoding="utf-8")
    block = text[text.index("static PReluGradUpdateTestParam testCases[]"):]
    dtype_key = {"float": 1, "float16": 0, "bf16": 2}
    dtype_size = {"float": 4, "float16": 2, "bf16": 2}
    checked = 0
    failed = 0
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith('{"'):
            continue
        name = line.split('"')[1]
        shape_match = re.search(r'\{"[^"]+", \{([^}]*)\}', line)
        shape = [int(v) for v in shape_match.group(1).split(",")]
        dtype = re.search(r"ge::DT_(\w+)", line).group(1).lower()
        key = int(re.search(r"GRAPH_SUCCESS, (\d+)UL,", line).group(1))
        expect = [int(v) for v in re.search(r'UL, "([^"]*)", \{0\}', line).group(1).split()]
        total = math.prod(shape)
        channels = shape[1]
        inner = math.prod(shape[2:]) if len(shape) > 2 else 1
        info = make_tiling(total, channels, inner, dtype_size[dtype])
        actual = [
            info["total"], info["blockFactor"], info["ubFactor"], info["innerSize"],
            info["channelNum"], info["patternSize"], info["usePattern"],
        ]
        checked += 1
        if actual != expect or key != dtype_key[dtype]:
            failed += 1
            print(f"FAIL tiling {name}: expect {expect} key={dtype_key[dtype]}, got {actual} key={key}")
    if failed:
        return False
    print(f"PASS tiling table: {checked} cases match host formula")
    return True


def float_to_half_bits(value):
    import struct
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = (bits >> 16) & 0x8000
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0xFF:
        return sign | (0x7E00 if mant else 0x7C00)
    e = exp - 127 + 15
    if e >= 31:
        return sign | 0x7C00
    if e <= 0:
        if e < -10:
            return sign
        mant |= 0x800000
        shift = 14 - e
        m = mant >> shift
        rem = mant & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and m & 1):
            m += 1
        return sign | m
    m = mant >> 13
    rem = mant & 0x1FFF
    if rem > 0x1000 or (rem == 0x1000 and m & 1):
        m += 1
        if m == 0x400:
            m = 0
            e += 1
            if e >= 31:
                return sign | 0x7C00
    return sign | (e << 10) | m


def half_bits_to_float(bits):
    import struct
    sign = (bits & 0x8000) << 16
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0:
        if mant == 0:
            fbits = sign
        else:
            e = -14
            while not (mant & 0x400):
                mant <<= 1
                e -= 1
            mant &= 0x3FF
            fbits = sign | ((e + 127) << 23) | (mant << 13)
    elif exp == 0x1F:
        fbits = sign | 0x7F800000 | (mant << 13)
    else:
        fbits = sign | ((exp - 15 + 127) << 23) | (mant << 13)
    return struct.unpack("<f", struct.pack("<I", fbits))[0]


def float_to_bf16_bits(value):
    import struct
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return (bits >> 16) & 0xFFFF


def bf16_bits_to_float(bits):
    import struct
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def quantize(value, dtype):
    if dtype == "float32":
        return value
    if dtype == "f16":
        return half_bits_to_float(float_to_half_bits(value))
    return bf16_bits_to_float(float_to_bf16_bits(value))


def simulate(shape, dtype, ub_size=UB_SIZE, seed=7):
    type_size = 4 if dtype == "float32" else 2
    total = math.prod(shape)
    channels = shape[1]
    inner = math.prod(shape[2:]) if len(shape) > 2 else 1
    info = make_tiling(total, channels, inner, type_size, ub_size=ub_size)
    rng = random.Random(seed)
    grads = [quantize(rng.uniform(-5.0, 5.0), dtype) for _ in range(total)]
    features = [quantize(rng.uniform(-5.0, 5.0), dtype) for _ in range(total)]
    weights = [quantize(rng.uniform(0.01, 0.5), dtype) for _ in range(channels)]
    dx = [0.0] * total
    update = [0.0] * total

    if info["usePattern"]:
        pattern = [weights[(i // inner) % channels] for i in range(info["patternSize"])]
        for block in range(info["blockDim"]):
            start = block * info["blockFactor"]
            end = min(total, start + info["blockFactor"])
            offset = start
            while offset < end:
                count = min(info["ubFactor"], end - offset)
                wv = [0.0] * count
                n = min(count, len(pattern))
                wv[:n] = pattern[:n]
                filled = n
                while filled < count:
                    k = min(filled, count - filled)
                    wv[filled:filled + k] = wv[:k]
                    filled += k
                for i in range(count):
                    f = features[offset + i]
                    g = grads[offset + i]
                    w = wv[i]
                    p = 1.0 if f > 0.0 else 0.0
                    dx[offset + i] = g * (p + (1.0 - p) * w)
                    update[offset + i] = f * g - p * f * g
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
                while offset < segment_end:
                    count = min(info["ubFactor"], segment_end - offset)
                    w = weights[channel]
                    for i in range(count):
                        f = features[offset + i]
                        g = grads[offset + i]
                        p = 1.0 if f > 0.0 else 0.0
                        dx[offset + i] = g * (p + (1.0 - p) * w)
                        update[offset + i] = f * g - p * f * g
                    offset += count

    tolerance = {"float32": 1e-4, "f16": 1e-3, "bf16": 4e-3}[dtype]
    bad = 0
    for i in range(total):
        channel = (i // inner) % channels
        f = features[i]
        g = grads[i]
        w = weights[channel]
        golden_dx = quantize(g if f > 0.0 else g * w, dtype)
        golden_update = quantize(0.0 if f > 0.0 else f * g, dtype)
        got_dx = quantize(dx[i], dtype)
        got_update = quantize(update[i], dtype)
        if abs(got_dx - golden_dx) > tolerance * (1.0 + abs(golden_dx)):
            bad += 1
        if abs(got_update - golden_update) > tolerance * (1.0 + abs(golden_update)):
            bad += 1
    return bad, info


def check_algorithm():
    shapes = [
        (2, 3), (4, 6, 8), (2, 3, 4, 5), (7, 5), (3, 5, 1),
        (2, 3, 5, 7), (13, 7, 11), (5, 7, 3), (9, 4, 6, 2),
    ]
    checked = 0
    for dtype in ("float32", "f16", "bf16"):
        for shape in shapes:
            bad, info = simulate(shape, dtype)
            checked += 1
            if bad:
                print(f"FAIL simulate {shape} {dtype}: {bad} mismatches")
                return False
    # Force channel mode with a small UB budget.
    for dtype in ("float32", "f16"):
        for shape in ((1, 8, 8, 8), (2, 5, 7, 3), (1, 4, 16, 16)):
            bad, info = simulate(shape, dtype, ub_size=16384)
            checked += 1
            if not info["usePattern"] and bad:
                print(f"FAIL simulate channel {shape} {dtype}: {bad} mismatches")
                return False
            if info["usePattern"]:
                print(f"FAIL expected channel mode for {shape} {dtype}")
                return False
    print(f"PASS algorithm simulation: {checked} shape/dtype cases")
    return True


def main():
    ok = True
    ok &= check_layout()
    ok &= check_tiling_cases()
    ok &= check_algorithm()
    print("SELF-CHECK PASSED" if ok else "SELF-CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

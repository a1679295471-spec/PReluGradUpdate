#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""Model the on-device gather algorithm and compare it to the reference.

The C++ kernel (``op_kernel/strided_slice_grad.h``) walks y, decomposes the
flat index into coordinates, applies begin/end clipping and the stride, then
reads one dy element (or 0).  This script reproduces that algorithm exactly
and checks it against ``samples/reference.py`` on randomized cases, including
negative strides, masks, ellipsis, new axes and shrink axes.

This is a semantics check only; it cannot validate Ascend C compilation.
"""

import random
import sys

import numpy as np

import reference


def _popcount(value):
    return bin(int(value)).count("1")


def build_mapping(rank, shape, begin, end, strides, begin_mask, end_mask,
                  ellipsis_mask, new_axis_mask, shrink_axis_mask):
    spec_len = len(begin)
    dims = [None] * rank
    new_count = _popcount(new_axis_mask)
    ellipsis_len = 0
    if ellipsis_mask:
        ellipsis_len = rank - (spec_len - 1 - new_count)
    elif spec_len - new_count > rank:
        raise ValueError("spec too long")

    y_dim = 0
    dy_axis = 0
    for s in range(spec_len):
        if (new_axis_mask >> s) & 1:
            dy_axis += 1
            continue
        if (ellipsis_mask >> s) & 1:
            for _ in range(ellipsis_len):
                dims[y_dim] = {"mode": 0, "spec": -1, "dy_axis": dy_axis}
                y_dim += 1
                dy_axis += 1
            continue
        if y_dim >= rank:
            raise ValueError("spec entries exceed rank")
        if (shrink_axis_mask >> s) & 1:
            dims[y_dim] = {"mode": 1, "spec": s, "dy_axis": -1}
            y_dim += 1
            continue
        dims[y_dim] = {"mode": 0, "spec": s, "dy_axis": dy_axis}
        y_dim += 1
        dy_axis += 1
    while y_dim < rank:
        dims[y_dim] = {"mode": 0, "spec": -1, "dy_axis": dy_axis}
        y_dim += 1
        dy_axis += 1
    return dims


def _adjust(index, dim, step):
    if index < 0:
        index += dim
    if step > 0:
        index = max(0, min(index, dim))
    else:
        index = max(-1, min(index, dim - 1))
    return index


def _count(start, stop, step):
    if step > 0:
        return 0 if start >= stop else (stop - start + step - 1) // step
    if start <= stop:
        return 0
    return (start - stop + (-step) - 1) // (-step)


def kernel_model(shape, begin, end, strides, dy, masks):
    begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask = masks
    shape = [int(v) for v in shape]
    dy = np.asarray(dy)
    rank = len(shape)
    dims = build_mapping(rank, shape, begin, end, strides, *masks)

    dy_shape = [int(v) for v in dy.shape]
    dy_strides = [1] * len(dy_shape)
    for i in range(len(dy_shape) - 2, -1, -1):
        dy_strides[i] = dy_strides[i + 1] * dy_shape[i + 1]
    total_dy = int(np.prod(dy_shape)) if dy_shape else 1

    start = [0] * rank
    step = [1] * rank
    count = [shape[d] for d in range(rank)]
    fixed = [0] * rank
    for d in range(rank):
        info = dims[d]
        s = info["spec"]
        if info["mode"] == 1:
            index = int(begin[s])
            if index < 0:
                index += shape[d]
            if index < 0 or index >= shape[d]:
                return np.zeros(shape, dtype=dy.dtype)
            fixed[d] = index
            continue
        if s < 0:
            continue
        st = int(strides[s])
        if st == 0:
            return np.zeros(shape, dtype=dy.dtype)
        b = 0 if (begin_mask >> s) & 1 else int(begin[s])
        e = 0 if (end_mask >> s) & 1 else int(end[s])
        if (begin_mask >> s) & 1:
            b = shape[d] - 1 if st < 0 else 0
        else:
            b = _adjust(b, shape[d], st)
        if (end_mask >> s) & 1:
            e = -1 if st < 0 else shape[d]
        else:
            e = _adjust(e, shape[d], st)
        start[d] = b
        step[d] = st
        count[d] = _count(b, e, st)

    y_strides = [1] * rank
    for i in range(rank - 2, -1, -1):
        y_strides[i] = y_strides[i + 1] * shape[i + 1]
    total_y = int(np.prod(shape)) if rank else 1
    out = np.zeros(total_y, dtype=dy.dtype)
    coords = [0] * rank
    for flat in range(total_y):
        remain = flat
        for d in range(rank):
            coords[d] = remain // y_strides[d]
            remain %= y_strides[d]
        offset = 0
        ok = True
        for d in range(rank):
            info = dims[d]
            if info["mode"] == 1:
                if coords[d] != fixed[d]:
                    ok = False
                    break
                continue
            diff = coords[d] - start[d]
            if step[d] > 0:
                if diff < 0 or diff % step[d]:
                    ok = False
                    break
                q = diff // step[d]
            else:
                diff = start[d] - coords[d]
                if diff < 0 or diff % (-step[d]):
                    ok = False
                    break
                q = diff // (-step[d])
            if q < 0 or q >= count[d]:
                ok = False
                break
            axis = info["dy_axis"]
            if axis < 0 or axis >= len(dy_shape):
                ok = False
                break
            offset += q * dy_strides[axis]
        if ok and 0 <= offset < total_dy:
            out[flat] = dy.reshape(-1)[offset]
    return out.reshape(shape)


def main():
    rng = random.Random(99173)
    checked = 0
    fixed_cases = [
        ([2, 4, 5, 6], [0, 1, 0, 0], [2, 4, 5, 6], [1, 1, 1, 1], (0, 0, 0, 0, 0)),
        ([3, 4, 5], [2, -1, 4], [-4, -5, -6], [-1, -2, -1], (0, 0, 0, 0, 0)),
        ([2, 3, 4], [0, 1], [0, 3], [1, 1], (0, 0, 1, 0, 0)),
        ([4, 5, 6], [0, 2, 0], [4, 3, 6], [2, 1, 3], (0, 0, 0, 0, 2)),
        ([3, 4], [1, 0, 2], [3, 1, 3], [1, 1, 1], (0, 0, 0, 2, 4)),
        ([2, 3, 4, 5], [0, 0, 0, 1], [2, 0, 4, 2], [2, 1, 1, 1], (0, 0, 2, 0, 8)),
        ([3, 4, 5], [1, 1, 1], [2, 2, 2], [1, 1, 1], (5, 2, 0, 0, 0)),
        ([4, 5], [2], [2], [1], (0, 0, 0, 0, 0)),
    ]
    cases = list(fixed_cases)
    while len(cases) < 60:
        rank = rng.randint(0, 5)
        shape = [rng.randint(1, 4) for _ in range(rank)]
        spec_len = rank if rank else 0
        if rank and rng.random() < 0.3:
            spec_len = rng.randint(0, rank)
        masks = [0, 0, 0, 0, 0]
        if spec_len and rng.random() < 0.3:
            masks[2] = 1 << rng.randrange(spec_len)
        if spec_len and rng.random() < 0.3:
            masks[3] = 1 << rng.randrange(spec_len)
        candidates = [p for p in range(spec_len) if not ((masks[3] >> p) & 1)]
        if rank and candidates and rng.random() < 0.3:
            masks[4] = 1 << rng.choice(candidates)
            masks[2] = 0
        if rng.random() < 0.4 and spec_len:
            masks[0] = rng.getrandbits(spec_len)
        if rng.random() < 0.4 and spec_len:
            masks[1] = rng.getrandbits(spec_len)
        begin = [rng.randint(-6, 6) for _ in range(spec_len)]
        end = [rng.randint(-6, 6) for _ in range(spec_len)]
        strides = [rng.choice([-2, -1, 1, 2]) for _ in range(spec_len)]
        for p in range(spec_len):
            if (masks[4] >> p) & 1:
                y_dim = 0
                for s in range(p):
                    if not ((masks[3] >> s) & 1):
                        y_dim += 1
                if y_dim < rank:
                    begin[p] = rng.randrange(shape[y_dim])
                    end[p] = begin[p] + 1
        try:
            info = reference.plan(shape, begin, end, strides, *masks)
        except ValueError:
            continue
        dy_shape = [int(v) for v in info["dy_shape"]]
        dy = np.arange(int(np.prod(dy_shape)) if dy_shape else 1,
                       dtype=np.float32).reshape(dy_shape or [])
        cases.append((shape, begin, end, strides, tuple(masks)))

    for shape, begin, end, strides, masks in cases:
        info = reference.plan(shape, begin, end, strides, *masks)
        dy_shape = [int(v) for v in info["dy_shape"]]
        dy = (np.arange(int(np.prod(dy_shape)) if dy_shape else 1, dtype=np.float32)
              .reshape(dy_shape or []) * 0.5 - 3.0)
        want = reference.impl(shape, begin, end, strides, dy, begin_mask=masks[0],
                              end_mask=masks[1], ellipsis_mask=masks[2],
                              new_axis_mask=masks[3], shrink_axis_mask=masks[4])
        got = kernel_model(shape, begin, end, strides, dy, masks)
        if got.shape != want.shape or not np.array_equal(got, want):
            print("FAIL:", shape, begin, end, strides, masks)
            print("kernel model:\n", got)
            print("reference:\n", want)
            return 1
        checked += 1

    print("kernel_model_check PASSED: %d cases" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
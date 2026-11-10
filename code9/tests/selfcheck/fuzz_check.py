#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""Randomized cross-check of the NumPy reference implementation.

Two independent formulations are compared for every generated case:

* ``reference.impl``  -- gather (assign the slice of y from dy);
* ``scatter_impl``    -- scatter (walk dy and write each element into y).

Both must agree, and every y element outside the slice must stay zero.  This
validates the semantics used by the kernel UT golden without needing a CANN
environment.
"""

import random
import sys

import numpy as np

import reference


def _decompose(flat, shape):
    coords = [0] * len(shape)
    remain = flat
    for i in range(len(shape)):
        stride = 1
        for j in range(i + 1, len(shape)):
            stride *= shape[j]
        coords[i] = remain // stride if stride else 0
        if stride:
            remain %= stride
    return coords


def scatter_impl(shape, begin, end, strides, dy, begin_mask=0, end_mask=0,
                 ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    info = reference.plan(shape, begin, end, strides, begin_mask, end_mask,
                          ellipsis_mask, new_axis_mask, shrink_axis_mask)
    shape = [int(v) for v in shape]
    y = np.zeros(shape, dtype=dy.dtype)
    touched = np.zeros(shape, dtype=bool)
    dy_shape = [int(v) for v in info["dy_shape"]]
    total = int(np.prod(dy_shape)) if dy_shape else 1
    for flat in range(total):
        coords = _decompose(flat, dy_shape)
        offset = 0
        for d, dim in enumerate(info["dims"]):
            if dim["mode"] == 1:
                coord = dim["fixed"]
            else:
                coord = dim["start"] + coords[dim["dy_axis"]] * dim["step"]
            stride = 1
            for b in range(d + 1, len(shape)):
                stride *= shape[b]
            offset += coord * stride
        if offset < 0 or offset >= y.size:
            raise AssertionError("reference produced out-of-range offset")
        if touched.flat[offset]:
            raise AssertionError("two dy elements map to the same y element")
        touched.flat[offset] = True
        y.flat[offset] = dy.flat[flat]
    return y


def random_case(rng):
    rank = rng.randint(0, 5)
    shape = [rng.randint(1, 5) for _ in range(rank)]

    masks = [0, 0, 0, 0, 0]
    kind = rng.random()
    spec_len = rank
    new_positions = []
    shrink_positions = []
    ellipsis_pos = None

    if rank == 0:
        spec_len = 0
    elif kind < 0.25:
        spec_len = rng.randint(0, rank)
    elif kind < 0.45:
        ellipsis_pos = rng.randrange(rank)
        spec_len = rank
    elif kind >= 0.65:
        spec_len = rank
        new_positions = [rng.randrange(spec_len)]
        if rng.random() < 0.5 and spec_len >= 2:
            candidates = [p for p in range(spec_len) if p not in new_positions]
            shrink_positions = [rng.choice(candidates)]

    if ellipsis_pos is not None:
        masks[2] = 1 << ellipsis_pos
        if rng.random() < 0.4:
            candidates = [p for p in range(spec_len) if p != ellipsis_pos]
            if candidates and rng.random() < 0.5:
                shrink_positions = [rng.choice(candidates)]
            if candidates and rng.random() < 0.5:
                new_positions = [rng.choice(candidates)]
        if shrink_positions and new_positions and shrink_positions[0] == new_positions[0]:
            new_positions = []

    for p in new_positions:
        masks[3] |= 1 << p
    for p in shrink_positions:
        masks[4] |= 1 << p

    begin = [rng.randint(-8, 8) for _ in range(spec_len)]
    end = [rng.randint(-8, 8) for _ in range(spec_len)]
    strides = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(spec_len)]
    if rng.random() < 0.3 and spec_len:
        masks[0] = rng.getrandbits(spec_len)
    if rng.random() < 0.3 and spec_len:
        masks[1] = rng.getrandbits(spec_len)

    for p in shrink_positions:
        y_dim = 0
        for s in range(p):
            if (masks[3] >> s) & 1:
                continue
            y_dim += 1
        if y_dim >= rank:
            return None
        index = rng.randrange(shape[y_dim])
        begin[p] = index
        end[p] = index + 1

    try:
        info = reference.plan(shape, begin, end, strides, masks[0], masks[1], masks[2],
                              masks[3], masks[4])
    except ValueError:
        return None
    dy_shape = [int(v) for v in info["dy_shape"]]
    if any(v < 0 for v in dy_shape):
        return None
    dy = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(dy_shape).astype(np.float32)
    return shape, begin, end, strides, dy, tuple(masks)


def main():
    rng = random.Random(20241110)
    checked = 0
    for _ in range(4000):
        case = random_case(rng)
        if case is None:
            continue
        shape, begin, end, strides, dy, masks = case
        got = reference.impl(shape, begin, end, strides, dy, begin_mask=masks[0],
                             end_mask=masks[1], ellipsis_mask=masks[2],
                             new_axis_mask=masks[3], shrink_axis_mask=masks[4])
        want = scatter_impl(shape, begin, end, strides, dy, *masks)
        if got.shape != want.shape or not np.array_equal(got, want):
            print("FAIL case:", shape, begin, end, strides, masks)
            print("gather:\n", got)
            print("scatter:\n", want)
            return 1
        checked += 1
        if checked >= 400:
            break

    fixed = [
        ([2, 4, 5, 6], [0, 1, 0, 0], [2, 4, 5, 6], [1, 1, 1, 1], (0, 0, 0, 0, 0)),
        ([3, 4, 5], [2, -1, 4], [-4, -5, -6], [-1, -2, -1], (0, 0, 0, 0, 0)),
        ([2, 3, 4], [0, 0], [0, 0], [1, 1], (0, 0, 1, 0, 0)),
        ([3, 4], [1, 0, 2], [3, 1, 3], [1, 1, 1], (0, 0, 0, 2, 4)),
        ([2, 3, 4, 5], [0, 0, 0, 1], [2, 0, 4, 2], [2, 1, 1, 1], (0, 0, 2, 0, 8)),
        ([4, 5], [2], [2], [1], (0, 0, 0, 0, 0)),
    ]
    for shape, begin, end, strides, masks in fixed:
        info = reference.plan(shape, begin, end, strides, *masks)
        dy_shape = info["dy_shape"]
        size = int(np.prod(dy_shape)) if dy_shape else 1
        dy = np.arange(size, dtype=np.float32).reshape(dy_shape or [])
        got = reference.impl(shape, begin, end, strides, dy, begin_mask=masks[0],
                             end_mask=masks[1], ellipsis_mask=masks[2],
                             new_axis_mask=masks[3], shrink_axis_mask=masks[4])
        want = scatter_impl(shape, begin, end, strides, dy, *masks)
        if not np.array_equal(got, want):
            print("FAIL fixed case:", shape, begin, end, strides, masks)
            return 1

    print("fuzz_check PASSED: %d random cases + %d fixed cases" % (checked, len(fixed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
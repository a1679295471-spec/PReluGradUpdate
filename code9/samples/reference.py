#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""Pure NumPy reference implementation of StridedSliceGrad.

Semantics follow TensorFlow ``strided_slice`` (and therefore the operator's
gradient):

* ``shape`` gives the output shape, ``begin``/``end``/``strides`` have one
  entry per spec position (they may be shorter than ``rank``; the remaining
  output dims are full slices).
* ``ellipsis_mask`` expands to the dimensions not covered by other specs.
* ``new_axis_mask`` inserts a size-1 axis in the slice result; it consumes no
  output dim and produces one size-1 ``dy`` axis.
* ``shrink_axis_mask`` picks a single index from ``begin`` and removes that
  axis from the slice result.
* Negative indices are normalized by ``+dim``; begin/end are clipped with the
  usual positive/negative-step rules; out-of-slice elements of ``y`` are 0.
"""

import numpy as np


def _value(x):
    """Accept either a raw array or an msopst-style {'value': ...} dict."""
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    return x


def _popcount(value):
    return bin(int(value) & ((1 << 64) - 1)).count("1")


def _clip(index, dim, step):
    if index < 0:
        index += dim
    if step > 0:
        return min(max(index, 0), dim)
    return min(max(index, -1), dim - 1)


def _slice_count(start, stop, step):
    if step > 0:
        return 0 if start >= stop else (stop - start + step - 1) // step
    if start <= stop:
        return 0
    return (start - stop + (-step) - 1) // (-step)


def plan(shape, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0,
         new_axis_mask=0, shrink_axis_mask=0):
    """Return the structural plan shared by the reference and the golden check."""
    shape = [int(v) for v in shape]
    begin = [int(v) for v in begin]
    end = [int(v) for v in end]
    strides = [int(v) for v in strides]
    rank = len(shape)
    spec_len = len(begin)

    if spec_len > 63:
        raise ValueError("spec length too large")
    limit = (1 << spec_len) - 1
    for name, mask in (("begin_mask", begin_mask), ("end_mask", end_mask),
                       ("ellipsis_mask", ellipsis_mask), ("new_axis_mask", new_axis_mask),
                       ("shrink_axis_mask", shrink_axis_mask)):
        if mask < 0 or (mask & ~limit) != 0:
            raise ValueError("%s has a bit outside the spec length" % name)
    if _popcount(ellipsis_mask) > 1:
        raise ValueError("ellipsis_mask must contain at most one bit")
    if new_axis_mask & shrink_axis_mask:
        raise ValueError("new_axis_mask and shrink_axis_mask overlap")

    new_count = _popcount(new_axis_mask)
    ellipsis_len = 0
    if ellipsis_mask:
        ellipsis_len = rank - (spec_len - 1 - new_count)
        if ellipsis_len < 0:
            raise ValueError("ellipsis expands to a negative number of dims")
    elif spec_len - new_count > rank:
        raise ValueError("begin is longer than the output rank")

    dims = []          # per y dim: dict(mode, spec, dy_axis, start, step, stop, count, fixed)
    dy_shape = []
    new_axis_positions = []
    y_dim = 0

    def add_normal(spec_idx):
        nonlocal y_dim
        if spec_idx is None:
            start, step, stop, count = 0, 1, shape[y_dim], shape[y_dim]
        else:
            step = strides[spec_idx]
            if step == 0:
                raise ValueError("strides[%d] must not be zero" % spec_idx)
            start = (shape[y_dim] - 1 if step < 0 else 0) if (begin_mask >> spec_idx) & 1 \
                else _clip(begin[spec_idx], shape[y_dim], step)
            stop = (-1 if step < 0 else shape[y_dim]) if (end_mask >> spec_idx) & 1 \
                else _clip(end[spec_idx], shape[y_dim], step)
            count = _slice_count(start, stop, step)
        dims.append({"mode": 0, "spec": -1 if spec_idx is None else spec_idx,
                     "dy_axis": len(dy_shape), "start": start, "step": step,
                     "stop": stop, "count": count, "fixed": 0})
        dy_shape.append(count)
        y_dim += 1

    for s in range(spec_len):
        if (new_axis_mask >> s) & 1:
            new_axis_positions.append(len(dy_shape))
            dy_shape.append(1)
            continue
        if (ellipsis_mask >> s) & 1:
            for _ in range(ellipsis_len):
                add_normal(None)
            continue
        if (shrink_axis_mask >> s) & 1:
            index = begin[s]
            if index < 0:
                index += shape[y_dim]
            dims.append({"mode": 1, "spec": s, "dy_axis": -1, "start": 0, "step": 1,
                         "stop": 0, "count": 1, "fixed": index})
            y_dim += 1
            continue
        add_normal(s)

    while y_dim < rank:
        add_normal(None)

    return {"rank": rank, "spec_len": spec_len, "dims": dims, "dy_shape": dy_shape,
            "new_axis_positions": new_axis_positions, "shape": shape}


def impl(shape, begin, end, strides, dy, output=None, begin_mask=0, end_mask=0,
         ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    """Compute ``y`` from ``dy``.  Extra ``output`` arg mirrors the msopst API."""
    shape = _value(shape)
    begin = _value(begin)
    end = _value(end)
    strides = _value(strides)
    dy = np.asarray(_value(dy))

    info = plan(shape, begin, end, strides, int(begin_mask), int(end_mask),
                int(ellipsis_mask), int(new_axis_mask), int(shrink_axis_mask))
    expected_dy = tuple(int(v) for v in info["dy_shape"])
    if tuple(dy.shape) != expected_dy:
        raise ValueError("dy shape %s does not match the slice %s" % (tuple(dy.shape), expected_dy))

    result = np.zeros(tuple(int(v) for v in info["shape"]), dtype=dy.dtype)
    if dy.size == 0:
        return result

    index_arrays = []
    dy_values = dy
    if info["new_axis_positions"]:
        dy_values = np.squeeze(dy_values, axis=tuple(info["new_axis_positions"]))
    for position, dim in enumerate(info["dims"]):
        if dim["mode"] == 1:
            index_arrays.append(np.asarray([dim["fixed"]], dtype=np.int64))
            dy_values = np.expand_dims(dy_values, axis=position)
        else:
            index_arrays.append(np.arange(dim["start"], dim["stop"], dim["step"], dtype=np.int64))
    if index_arrays:
        result[np.ix_(*index_arrays)] = dy_values
    else:
        result[()] = dy_values
    return result


if __name__ == "__main__":
    y = impl([2, 4, 5, 6], [0, 1, 0, 0], [2, 4, 5, 6], [1, 1, 1, 1],
             np.arange(180, dtype=np.float32).reshape(2, 3, 5, 6))
    assert y.shape == (2, 4, 5, 6)
    assert np.all(y[:, 0, :, :] == 0)
    print("reference self-check OK, out shape:", y.shape)
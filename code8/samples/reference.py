#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""QuantUpdateScatter（aclnn 侧是 aclnnInplaceQuantScatter）的参考实现。

语义按内置 AscendC 内核（cann-9.1.0 .../ops_nn/ascendc/quant_update_scatter/arch35/
quant_update_scatter_regbase.h）+ 实测探针（st/QuantUpdateScatter/probe）确定：

  for p, idx in enumerate(indices):                 # indices 1 维，长度 == updates.shape(0)
      for k in range(updates.shape(axis)):          # axis 只能是 -2，是「块」大小
          y[p, ..., idx + k, :] = cast_int8(
              fp32(updates[p, ..., k, :]) / fp32(scales) + fp32(zero_points))

即：量化只做「除以 scale 加 zero_point」，不跟 var 里原有的值发生关系；
updates 的 dim0 决定写在 var 的哪个 dim0 分片上，dim1 及以后逐维对齐；
被写的行只有 indices[p] 起 updates.shape(axis) 行，其余元素保持 var 原值。
取整是 CAST_RINT（就近取偶），除法在 fp32 里做（内核先 cast 到 fp32）。
"""
import numpy as np


def impl(var, indices, updates, quant_scales, quant_zero_points=None, reduce="update",
         axis=-2, quant_axis=-1, reciprocal_scale=False):
    var_np = np.asarray(var)
    indices_np = np.asarray(indices).reshape(-1)
    updates_np = np.asarray(updates)
    scales_np = np.asarray(quant_scales).reshape(-1)
    if quant_zero_points is None:
        zeros_np = np.zeros(scales_np.shape, dtype=np.float64)
    else:
        zeros_np = np.asarray(quant_zero_points).reshape(-1).astype(np.float64)

    ndim = updates_np.ndim
    ax = axis % ndim                       # 内置只支持 -2
    qax = quant_axis % ndim                # 内置只支持 -1（== ndim-1）

    if scales_np.size != updates_np.shape[qax]:
        raise ValueError(f"quant_scales 元素数 {scales_np.size} != updates 在量化轴上的长度 "
                         f"{updates_np.shape[qax]}")
    if reciprocal_scale:                   # 内置的 DivMode=MUL 分支（aclnn 接口没有这个开关）
        scales_np = 1.0 / scales_np

    shape = [1] * ndim
    shape[qax] = scales_np.size
    q = np.rint(updates_np.astype(np.float64) / scales_np.reshape(shape)
                + zeros_np.reshape(shape))
    q = np.clip(q, -128.0, 127.0)

    y = var_np.astype(np.int8).copy()
    for p, idx in enumerate(indices_np):
        if p >= q.shape[0]:                # indices 比 updates.dim0 长时多出来的不写
            break
        for k in range(q.shape[ax]):
            src = [slice(None)] * ndim
            dst = [slice(None)] * ndim
            src[0] = dst[0] = int(p)
            src[ax] = k
            dst[ax] = int(idx) + k
            y[tuple(dst)] = q[tuple(src)].astype(np.int8)
    return y


if __name__ == "__main__":
    # 与 st/QuantUpdateScatter/probe 实测结果对拍的三条探针
    var = np.full((2, 4, 32), -1, dtype=np.int8)
    upd = np.zeros((2, 1, 32), dtype=np.float16)
    upd[0, 0, :] = np.arange(32) - 50
    upd[1, 0, :] = np.arange(32) + 50
    y = impl(var, [0, 2], upd, np.ones((1, 1, 32), np.float32), np.zeros((1, 1, 32), np.int32))
    assert y[0, 0, :].tolist() == (np.arange(32) - 50).tolist()
    assert y[1, 2, :].tolist() == (np.arange(32) + 50).tolist()
    assert (y[0, 1] == -1).all() and (y[1, 0] == -1).all()

    upd = np.zeros((2, 1, 32), dtype=np.float16)
    upd[0, 0, :] = (np.arange(32) * 0.5).astype(np.float16)
    upd[1, 0, :] = (np.arange(32) * 0.5 + 20).astype(np.float16)
    y = impl(var, [0, 2], upd, np.full((1, 1, 32), 2.0, np.float32),
             np.full((1, 1, 32), 10, np.int32))
    assert y[0, 0, :].tolist() == (np.round(np.arange(32) * 0.25) + 10).astype(int).tolist()
    print("探针对拍通过")

/*!
 * \file quant_update_scatter.h
 * \brief QuantUpdateScatter Ascend C kernel
 *
 * Semantics (axis=-2, quant_axis=-1):
 *   for p, idx in enumerate(indices):
 *       for k in range(updates.shape[axis]):
 *           var_out[p, ..., idx + k, :] =
 *               int8_rne(clip(updates[p, ..., k, :] / scale + zero_point))
 *
 * Elements that are not covered by an update keep their value from var.  The
 * kernel writes every output element exactly once, so no separate GM->GM
 * initialisation pass (and no cross-core barrier) is required.
 *
 * CANN 9.0.0 notes:
 *   - DataCopyExtParams always uses the five-argument form.
 *   - Tensor + tensor uses Add(dst, src0, src1, n), never Adds.
 *   - bf16 is widened only with the vector Cast API (CAST_NONE); the scalar
 *     unit is never asked to convert bf16, which avoids the bisheng backend
 *     "not support bf16 type cast" error.
 *   - Unchanged int8 var rows are copied directly in UB; no Cast is needed.
 *   - fp32 -> int8 is a two-step conversion: fp32 -> half (CAST_RINT), then
 *     half -> int8 (CAST_RINT).  CANN 9.0.0 does not expose a direct
 *     float -> int8 vconv overload.
 */

#ifndef QUANTUPDATESCATTER_H
#define QUANTUPDATESCATTER_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_update_scatter_tiling_data.h"
#include "quant_update_scatter_tiling_key.h"

namespace NsQuantUpdateScatter {

using namespace AscendC;

template <typename A, typename B>
struct IsSameType {
    static constexpr bool value = false;
};
template <typename A>
struct IsSameType<A, A> {
    static constexpr bool value = true;
};

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t UB_ALIGN_BYTES = 32;

template <typename T, typename IDX_T, typename SCALE_T, typename ZP_T>
class QuantUpdateScatter {
public:
    __aicore__ inline QuantUpdateScatter() = default;

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                GM_ADDR quantScales, GM_ADDR quantZeroPoints, GM_ADDR varOut,
                                const QuantUpdateScatterTilingData* tilingData);
    __aicore__ inline void Process();

private:
    static __aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline void LoadScaleTile(int64_t colOffset, int32_t count);
    __aicore__ inline void LoadZeroPointTile(int64_t colOffset, int32_t count);
    __aicore__ inline void CopyVarRow(int64_t srcOffset, int64_t dstOffset, int32_t count);
    __aicore__ inline void ComputeUpdateRow(int64_t srcOffset, int64_t dstOffset, int32_t count);
    __aicore__ inline void ProcessRankN();
    __aicore__ inline void ProcessRank2();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> updateInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> varInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> scaleInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> zpInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    TBuf<QuePosition::VECCALC> updateF32Buf;
    TBuf<QuePosition::VECCALC> scaleF32Buf;
    TBuf<QuePosition::VECCALC> zpF32Buf;
    TBuf<QuePosition::VECCALC> oneBuf;
    TBuf<QuePosition::VECCALC> outHalfBuf;

    GlobalTensor<int8_t> varGm;
    GlobalTensor<IDX_T> indicesGm;
    GlobalTensor<T> updatesGm;
    GlobalTensor<SCALE_T> scalesGm;
    GlobalTensor<ZP_T> zpGm;
    GlobalTensor<int8_t> outGm;

    int64_t varDim0 = 0;
    int64_t varAxis = 0;
    int64_t updDim0 = 0;
    int64_t updAxis = 0;
    int64_t outer = 0;
    int64_t rowLen = 0;
    int64_t indicesNum = 0;
    int64_t colTile = 0;
    int64_t varPStride = 0;
    int64_t updPStride = 0;
    int64_t start = 0;
    int64_t end = 0;

    bool rank2 = false;
    bool reciprocalScale = false;
    bool zpPresent = false;
    bool ready = false;
};

template <typename T, typename IDX_T, typename SCALE_T, typename ZP_T>
__aicore__ inline void QuantUpdateScatter<T, IDX_T, SCALE_T, ZP_T>::Init(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR quantScales,
    GM_ADDR quantZeroPoints, GM_ADDR varOut, const QuantUpdateScatterTilingData* tilingData)
{
    varDim0 = tilingData->varDim0;
    varAxis = tilingData->varAxis;
    updDim0 = tilingData->updDim0;
    updAxis = tilingData->updAxis;
    outer = tilingData->outer;
    rowLen = tilingData->rowLen;
    indicesNum = tilingData->indicesNum;
    colTile = tilingData->colTile > 0 ? tilingData->colTile : 1;
    varPStride = tilingData->varPStride;
    updPStride = tilingData->updPStride;
    rank2 = tilingData->rank2 != 0;
    reciprocalScale = tilingData->reciprocalScale != 0;
    zpPresent = (tilingData->hasZeroPoint != 0) && (quantZeroPoints != nullptr);

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    start = blockIdx * tilingData->pPerCore;
    end = start + tilingData->pPerCore;
    if (start > varDim0) {
        start = varDim0;
    }
    if (end > varDim0) {
        end = varDim0;
    }
    if (start >= end || rowLen <= 0 || varAxis <= 0) {
        return;
    }

    varGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(var), varDim0 * varPStride);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(varOut), varDim0 * varPStride);
    if (indicesNum > 0) {
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ IDX_T*>(indices), indicesNum);
    }
    if (updDim0 > 0) {
        updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updates), updDim0 * updPStride);
    }
    scalesGm.SetGlobalBuffer(reinterpret_cast<__gm__ SCALE_T*>(quantScales), rowLen);
    if (zpPresent) {
        zpGm.SetGlobalBuffer(reinterpret_cast<__gm__ ZP_T*>(quantZeroPoints), rowLen);
    }

    const int64_t tileElems = colTile;
    const int64_t tileBytesT = AlignUp(tileElems * static_cast<int64_t>(sizeof(T)), UB_ALIGN_BYTES);
    const int64_t tileBytesS = AlignUp(tileElems * static_cast<int64_t>(sizeof(SCALE_T)), UB_ALIGN_BYTES);
    const int64_t tileBytesZ = AlignUp(tileElems * static_cast<int64_t>(sizeof(ZP_T)), UB_ALIGN_BYTES);
    const int64_t tileBytesF = AlignUp(tileElems * static_cast<int64_t>(sizeof(float)), UB_ALIGN_BYTES);
    const int64_t tileBytesI8 = AlignUp(tileElems * static_cast<int64_t>(sizeof(int8_t)), UB_ALIGN_BYTES);
    const int64_t tileBytesH = AlignUp(tileElems * static_cast<int64_t>(sizeof(half)), UB_ALIGN_BYTES);

    pipe.InitBuffer(updateInQueue, BUFFER_NUM, tileBytesT);
    pipe.InitBuffer(varInQueue, BUFFER_NUM, tileBytesI8);
    pipe.InitBuffer(scaleInQueue, BUFFER_NUM, tileBytesS);
    pipe.InitBuffer(zpInQueue, BUFFER_NUM, tileBytesZ);
    pipe.InitBuffer(outQueue, BUFFER_NUM, tileBytesI8);
    pipe.InitBuffer(updateF32Buf, tileBytesF);
    pipe.InitBuffer(scaleF32Buf, tileBytesF);
    pipe.InitBuffer(zpF32Buf, tileBytesF);
    pipe.InitBuffer(oneBuf, tileBytesF);
    pipe.InitBuffer(outHalfBuf, tileBytesH);
    ready = true;
}

template <typename T, typename IDX_T, typename SCALE_T, typename ZP_T>
__aicore__ inline void QuantUpdateScatter<T, IDX_T, SCALE_T, ZP_T>::LoadScaleTile(
    int64_t colOffset, int32_t count)
{
    LocalTensor<SCALE_T> scaleIn = scaleInQueue.AllocTensor<SCALE_T>();
    DataCopyExtParams copyParams{1,
        static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(SCALE_T))), 0, 0, 0};
    DataCopyPadExtParams<SCALE_T> padParams{false, 0, 0, SCALE_T{}};
    DataCopyPad(scaleIn, scalesGm[colOffset], copyParams, padParams);
    scaleInQueue.EnQue(scaleIn);
    scaleIn = scaleInQueue.DeQue<SCALE_T>();

    LocalTensor<float> scaleF32 = scaleF32Buf.Get<float>();
    if constexpr (IsSameType<SCALE_T, float>::value) {
        Muls(scaleF32, scaleIn, 1.0f, count);
    } else {
        Cast(scaleF32, scaleIn, RoundMode::CAST_NONE, count);
    }
    scaleInQueue.FreeTensor(scaleIn);
    if (reciprocalScale) {
        LocalTensor<float> one = oneBuf.Get<float>();
        Duplicate(one, 1.0f, count);
        Div(scaleF32, one, scaleF32, count);
    }
}

template <typename T, typename IDX_T, typename SCALE_T, typename ZP_T>
__aicore__ inline void QuantUpdateScatter<T, IDX_T, SCALE_T, ZP_T>::LoadZeroPointTile(
    int64_t colOffset, int32_t count)
{
    if (!zpPresent) {
        return;
    }
    LocalTensor<ZP_T> zpIn = zpInQueue.AllocTensor<ZP_T>();
    DataCopyExtParams copyParams{1,
        static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(ZP_T))), 0, 0, 0};
    DataCopyPadExtParams<ZP_T> padParams{false, 0, 0, ZP_T{}};
    DataCopyPad(zpIn, zpGm[colOffset], copyParams, padParams);
    zpInQueue.EnQue(zpIn);
    zpIn = zpInQueue.DeQue<ZP_T>();

    LocalTensor<float> zpF32 = zpF32Buf.Get<float>();
    if constexpr (IsSameType<ZP_T, float>::value) {
        Muls(zpF32, zpIn, 1.0f, count);
    } else {
        Cast(zpF32, zpIn, RoundMode::CAST_NONE, count);
    }
    zpInQueue.FreeTensor(zpIn);
}

template <typename T, typename IDX_T, typename SCALE_T, typename ZP_T>
__aicore__ inline void QuantUpdateScatter<T, IDX_T, SCALE_T, ZP_T>::CopyVarRow(
    int64_t srcOffset, int64_t dstOffset, int32_t count)
{
    LocalTensor<int8_t> varLocal = varInQueue.AllocTensor<int8_t>();
    DataCopyExtParams inParams{1, static_cast<uint32_t>(count), 0, 0, 0};
    DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
    DataCopyPad(varLocal, varGm[srcOffset], inParams, padParams);

    // This row is not updated; copy the int8 bytes through UB unchanged.
    // Avoid the redundant int8 -> half -> int8 round trip here.
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopyExtParams outParams{1, static_cast<uint32_t>(count), 0, 0, 0};
    DataCopyPad(outGm[dstOffset], varLocal, outParams);
    varInQueue.FreeTensor(varLocal);
}

template <typename T, typename IDX_T, typename SCALE_T, typename ZP_T>
__aicore__ inline void QuantUpdateScatter<T, IDX_T, SCALE_T, ZP_T>::ComputeUpdateRow(
    int64_t srcOffset, int64_t dstOffset, int32_t count)
{
    LocalTensor<T> updateLocal = updateInQueue.AllocTensor<T>();
    DataCopyExtParams inParams{1,
        static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, T{}};
    DataCopyPad(updateLocal, updatesGm[srcOffset], inParams, padParams);
    updateInQueue.EnQue(updateLocal);
    updateLocal = updateInQueue.DeQue<T>();

    LocalTensor<float> value = updateF32Buf.Get<float>();
    if constexpr (IsSameType<T, float>::value) {
        Muls(value, updateLocal, 1.0f, count);
    } else {
        Cast(value, updateLocal, RoundMode::CAST_NONE, count);
    }
    updateInQueue.FreeTensor(updateLocal);

    Div(value, value, scaleF32Buf.Get<float>(), count);
    if (zpPresent) {
        Add(value, value, zpF32Buf.Get<float>(), count);
    }
    Mins(value, value, 127.0f, count);
    Maxs(value, value, -128.0f, count);

    // CANN 9.0.0 has no direct float -> int8 vconv overload.  Round to half
    // first, then narrow the half tensor to int8.
    LocalTensor<half> outHalf = outHalfBuf.Get<half>();
    Cast(outHalf, value, RoundMode::CAST_RINT, count);
    LocalTensor<int8_t> outLocal = outQueue.AllocTensor<int8_t>();
    Cast(outLocal, outHalf, RoundMode::CAST_RINT, count);
    outQueue.EnQue(outLocal);
    outLocal = outQueue.DeQue<int8_t>();
    DataCopyExtParams outParams{1, static_cast<uint32_t>(count), 0, 0, 0};
    DataCopyPad(outGm[dstOffset], outLocal, outParams);
    outQueue.FreeTensor(outLocal);
}

template <typename T, typename IDX_T, typename SCALE_T, typename ZP_T>
__aicore__ inline void QuantUpdateScatter<T, IDX_T, SCALE_T, ZP_T>::ProcessRankN()
{
    for (int64_t p = start; p < end; ++p) {
        bool hasUpdate = (p < updDim0) && (p < indicesNum);
        int64_t idx = 0;
        if (hasUpdate) {
            idx = static_cast<int64_t>(indicesGm.GetValue(p));
            if (idx < 0) {
                idx += varAxis;
            }
        }
        const int64_t varPBase = p * varPStride;
        const int64_t updPBase = p * updPStride;

        for (int64_t col = 0; col < rowLen; col += colTile) {
            const int64_t remain = rowLen - col;
            const int32_t n = static_cast<int32_t>(remain < colTile ? remain : colTile);
            if (hasUpdate) {
                LoadScaleTile(col, n);
                LoadZeroPointTile(col, n);
            }
            for (int64_t j = 0; j < outer; ++j) {
                for (int64_t r = 0; r < varAxis; ++r) {
                    const int64_t varOffset = varPBase + (j * varAxis + r) * rowLen + col;
                    const int64_t k = r - idx;
                    if (hasUpdate && k >= 0 && k < updAxis) {
                        const int64_t updOffset = updPBase + (j * updAxis + k) * rowLen + col;
                        ComputeUpdateRow(updOffset, varOffset, n);
                    } else {
                        CopyVarRow(varOffset, varOffset, n);
                    }
                }
            }
        }
    }
}

template <typename T, typename IDX_T, typename SCALE_T, typename ZP_T>
__aicore__ inline void QuantUpdateScatter<T, IDX_T, SCALE_T, ZP_T>::ProcessRank2()
{
    // axis == 0: the scatter dimension is also dim0.  Reference semantics use
    // updates[k] for every indices[p], and later records win on overlap.
    for (int64_t col = 0; col < rowLen; col += colTile) {
        const int64_t remain = rowLen - col;
        const int32_t n = static_cast<int32_t>(remain < colTile ? remain : colTile);
        LoadScaleTile(col, n);
        LoadZeroPointTile(col, n);
        for (int64_t r = start; r < end; ++r) {
            int64_t srcRow = -1;
            for (int64_t p = indicesNum - 1; p >= 0; --p) {
                int64_t idx = static_cast<int64_t>(indicesGm.GetValue(p));
                if (idx < 0) {
                    idx += varAxis;
                }
                if (r >= idx && r < idx + updAxis) {
                    srcRow = r - idx;
                    break;
                }
            }
            const int64_t dstOffset = r * rowLen + col;
            if (srcRow >= 0 && srcRow < updDim0) {
                ComputeUpdateRow(srcRow * rowLen + col, dstOffset, n);
            } else {
                CopyVarRow(dstOffset, dstOffset, n);
            }
        }
    }
}

template <typename T, typename IDX_T, typename SCALE_T, typename ZP_T>
__aicore__ inline void QuantUpdateScatter<T, IDX_T, SCALE_T, ZP_T>::Process()
{
    if (!ready) {
        return;
    }
    if (rank2) {
        ProcessRank2();
    } else {
        ProcessRankN();
    }
}

} // namespace NsQuantUpdateScatter

#endif // QUANTUPDATESCATTER_H

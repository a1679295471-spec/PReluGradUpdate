/*!
 * \file p_relu_grad_update.h
 * \brief PReluGradUpdate 算子 kernel 类定义
 *
 * dx     = features > 0 ? grads : grads * weights
 * update = features > 0 ? 0     : features * grads
 *
 * weights is [C] and is broadcast as [1, C, 1, ...].  Two scheduling modes
 * are used by the host tiling:
 *   - pattern mode: C * innerSize fits in UB.  One period of weights is
 *     materialised once and replicated in UB.  This also covers the 2-D
 *     [N, C] case where innerSize == 1.
 *   - channel mode: each tile is kept inside a single channel, so the weight
 *     is a scalar and no per-element weight expansion is needed.
 */

#ifndef PRELUGRADUPDATE_H
#define PRELUGRADUPDATE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "p_relu_grad_update_tiling_data.h"
#include "p_relu_grad_update_tiling_key.h"

namespace NsPReluGradUpdate {

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

template <typename T>
class PReluGradUpdate {
public:
    __aicore__ inline PReluGradUpdate() = default;

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx,
                                GM_ADDR update, const PReluGradUpdateTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count);
    __aicore__ inline void PrepareFloatInputs(int64_t count);
    __aicore__ inline void BuildPattern();
    __aicore__ inline void BuildWeightVector(int64_t count);
    __aicore__ inline void ComputeWithVectorWeight(int64_t count);
    __aicore__ inline void ComputeWithScalarWeight(int64_t count, float weight);
    __aicore__ inline void StoreOutputs(int64_t count);
    __aicore__ inline void CopyOut(int64_t offset, int64_t count);

    static __aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
    {
        return (value + align - 1) / align * align;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGrads;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueFeatures;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueDx;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueUpdate;

    TBuf<QuePosition::VECCALC> grads32Buf;
    TBuf<QuePosition::VECCALC> features32Buf;
    TBuf<QuePosition::VECCALC> selectBuf;
    TBuf<QuePosition::VECCALC> tempBuf;
    TBuf<QuePosition::VECCALC> weightBuf;
    TBuf<QuePosition::VECCALC> patternBuf;

    GlobalTensor<T> gradsGm;
    GlobalTensor<T> featuresGm;
    GlobalTensor<T> weightsGm;
    GlobalTensor<T> dxGm;
    GlobalTensor<T> updateGm;

    LocalTensor<float> grads32;
    LocalTensor<float> features32;
    LocalTensor<float> selectMask;
    LocalTensor<float> temp;
    LocalTensor<float> weightVec;
    LocalTensor<float> pattern;

    int64_t start = 0;
    int64_t end = 0;
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    int64_t innerSize = 1;
    int64_t channelNum = 1;
    int64_t patternSize = 0;
    bool usePattern = false;
};

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::Init(GM_ADDR grads, GM_ADDR features, GM_ADDR weights,
                                                GM_ADDR dx, GM_ADDR update,
                                                const PReluGradUpdateTilingData* tilingData)
{
    totalNum = tilingData->totalNum;
    blockFactor = tilingData->blockFactor;
    ubFactor = tilingData->ubFactor;
    innerSize = tilingData->innerSize;
    channelNum = tilingData->channelNum;
    patternSize = tilingData->patternSize;
    usePattern = tilingData->usePattern != 0;

    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    start = blockIdx * blockFactor;
    end = start + blockFactor;
    if (start > totalNum) {
        start = totalNum;
    }
    if (end > totalNum) {
        end = totalNum;
    }
    if (start >= end) {
        return;
    }

    gradsGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grads), totalNum);
    featuresGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(features), totalNum);
    weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weights), channelNum);
    dxGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dx), totalNum);
    updateGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(update), totalNum);

    const int64_t tileBytesT = AlignUp(ubFactor * static_cast<int64_t>(sizeof(T)), UB_ALIGN_BYTES);
    const int64_t tileBytesF = AlignUp(ubFactor * static_cast<int64_t>(sizeof(float)), UB_ALIGN_BYTES);

    // Build the periodic weight table before any TQue is created, so the
    // manual S->V event ids cannot collide with TPipe's queue events.
    if (usePattern) {
        pipe.InitBuffer(patternBuf, AlignUp(patternSize * static_cast<int64_t>(sizeof(float)), UB_ALIGN_BYTES));
        pattern = patternBuf.Get<float>();
        BuildPattern();
    }

    pipe.InitBuffer(inQueueGrads, BUFFER_NUM, tileBytesT);
    pipe.InitBuffer(inQueueFeatures, BUFFER_NUM, tileBytesT);
    pipe.InitBuffer(outQueueDx, BUFFER_NUM, tileBytesT);
    pipe.InitBuffer(outQueueUpdate, BUFFER_NUM, tileBytesT);
    pipe.InitBuffer(grads32Buf, tileBytesF);
    pipe.InitBuffer(features32Buf, tileBytesF);
    pipe.InitBuffer(selectBuf, tileBytesF);
    pipe.InitBuffer(tempBuf, tileBytesF);
    pipe.InitBuffer(weightBuf, tileBytesF);

    grads32 = grads32Buf.Get<float>();
    features32 = features32Buf.Get<float>();
    selectMask = selectBuf.Get<float>();
    temp = tempBuf.Get<float>();
    weightVec = weightBuf.Get<float>();
}


template <typename T>
__aicore__ inline void PReluGradUpdate<T>::BuildPattern()
{
    const int64_t period = channelNum * innerSize;
    const int64_t repeat = patternSize / period;
    int64_t index = 0;
    for (int64_t r = 0; r < repeat; ++r) {
        for (int64_t c = 0; c < channelNum; ++c) {
            const float value = static_cast<float>(weightsGm.GetValue(c));
            for (int64_t k = 0; k < innerSize; ++k) {
                pattern.SetValue(index, value);
                ++index;
            }
        }
    }
    // scalar writes -> vector reads
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::CopyIn(int64_t offset, int64_t count)
{
    LocalTensor<T> gradsLocal = inQueueGrads.AllocTensor<T>();
    LocalTensor<T> featuresLocal = inQueueFeatures.AllocTensor<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0.0f)};
    DataCopyPad(gradsLocal, gradsGm[offset], copyParams, padParams);
    DataCopyPad(featuresLocal, featuresGm[offset], copyParams, padParams);

    inQueueGrads.EnQue(gradsLocal);
    inQueueFeatures.EnQue(featuresLocal);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::PrepareFloatInputs(int64_t count)
{
    LocalTensor<T> gradsLocal = inQueueGrads.DeQue<T>();
    LocalTensor<T> featuresLocal = inQueueFeatures.DeQue<T>();

    if constexpr (IsSameType<T, float>::value) {
        Muls(grads32, gradsLocal, 1.0f, static_cast<int32_t>(count));
        Muls(features32, featuresLocal, 1.0f, static_cast<int32_t>(count));
    } else {
        Cast(grads32, gradsLocal, RoundMode::CAST_NONE, static_cast<int32_t>(count));
        Cast(features32, featuresLocal, RoundMode::CAST_NONE, static_cast<int32_t>(count));
    }

    inQueueGrads.FreeTensor(gradsLocal);
    inQueueFeatures.FreeTensor(featuresLocal);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::BuildWeightVector(int64_t count)
{
    int64_t copyCount = count < patternSize ? count : patternSize;
    Adds(weightVec, pattern, 0.0f, static_cast<int32_t>(copyCount));
    int64_t filled = copyCount;
    while (filled < count) {
        int64_t remain = count - filled;
        int64_t n = remain < filled ? remain : filled;
        Adds(weightVec[filled], weightVec, 0.0f, static_cast<int32_t>(n));
        filled += n;
    }
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::ComputeWithVectorWeight(int64_t count)
{
    const int32_t n = static_cast<int32_t>(count);
    // selectMask = 1 for features > 0, 0 otherwise.
    Sign(temp, features32, n);
    Maxs(selectMask, temp, 0.0f, n);

    // coeff = p + (1 - p) * weight.  p is exactly 0/1, so the two branches
    // are exact: f > 0 -> 1, f <= 0 -> weight.
    Muls(temp, selectMask, -1.0f, n);
    Adds(temp, temp, 1.0f, n);
    Mul(temp, temp, weightVec, n);
    Add(temp, temp, selectMask, n);

    // update = features * grads * (1 - p)
    Mul(features32, features32, grads32, n);
    Mul(selectMask, selectMask, features32, n);
    Sub(features32, features32, selectMask, n);

    // dx = grads * coeff
    Mul(grads32, grads32, temp, n);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::ComputeWithScalarWeight(int64_t count, float weight)
{
    const int32_t n = static_cast<int32_t>(count);
    Sign(temp, features32, n);
    Maxs(selectMask, temp, 0.0f, n);

    // coeff = p + (1 - p) * weight (exact for the 0/1 indicator).
    Muls(temp, selectMask, -1.0f, n);
    Adds(temp, temp, 1.0f, n);
    Muls(temp, temp, weight, n);
    Add(temp, temp, selectMask, n);

    Mul(features32, features32, grads32, n);
    Mul(selectMask, selectMask, features32, n);
    Sub(features32, features32, selectMask, n);

    Mul(grads32, grads32, temp, n);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::StoreOutputs(int64_t count)
{
    LocalTensor<T> dxLocal = outQueueDx.AllocTensor<T>();
    LocalTensor<T> updateLocal = outQueueUpdate.AllocTensor<T>();

    if constexpr (IsSameType<T, float>::value) {
        Muls(dxLocal, grads32, 1.0f, static_cast<int32_t>(count));
        Muls(updateLocal, features32, 1.0f, static_cast<int32_t>(count));
    } else {
        Cast(dxLocal, grads32, RoundMode::CAST_RINT, static_cast<int32_t>(count));
        Cast(updateLocal, features32, RoundMode::CAST_RINT, static_cast<int32_t>(count));
    }

    outQueueDx.EnQue(dxLocal);
    outQueueUpdate.EnQue(updateLocal);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::CopyOut(int64_t offset, int64_t count)
{
    LocalTensor<T> dxLocal = outQueueDx.DeQue<T>();
    LocalTensor<T> updateLocal = outQueueUpdate.DeQue<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(dxGm[offset], dxLocal, copyParams);
    DataCopyPad(updateGm[offset], updateLocal, copyParams);

    outQueueDx.FreeTensor(dxLocal);
    outQueueUpdate.FreeTensor(updateLocal);
}

template <typename T>
__aicore__ inline void PReluGradUpdate<T>::Process()
{
    if (start >= end) {
        return;
    }

    if (usePattern) {
        int64_t offset = start;
        while (offset < end) {
            const int64_t count = (end - offset) < ubFactor ? (end - offset) : ubFactor;
            CopyIn(offset, count);
            PrepareFloatInputs(count);
            BuildWeightVector(count);
            ComputeWithVectorWeight(count);
            StoreOutputs(count);
            CopyOut(offset, count);
            offset += count;
        }
        return;
    }

    int64_t offset = start;
    while (offset < end) {
        const int64_t channel = (offset / innerSize) % channelNum;
        const int64_t channelEnd = offset + innerSize - (offset % innerSize);
        const int64_t segmentEnd = channelEnd < end ? channelEnd : end;
        const float weight = static_cast<float>(weightsGm.GetValue(channel));
        while (offset < segmentEnd) {
            const int64_t count = (segmentEnd - offset) < ubFactor ? (segmentEnd - offset) : ubFactor;
            CopyIn(offset, count);
            PrepareFloatInputs(count);
            ComputeWithScalarWeight(count, weight);
            StoreOutputs(count);
            CopyOut(offset, count);
            offset += count;
        }
    }
}

} // namespace NsPReluGradUpdate
#endif // PRELUGRADUPDATE_H


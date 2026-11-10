/*!
 * \file strided_slice_grad.h
 * \brief StridedSliceGrad Ascend C kernel.
 *
 * The kernel does a gather over the output tensor y.  Every output element is
 * written exactly once by exactly one core: it is either zero (outside the
 * slice) or the corresponding dy element.  This avoids any zero/scatter race
 * between cores.
 *
 * RAW_T is the raw storage type (uint16_t for fp16/bf16, uint32_t for fp32).
 * Using the raw type keeps bf16 away from the scalar Cast path entirely.
 */

#ifndef STRIDEDSLICEGRAD_H
#define STRIDEDSLICEGRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "strided_slice_grad_tiling_data.h"
#include "strided_slice_grad_tiling_key.h"

namespace NsStridedSliceGrad {

using namespace AscendC;

template <typename RAW_T, typename IDX_T>
class StridedSliceGrad {
public:
    __aicore__ inline StridedSliceGrad() = default;

    __aicore__ inline void Init(GM_ADDR shape, GM_ADDR begin, GM_ADDR end, GM_ADDR strides,
                                GM_ADDR dy, GM_ADDR y,
                                const StridedSliceGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    static __aicore__ inline int64_t MinI64(int64_t a, int64_t b) { return a < b ? a : b; }
    static __aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline int64_t AdjustIndex(int64_t index, int64_t dim, int64_t step) const;
    __aicore__ inline int64_t SliceCount(int64_t start, int64_t stop, int64_t step) const;
    __aicore__ inline bool BuildSliceParams();
    __aicore__ inline bool MapYIndex(int64_t yIndex, int64_t& dyOffset) const;
    __aicore__ inline void ProcessTile(int64_t yBase, int32_t count);

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, 2> outQueue;

    GlobalTensor<RAW_T> dyGm;
    GlobalTensor<RAW_T> yGm;
    GlobalTensor<IDX_T> beginGm;
    GlobalTensor<IDX_T> endGm;
    GlobalTensor<IDX_T> stridesGm;

    int32_t rank_ = 0;
    int32_t dyRank_ = 0;
    int32_t specLen_ = 0;
    int64_t totalY_ = 0;
    int64_t totalDy_ = 0;
    int64_t yBlockFactor_ = 1;
    int64_t ubFactor_ = 1;

    int64_t yDims_[SSG_MAX_RANK] = {};
    int64_t yStrides_[SSG_MAX_RANK] = {};
    int64_t dyStrides_[SSG_MAX_DY_RANK] = {};
    int32_t inputMode_[SSG_MAX_RANK] = {};
    int32_t inputSpecIdx_[SSG_MAX_RANK] = {};
    int32_t inputDyAxis_[SSG_MAX_RANK] = {};

    int64_t beginMask_ = 0;
    int64_t endMask_ = 0;

    int64_t start_[SSG_MAX_RANK] = {};
    int64_t step_[SSG_MAX_RANK] = {};
    int64_t count_[SSG_MAX_RANK] = {};
    int64_t fixed_[SSG_MAX_RANK] = {};

    int64_t yStart_ = 0;
    int64_t yEnd_ = 0;
    bool valid_ = false;
};

template <typename RAW_T, typename IDX_T>
__aicore__ inline void StridedSliceGrad<RAW_T, IDX_T>::Init(
    GM_ADDR shape, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR dy, GM_ADDR y,
    const StridedSliceGradTilingData* tilingData)
{
    (void)shape;

    rank_ = tilingData->rank;
    dyRank_ = tilingData->dyRank;
    specLen_ = tilingData->specLen;
    totalY_ = tilingData->totalY;
    totalDy_ = tilingData->totalDy;
    yBlockFactor_ = tilingData->yBlockFactor > 0 ? tilingData->yBlockFactor : 1;
    ubFactor_ = tilingData->ubFactor > 0 ? tilingData->ubFactor : 1;

    for (int32_t i = 0; i < SSG_MAX_RANK; ++i) {
        yDims_[i] = tilingData->yDims[i];
        yStrides_[i] = tilingData->yStrides[i];
        inputMode_[i] = tilingData->inputMode[i];
        inputSpecIdx_[i] = tilingData->inputSpecIdx[i];
        inputDyAxis_[i] = tilingData->inputDyAxis[i];
    }
    for (int32_t i = 0; i < SSG_MAX_DY_RANK; ++i) {
        dyStrides_[i] = tilingData->dyStrides[i];
    }
    beginMask_ = tilingData->beginMask;
    endMask_ = tilingData->endMask;

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    yStart_ = blockIdx * yBlockFactor_;
    yEnd_ = MinI64(totalY_, yStart_ + yBlockFactor_);

    if (yStart_ >= yEnd_ || totalY_ <= 0) {
        return;
    }

    beginGm.SetGlobalBuffer(reinterpret_cast<__gm__ IDX_T*>(begin), specLen_);
    endGm.SetGlobalBuffer(reinterpret_cast<__gm__ IDX_T*>(end), specLen_);
    stridesGm.SetGlobalBuffer(reinterpret_cast<__gm__ IDX_T*>(strides), specLen_);
    // An empty slice leaves totalDy_ == 0; keep the buffer length legal, the
    // mapping never dereferences it in that case.
    dyGm.SetGlobalBuffer(reinterpret_cast<__gm__ RAW_T*>(dy), totalDy_ > 0 ? totalDy_ : 1);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ RAW_T*>(y), totalY_);

    const int64_t tileBytes = AlignUp(ubFactor_ * static_cast<int64_t>(sizeof(RAW_T)), 32);
    pipe.InitBuffer(outQueue, 2, static_cast<uint32_t>(tileBytes));
    valid_ = BuildSliceParams();
}

template <typename RAW_T, typename IDX_T>
__aicore__ inline int64_t StridedSliceGrad<RAW_T, IDX_T>::AdjustIndex(
    int64_t index, int64_t dim, int64_t step) const
{
    if (index < 0) {
        index += dim;
    }
    if (step > 0) {
        if (index < 0) {
            index = 0;
        }
        if (index > dim) {
            index = dim;
        }
    } else {
        if (index < -1) {
            index = -1;
        }
        if (index > dim - 1) {
            index = dim - 1;
        }
    }
    return index;
}

template <typename RAW_T, typename IDX_T>
__aicore__ inline int64_t StridedSliceGrad<RAW_T, IDX_T>::SliceCount(
    int64_t start, int64_t stop, int64_t step) const
{
    if (step > 0) {
        if (start >= stop) {
            return 0;
        }
        return (stop - start + step - 1) / step;
    }
    if (start <= stop) {
        return 0;
    }
    const int64_t absStep = -step;
    return (start - stop + absStep - 1) / absStep;
}

template <typename RAW_T, typename IDX_T>
__aicore__ inline bool StridedSliceGrad<RAW_T, IDX_T>::BuildSliceParams()
{
    for (int32_t d = 0; d < rank_; ++d) {
        start_[d] = 0;
        step_[d] = 1;
        count_[d] = yDims_[d];
        fixed_[d] = 0;

        if (inputMode_[d] == 1) {
            const int32_t specIdx = inputSpecIdx_[d];
            if (specIdx < 0 || specIdx >= specLen_) {
                return false;
            }
            int64_t index = static_cast<int64_t>(beginGm.GetValue(specIdx));
            if (index < 0) {
                index += yDims_[d];
            }
            if (index < 0 || index >= yDims_[d]) {
                return false;
            }
            fixed_[d] = index;
            continue;
        }

        const int32_t specIdx = inputSpecIdx_[d];
        if (specIdx < 0) {
            continue;
        }
        if (specIdx >= specLen_) {
            return false;
        }

        const int64_t beginValue = static_cast<int64_t>(beginGm.GetValue(specIdx));
        const int64_t endValue = static_cast<int64_t>(endGm.GetValue(specIdx));
        const int64_t stepValue = static_cast<int64_t>(stridesGm.GetValue(specIdx));
        if (stepValue == 0) {
            return false;
        }

        const bool beginIgnored = ((beginMask_ >> specIdx) & 1) != 0;
        const bool endIgnored = ((endMask_ >> specIdx) & 1) != 0;
        int64_t start = 0;
        int64_t stop = 0;
        if (beginIgnored) {
            start = stepValue < 0 ? yDims_[d] - 1 : 0;
        } else {
            start = AdjustIndex(beginValue, yDims_[d], stepValue);
        }
        if (endIgnored) {
            stop = stepValue < 0 ? -1 : yDims_[d];
        } else {
            stop = AdjustIndex(endValue, yDims_[d], stepValue);
        }
        start_[d] = start;
        step_[d] = stepValue;
        count_[d] = SliceCount(start, stop, stepValue);
    }
    return true;
}

template <typename RAW_T, typename IDX_T>
__aicore__ inline bool StridedSliceGrad<RAW_T, IDX_T>::MapYIndex(
    int64_t yIndex, int64_t& dyOffset) const
{
    int64_t coords[SSG_MAX_RANK] = {};
    int64_t remain = yIndex;
    for (int32_t d = 0; d < rank_; ++d) {
        const int64_t stride = yStrides_[d];
        coords[d] = stride > 0 ? remain / stride : 0;
        if (stride > 0) {
            remain = remain % stride;
        }
    }

    dyOffset = 0;
    for (int32_t d = 0; d < rank_; ++d) {
        if (inputMode_[d] == 1) {
            if (coords[d] != fixed_[d]) {
                return false;
            }
            continue;
        }

        int64_t diff = coords[d] - start_[d];
        int64_t q = 0;
        if (step_[d] > 0) {
            if (diff < 0 || (diff % step_[d]) != 0) {
                return false;
            }
            q = diff / step_[d];
        } else {
            const int64_t absStep = -step_[d];
            diff = start_[d] - coords[d];
            if (diff < 0 || (diff % absStep) != 0) {
                return false;
            }
            q = diff / absStep;
        }
        if (q < 0 || q >= count_[d]) {
            return false;
        }

        const int32_t axis = inputDyAxis_[d];
        if (axis < 0 || axis >= dyRank_) {
            return false;
        }
        dyOffset += q * dyStrides_[axis];
    }

    return dyOffset >= 0 && dyOffset < totalDy_;
}

template <typename RAW_T, typename IDX_T>
__aicore__ inline void StridedSliceGrad<RAW_T, IDX_T>::ProcessTile(
    int64_t yBase, int32_t count)
{
    LocalTensor<RAW_T> local = outQueue.AllocTensor<RAW_T>();
    // The previous tile may still be in flight on MTE3; wait before scalar stores.
    pipe_barrier(PIPE_ALL);
    for (int32_t i = 0; i < count; ++i) {
        RAW_T value{};
        int64_t dyOffset = 0;
        if (valid_ && MapYIndex(yBase + i, dyOffset)) {
            value = dyGm.GetValue(dyOffset);
        }
        local.SetValue(i, value);
    }

    // The tile is filled by scalar stores; make them visible before MTE3 reads it.
    pipe_barrier(PIPE_ALL);

    DataCopyExtParams outParams{1,
        static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(RAW_T))), 0, 0, 0};
    DataCopyPad(yGm[yBase], local, outParams);
    outQueue.FreeTensor(local);
}

template <typename RAW_T, typename IDX_T>
__aicore__ inline void StridedSliceGrad<RAW_T, IDX_T>::Process()
{
    if (yStart_ >= yEnd_ || totalY_ <= 0) {
        return;
    }
    for (int64_t offset = yStart_; offset < yEnd_; offset += ubFactor_) {
        const int64_t remain = yEnd_ - offset;
        const int32_t count = static_cast<int32_t>(remain < ubFactor_ ? remain : ubFactor_);
        ProcessTile(offset, count);
    }
}

} // namespace NsStridedSliceGrad

#endif // STRIDEDSLICEGRAD_H

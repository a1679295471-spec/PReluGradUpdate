/*!
 * \file strided_slice_grad.cpp
 * \brief StridedSliceGrad kernel entry and dtype dispatch.
 */

#include "strided_slice_grad.h"

enum class StridedSliceGradTilingKey : uint32_t {
    I32_F16 = 0,
    I32_F32 = 1,
    I32_BF16 = 2,
    I64_F16 = 3,
    I64_F32 = 4,
    I64_BF16 = 5,
};

template <uint32_t schMode>
__global__ __aicore__ void strided_slice_grad(GM_ADDR shape, GM_ADDR begin, GM_ADDR end,
                                              GM_ADDR strides, GM_ADDR dy, GM_ADDR y,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StridedSliceGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(StridedSliceGradTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(StridedSliceGradTilingKey::I32_F16)) {
        NsStridedSliceGrad::StridedSliceGrad<uint16_t, int32_t> op;
        op.Init(shape, begin, end, strides, dy, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(StridedSliceGradTilingKey::I32_F32)) {
        NsStridedSliceGrad::StridedSliceGrad<uint32_t, int32_t> op;
        op.Init(shape, begin, end, strides, dy, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(StridedSliceGradTilingKey::I32_BF16)) {
        NsStridedSliceGrad::StridedSliceGrad<uint16_t, int32_t> op;
        op.Init(shape, begin, end, strides, dy, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(StridedSliceGradTilingKey::I64_F16)) {
        NsStridedSliceGrad::StridedSliceGrad<uint16_t, int64_t> op;
        op.Init(shape, begin, end, strides, dy, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(StridedSliceGradTilingKey::I64_F32)) {
        NsStridedSliceGrad::StridedSliceGrad<uint32_t, int64_t> op;
        op.Init(shape, begin, end, strides, dy, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(StridedSliceGradTilingKey::I64_BF16)) {
        NsStridedSliceGrad::StridedSliceGrad<uint16_t, int64_t> op;
        op.Init(shape, begin, end, strides, dy, y, &tilingData);
        op.Process();
    }
}

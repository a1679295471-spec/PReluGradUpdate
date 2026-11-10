/*!
 * \file strided_slice_grad_tiling_data.h
 * \brief StridedSliceGrad tiling data shared by host and kernel.
 */

#ifndef STRIDED_SLICE_GRAD_TILING_DATA_H
#define STRIDED_SLICE_GRAD_TILING_DATA_H

#include <cstdint>

constexpr int32_t SSG_MAX_RANK = 8;
constexpr int32_t SSG_MAX_SPEC = 16;
// A new axis adds one dy dim without consuming a y dim, so dy rank can exceed
// the output rank.
constexpr int32_t SSG_MAX_DY_RANK = SSG_MAX_RANK + SSG_MAX_SPEC;

struct StridedSliceGradTilingData {
    int64_t totalY = 0;
    int64_t totalDy = 0;

    int64_t yDims[SSG_MAX_RANK] = {};
    int64_t yStrides[SSG_MAX_RANK] = {};
    int64_t dyDims[SSG_MAX_DY_RANK] = {};
    int64_t dyStrides[SSG_MAX_DY_RANK] = {};

    int32_t rank = 0;
    int32_t dyRank = 0;
    int32_t specLen = 0;

    // inputMode: 0 = normal slice, 1 = shrink axis.
    // inputSpecIdx: index into begin/end/strides, -1 = implicit full slice.
    // inputDyAxis: dy axis consumed by this input dim, -1 = shrink axis.
    int32_t inputMode[SSG_MAX_RANK] = {};
    int32_t inputSpecIdx[SSG_MAX_RANK] = {};
    int32_t inputDyAxis[SSG_MAX_RANK] = {};

    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;

    int64_t yBlockFactor = 1;
    int64_t ubFactor = 1;
};

#endif // STRIDED_SLICE_GRAD_TILING_DATA_H

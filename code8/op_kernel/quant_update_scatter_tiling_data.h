/*!
 * \file quant_update_scatter_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _QUANTUPDATESCATTER_TILING_DATA_H_
#define _QUANTUPDATESCATTER_TILING_DATA_H_

#include <cstdint>

struct QuantUpdateScatterTilingData {
    int64_t varDim0 = 0;
    int64_t varAxis = 0;
    int64_t updDim0 = 0;
    int64_t updAxis = 0;
    int64_t outer = 0;
    int64_t rowLen = 0;
    int64_t indicesNum = 0;
    int64_t colTile = 0;
    int64_t pPerCore = 0;
    int64_t varPStride = 0;
    int64_t updPStride = 0;
    int64_t rank2 = 0;
    int64_t reciprocalScale = 0;
    int64_t hasZeroPoint = 0;
};
#endif

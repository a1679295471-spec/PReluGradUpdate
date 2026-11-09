/*!
 * \file p_relu_grad_update_tiling_data.h
 * \brief tiling data struct
 *
 * Fields are shared by op_host and op_kernel.  All values are element counts
 * unless stated otherwise.
 */

#ifndef _PRELUGRADUPDATE_TILING_DATA_H_
#define _PRELUGRADUPDATE_TILING_DATA_H_

struct PReluGradUpdateTilingData {
    int64_t totalNum = 0;      // number of elements of grads / features
    int64_t blockFactor = 0;   // elements handled by one AI core
    int64_t ubFactor = 0;      // elements handled by one UB tile
    int64_t innerSize = 1;     // product(shape[2:]), i.e. elements per channel in one sample
    int64_t channelNum = 1;    // shape[1]
    int64_t patternSize = 0;   // aligned weight period (C * innerSize rounded to 32B)
    int64_t usePattern = 0;    // 1: weights are expanded through the periodic pattern
};

#endif

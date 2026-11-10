/*!
 * \file strided_slice_grad_tiling_key.h
 * \brief Tiling template arguments for StridedSliceGrad.
 *
 * key = index_selector * 3 + value_selector
 *   index_selector: 0 = int32, 1 = int64
 *   value_selector: 0 = float16, 1 = float32, 2 = bfloat16
 */

#ifndef STRIDED_SLICE_GRAD_TILING_KEY_H
#define STRIDED_SLICE_GRAD_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define SSG_TPL_MODE_I32_F16 0
#define SSG_TPL_MODE_I32_F32 1
#define SSG_TPL_MODE_I32_BF16 2
#define SSG_TPL_MODE_I64_F16 3
#define SSG_TPL_MODE_I64_F32 4
#define SSG_TPL_MODE_I64_BF16 5

ASCENDC_TPL_ARGS_DECL(
    StridedSliceGrad,
    ASCENDC_TPL_UINT_DECL(schMode, 3, ASCENDC_TPL_UI_LIST,
                          SSG_TPL_MODE_I32_F16, SSG_TPL_MODE_I32_F32, SSG_TPL_MODE_I32_BF16,
                          SSG_TPL_MODE_I64_F16, SSG_TPL_MODE_I64_F32, SSG_TPL_MODE_I64_BF16));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                         SSG_TPL_MODE_I32_F16, SSG_TPL_MODE_I32_F32, SSG_TPL_MODE_I32_BF16,
                         SSG_TPL_MODE_I64_F16, SSG_TPL_MODE_I64_F32, SSG_TPL_MODE_I64_BF16)));

#endif // STRIDED_SLICE_GRAD_TILING_KEY_H

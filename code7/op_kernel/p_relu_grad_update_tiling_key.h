/*!
 * \file p_relu_grad_update_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __PRELUGRADUPDATE_TILING_KEY_H__
#define __PRELUGRADUPDATE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define PRELUGRADUPDATE_TPL_SCH_MODE_0 0  // float16
#define PRELUGRADUPDATE_TPL_SCH_MODE_1 1  // float32
#define PRELUGRADUPDATE_TPL_SCH_MODE_2 2  // bfloat16

ASCENDC_TPL_ARGS_DECL(
    PReluGradUpdate,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST,
                          PRELUGRADUPDATE_TPL_SCH_MODE_0,
                          PRELUGRADUPDATE_TPL_SCH_MODE_1,
                          PRELUGRADUPDATE_TPL_SCH_MODE_2));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                         PRELUGRADUPDATE_TPL_SCH_MODE_0,
                         PRELUGRADUPDATE_TPL_SCH_MODE_1,
                         PRELUGRADUPDATE_TPL_SCH_MODE_2)));

#endif

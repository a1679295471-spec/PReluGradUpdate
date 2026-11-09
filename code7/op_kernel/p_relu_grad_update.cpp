/*!
 * \file p_relu_grad_update.cpp
 * \brief PReluGradUpdate 算子 kernel 入口
 */

#include "p_relu_grad_update.h"

using namespace AscendC;

enum class PReluGradUpdateTilingKey : uint32_t
{
    TILING_KEY_PRELUGRADUPDATE_MODE_0 = 0,  // float16
    TILING_KEY_PRELUGRADUPDATE_MODE_1 = 1,  // float32
    TILING_KEY_PRELUGRADUPDATE_MODE_2 = 2,  // bfloat16
};

template <uint32_t schMode>
__global__ __aicore__ void p_relu_grad_update(GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx, GM_ADDR update, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PReluGradUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(PReluGradUpdateTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(PReluGradUpdateTilingKey::TILING_KEY_PRELUGRADUPDATE_MODE_0)) {
        NsPReluGradUpdate::PReluGradUpdate<half> op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(PReluGradUpdateTilingKey::TILING_KEY_PRELUGRADUPDATE_MODE_1)) {
        NsPReluGradUpdate::PReluGradUpdate<float> op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(PReluGradUpdateTilingKey::TILING_KEY_PRELUGRADUPDATE_MODE_2)) {
        NsPReluGradUpdate::PReluGradUpdate<bfloat16_t> op;
        op.Init(grads, features, weights, dx, update, &tilingData);
        op.Process();
    }
}

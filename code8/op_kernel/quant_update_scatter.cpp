/*!
 * \file quant_update_scatter.cpp
 * \brief QuantUpdateScatter kernel entry and dtype dispatch
 */

#include "quant_update_scatter.h"

enum class QuantUpdateScatterTilingKey : uint32_t {
    TILING_KEY_QUANTUPDATESCATTER_MODE_0 = 0, // fp16 updates / fp32 scale / int32 zp / int32 idx
    TILING_KEY_QUANTUPDATESCATTER_MODE_1 = 1, // bf16 updates / bf16 scale / bf16 zp / int64 idx
};

template <uint32_t schMode>
__global__ __aicore__ void quant_update_scatter(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                GM_ADDR quant_scales, GM_ADDR quant_zero_points,
                                                GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantUpdateScatterTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantUpdateScatterTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(QuantUpdateScatterTilingKey::TILING_KEY_QUANTUPDATESCATTER_MODE_0)) {
        NsQuantUpdateScatter::QuantUpdateScatter<half, int32_t, float, int32_t> op;
        op.Init(var, indices, updates, quant_scales, quant_zero_points, var_out, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantUpdateScatterTilingKey::TILING_KEY_QUANTUPDATESCATTER_MODE_1)) {
        NsQuantUpdateScatter::QuantUpdateScatter<bfloat16_t, int64_t, bfloat16_t, bfloat16_t> op;
        op.Init(var, indices, updates, quant_scales, quant_zero_points, var_out, &tilingData);
        op.Process();
    }
}

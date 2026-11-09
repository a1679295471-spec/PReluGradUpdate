/*!
 * \file p_relu_grad_update_tiling.cpp
 * \brief PReluGradUpdate 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../op_kernel/p_relu_grad_update_tiling_data.h"
#include "../op_kernel/p_relu_grad_update_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t RESERVED_UB_BYTES = 8192;
constexpr int64_t FLOAT_TEMP_BYTES = 5 * 4;  // grads32/features32/select/temp/weight
constexpr int64_t DOUBLE_BUFFER_FACTOR = 8;  // 4 tensors * 2 buffers

static int64_t Gcd(int64_t a, int64_t b)
{
    while (b != 0) {
        const int64_t r = a % b;
        a = b;
        b = r;
    }
    return a == 0 ? 1 : a;
}

static int64_t CeilDivI(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

static int64_t CeilAlignI(int64_t a, int64_t b)
{
    return CeilDivI(a, b) * b;
}

static int64_t FloorAlignI(int64_t a, int64_t b)
{
    return a / b * b;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndGetShape(gert::TilingContext* context, int64_t& totalNum,
                                        int64_t& channelNum, int64_t& innerSize)
{
    const gert::StorageShape* gradsStorage = context->GetInputShape(0);
    const gert::StorageShape* featuresStorage = context->GetInputShape(1);
    const gert::StorageShape* weightsStorage = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsStorage);

    const gert::Shape& gradsShape = gradsStorage->GetStorageShape();
    const gert::Shape& featuresShape = featuresStorage->GetStorageShape();
    const gert::Shape& weightsShape = weightsStorage->GetStorageShape();

    const int64_t rank = static_cast<int64_t>(gradsShape.GetDimNum());
    OP_CHECK_IF(rank < 2, OP_LOGE(context, "grads rank must be >= 2, got %ld", rank), return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(featuresShape.GetDimNum()) != rank,
                OP_LOGE(context, "features rank mismatch"), return ge::GRAPH_FAILED);
    for (int64_t i = 0; i < rank; ++i) {
        OP_CHECK_IF(gradsShape.GetDim(i) != featuresShape.GetDim(i),
                    OP_LOGE(context, "grads/features shape mismatch at dim %ld", i), return ge::GRAPH_FAILED);
        OP_CHECK_IF(gradsShape.GetDim(i) < 0,
                    OP_LOGE(context, "dynamic shape is not supported in tiling"), return ge::GRAPH_FAILED);
    }

    totalNum = gradsShape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "invalid total element count"), return ge::GRAPH_FAILED);

    channelNum = gradsShape.GetDim(1);
    innerSize = 1;
    for (int64_t i = 2; i < rank; ++i) {
        innerSize *= gradsShape.GetDim(i);
    }

    if (totalNum == 0) {
        // Empty tensor: the kernel exits immediately; keep every divisor non-zero.
        channelNum = channelNum > 0 ? channelNum : 1;
        innerSize = 1;
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(channelNum <= 0, OP_LOGE(context, "channel dim must be > 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(innerSize <= 0, OP_LOGE(context, "inner size must be > 0"), return ge::GRAPH_FAILED);

    const int64_t weightsRank = static_cast<int64_t>(weightsShape.GetDimNum());
    const int64_t weightNum = weightsShape.GetShapeSize();
    OP_CHECK_IF(weightNum != channelNum,
                OP_LOGE(context, "weights element count %ld != channelNum %ld", weightNum, channelNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightsRank != 1 && !(weightsRank == 2 && weightsShape.GetDim(0) == 1),
                OP_LOGE(context, "weights must be [C] or [1, C]"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PReluGradUpdateTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    int64_t channelNum = 0;
    int64_t innerSize = 0;
    OP_CHECK_IF(CheckAndGetShape(context, totalNum, channelNum, innerSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "invalid input shape"), return ge::GRAPH_FAILED);

    auto gradsDesc = context->GetInputDesc(0);
    auto featuresDesc = context->GetInputDesc(1);
    auto weightsDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsDesc);
    const ge::DataType dtype = gradsDesc->GetDataType();
    OP_CHECK_IF(featuresDesc->GetDataType() != dtype || weightsDesc->GetDataType() != dtype,
                OP_LOGE(context, "all inputs must have the same dtype"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0;
    int64_t typeSize = 0;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_SCH_MODE_0);
        typeSize = 2;
    } else if (dtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_SCH_MODE_1);
        typeSize = 4;
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_SCH_MODE_2);
        typeSize = 2;
    } else {
        OP_LOGE(context, "unsupported dtype %d", static_cast<int32_t>(dtype));
        return ge::GRAPH_FAILED;
    }

    const int64_t alignElems = 32 / typeSize;
    const int64_t bytesPerElem = DOUBLE_BUFFER_FACTOR * typeSize + FLOAT_TEMP_BYTES;
    const int64_t ubBudget = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(ubBudget <= RESERVED_UB_BYTES, OP_LOGE(context, "UB size too small"), return ge::GRAPH_FAILED);

    int64_t maxTileBase = FloorAlignI((ubBudget - RESERVED_UB_BYTES) / bytesPerElem, alignElems);
    if (maxTileBase < alignElems) {
        maxTileBase = alignElems;
    }

    const int64_t period = channelNum * innerSize;
    const int64_t patternSize = period / Gcd(period, alignElems) * alignElems;
    const int64_t patternBytes = patternSize * static_cast<int64_t>(sizeof(float));

    bool usePattern = false;
    int64_t maxTilePattern = 0;
    if (patternBytes + RESERVED_UB_BYTES < ubBudget) {
        maxTilePattern = FloorAlignI((ubBudget - RESERVED_UB_BYTES - patternBytes) / bytesPerElem, alignElems);
        usePattern = maxTilePattern >= patternSize;
    }

    int64_t ubFactor = 0;
    if (usePattern) {
        ubFactor = FloorAlignI(maxTilePattern, patternSize);
    } else {
        ubFactor = maxTileBase < innerSize ? maxTileBase : innerSize;
        if (ubFactor < 1) {
            ubFactor = 1;
        }
    }

    int64_t blockFactor = 0;
    int64_t blockDim = 1;
    if (totalNum > 0) {
        const int64_t perCore = CeilDivI(totalNum, coreNum);
        if (usePattern) {
            blockFactor = CeilAlignI(perCore, period);
        } else {
            blockFactor = perCore;
        }
        blockDim = CeilDivI(totalNum, blockFactor);
        if (blockDim < 1) {
            blockDim = 1;
        }
        if (blockDim > coreNum) {
            blockDim = coreNum;
        }
    }

    PReluGradUpdateTilingData* tiling = context->GetTilingData<PReluGradUpdateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->innerSize = innerSize;
    tiling->channelNum = channelNum;
    tiling->patternSize = usePattern ? patternSize : 0;
    tiling->usePattern = usePattern ? 1 : 0;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPReluGradUpdate([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct PReluGradUpdateCompileInfo {};

IMPL_OP_OPTILING(PReluGradUpdate).Tiling(PReluGradUpdateTilingFunc).TilingParse<PReluGradUpdateCompileInfo>(TilingParseForPReluGradUpdate);

} // namespace optiling


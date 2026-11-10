/*!
 * \file quant_update_scatter_tiling.cpp
 * \brief QuantUpdateScatter host tiling
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/quant_update_scatter_tiling_data.h"
#include "../op_kernel/quant_update_scatter_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t RESERVED_UB_BYTES = 16384;
constexpr int64_t QUEUE_AND_TEMP_BYTES = 22;

static int64_t CeilDivI(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

static int64_t NormalizeAxis(int64_t axis, int64_t rank)
{
    int64_t result = axis % rank;
    if (result < 0) {
        result += rank;
    }
    return result;
}

static int64_t ProductDims(const gert::Shape& shape, int64_t begin, int64_t end)
{
    int64_t value = 1;
    for (int64_t i = begin; i < end; ++i) {
        value *= shape.GetDim(i);
    }
    return value;
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

static ge::graphStatus CheckDtypeProfile(gert::TilingContext* context, uint64_t& tilingKey,
                                         int64_t& typeSizeT, int64_t& typeSizeS, int64_t& typeSizeZ)
{
    auto varDesc = context->GetInputDesc(0);
    auto indicesDesc = context->GetInputDesc(1);
    auto updatesDesc = context->GetInputDesc(2);
    auto scalesDesc = context->GetInputDesc(3);
    auto outDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);

    OP_CHECK_IF(varDesc->GetDataType() != ge::DT_INT8,
                OP_LOGE(context, "var dtype must be int8"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outDesc->GetDataType() != ge::DT_INT8,
                OP_LOGE(context, "var_out dtype must be int8"), return ge::GRAPH_FAILED);

    const ge::DataType updatesDtype = updatesDesc->GetDataType();
    if (updatesDtype == ge::DT_FLOAT16) {
        OP_CHECK_IF(indicesDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(context, "fp16 profile requires int32 indices"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(scalesDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context, "fp16 profile requires fp32 quant_scales"), return ge::GRAPH_FAILED);
        auto zpDesc = context->GetInputDesc(4);
        if (context->GetInputShape(4) != nullptr && zpDesc != nullptr) {
            OP_CHECK_IF(zpDesc->GetDataType() != ge::DT_INT32,
                        OP_LOGE(context, "fp16 profile requires int32 quant_zero_points"),
                        return ge::GRAPH_FAILED);
        }
        tilingKey = GET_TPL_TILING_KEY(QUANTUPDATESCATTER_TPL_SCH_MODE_0);
        typeSizeT = 2;
        typeSizeS = 4;
        typeSizeZ = 4;
        return ge::GRAPH_SUCCESS;
    }
    if (updatesDtype == ge::DT_BF16) {
        OP_CHECK_IF(indicesDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE(context, "bf16 profile requires int64 indices"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(scalesDesc->GetDataType() != ge::DT_BF16,
                    OP_LOGE(context, "bf16 profile requires bf16 quant_scales"), return ge::GRAPH_FAILED);
        auto zpDesc = context->GetInputDesc(4);
        if (context->GetInputShape(4) != nullptr && zpDesc != nullptr) {
            OP_CHECK_IF(zpDesc->GetDataType() != ge::DT_BF16,
                        OP_LOGE(context, "bf16 profile requires bf16 quant_zero_points"),
                        return ge::GRAPH_FAILED);
        }
        tilingKey = GET_TPL_TILING_KEY(QUANTUPDATESCATTER_TPL_SCH_MODE_1);
        typeSizeT = 2;
        typeSizeS = 2;
        typeSizeZ = 2;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(context, "updates dtype must be float16 or bfloat16");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus CheckAndFillShapes(gert::TilingContext* context,
                                          QuantUpdateScatterTilingData* tiling)
{
    const gert::StorageShape* varStorage = context->GetInputShape(0);
    const gert::StorageShape* indicesStorage = context->GetInputShape(1);
    const gert::StorageShape* updatesStorage = context->GetInputShape(2);
    const gert::StorageShape* scalesStorage = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, varStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalesStorage);

    const gert::Shape& varShape = varStorage->GetStorageShape();
    const gert::Shape& indicesShape = indicesStorage->GetStorageShape();
    const gert::Shape& updatesShape = updatesStorage->GetStorageShape();
    const gert::Shape& scalesShape = scalesStorage->GetStorageShape();
    const int64_t rank = static_cast<int64_t>(varShape.GetDimNum());
    OP_CHECK_IF(rank < 2, OP_LOGE(context, "rank must be >= 2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(updatesShape.GetDimNum()) != rank,
                OP_LOGE(context, "var/updates rank mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(indicesShape.GetDimNum()) != 1,
                OP_LOGE(context, "indices must be 1-D"), return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < rank; ++i) {
        OP_CHECK_IF(varShape.GetDim(i) < 0 || updatesShape.GetDim(i) < 0,
                    OP_LOGE(context, "dynamic shape is not supported in tiling"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(indicesShape.GetDim(0) < 0,
                OP_LOGE(context, "dynamic indices shape is not supported"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* quantAxisPtr = attrs->GetAttrPointer<int64_t>(2);
    const bool* reciprocalPtr = attrs->GetAttrPointer<bool>(3);
    const int64_t axis = NormalizeAxis(axisPtr == nullptr ? -2 : *axisPtr, rank);
    const int64_t quantAxis = NormalizeAxis(quantAxisPtr == nullptr ? -1 : *quantAxisPtr, rank);
    OP_CHECK_IF(axis != rank - 2, OP_LOGE(context, "only axis=-2 is supported"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(quantAxis != rank - 1, OP_LOGE(context, "only quant_axis=-1 is supported"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(updatesShape.GetDim(0) > varShape.GetDim(0),
                OP_LOGE(context, "updates.shape[0] must not exceed var.shape[0]"),
                return ge::GRAPH_FAILED);
    for (int64_t i = 1; i < rank; ++i) {
        if (i == axis) {
            continue;
        }
        OP_CHECK_IF(varShape.GetDim(i) != updatesShape.GetDim(i),
                    OP_LOGE(context, "var/updates shape mismatch on a non-axis dimension"),
                    return ge::GRAPH_FAILED);
    }

    const int64_t indicesNum = indicesShape.GetDim(0);
    const int64_t updDim0 = updatesShape.GetDim(0);
    if (rank == 2) {
        // Rank-2 axis=0: updates.shape[0] is the update block length, while
        // indices.shape[0] is the number of starting positions.  A block may
        // be longer than the number of indices, so equality is not required.
        OP_CHECK_IF(indicesNum > updDim0,
                    OP_LOGE(context, "indices length must not exceed updates.shape[0] for rank-2"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(indicesNum != updDim0,
                    OP_LOGE(context, "indices length must equal updates.shape[0]"),
                    return ge::GRAPH_FAILED);
    }

    const int64_t rowLen = updatesShape.GetDim(rank - 1);
    const int64_t scalesNum = scalesShape.GetShapeSize();
    OP_CHECK_IF(scalesNum != rowLen,
                OP_LOGE(context, "quant_scales element count must equal row length"),
                return ge::GRAPH_FAILED);

    const gert::StorageShape* zpStorage = context->GetInputShape(4);
    if (zpStorage != nullptr) {
        const int64_t zpNum = zpStorage->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(zpNum != rowLen,
                    OP_LOGE(context, "quant_zero_points element count must equal row length"),
                    return ge::GRAPH_FAILED);
    }
    tiling->hasZeroPoint = zpStorage != nullptr ? 1 : 0;

    tiling->varDim0 = varShape.GetDim(0);
    tiling->varAxis = varShape.GetDim(axis);
    tiling->updDim0 = updDim0;
    tiling->updAxis = updatesShape.GetDim(axis);
    tiling->outer = ProductDims(updatesShape, 1, axis);
    tiling->rowLen = rowLen;
    tiling->indicesNum = indicesNum;
    tiling->varPStride = ProductDims(varShape, 1, rank);
    tiling->updPStride = ProductDims(updatesShape, 1, rank);
    tiling->rank2 = rank == 2 ? 1 : 0;
    tiling->reciprocalScale = (reciprocalPtr != nullptr && *reciprocalPtr) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus QuantUpdateScatterTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    QuantUpdateScatterTilingData* tiling = context->GetTilingData<QuantUpdateScatterTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    uint64_t tilingKey = 0;
    int64_t typeSizeT = 0;
    int64_t typeSizeS = 0;
    int64_t typeSizeZ = 0;
    OP_CHECK_IF(CheckDtypeProfile(context, tilingKey, typeSizeT, typeSizeS, typeSizeZ) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "unsupported dtype profile"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAndFillShapes(context, tiling) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "invalid shapes or attributes"), return ge::GRAPH_FAILED);

    const int64_t bytesPerElem = 2 * typeSizeT + 2 * typeSizeS + 2 * typeSizeZ +
                                 QUEUE_AND_TEMP_BYTES;
    OP_CHECK_IF(static_cast<int64_t>(ubSize) <= RESERVED_UB_BYTES,
                OP_LOGE(context, "UB size too small"), return ge::GRAPH_FAILED);
    int64_t maxCol = (static_cast<int64_t>(ubSize) - RESERVED_UB_BYTES) / bytesPerElem;
    if (maxCol < 1) {
        maxCol = 1;
    }
    tiling->colTile = tiling->rowLen > 0 ? (tiling->rowLen < maxCol ? tiling->rowLen : maxCol) : 1;

    int64_t blockDim = 1;
    if (tiling->varDim0 > 0) {
        tiling->pPerCore = CeilDivI(tiling->varDim0, coreNum);
        if (tiling->pPerCore < 1) {
            tiling->pPerCore = 1;
        }
        blockDim = CeilDivI(tiling->varDim0, tiling->pPerCore);
        if (blockDim > coreNum) {
            blockDim = coreNum;
        }
    } else {
        tiling->pPerCore = 0;
        blockDim = 1;
    }

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantUpdateScatter([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantUpdateScatterCompileInfo {};

IMPL_OP_OPTILING(QuantUpdateScatter)
    .Tiling(QuantUpdateScatterTilingFunc)
    .TilingParse<QuantUpdateScatterCompileInfo>(TilingParseForQuantUpdateScatter);

} // namespace optiling

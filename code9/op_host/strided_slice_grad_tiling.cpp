/*!
 * \file strided_slice_grad_tiling.cpp
 * \brief StridedSliceGrad host tiling.
 *
 * Tiling decides the structural mapping between output dims `y` and gradient
 * dims `dy`, the block split, and the UB tile size.  begin/end/strides stay
 * runtime tensors: the kernel recomputes start/step/count on device, so tiling
 * never depends on their values.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/strided_slice_grad_tiling_data.h"
#include "../op_kernel/strided_slice_grad_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t RESERVED_UB_BYTES = 16384;
constexpr int64_t NUM_UB_BUFFERS = 2;
constexpr int64_t BLOCK_BYTES = 32;

static int64_t CeilDivI(int64_t a, int64_t b)
{
    return b > 0 ? (a + b - 1) / b : 0;
}

static int64_t FloorAlignI(int64_t value, int64_t align)
{
    if (align <= 0) {
        return value;
    }
    return value / align * align;
}

static int32_t PopCount64(uint64_t value)
{
    int32_t count = 0;
    while (value != 0U) {
        count += static_cast<int32_t>(value & 1U);
        value >>= 1U;
    }
    return count;
}

static int64_t GetShapeDim(const gert::StorageShape* storageShape, size_t index)
{
    const gert::Shape& shape = storageShape->GetStorageShape();
    if (index >= shape.GetDimNum()) {
        return -1;
    }
    return shape.GetDim(index);
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

static int64_t GetMask(const gert::RuntimeAttrs* attrs, size_t index)
{
    const int64_t* value = attrs->GetAttrPointer<int64_t>(index);
    return value == nullptr ? 0 : *value;
}

// Compile-safe access to the optional input-tensor API: if this CANN version
// does not expose GetInputTensor on the tiling context, the fallback simply
// reports "not available" instead of failing to compile.
template <typename Ctx>
static auto GetInputTensorAt(Ctx* context, size_t index, int)
    -> decltype(context->GetInputTensor(index))
{
    return context->GetInputTensor(index);
}

template <typename Ctx>
static const gert::Tensor* GetInputTensorAt(Ctx*, size_t, ...)
{
    return nullptr;
}

/*!
 * \brief Read a runtime index tensor into host memory.
 * \return true when the data is actually readable at tiling time.
 */
static bool ReadIndexValues(gert::TilingContext* context, size_t inputIdx,
                            ge::DataType dtype, int64_t* values, int32_t count)
{
    if (count <= 0) {
        return true;
    }
    const gert::Tensor* tensor = GetInputTensorAt(context, inputIdx, 0);
    if (tensor == nullptr) {
        return false;
    }
    if (dtype == ge::DT_INT32) {
        const int32_t* data = tensor->GetData<int32_t>();
        if (data == nullptr) {
            return false;
        }
        for (int32_t i = 0; i < count; ++i) {
            values[i] = static_cast<int64_t>(data[i]);
        }
        return true;
    }
    if (dtype == ge::DT_INT64) {
        const int64_t* data = tensor->GetData<int64_t>();
        if (data == nullptr) {
            return false;
        }
        for (int32_t i = 0; i < count; ++i) {
            values[i] = data[i];
        }
        return true;
    }
    return false;
}

/*!
 * \brief Structural mapping: which spec entry feeds which y dim, and which
 *        dy axis that y dim consumes.
 */
static ge::graphStatus BuildMapping(gert::TilingContext* context, StridedSliceGradTilingData* tiling,
                                    int32_t rank, int32_t specLen)
{
    for (int32_t i = 0; i < SSG_MAX_RANK; ++i) {
        tiling->inputMode[i] = 0;
        tiling->inputSpecIdx[i] = -1;
        tiling->inputDyAxis[i] = -1;
    }

    const int64_t beginMask = tiling->beginMask;
    const int64_t endMask = tiling->endMask;
    const int64_t ellipsisMask = tiling->ellipsisMask;
    const int64_t newAxisMask = tiling->newAxisMask;
    const int64_t shrinkAxisMask = tiling->shrinkAxisMask;

    OP_CHECK_IF(beginMask < 0 || endMask < 0 || ellipsisMask < 0 || newAxisMask < 0 || shrinkAxisMask < 0,
                OP_LOGE(context, "mask attributes must be non-negative"), return ge::GRAPH_FAILED);

    const uint64_t specLimit =
        specLen >= 63 ? ~0ULL : ((1ULL << static_cast<uint64_t>(specLen)) - 1ULL);
    if ((static_cast<uint64_t>(beginMask) & ~specLimit) != 0U ||
        (static_cast<uint64_t>(endMask) & ~specLimit) != 0U ||
        (static_cast<uint64_t>(ellipsisMask) & ~specLimit) != 0U ||
        (static_cast<uint64_t>(newAxisMask) & ~specLimit) != 0U ||
        (static_cast<uint64_t>(shrinkAxisMask) & ~specLimit) != 0U) {
        OP_LOGE(context, "mask bit is out of range for spec length %d", specLen);
        return ge::GRAPH_FAILED;
    }
    if (PopCount64(static_cast<uint64_t>(ellipsisMask)) > 1) {
        OP_LOGE(context, "ellipsis_mask must contain at most one bit");
        return ge::GRAPH_FAILED;
    }
    if ((newAxisMask & shrinkAxisMask) != 0) {
        OP_LOGE(context, "new_axis_mask and shrink_axis_mask overlap");
        return ge::GRAPH_FAILED;
    }
    if ((ellipsisMask & (newAxisMask | shrinkAxisMask)) != 0) {
        OP_LOGE(context, "ellipsis_mask overlaps new_axis_mask or shrink_axis_mask");
        return ge::GRAPH_FAILED;
    }

    const int32_t newAxisCount = PopCount64(static_cast<uint64_t>(newAxisMask));
    int32_t ellipsisLen = 0;
    if (ellipsisMask != 0) {
        const int32_t consumedBySpecs = specLen - 1 - newAxisCount;
        ellipsisLen = rank - consumedBySpecs;
        if (ellipsisLen < 0) {
            OP_LOGE(context, "ellipsis expands to a negative number of dims");
            return ge::GRAPH_FAILED;
        }
    } else {
        const int32_t consumedBySpecs = specLen - newAxisCount;
        if (consumedBySpecs > rank) {
            OP_LOGE(context, "begin length %d exceeds output rank %d", specLen, rank);
            return ge::GRAPH_FAILED;
        }
    }

    int32_t yDim = 0;
    int32_t dyAxis = 0;
    for (int32_t s = 0; s < specLen; ++s) {
        const uint64_t bit = 1ULL << static_cast<uint64_t>(s);
        if ((static_cast<uint64_t>(newAxisMask) & bit) != 0U) {
            ++dyAxis;
            continue;
        }
        if ((static_cast<uint64_t>(ellipsisMask) & bit) != 0U) {
            for (int32_t k = 0; k < ellipsisLen; ++k) {
                if (yDim >= rank) {
                    OP_LOGE(context, "ellipsis consumes more dims than rank");
                    return ge::GRAPH_FAILED;
                }
                tiling->inputMode[yDim] = 0;
                tiling->inputSpecIdx[yDim] = -1;
                tiling->inputDyAxis[yDim] = dyAxis;
                ++yDim;
                ++dyAxis;
            }
            continue;
        }
        if (yDim >= rank) {
            OP_LOGE(context, "spec entries consume more dims than rank");
            return ge::GRAPH_FAILED;
        }
        if ((static_cast<uint64_t>(shrinkAxisMask) & bit) != 0U) {
            tiling->inputMode[yDim] = 1;
            tiling->inputSpecIdx[yDim] = s;
            tiling->inputDyAxis[yDim] = -1;
            ++yDim;
            continue;
        }
        tiling->inputMode[yDim] = 0;
        tiling->inputSpecIdx[yDim] = s;
        tiling->inputDyAxis[yDim] = dyAxis;
        ++yDim;
        ++dyAxis;
    }

    while (yDim < rank) {
        tiling->inputMode[yDim] = 0;
        tiling->inputSpecIdx[yDim] = -1;
        tiling->inputDyAxis[yDim] = dyAxis;
        ++yDim;
        ++dyAxis;
    }
    if (dyAxis != tiling->dyRank) {
        OP_LOGE(context, "dy rank %d does not match the slice rank %d", tiling->dyRank, dyAxis);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus StridedSliceGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    StridedSliceGradTilingData* tiling = context->GetTilingData<StridedSliceGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto shapeDesc = context->GetInputDesc(0);
    auto beginDesc = context->GetInputDesc(1);
    auto endDesc = context->GetInputDesc(2);
    auto stridesDesc = context->GetInputDesc(3);
    auto dyDesc = context->GetInputDesc(4);
    auto yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, beginDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, endDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    // `shape` and the begin/end/strides tensors have independent dtypes in the
    // original op (Tshape vs Index).  Only begin/end/strides must agree with
    // each other, because they are read by the same kernel template parameter.
    const ge::DataType shapeDtype = shapeDesc->GetDataType();
    const ge::DataType idxDtype = beginDesc->GetDataType();
    OP_CHECK_IF(shapeDtype != ge::DT_INT32 && shapeDtype != ge::DT_INT64,
                OP_LOGE(context, "shape dtype must be int32 or int64"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(idxDtype != ge::DT_INT32 && idxDtype != ge::DT_INT64,
                OP_LOGE(context, "begin/end/strides dtype must be int32 or int64"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(endDesc->GetDataType() != idxDtype || stridesDesc->GetDataType() != idxDtype,
                OP_LOGE(context, "begin/end/strides dtype must match each other"),
                return ge::GRAPH_FAILED);

    const ge::DataType valueDtype = dyDesc->GetDataType();
    OP_CHECK_IF(valueDtype != ge::DT_FLOAT16 && valueDtype != ge::DT_FLOAT &&
                    valueDtype != ge::DT_BF16,
                OP_LOGE(context, "dy dtype must be float16, float32 or bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yDesc->GetDataType() != valueDtype,
                OP_LOGE(context, "y dtype must match dy dtype"), return ge::GRAPH_FAILED);

    int32_t valueSel = 0;
    int64_t elemBytes = 2;
    if (valueDtype == ge::DT_FLOAT) {
        valueSel = 1;
        elemBytes = 4;
    } else if (valueDtype == ge::DT_BF16) {
        valueSel = 2;
    }
    uint64_t tilingKey = 0;
    if (idxDtype == ge::DT_INT32) {
        tilingKey = valueSel == 0 ? GET_TPL_TILING_KEY(SSG_TPL_MODE_I32_F16)
                 : (valueSel == 1 ? GET_TPL_TILING_KEY(SSG_TPL_MODE_I32_F32)
                                  : GET_TPL_TILING_KEY(SSG_TPL_MODE_I32_BF16));
    } else {
        tilingKey = valueSel == 0 ? GET_TPL_TILING_KEY(SSG_TPL_MODE_I64_F16)
                 : (valueSel == 1 ? GET_TPL_TILING_KEY(SSG_TPL_MODE_I64_F32)
                                  : GET_TPL_TILING_KEY(SSG_TPL_MODE_I64_BF16));
    }

    const gert::StorageShape* shapeInput = context->GetInputShape(0);
    const gert::StorageShape* beginInput = context->GetInputShape(1);
    const gert::StorageShape* endInput = context->GetInputShape(2);
    const gert::StorageShape* stridesInput = context->GetInputShape(3);
    const gert::StorageShape* dyInput = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, beginInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, endInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyInput);

    int64_t rank = GetShapeDim(shapeInput, 0);
    const int64_t specLen64 = GetShapeDim(beginInput, 0);
    OP_CHECK_IF(specLen64 < 0, OP_LOGE(context, "dynamic begin length is not supported"),
                return ge::GRAPH_FAILED);
    const int64_t endLen = GetShapeDim(endInput, 0);
    const int64_t stridesLen = GetShapeDim(stridesInput, 0);
    OP_CHECK_IF(endLen >= 0 && endLen != specLen64,
                OP_LOGE(context, "begin/end length mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(stridesLen >= 0 && stridesLen != specLen64,
                OP_LOGE(context, "begin/strides length mismatch"), return ge::GRAPH_FAILED);

    if (rank < 0) {
        const gert::StorageShape* yShape = context->GetOutputShape(0);
        if (yShape != nullptr) {
            rank = static_cast<int64_t>(yShape->GetStorageShape().GetDimNum());
        }
    }
    OP_CHECK_IF(rank < 0, OP_LOGE(context, "cannot determine output rank"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(rank > SSG_MAX_RANK || specLen64 > SSG_MAX_SPEC,
                OP_LOGE(context, "rank %ld or spec length %ld exceeds supported maximum (%d/%d)",
                        static_cast<long>(rank), static_cast<long>(specLen64),
                        SSG_MAX_RANK, SSG_MAX_SPEC),
                return ge::GRAPH_FAILED);

    const int32_t rank32 = static_cast<int32_t>(rank);
    const int32_t specLen = static_cast<int32_t>(specLen64);

    tiling->rank = rank32;
    tiling->specLen = specLen;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    tiling->beginMask = GetMask(attrs, 0);
    tiling->endMask = GetMask(attrs, 1);
    tiling->ellipsisMask = GetMask(attrs, 2);
    tiling->newAxisMask = GetMask(attrs, 3);
    tiling->shrinkAxisMask = GetMask(attrs, 4);

    // The inferred output shape is authoritative.  Reading the `shape` tensor is
    // only a fallback for the case where shape inference could not bind a dim;
    // in the normal path tiling never touches the tensor data.
    const gert::StorageShape* yStorage = context->GetOutputShape(0);
    int64_t shapeVals[SSG_MAX_RANK] = {};
    bool haveShapeValues = false;
    bool triedShapeValues = false;
    for (int32_t d = 0; d < rank32; ++d) {
        int64_t dim = -1;
        if (yStorage != nullptr && d < static_cast<int32_t>(yStorage->GetStorageShape().GetDimNum())) {
            dim = yStorage->GetStorageShape().GetDim(d);
        }
        if (dim < 0 && !triedShapeValues) {
            triedShapeValues = true;
            haveShapeValues = ReadIndexValues(context, 0, shapeDtype, shapeVals, rank32);
        }
        if (dim < 0 && haveShapeValues) {
            dim = shapeVals[d];
        }
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "dim %d of shape is unknown", d),
                    return ge::GRAPH_FAILED);
        tiling->yDims[d] = dim;
    }

    const int64_t dyRank64 = static_cast<int64_t>(dyInput->GetStorageShape().GetDimNum());
    OP_CHECK_IF(dyRank64 > SSG_MAX_DY_RANK,
                OP_LOGE(context, "dy rank exceeds %d", SSG_MAX_DY_RANK), return ge::GRAPH_FAILED);
    tiling->dyRank = static_cast<int32_t>(dyRank64);
    for (int32_t d = 0; d < tiling->dyRank; ++d) {
        const int64_t dim = dyInput->GetStorageShape().GetDim(d);
        OP_CHECK_IF(dim < 0, OP_LOGE(context, "dy dim %d is unknown", d), return ge::GRAPH_FAILED);
        tiling->dyDims[d] = dim;
    }

    OP_CHECK_IF(BuildMapping(context, tiling, rank32, specLen) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to build dim mapping"), return ge::GRAPH_FAILED);

    int64_t yStride = 1;
    for (int32_t d = rank32 - 1; d >= 0; --d) {
        tiling->yStrides[d] = yStride;
        yStride *= tiling->yDims[d];
    }
    int64_t dyStride = 1;
    for (int32_t d = tiling->dyRank - 1; d >= 0; --d) {
        tiling->dyStrides[d] = dyStride;
        dyStride *= tiling->dyDims[d];
    }
    tiling->totalY = rank32 == 0 ? 1 : yStride;
    tiling->totalDy = tiling->dyRank == 0 ? 1 : dyStride;

    const int64_t alignElems = BLOCK_BYTES / elemBytes;
    int64_t ubFactor = FloorAlignI((static_cast<int64_t>(ubSize) - RESERVED_UB_BYTES) /
                                       (NUM_UB_BUFFERS * elemBytes),
                                   alignElems);
    if (ubFactor < 1) {
        ubFactor = 1;
    }
    if (ubFactor > tiling->totalY) {
        ubFactor = tiling->totalY > 0 ? tiling->totalY : 1;
    }

    int64_t blockDim = 1;
    int64_t yBlockFactor = 1;
    if (tiling->totalY > 0) {
        yBlockFactor = CeilDivI(CeilDivI(tiling->totalY, coreNum), alignElems) * alignElems;
        if (yBlockFactor < 1) {
            yBlockFactor = 1;
        }
        blockDim = CeilDivI(tiling->totalY, yBlockFactor);
        if (blockDim > coreNum) {
            blockDim = coreNum;
        }
        if (blockDim < 1) {
            blockDim = 1;
        }
    }

    tiling->ubFactor = ubFactor;
    tiling->yBlockFactor = yBlockFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForStridedSliceGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct StridedSliceGradCompileInfo {};

IMPL_OP_OPTILING(StridedSliceGrad)
    .Tiling(StridedSliceGradTilingFunc)
    .TilingParse<StridedSliceGradCompileInfo>(TilingParseForStridedSliceGrad);

} // namespace optiling

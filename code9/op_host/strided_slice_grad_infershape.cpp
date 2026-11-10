/*!
 * \file strided_slice_grad_infershape.cpp
 * \brief StridedSliceGrad shape inference.
 *
 * `shape` is a runtime tensor (rank 1, length = output rank) whose values are
 * the output dims.  When the values are available (they are declared
 * ValueDepend in the op definition) the output shape is exact; otherwise the
 * rank is still known and every dim is left unknown so that tiling can fall
 * back to the same runtime values.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

// Compile-safe access to the optional input-tensor API.  When the CANN version
// does not expose GetInputTensor on the infer-shape context, the fallback keeps
// the rank known and leaves the dims unknown instead of failing to compile.
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

static ge::graphStatus InferShapeStridedSliceGrad(gert::InferShapeContext* context)
{
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* shapeInput = context->GetInputShape(0);
    if (shapeInput == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t rank = -1;
    if (shapeInput->GetDimNum() >= 1) {
        rank = shapeInput->GetDim(0);
    } else {
        rank = 0;
    }
    if (rank < 0) {
        // Unknown length of the shape tensor: cannot infer anything.
        outShape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }

    // In single-op (aclnn) mode the caller already allocated `y`, so the output
    // shape can arrive fully populated.  Prefer it: infershape then never has
    // to dereference the `shape` tensor, which is important when that input
    // lives in device memory.
    bool prefilled = outShape->GetDimNum() == static_cast<size_t>(rank);
    if (prefilled) {
        for (int64_t d = 0; d < rank; ++d) {
            if (outShape->GetDim(static_cast<size_t>(d)) < 0) {
                prefilled = false;
                break;
            }
        }
    }
    if (prefilled) {
        return ge::GRAPH_SUCCESS;
    }

    outShape->SetDimNum(static_cast<size_t>(rank));
    for (int64_t d = 0; d < rank; ++d) {
        outShape->SetDim(static_cast<size_t>(d), -1);
    }

    ge::DataType dtype = ge::DT_INT32;
    auto desc = context->GetInputDesc(0);
    if (desc != nullptr) {
        dtype = desc->GetDataType();
    }

    const gert::Tensor* shapeTensor = GetInputTensorAt(context, 0, 0);
    if (shapeTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (dtype == ge::DT_INT32) {
        const int32_t* data = shapeTensor->GetData<int32_t>();
        if (data == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        for (int64_t d = 0; d < rank; ++d) {
            outShape->SetDim(static_cast<size_t>(d), static_cast<int64_t>(data[d]));
        }
    } else if (dtype == ge::DT_INT64) {
        const int64_t* data = shapeTensor->GetData<int64_t>();
        if (data == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        for (int64_t d = 0; d < rank; ++d) {
            outShape->SetDim(static_cast<size_t>(d), data[d]);
        }
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StridedSliceGrad).InferShape(InferShapeStridedSliceGrad);

} // namespace ops
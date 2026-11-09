/*!
 * \file p_relu_grad_update_infershape.cpp
 * \brief PReluGradUpdate 算子形状/数据类型推导
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapePReluGradUpdate(gert::InferShapeContext* context)
{
    const gert::Shape* gradsShape = context->GetInputShape(0);
    const gert::Shape* featuresShape = context->GetInputShape(1);
    if (gradsShape == nullptr || featuresShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (gradsShape->GetDimNum() != featuresShape->GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < gradsShape->GetDimNum(); ++i) {
        if (gradsShape->GetDim(i) != featuresShape->GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }

    gert::Shape* dxShape = context->GetOutputShape(0);
    gert::Shape* updateShape = context->GetOutputShape(1);
    if (dxShape == nullptr || updateShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *dxShape = *gradsShape;
    *updateShape = *featuresShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePReluGradUpdate(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PReluGradUpdate)
    .InferShape(InferShapePReluGradUpdate)
    .InferDataType(InferDataTypePReluGradUpdate);

} // namespace ops

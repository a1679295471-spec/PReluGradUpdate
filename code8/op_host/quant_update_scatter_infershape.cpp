/*!
 * \file quant_update_scatter_infershape.cpp
 * \brief QuantUpdateScatter shape inference
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeQuantUpdateScatter(gert::InferShapeContext* context)
{
    const gert::Shape* varShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    if (varShape == nullptr || outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *varShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantUpdateScatter).InferShape(InferShapeQuantUpdateScatter);

} // namespace ops

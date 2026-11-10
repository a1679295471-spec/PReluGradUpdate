/*!
 * \file strided_slice_grad_def.cpp
 * \brief StridedSliceGrad operator definition.
 *
 * Dtype profiles (shape dtype x spec dtype x value dtype):
 *   {int32, int64} x {int32, int64} x {float16, float32, bfloat16}
 *
 * The shape dtype and the begin/end/strides dtype are independent (TF exposes
 * them as separate Tshape/Index attrs), so all 12 combinations are registered.
 * The kernel only depends on the spec dtype and the value dtype, which is why
 * there are still 6 tiling keys.
 *
 * shape/begin/end/strides are runtime index tensors.  `shape` is marked
 * ValueDepend because both shape inference and tiling must read its values.
 */

#include "register/op_def_registry.h"

namespace ops {
class StridedSliceGrad : public OpDef {
public:
    explicit StridedSliceGrad(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> shapeTypes = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
        const std::vector<ge::DataType> indexTypes = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
        const std::vector<ge::DataType> valueTypes = {
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        const std::vector<ge::Format> formats(12, ge::FORMAT_ND);

        this->Input("shape")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(shapeTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("begin")
            .ParamType(REQUIRED)
            .DataType(indexTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType(indexTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("strides")
            .ParamType(REQUIRED)
            .DataType(indexTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType(valueTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(valueTypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Attr("begin_mask").AttrType(OPTIONAL).Int(0);
        this->Attr("end_mask").AttrType(OPTIONAL).Int(0);
        this->Attr("ellipsis_mask").AttrType(OPTIONAL).Int(0);
        this->Attr("new_axis_mask").AttrType(OPTIONAL).Int(0);
        this->Attr("shrink_axis_mask").AttrType(OPTIONAL).Int(0);

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(StridedSliceGrad);
} // namespace ops
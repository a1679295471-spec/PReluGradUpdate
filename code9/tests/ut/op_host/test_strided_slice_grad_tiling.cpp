/*!
 * \file test_strided_slice_grad_tiling.cpp
 * \brief Host tiling UT for StridedSliceGrad.
 *
 * Covers all six dtype profiles (int32/int64 x fp16/fp32/bf16), rank 1..5,
 * negative strides, clipped begin/end, begin/end masks, ellipsis, new axis,
 * shrink axis, combined masks, empty slices and invalid inputs.
 */

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "strided_slice_grad_tiling_data.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace StridedSliceGradUT {
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "StridedSliceGrad";

static gert::StorageShape ToStorage(const std::vector<int64_t>& dims)
{
    // CANN 9.0.0 only exposes the two-initializer_list constructor for
    // StorageShape; the (Shape, Shape) overload does not exist. Dispatch on
    // rank so every supported rank still uses that exact overload.
    const int64_t* p = dims.data();
    switch (dims.size()) {
        case 0:
            return gert::StorageShape({}, {});
        case 1:
            return gert::StorageShape({p[0]}, {p[0]});
        case 2:
            return gert::StorageShape({p[0], p[1]}, {p[0], p[1]});
        case 3:
            return gert::StorageShape({p[0], p[1], p[2]}, {p[0], p[1], p[2]});
        case 4:
            return gert::StorageShape({p[0], p[1], p[2], p[3]}, {p[0], p[1], p[2], p[3]});
        case 5:
            return gert::StorageShape({p[0], p[1], p[2], p[3], p[4]}, {p[0], p[1], p[2], p[3], p[4]});
        case 6:
            return gert::StorageShape({p[0], p[1], p[2], p[3], p[4], p[5]},
                                      {p[0], p[1], p[2], p[3], p[4], p[5]});
        case 7:
            return gert::StorageShape({p[0], p[1], p[2], p[3], p[4], p[5], p[6]},
                                      {p[0], p[1], p[2], p[3], p[4], p[5], p[6]});
        case 8:
            return gert::StorageShape({p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]},
                                      {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]});
        default:
            throw std::runtime_error("StridedSliceGrad UT: unsupported rank");
    }
}

static int64_t Product(const std::vector<int64_t>& dims)
{
    int64_t value = 1;
    for (int64_t dim : dims) {
        value *= dim;
    }
    return value;
}

struct TilingCase {
    std::string name;
    std::vector<int64_t> yShape;
    std::vector<int64_t> dyShape;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    bool int64Index = false;
    DataType valueType = DT_FLOAT16;
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    bool expectSuccess = true;
    uint64_t expectKey = 0;
    std::vector<int32_t> expectMode;
    std::vector<int32_t> expectSpec;
    std::vector<int32_t> expectDyAxis;
    bool int64Shape = false;
};

static TilingContextPara MakePara(const TilingCase& c)
{
    const DataType idxType = c.int64Index ? DT_INT64 : DT_INT32;
    const DataType shapeType = c.int64Shape ? DT_INT64 : DT_INT32;
    const std::vector<int64_t> shapeLen = {static_cast<int64_t>(c.yShape.size())};
    const std::vector<int64_t> specLen = {static_cast<int64_t>(c.begin.size())};

    std::vector<TilingContextPara::TensorDescription> inputs = {
        {ToStorage(shapeLen), shapeType, FORMAT_ND},
        {ToStorage(specLen), idxType, FORMAT_ND},
        {ToStorage(specLen), idxType, FORMAT_ND},
        {ToStorage(specLen), idxType, FORMAT_ND},
        {ToStorage(c.dyShape), c.valueType, FORMAT_ND},
    };
    std::vector<TilingContextPara::TensorDescription> outputs = {
        {ToStorage(c.yShape), c.valueType, FORMAT_ND},
    };
    std::vector<TilingContextPara::OpAttr> attrs = {
        {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(c.beginMask)},
        {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(c.endMask)},
        {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(c.ellipsisMask)},
        {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(c.newAxisMask)},
        {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(c.shrinkAxisMask)},
    };
    return TilingContextPara(OP_NAME, inputs, outputs, attrs, nullptr, 64, 196608, 4096);
}

static void RunSuccess(const TilingCase& c)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakePara(c), info)) << c.name;
    EXPECT_EQ(info.tilingKey, c.expectKey) << c.name;
    EXPECT_GE(info.blockNum, 1U) << c.name;
    EXPECT_LE(info.blockNum, 64U) << c.name;
    ASSERT_GE(info.tilingDataSize, sizeof(StridedSliceGradTilingData)) << c.name;

    StridedSliceGradTilingData data;
    std::memcpy(&data, info.tilingData.get(), sizeof(data));

    EXPECT_EQ(data.rank, static_cast<int32_t>(c.yShape.size())) << c.name;
    EXPECT_EQ(data.specLen, static_cast<int32_t>(c.begin.size())) << c.name;
    EXPECT_EQ(data.dyRank, static_cast<int32_t>(c.dyShape.size())) << c.name;
    EXPECT_EQ(data.totalY, Product(c.yShape)) << c.name;
    EXPECT_EQ(data.totalDy, Product(c.dyShape)) << c.name;
    EXPECT_GE(data.ubFactor, 1) << c.name;
    EXPECT_GE(data.yBlockFactor, 1) << c.name;
    for (int32_t d = 0; d < data.rank; ++d) {
        EXPECT_EQ(data.yDims[d], c.yShape[static_cast<size_t>(d)]) << c.name;
    }
    for (int32_t d = 0; d < data.dyRank; ++d) {
        EXPECT_EQ(data.dyDims[d], c.dyShape[static_cast<size_t>(d)]) << c.name;
    }
    for (size_t i = 0; i < info.workspaceSizes.size(); ++i) {
        EXPECT_EQ(info.workspaceSizes[i], 0U) << c.name;
    }

    if (!c.expectMode.empty()) {
        ASSERT_EQ(c.expectMode.size(), static_cast<size_t>(data.rank)) << c.name;
        for (int32_t d = 0; d < data.rank; ++d) {
            EXPECT_EQ(data.inputMode[d], c.expectMode[static_cast<size_t>(d)]) << c.name << " dim " << d;
        }
    }
    if (!c.expectSpec.empty()) {
        ASSERT_EQ(c.expectSpec.size(), static_cast<size_t>(data.rank)) << c.name;
        for (int32_t d = 0; d < data.rank; ++d) {
            EXPECT_EQ(data.inputSpecIdx[d], c.expectSpec[static_cast<size_t>(d)])
                << c.name << " dim " << d;
        }
    }
    if (!c.expectDyAxis.empty()) {
        ASSERT_EQ(c.expectDyAxis.size(), static_cast<size_t>(data.rank)) << c.name;
        for (int32_t d = 0; d < data.rank; ++d) {
            EXPECT_EQ(data.inputDyAxis[d], c.expectDyAxis[static_cast<size_t>(d)])
                << c.name << " dim " << d;
        }
    }
}

static void RunFailure(const TilingCase& c)
{
    ExecuteTestCase(MakePara(c), GRAPH_FAILED, 0UL, EMPTY_EXPECT_TILING_DATA, {});
}

TEST(StridedSliceGradTilingTest, i32_f16_basic)
{
    RunSuccess({"i32_f16_basic", {2, 4, 5, 6}, {2, 3, 5, 6}, {0, 1, 0, 0}, {2, 4, 5, 6},
                {1, 1, 1, 1}, false, DT_FLOAT16, 0, 0, 0, 0, 0, true, 0,
                {0, 0, 0, 0}, {0, 1, 2, 3}, {0, 1, 2, 3}});
}

TEST(StridedSliceGradTilingTest, i32_f32_partial_spec_rank2)
{
    RunSuccess({"i32_f32_partial_spec_rank2", {8, 13}, {3, 13}, {2}, {7}, {2}, false, DT_FLOAT,
                0, 0, 0, 0, 0, true, 1,
                {0, 0}, {0, -1}, {0, 1}});
}

TEST(StridedSliceGradTilingTest, i32_bf16_negative_strides)
{
    RunSuccess({"i32_bf16_negative_strides", {3, 4, 5}, {3, 2, 5}, {2, -1, 4}, {-4, -5, -6},
                {-1, -2, -1}, false, DT_BF16, 0, 0, 0, 0, 0, true, 2});
}

TEST(StridedSliceGradTilingTest, i32_f16_ellipsis_middle)
{
    RunSuccess({"i32_f16_ellipsis_middle", {2, 3, 4, 5}, {2, 3, 4, 4}, {0, 1, 0}, {2, 3, 4},
                {1, 1, 1}, false, DT_FLOAT16, 0, 0, 0x2, 0, 0, true, 0,
                {0, 0, 0, 0}, {0, -1, -1, 2}, {0, 1, 2, 3}});
}

TEST(StridedSliceGradTilingTest, i32_f32_new_axis_middle)
{
    RunSuccess({"i32_f32_new_axis_middle", {3, 4}, {2, 1, 2}, {1, 0, 0}, {3, 1, 2},
                {1, 1, 1}, false, DT_FLOAT, 0, 0, 0, 0x2, 0, true, 1,
                {0, 0}, {0, 2}, {0, 2}});
}

TEST(StridedSliceGradTilingTest, i32_bf16_shrink_middle)
{
    RunSuccess({"i32_bf16_shrink_middle", {4, 5, 6}, {2, 2}, {0, 2, 0}, {4, 3, 6},
                {2, 1, 3}, false, DT_BF16, 0, 0, 0, 0, 0x2, true, 2,
                {0, 1, 0}, {0, 1, 2}, {0, -1, 1}});
}

TEST(StridedSliceGradTilingTest, i64_f16_basic)
{
    RunSuccess({"i64_f16_basic", {5, 6}, {5, 3}, {0, 1}, {5, 6}, {1, 2}, true, DT_FLOAT16,
                0, 0, 0, 0, 0, true, 3});
}

TEST(StridedSliceGradTilingTest, i64_f32_rank5)
{
    RunSuccess({"i64_f32_rank5", {2, 2, 2, 2, 3}, {2, 2, 2, 2, 3}, {0, 0, 0, 0, 0},
                {2, 2, 2, 2, 3}, {1, 1, 1, 1, 1}, true, DT_FLOAT, 0, 0, 0, 0, 0, true, 4});
}

TEST(StridedSliceGradTilingTest, i64_bf16_begin_end_mask)
{
    RunSuccess({"i64_bf16_begin_end_mask", {3, 4, 5}, {2, 3, 2}, {1, 1, 1}, {2, 2, 2},
                {1, 1, 1}, true, DT_BF16, 0x5, 0x2, 0, 0, 0, true, 5,
                {0, 0, 0}, {0, 1, 2}, {0, 1, 2}});
}

TEST(StridedSliceGradTilingTest, i32_f16_clipped_bounds)
{
    RunSuccess({"i32_f16_clipped_bounds", {3, 4}, {3, 3}, {-10, 1}, {100, 4}, {1, 1},
                false, DT_FLOAT16, 0, 0, 0, 0, 0, true, 0});
}

TEST(StridedSliceGradTilingTest, i32_f32_ellipsis_first)
{
    RunSuccess({"i32_f32_ellipsis_first", {2, 3, 4}, {2, 3, 2}, {0, 1}, {0, 3}, {1, 1},
                false, DT_FLOAT, 0, 0, 0x1, 0, 0, true, 1,
                {0, 0, 0}, {-1, -1, 1}, {0, 1, 2}});
}

TEST(StridedSliceGradTilingTest, i32_bf16_new_axis_first)
{
    RunSuccess({"i32_bf16_new_axis_first", {4, 5}, {1, 2, 5}, {0, 1, 0}, {0, 3, 5},
                {1, 1, 1}, false, DT_BF16, 0, 0, 0, 0x1, 0, true, 2,
                {0, 0}, {1, 2}, {1, 2}});
}

TEST(StridedSliceGradTilingTest, mixed_shape_i64_spec_i32)
{
    RunSuccess({"mixed_shape_i64_spec_i32", {5, 6}, {5, 3}, {0, 1}, {5, 6}, {1, 2}, false,
                DT_FLOAT, 0, 0, 0, 0, 0, true, 1, {}, {}, {}, true});
}

TEST(StridedSliceGradTilingTest, i32_f16_empty_slice)
{
    RunSuccess({"i32_f16_empty_slice", {4, 5}, {0, 5}, {2}, {2}, {1}, false, DT_FLOAT16,
                0, 0, 0, 0, 0, true, 0});
}

TEST(StridedSliceGradTilingTest, i32_f32_rank1)
{
    RunSuccess({"i32_f32_rank1", {10}, {3}, {1}, {9}, {3}, false, DT_FLOAT, 0, 0, 0, 0, 0,
                true, 1});
}

TEST(StridedSliceGradTilingTest, i64_f16_new_axis_and_shrink)
{
    RunSuccess({"i64_f16_new_axis_and_shrink", {3, 4}, {2, 1}, {1, 0, 2}, {3, 1, 3},
                {1, 1, 1}, true, DT_FLOAT16, 0, 0, 0, 0x2, 0x4, true, 3,
                {0, 1}, {0, 2}, {0, -1}});
}

TEST(StridedSliceGradTilingTest, i64_f32_negative_strides_rank2)
{
    RunSuccess({"i64_f32_negative_strides_rank2", {2, 6}, {2, 3}, {1, 5}, {-3, -7},
                {-1, -2}, true, DT_FLOAT, 0, 0, 0, 0, 0, true, 4});
}

TEST(StridedSliceGradTilingTest, i64_bf16_ellipsis_and_shrink)
{
    RunSuccess({"i64_bf16_ellipsis_and_shrink", {2, 3, 4, 5}, {1, 3, 4}, {0, 0, 0, 1},
                {2, 0, 4, 2}, {2, 1, 1, 1}, true, DT_BF16, 0, 0, 0x2, 0, 0x8, true, 5,
                {0, 0, 0, 1}, {0, -1, 2, 3}, {0, 1, 2, -1}});
}

TEST(StridedSliceGradTilingTest, i32_f16_shrink_first_trailing)
{
    RunSuccess({"i32_f16_shrink_first_trailing", {3, 4, 5}, {4, 5}, {1}, {2}, {1},
                false, DT_FLOAT16, 0, 0, 0, 0, 0x1, true, 0,
                {1, 0, 0}, {0, -1, -1}, {-1, 0, 1}});
}

TEST(StridedSliceGradTilingTest, i32_f32_begin_mask_shrink_combo)
{
    RunSuccess({"i32_f32_begin_mask_shrink_combo", {2, 3, 4}, {2, 4}, {0, 1, 0}, {2, 2, 0},
                {1, 1, 1}, false, DT_FLOAT, 0x1, 0x4, 0, 0, 0x2, true, 1,
                {0, 1, 0}, {0, 1, 2}, {0, -1, 1}});
}

TEST(StridedSliceGradTilingTest, invalid_dy_rank)
{
    RunFailure({"invalid_dy_rank", {3, 4}, {3}, {0, 0}, {3, 4}, {1, 1}, false, DT_FLOAT16,
                0, 0, 0, 0, 0, false});
}

TEST(StridedSliceGradTilingTest, invalid_mask_out_of_range)
{
    RunFailure({"invalid_mask_out_of_range", {3, 4}, {3, 4}, {0, 0}, {3, 4}, {1, 1},
                false, DT_FLOAT16, 0x20, 0, 0, 0, 0, false});
}

TEST(StridedSliceGradTilingTest, invalid_mask_overlap)
{
    RunFailure({"invalid_mask_overlap", {3, 4}, {3}, {0, 0}, {3, 4}, {1, 1},
                false, DT_FLOAT16, 0, 0, 0, 0x1, 0x1, false});
}

TEST(StridedSliceGradTilingTest, invalid_spec_longer_than_rank)
{
    RunFailure({"invalid_spec_longer_than_rank", {3}, {3}, {0, 0}, {3, 4}, {1, 1},
                false, DT_FLOAT16, 0, 0, 0, 0, 0, false});
}

} // namespace StridedSliceGradUT
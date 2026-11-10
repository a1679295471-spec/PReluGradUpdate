#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "quant_update_scatter_tiling_data.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace QuantUpdateScatterUT {
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "QuantUpdateScatter";

enum class Profile { FP16 = 0, BF16 = 1 };

struct TilingCase {
    std::string name;
    Profile profile;
    std::vector<int64_t> varShape;
    std::vector<int64_t> indicesShape;
    std::vector<int64_t> updatesShape;
    std::vector<int64_t> scalesShape;
    std::vector<int64_t> zpShape; // empty: optional input omitted
    int64_t axis = -2;
    int64_t quantAxis = -1;
    bool reciprocal = false;
    bool expectSuccess = true;
    uint64_t expectKey = 0;
    int64_t expectVarDim0 = 0;
    int64_t expectVarAxis = 0;
    int64_t expectUpdAxis = 0;
    int64_t expectOuter = 0;
    int64_t expectRowLen = 0;
    int64_t expectIndicesNum = 0;
    int64_t expectRank2 = 0;
};

struct QuantUpdateScatterCompileInfo {};
static QuantUpdateScatterCompileInfo compileInfo;

static TilingContextPara MakePara(const TilingCase& testCase)
{
    const bool bf16 = testCase.profile == Profile::BF16;
    const DataType idxType = bf16 ? DT_INT64 : DT_INT32;
    const DataType updType = bf16 ? DT_BF16 : DT_FLOAT16;
    const DataType scaleType = bf16 ? DT_BF16 : DT_FLOAT;
    const DataType zpType = bf16 ? DT_BF16 : DT_INT32;

    StorageShape varShape(testCase.varShape, testCase.varShape);
    StorageShape idxShape(testCase.indicesShape, testCase.indicesShape);
    StorageShape updShape(testCase.updatesShape, testCase.updatesShape);
    StorageShape scaleShape(testCase.scalesShape, testCase.scalesShape);
    std::vector<TilingContextPara::TensorDescription> inputs = {
        {varShape, DT_INT8, FORMAT_ND},
        {idxShape, idxType, FORMAT_ND},
        {updShape, updType, FORMAT_ND},
        {scaleShape, scaleType, FORMAT_ND},
    };
    if (!testCase.zpShape.empty()) {
        StorageShape zpShape(testCase.zpShape, testCase.zpShape);
        inputs.emplace_back(zpShape, zpType, FORMAT_ND);
    }
    std::vector<TilingContextPara::TensorDescription> outputs = {
        {varShape, DT_INT8, FORMAT_ND},
    };
    std::vector<TilingContextPara::OpAttr> attrs = {
        {"reduce", Ops::Math::AnyValue::CreateFrom<std::string>("update")},
        {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(testCase.axis)},
        {"quant_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(testCase.quantAxis)},
        {"reciprocal_scale", Ops::Math::AnyValue::CreateFrom<bool>(testCase.reciprocal)},
    };
    return TilingContextPara(OP_NAME, inputs, outputs, attrs, &compileInfo, 64, 196608, 4096);
}

static void RunSuccessCase(const TilingCase& testCase)
{
    TilingContextPara para = MakePara(testCase);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << testCase.name;
    EXPECT_EQ(info.tilingKey, testCase.expectKey) << testCase.name;
    EXPECT_GE(info.blockNum, 1U) << testCase.name;
    EXPECT_LE(info.blockNum, 64U) << testCase.name;
    ASSERT_GE(info.tilingDataSize, sizeof(QuantUpdateScatterTilingData)) << testCase.name;

    QuantUpdateScatterTilingData data;
    std::memcpy(&data, info.tilingData.get(), sizeof(data));
    EXPECT_EQ(data.varDim0, testCase.expectVarDim0) << testCase.name;
    EXPECT_EQ(data.varAxis, testCase.expectVarAxis) << testCase.name;
    EXPECT_EQ(data.updAxis, testCase.expectUpdAxis) << testCase.name;
    EXPECT_EQ(data.outer, testCase.expectOuter) << testCase.name;
    EXPECT_EQ(data.rowLen, testCase.expectRowLen) << testCase.name;
    EXPECT_EQ(data.indicesNum, testCase.expectIndicesNum) << testCase.name;
    EXPECT_EQ(data.rank2, testCase.expectRank2) << testCase.name;
    EXPECT_GE(data.colTile, 1) << testCase.name;
    if (data.rowLen > 0) {
        EXPECT_LE(data.colTile, data.rowLen) << testCase.name;
    }
    if (data.varDim0 > 0) {
        EXPECT_GE(data.pPerCore, 1) << testCase.name;
    }
    for (size_t size : info.workspaceSizes) {
        EXPECT_EQ(size, 0U) << testCase.name;
    }
}

static void RunFailureCase(const TilingCase& testCase)
{
    TilingContextPara para = MakePara(testCase);
    ExecuteTestCase(para, GRAPH_FAILED, 0UL, EMPTY_EXPECT_TILING_DATA, {});
}

TEST(QuantUpdateScatterTilingTest, fp16_sample)
{
    RunSuccessCase({"fp16_sample", Profile::FP16, {1, 2, 32}, {1}, {1, 1, 32},
                    {1, 1, 32}, {1, 1, 32}, -2, -1, false, true, 0, 1, 2, 1, 1, 32, 1, 0});
}

TEST(QuantUpdateScatterTilingTest, fp16_4d_outer)
{
    RunSuccessCase({"fp16_4d_outer", Profile::FP16, {2, 3, 4, 7}, {2}, {2, 3, 2, 7},
                    {7}, {7}, -2, -1, false, true, 0, 2, 4, 2, 3, 7, 2, 0});
}

TEST(QuantUpdateScatterTilingTest, fp16_5d_outer)
{
    RunSuccessCase({"fp16_5d_outer", Profile::FP16, {2, 3, 2, 4, 9}, {2},
                    {2, 3, 2, 2, 9}, {9}, {9}, -2, -1, false, true, 0,
                    2, 4, 2, 6, 9, 2, 0});
}

TEST(QuantUpdateScatterTilingTest, fp16_column_one)
{
    RunSuccessCase({"fp16_column_one", Profile::FP16, {3, 5, 1}, {3}, {3, 2, 1},
                    {1}, {1}, -2, -1, true, true, 0, 3, 5, 2, 1, 1, 3, 0});
}

TEST(QuantUpdateScatterTilingTest, fp16_rank2_axis0)
{
    RunSuccessCase({"fp16_rank2_axis0", Profile::FP16, {8, 13}, {4}, {4, 13},
                    {13}, {13}, -2, -1, false, true, 0, 8, 8, 4, 1, 13, 4, 1});
}

TEST(QuantUpdateScatterTilingTest, fp16_without_zero_points)
{
    RunSuccessCase({"fp16_without_zero_points", Profile::FP16, {2, 4, 16}, {2}, {2, 1, 16},
                    {16}, {}, -2, -1, false, true, 0, 2, 4, 1, 1, 16, 2, 0});
}

TEST(QuantUpdateScatterTilingTest, fp16_empty_updates)
{
    RunSuccessCase({"fp16_empty_updates", Profile::FP16, {2, 3, 16}, {0}, {0, 1, 16},
                    {16}, {}, -2, -1, false, true, 0, 2, 3, 1, 1, 16, 0, 0});
}

TEST(QuantUpdateScatterTilingTest, fp16_partial_dim0)
{
    RunSuccessCase({"fp16_partial_dim0", Profile::FP16, {3, 4, 16}, {1}, {1, 1, 16},
                    {16}, {}, -2, -1, false, true, 0, 3, 4, 1, 1, 16, 1, 0});
}

TEST(QuantUpdateScatterTilingTest, fp16_wide_row)
{
    RunSuccessCase({"fp16_wide_row", Profile::FP16, {1, 3, 9000}, {1}, {1, 1, 9000},
                    {9000}, {9000}, -2, -1, false, true, 0, 1, 3, 1, 1, 9000, 1, 0});
}

TEST(QuantUpdateScatterTilingTest, bf16_sample)
{
    RunSuccessCase({"bf16_sample", Profile::BF16, {2, 3, 32}, {2}, {2, 1, 32},
                    {32}, {32}, -2, -1, false, true, 1, 2, 3, 1, 1, 32, 2, 0});
}

TEST(QuantUpdateScatterTilingTest, bf16_4d_outer)
{
    RunSuccessCase({"bf16_4d_outer", Profile::BF16, {2, 3, 4, 6}, {2}, {2, 3, 2, 6},
                    {6}, {6}, -2, -1, true, true, 1, 2, 4, 2, 3, 6, 2, 0});
}

TEST(QuantUpdateScatterTilingTest, bf16_rank2_axis0)
{
    RunSuccessCase({"bf16_rank2_axis0", Profile::BF16, {6, 9}, {2}, {3, 9},
                    {9}, {9}, -2, -1, false, true, 1, 6, 6, 3, 1, 9, 2, 1});
}

TEST(QuantUpdateScatterTilingTest, invalid_axis)
{
    RunFailureCase({"invalid_axis", Profile::FP16, {1, 2, 32}, {1}, {1, 1, 32},
                    {32}, {32}, -1, -1, false, false});
}

TEST(QuantUpdateScatterTilingTest, invalid_quant_axis)
{
    RunFailureCase({"invalid_quant_axis", Profile::FP16, {1, 2, 32}, {1}, {1, 1, 32},
                    {32}, {32}, -2, 1, false, false});
}

TEST(QuantUpdateScatterTilingTest, invalid_indices_length)
{
    RunFailureCase({"invalid_indices_length", Profile::FP16, {2, 3, 16}, {1}, {2, 1, 16},
                    {16}, {16}, -2, -1, false, false});
}

TEST(QuantUpdateScatterTilingTest, invalid_scale_count)
{
    RunFailureCase({"invalid_scale_count", Profile::FP16, {1, 2, 32}, {1}, {1, 1, 32},
                    {16}, {32}, -2, -1, false, false});
}

TEST(QuantUpdateScatterTilingTest, invalid_non_axis_dim)
{
    RunFailureCase({"invalid_non_axis_dim", Profile::FP16, {2, 3, 4, 7}, {2}, {2, 2, 2, 7},
                    {7}, {7}, -2, -1, false, false});
}

TEST(QuantUpdateScatterTilingTest, invalid_rank)
{
    RunFailureCase({"invalid_rank", Profile::FP16, {32}, {1}, {32},
                    {32}, {32}, -2, -1, false, false});
}

} // namespace QuantUpdateScatterUT

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "p_relu_grad_update_tiling_data.h"

namespace PReluGradUpdateUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "PReluGradUpdate";

struct PReluGradUpdateTestParam {
    std::string caseName;
    std::initializer_list<int64_t> gradsShape;
    ge::DataType gradsDtype;
    ge::Format gradsFormat;
    std::initializer_list<int64_t> featuresShape;
    ge::DataType featuresDtype;
    ge::Format featuresFormat;
    std::initializer_list<int64_t> weightsShape;
    ge::DataType weightsDtype;
    ge::Format weightsFormat;
    std::initializer_list<int64_t> dxShape;
    ge::DataType dxDtype;
    ge::Format dxFormat;
    std::initializer_list<int64_t> updateShape;
    ge::DataType updateDtype;
    ge::Format updateFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};


// NOTE: 以下期望值由当前 tiling 实现推导：
//   expectTilingKey: 0=fp16, 1=fp32, 2=bf16
//   expectTilingData: totalNum blockFactor ubFactor innerSize channelNum patternSize usePattern
//   expectWorkspaces: 本算子不使用 workspace
static PReluGradUpdateTestParam testCases[] = {
    // pattern mode, 3-D, fp32
    {"p_relu_grad_update_f32_3d", {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "192 48 4848 8 6 48 1 ", {0}, 64, 262144, 4096},
    // pattern mode, same shape, fp16 / bf16
    {"p_relu_grad_update_f16_3d", {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {6}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "192 48 7008 8 6 48 1 ", {0}, 64, 262144, 4096},
    {"p_relu_grad_update_bf16_3d", {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND, {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND, {6}, ge::DT_BF16, ge::FORMAT_ND, {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND, {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 2UL, "192 48 7008 8 6 48 1 ", {0}, 64, 262144, 4096},
    // 2-D [N, C]: innerSize == 1, the periodic path must still work
    {"p_relu_grad_update_f32_2d", {2, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {3}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 3}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "6 3 4872 1 3 24 1 ", {0}, 64, 262144, 4096},
    // 4-D with a small channel plane, pattern mode
    {"p_relu_grad_update_f32_4d", {2, 3, 4, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 3, 4, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {3}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 3, 4, 5}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 3, 4, 5}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "120 60 4800 20 3 120 1 ", {0}, 64, 262144, 4096},
    // 4-D large channel plane, channel mode
    {"p_relu_grad_update_f32_channel", {1, 8, 64, 64}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 8, 64, 64}, ge::DT_FLOAT, ge::FORMAT_ND, {8}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 8, 64, 64}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 8, 64, 64}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "32768 512 4096 4096 8 0 0 ", {0}, 64, 262144, 4096},
    {"p_relu_grad_update_f32_channel2", {1, 16, 56, 56}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 16, 56, 56}, ge::DT_FLOAT, ge::FORMAT_ND, {16}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 16, 56, 56}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 16, 56, 56}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "50176 784 3136 3136 16 0 0 ", {0}, 64, 262144, 4096},
    {"p_relu_grad_update_f16_channel", {1, 8, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 8, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {8}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 8, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 8, 64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "32768 512 4096 4096 8 0 0 ", {0}, 64, 262144, 4096},
    // 1-element edge case
    {"p_relu_grad_update_f32_single", {1, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "1 1 4880 1 1 8 1 ", {0}, 64, 262144, 4096},
    // fp16 odd inner size
    {"p_relu_grad_update_f16_odd", {2, 3, 5, 7}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 5, 7}, ge::DT_FLOAT16, ge::FORMAT_ND, {3}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 5, 7}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3, 5, 7}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "210 105 6720 35 3 1680 1 ", {0}, 64, 262144, 4096},
    // bf16 2-D
    {"p_relu_grad_update_bf16_2d", {7, 5}, ge::DT_BF16, ge::FORMAT_ND, {7, 5}, ge::DT_BF16, ge::FORMAT_ND, {5}, ge::DT_BF16, ge::FORMAT_ND, {7, 5}, ge::DT_BF16, ge::FORMAT_ND, {7, 5}, ge::DT_BF16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 2UL, "35 5 7040 1 5 80 1 ", {0}, 64, 262144, 4096},
    // rank-3 [N, C, 1]
    {"p_relu_grad_update_f16_rank3", {3, 5, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {3, 5, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {5}, ge::DT_FLOAT16, ge::FORMAT_ND, {3, 5, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {3, 5, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "15 5 7040 1 5 80 1 ", {0}, 64, 262144, 4096},
};

class PReluGradUpdateTilingTest : public testing::TestWithParam<PReluGradUpdateTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "PReluGradUpdateTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "PReluGradUpdateTilingTest TearDown." << std::endl;
    }
};

struct PReluGradUpdateCompileInfo {} compileInfo;

static void TestOneParamCase(const PReluGradUpdateTestParam &param)
{
    gert::StorageShape gradsShape = {param.gradsShape, param.gradsShape};
    gert::StorageShape featuresShape = {param.featuresShape, param.featuresShape};
    gert::StorageShape weightsShape = {param.weightsShape, param.weightsShape};
    gert::StorageShape dxShape = {param.dxShape, param.dxShape};
    gert::StorageShape updateShape = {param.updateShape, param.updateShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{gradsShape, param.gradsDtype, param.gradsFormat},
        {featuresShape, param.featuresDtype, param.featuresFormat},
        {weightsShape, param.weightsDtype, param.weightsFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{dxShape, param.dxDtype, param.dxFormat},
        {updateShape, param.updateDtype, param.updateFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(PReluGradUpdateTilingTest, tiling_test)
{
    const PReluGradUpdateTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    PReluGradUpdateTilingTests,
    PReluGradUpdateTilingTest,
    testing::ValuesIn(testCases));

}


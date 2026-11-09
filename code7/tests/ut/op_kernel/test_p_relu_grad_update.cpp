/*!
 * \file test_p_relu_grad_update.cpp
 * \brief PReluGradUpdate 算子 kernel UT（CPU 仿真）
 *
 * 覆盖 fp16 / fp32 / bf16、2-D~4-D、inner=1、奇数 inner、period 模式与
 * channel 模式、单元素边界；输入按类型量化后再计算 golden，比较前按输出
 * 类型做相同的 RNE 量化，避免“二进制比较”带来的假失败。
 */

#include "p_relu_grad_update_tiling.h"
#include "../../../op_kernel/p_relu_grad_update.cpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;
using namespace AscendC;

namespace {

enum class DType { F16 = 0, F32 = 1, BF16 = 2 };

struct KernelCase {
    std::vector<int64_t> shape;
    DType dtype;
};

struct TilingInfo {
    PReluGradUpdateTilingData data;
    int64_t blockDim;
};

static int64_t GcdI(int64_t a, int64_t b)
{
    while (b != 0) {
        const int64_t r = a % b;
        a = b;
        b = r;
    }
    return a == 0 ? 1 : a;
}
static int64_t CeilDivI(int64_t a, int64_t b) { return (a + b - 1) / b; }
static int64_t CeilAlignI(int64_t a, int64_t b) { return CeilDivI(a, b) * b; }
static int64_t FloorAlignI(int64_t a, int64_t b) { return a / b * b; }

static TilingInfo MakeTiling(int64_t total, int64_t channelNum, int64_t innerSize, int64_t typeSize)
{
    constexpr int64_t RESERVED_UB_BYTES = 8192;
    constexpr int64_t FLOAT_TEMP_BYTES = 20;
    constexpr int64_t DOUBLE_BUFFER_FACTOR = 8;
    constexpr int64_t UB_SIZE = 262144;
    constexpr int64_t CORE_NUM = 64;

    const int64_t alignElems = 32 / typeSize;
    const int64_t bytesPerElem = DOUBLE_BUFFER_FACTOR * typeSize + FLOAT_TEMP_BYTES;
    int64_t maxTileBase = FloorAlignI((UB_SIZE - RESERVED_UB_BYTES) / bytesPerElem, alignElems);
    if (maxTileBase < alignElems) {
        maxTileBase = alignElems;
    }

    const int64_t period = channelNum * innerSize;
    const int64_t patternSize = period / GcdI(period, alignElems) * alignElems;
    const int64_t patternBytes = patternSize * static_cast<int64_t>(sizeof(float));
    bool usePattern = false;
    int64_t maxTilePattern = 0;
    if (patternBytes + RESERVED_UB_BYTES < UB_SIZE) {
        maxTilePattern = FloorAlignI((UB_SIZE - RESERVED_UB_BYTES - patternBytes) / bytesPerElem, alignElems);
        usePattern = maxTilePattern >= patternSize;
    }

    TilingInfo info{};
    if (usePattern) {
        info.data.ubFactor = FloorAlignI(maxTilePattern, patternSize);
    } else {
        info.data.ubFactor = std::min(maxTileBase, innerSize);
        if (info.data.ubFactor < 1) {
            info.data.ubFactor = 1;
        }
    }

    info.blockDim = 1;
    if (total > 0) {
        const int64_t perCore = CeilDivI(total, CORE_NUM);
        info.data.blockFactor = usePattern ? CeilAlignI(perCore, period) : perCore;
        info.blockDim = CeilDivI(total, info.data.blockFactor);
        info.blockDim = std::max<int64_t>(1, std::min<int64_t>(info.blockDim, CORE_NUM));
    }
    info.data.totalNum = total;
    info.data.innerSize = innerSize;
    info.data.channelNum = channelNum;
    info.data.patternSize = usePattern ? patternSize : 0;
    info.data.usePattern = usePattern ? 1 : 0;
    return info;
}

static uint16_t FloatToHalfBits(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(float));
    const uint32_t sign = (x >> 16) & 0x8000U;
    const uint32_t abs = x & 0x7fffffffU;
    if (abs >= 0x7f800000U) {
        return static_cast<uint16_t>(sign | (abs > 0x7f800000U ? 0x7e00U : 0x7c00U));
    }
    int32_t e = static_cast<int32_t>((x >> 23) & 0xffU) - 127 + 15;
    uint32_t mant = x & 0x7fffffU;
    if (e <= 0) {
        if (e < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000U;
        const int32_t shift = 14 - e;
        uint32_t m = mant >> shift;
        const uint32_t rem = mant & ((1U << shift) - 1U);
        const uint32_t half = 1U << (shift - 1);
        if (rem > half || (rem == half && (m & 1U) != 0U)) {
            ++m;
        }
        return static_cast<uint16_t>(sign | m);
    }
    if (e >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    uint32_t m = mant >> 13;
    const uint32_t rem = mant & 0x1fffU;
    if (rem > 0x1000U || (rem == 0x1000U && (m & 1U) != 0U)) {
        ++m;
        if (m == 0x400U) {
            m = 0;
            ++e;
            if (e >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00U);
            }
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | m);
}

static float HalfBitsToFloat(uint16_t h)
{
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000U)) << 16;
    const uint32_t exp = (h >> 10) & 0x1fU;
    const uint32_t mant = h & 0x3ffU;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int32_t e = -14;
            uint32_t m = mant;
            while ((m & 0x400U) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffU;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 0x1fU) {
        bits = sign | 0x7f800000U | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

static uint16_t FloatToBF16Bits(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(float));
    const uint32_t lsb = (x >> 16) & 1U;
    x += 0x7fffU + lsb;
    return static_cast<uint16_t>(x >> 16);
}

static float BF16BitsToFloat(uint16_t b)
{
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

static float LoadFloat(const uint8_t* base, int64_t index, DType dtype)
{
    const uint8_t* p = base + index * (dtype == DType::F32 ? 4 : 2);
    if (dtype == DType::F32) {
        float f;
        memcpy(&f, p, sizeof(float));
        return f;
    }
    uint16_t bits;
    memcpy(&bits, p, sizeof(uint16_t));
    return dtype == DType::F16 ? HalfBitsToFloat(bits) : BF16BitsToFloat(bits);
}

static void StoreFloat(uint8_t* base, int64_t index, float value, DType dtype)
{
    uint8_t* p = base + index * (dtype == DType::F32 ? 4 : 2);
    if (dtype == DType::F32) {
        memcpy(p, &value, sizeof(float));
        return;
    }
    const uint16_t bits = dtype == DType::F16 ? FloatToHalfBits(value) : FloatToBF16Bits(value);
    memcpy(p, &bits, sizeof(uint16_t));
}

static float QuantizeToType(float value, DType dtype)
{
    uint8_t buffer[4] = {0, 0, 0, 0};
    StoreFloat(buffer, 0, value, dtype);
    return LoadFloat(buffer, 0, dtype);
}

template <typename T, uint32_t Key>
static bool RunTyped(const std::vector<int64_t>& shape, DType dtype)
{
    int64_t total = 1;
    for (auto d : shape) {
        total *= d;
    }
    const int64_t channelNum = shape[1];
    int64_t innerSize = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        innerSize *= shape[i];
    }
    const int64_t typeSize = static_cast<int64_t>(sizeof(T));
    const TilingInfo tiling = MakeTiling(total, channelNum, innerSize, typeSize);

    std::vector<uint8_t> grads(total * typeSize);
    std::vector<uint8_t> features(total * typeSize);
    std::vector<uint8_t> weights(channelNum * typeSize);
    std::vector<uint8_t> dx(total * typeSize);
    std::vector<uint8_t> update(total * typeSize);
    std::vector<float> gradsValue(total);
    std::vector<float> featuresValue(total);
    std::vector<float> weightValue(channelNum);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> wdist(0.01f, 0.5f);
    for (int64_t i = 0; i < total; ++i) {
        const float g = dist(rng);
        const float f = dist(rng);
        StoreFloat(grads.data(), i, g, dtype);
        StoreFloat(features.data(), i, f, dtype);
        gradsValue[i] = LoadFloat(grads.data(), i, dtype);
        featuresValue[i] = LoadFloat(features.data(), i, dtype);
    }
    for (int64_t i = 0; i < channelNum; ++i) {
        StoreFloat(weights.data(), i, wdist(rng), dtype);
        weightValue[i] = LoadFloat(weights.data(), i, dtype);
    }

    auto* gradsGm = static_cast<uint8_t*>(AscendC::GmAlloc(grads.size()));
    auto* featuresGm = static_cast<uint8_t*>(AscendC::GmAlloc(features.size()));
    auto* weightsGm = static_cast<uint8_t*>(AscendC::GmAlloc(weights.size()));
    auto* dxGm = static_cast<uint8_t*>(AscendC::GmAlloc(dx.size()));
    auto* updateGm = static_cast<uint8_t*>(AscendC::GmAlloc(update.size()));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tilingGm = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(PReluGradUpdateTilingData)));
    memcpy(gradsGm, grads.data(), grads.size());
    memcpy(featuresGm, features.data(), features.size());
    memcpy(weightsGm, weights.data(), weights.size());
    memcpy(tilingGm, &tiling.data, sizeof(PReluGradUpdateTilingData));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
#define RUN_PRELUGRADUPDATE_KF(KEY)                                                                 \
    ICPU_SET_TILING_KEY(KEY);                                                                       \
    ICPU_RUN_KF((p_relu_grad_update<KEY>), static_cast<uint32_t>(tiling.blockDim), gradsGm,          \
                featuresGm, weightsGm, dxGm, updateGm, workspace, tilingGm)
    if constexpr (Key == 0) {
        RUN_PRELUGRADUPDATE_KF(0);
    } else if constexpr (Key == 1) {
        RUN_PRELUGRADUPDATE_KF(1);
    } else {
        RUN_PRELUGRADUPDATE_KF(2);
    }
#undef RUN_PRELUGRADUPDATE_KF

    int64_t badDx = 0;
    int64_t badUpdate = 0;
    const double tolerance = (dtype == DType::F32) ? 1e-4 : ((dtype == DType::F16) ? 1e-3 : 4e-3);
    for (int64_t i = 0; i < total; ++i) {
        const int64_t channel = (i / innerSize) % channelNum;
        const float f = featuresValue[i];
        const float g = gradsValue[i];
        const float w = weightValue[channel];
        const float expectDx = f > 0.0f ? g : g * w;
        const float expectUpdate = f > 0.0f ? 0.0f : f * g;
        const float expectDxT = QuantizeToType(expectDx, dtype);
        const float expectUpdateT = QuantizeToType(expectUpdate, dtype);
        const float gotDx = LoadFloat(dxGm, i, dtype);
        const float gotUpdate = LoadFloat(updateGm, i, dtype);
        const double boundDx = tolerance * (1.0 + std::fabs(static_cast<double>(expectDxT)));
        const double boundUpdate = tolerance * (1.0 + std::fabs(static_cast<double>(expectUpdateT)));
        if (std::fabs(static_cast<double>(gotDx) - expectDxT) > boundDx) {
            if (badDx < 5) {
                std::cerr << "dx mismatch idx=" << i << " got=" << gotDx << " expect=" << expectDxT
                          << " f=" << f << " g=" << g << " w=" << w << std::endl;
            }
            ++badDx;
        }
        if (std::fabs(static_cast<double>(gotUpdate) - expectUpdateT) > boundUpdate) {
            if (badUpdate < 5) {
                std::cerr << "update mismatch idx=" << i << " got=" << gotUpdate
                          << " expect=" << expectUpdateT << " f=" << f << " g=" << g << std::endl;
            }
            ++badUpdate;
        }
    }

    AscendC::GmFree(gradsGm);
    AscendC::GmFree(featuresGm);
    AscendC::GmFree(weightsGm);
    AscendC::GmFree(dxGm);
    AscendC::GmFree(updateGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingGm);
    return badDx == 0 && badUpdate == 0;
}

static bool RunCase(const KernelCase& testCase)
{
    switch (testCase.dtype) {
        case DType::F16:
            return RunTyped<half, 0>(testCase.shape, testCase.dtype);
        case DType::F32:
            return RunTyped<float, 1>(testCase.shape, testCase.dtype);
        case DType::BF16:
            return RunTyped<bfloat16_t, 2>(testCase.shape, testCase.dtype);
    }
    return false;
}

} // namespace

class PReluGradUpdateKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "PReluGradUpdateKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "PReluGradUpdateKernelTest TearDown" << endl; }
};

TEST_F(PReluGradUpdateKernelTest, period_mode_shapes)
{
    const std::vector<KernelCase> cases = {
        {{2, 3}, DType::F32},
        {{4, 6, 8}, DType::F32},
        {{2, 3, 4, 5}, DType::F32},
        {{1, 1, 1, 1}, DType::F32},
        {{2, 3, 5, 7}, DType::F16},
        {{7, 5}, DType::BF16},
        {{3, 5, 1}, DType::F16},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE("shape rank=" + std::to_string(c.shape.size()) + " dtype=" +
                     std::to_string(static_cast<int>(c.dtype)));
        EXPECT_TRUE(RunCase(c));
    }
}

TEST_F(PReluGradUpdateKernelTest, channel_mode_shapes)
{
    const std::vector<KernelCase> cases = {
        {{1, 8, 64, 64}, DType::F32},
        {{1, 16, 56, 56}, DType::F32},
        {{1, 8, 64, 64}, DType::F16},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE("shape rank=" + std::to_string(c.shape.size()) + " dtype=" +
                     std::to_string(static_cast<int>(c.dtype)));
        EXPECT_TRUE(RunCase(c));
    }
}

TEST_F(PReluGradUpdateKernelTest, block_boundary_shapes)
{
    const std::vector<KernelCase> cases = {
        {{5, 7, 3}, DType::F32},
        {{9, 4, 6, 2}, DType::F16},
        {{3, 9, 5}, DType::BF16},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE("shape rank=" + std::to_string(c.shape.size()) + " dtype=" +
                     std::to_string(static_cast<int>(c.dtype)));
        EXPECT_TRUE(RunCase(c));
    }
}

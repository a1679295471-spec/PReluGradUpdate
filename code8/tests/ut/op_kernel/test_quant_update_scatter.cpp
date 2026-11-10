/*!
 * \file test_quant_update_scatter.cpp
 * \brief QuantUpdateScatter kernel CPU UT (tikicpulib)
 *
 * The expected result is built with the same logical semantics as the kernel:
 * fp32 divide/add, round-to-nearest-even, clip to [-128, 127], then int8.
 * Comparison is on parsed int8 values, not on raw binary bytes.
 */

#include "quant_update_scatter_tiling.h"
#include "../../../op_kernel/quant_update_scatter.cpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;
using namespace AscendC;

namespace {

enum class NumType { F16 = 0, BF16 = 1, F32 = 2, I32 = 3 };
enum class Profile { FP16 = 0, BF16 = 1 };

struct KernelCase {
    string name;
    Profile profile;
    vector<int64_t> varShape;
    vector<int64_t> updatesShape;
    vector<int64_t> indices;
    bool hasZp;
    bool reciprocal;
};

struct TilingInfo {
    QuantUpdateScatterTilingData data;
    int64_t blockDim;
};

static int64_t CeilDivI(int64_t a, int64_t b) { return (a + b - 1) / b; }

static int64_t Prod(const vector<int64_t>& shape, size_t begin, size_t end)
{
    int64_t value = 1;
    for (size_t i = begin; i < end; ++i) {
        value *= shape[i];
    }
    return value;
}

static int64_t ShapeSize(const vector<int64_t>& shape) { return Prod(shape, 0, shape.size()); }

static size_t NumSize(NumType type)
{
    switch (type) {
        case NumType::F32:
        case NumType::I32:
            return 4;
        default:
            return 2;
    }
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

static void StoreNum(uint8_t* base, int64_t index, NumType type, float value)
{
    uint8_t* p = base + index * static_cast<int64_t>(NumSize(type));
    if (type == NumType::F32) {
        memcpy(p, &value, sizeof(float));
    } else if (type == NumType::I32) {
        int32_t v = static_cast<int32_t>(value);
        memcpy(p, &v, sizeof(int32_t));
    } else {
        uint16_t bits = type == NumType::F16 ? FloatToHalfBits(value) : FloatToBF16Bits(value);
        memcpy(p, &bits, sizeof(uint16_t));
    }
}

static float LoadNum(const uint8_t* base, int64_t index, NumType type)
{
    const uint8_t* p = base + index * static_cast<int64_t>(NumSize(type));
    if (type == NumType::F32) {
        float v;
        memcpy(&v, p, sizeof(float));
        return v;
    }
    if (type == NumType::I32) {
        int32_t v;
        memcpy(&v, p, sizeof(int32_t));
        return static_cast<float>(v);
    }
    uint16_t bits;
    memcpy(&bits, p, sizeof(uint16_t));
    return type == NumType::F16 ? HalfBitsToFloat(bits) : BF16BitsToFloat(bits);
}

static TilingInfo MakeTiling(const KernelCase& testCase, NumType updType, NumType scaleType,
                             NumType zpType)
{
    constexpr int64_t UB_SIZE = 196608;
    constexpr int64_t RESERVED_UB_BYTES = 16384;
    constexpr int64_t CORE_NUM = 16;
    constexpr int64_t QUEUE_AND_TEMP_BYTES = 22;

    const int64_t rank = static_cast<int64_t>(testCase.updatesShape.size());
    const int64_t axis = rank - 2;
    TilingInfo info{};
    info.data.varDim0 = testCase.varShape[0];
    info.data.varAxis = testCase.varShape[axis];
    info.data.updDim0 = testCase.updatesShape[0];
    info.data.updAxis = testCase.updatesShape[axis];
    info.data.outer = Prod(testCase.updatesShape, 1, axis);
    info.data.rowLen = testCase.updatesShape.back();
    info.data.indicesNum = static_cast<int64_t>(testCase.indices.size());
    info.data.varPStride = Prod(testCase.varShape, 1, rank);
    info.data.updPStride = Prod(testCase.updatesShape, 1, rank);
    info.data.rank2 = rank == 2 ? 1 : 0;
    info.data.reciprocalScale = testCase.reciprocal ? 1 : 0;
    info.data.hasZeroPoint = testCase.hasZp ? 1 : 0;

    const int64_t bytesPerElem = 2 * static_cast<int64_t>(NumSize(updType)) +
                                 2 * static_cast<int64_t>(NumSize(scaleType)) +
                                 2 * static_cast<int64_t>(NumSize(zpType)) + QUEUE_AND_TEMP_BYTES;
    int64_t maxCol = (UB_SIZE - RESERVED_UB_BYTES) / bytesPerElem;
    if (maxCol < 1) {
        maxCol = 1;
    }
    info.data.colTile = info.data.rowLen > 0 ?
        std::min(info.data.rowLen, maxCol) : 1;
    if (info.data.varDim0 > 0) {
        info.data.pPerCore = CeilDivI(info.data.varDim0, CORE_NUM);
        info.blockDim = std::min<int64_t>(CORE_NUM,
            CeilDivI(info.data.varDim0, info.data.pPerCore));
    } else {
        info.data.pPerCore = 0;
        info.blockDim = 1;
    }
    return info;
}

static int8_t QuantValue(float value)
{
    value = std::nearbyintf(value);
    value = std::fmin(value, 127.0f);
    value = std::fmax(value, -128.0f);
    return static_cast<int8_t>(value);
}

template <typename T, typename IDX_T, typename SCALE_T, typename ZP_T, uint32_t Key>
static bool RunTyped(const KernelCase& testCase, NumType updType, NumType scaleType,
                     NumType zpType)
{
    const TilingInfo info = MakeTiling(testCase, updType, scaleType, zpType);
    const int64_t varNum = ShapeSize(testCase.varShape);
    const int64_t updNum = ShapeSize(testCase.updatesShape);
    const int64_t idxNum = static_cast<int64_t>(testCase.indices.size());
    const int64_t rowLen = info.data.rowLen;

    vector<uint8_t> varHost(static_cast<size_t>(varNum));
    vector<uint8_t> updHost(static_cast<size_t>(updNum * NumSize(updType)));
    vector<uint8_t> scaleHost(static_cast<size_t>(rowLen * NumSize(scaleType)));
    vector<uint8_t> zpHost(static_cast<size_t>(rowLen * NumSize(zpType)));
    vector<uint8_t> idxHost(static_cast<size_t>(idxNum * sizeof(IDX_T)));
    vector<uint8_t> outHost(static_cast<size_t>(varNum));

    mt19937 rng(20241109U + static_cast<uint32_t>(testCase.name.size()) * 97U);
    uniform_int_distribution<int> varDist(-128, 127);
    uniform_real_distribution<float> updDist(-20.0f, 20.0f);
    uniform_real_distribution<float> scaleDist(0.25f, 4.0f);
    uniform_real_distribution<float> zpDist(-8.0f, 8.0f);
    for (int64_t i = 0; i < varNum; ++i) {
        varHost[static_cast<size_t>(i)] = static_cast<uint8_t>(varDist(rng));
    }
    for (int64_t i = 0; i < updNum; ++i) {
        StoreNum(updHost.data(), i, updType, updDist(rng));
    }
    for (int64_t i = 0; i < rowLen; ++i) {
        StoreNum(scaleHost.data(), i, scaleType, scaleDist(rng));
        if (testCase.hasZp) {
            StoreNum(zpHost.data(), i, zpType, std::round(zpDist(rng)));
        }
    }
    for (int64_t i = 0; i < idxNum; ++i) {
        IDX_T value = static_cast<IDX_T>(testCase.indices[static_cast<size_t>(i)]);
        memcpy(idxHost.data() + i * static_cast<int64_t>(sizeof(IDX_T)), &value, sizeof(IDX_T));
    }

    uint8_t* varGm = static_cast<uint8_t*>(AscendC::GmAlloc(static_cast<size_t>(varNum)));
    uint8_t* idxGm = static_cast<uint8_t*>(AscendC::GmAlloc(static_cast<size_t>(max<int64_t>(1, idxNum * sizeof(IDX_T)))));
    uint8_t* updGm = static_cast<uint8_t*>(AscendC::GmAlloc(static_cast<size_t>(max<int64_t>(1, updNum * NumSize(updType)))));
    uint8_t* scaleGm = static_cast<uint8_t*>(AscendC::GmAlloc(static_cast<size_t>(max<int64_t>(1, rowLen * NumSize(scaleType)))));
    uint8_t* zpGm = nullptr;
    if (testCase.hasZp) {
        zpGm = static_cast<uint8_t*>(AscendC::GmAlloc(static_cast<size_t>(max<int64_t>(1, rowLen * NumSize(zpType)))));
    }
    uint8_t* outGm = static_cast<uint8_t*>(AscendC::GmAlloc(static_cast<size_t>(varNum)));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tilingGm = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(QuantUpdateScatterTilingData)));

    memcpy(varGm, varHost.data(), varHost.size());
    if (!idxHost.empty()) memcpy(idxGm, idxHost.data(), idxHost.size());
    memcpy(updGm, updHost.data(), updHost.size());
    memcpy(scaleGm, scaleHost.data(), scaleHost.size());
    if (testCase.hasZp) memcpy(zpGm, zpHost.data(), zpHost.size());
    memcpy(tilingGm, &info.data, sizeof(QuantUpdateScatterTilingData));
    memset(outGm, 0, static_cast<size_t>(varNum));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(Key);
    ICPU_RUN_KF((quant_update_scatter<Key>), static_cast<uint32_t>(info.blockDim), varGm, idxGm,
                updGm, scaleGm, zpGm, outGm, workspace, tilingGm);

    vector<float> scales(static_cast<size_t>(rowLen));
    vector<float> zeros(static_cast<size_t>(rowLen), 0.0f);
    for (int64_t i = 0; i < rowLen; ++i) {
        scales[static_cast<size_t>(i)] = LoadNum(scaleHost.data(), i, scaleType);
        if (testCase.hasZp) {
            zeros[static_cast<size_t>(i)] = LoadNum(zpHost.data(), i, zpType);
        }
    }
    vector<uint8_t> expected = varHost;
    const int64_t rank = static_cast<int64_t>(testCase.varShape.size());
    const int64_t axis = rank - 2;
    const int64_t varAxis = info.data.varAxis;
    const int64_t updAxis = info.data.updAxis;
    const int64_t outer = info.data.outer;
    const int64_t updPStride = info.data.updPStride;
    const int64_t varPStride = info.data.varPStride;

    for (int64_t p = 0; p < idxNum; ++p) {
        int64_t idx = testCase.indices[static_cast<size_t>(p)];
        for (int64_t k = 0; k < updAxis; ++k) {
            if (rank == 2) {
                const int64_t dstRow = idx + k;
                if (dstRow < 0 || dstRow >= varAxis) {
                    continue;
                }
                const int64_t srcBase = k * rowLen;
                const int64_t dstBase = dstRow * rowLen;
                for (int64_t c = 0; c < rowLen; ++c) {
                    float denom = testCase.reciprocal ? (1.0f / scales[static_cast<size_t>(c)])
                                                      : scales[static_cast<size_t>(c)];
                    float q = LoadNum(updHost.data(), srcBase + c, updType) / denom +
                              zeros[static_cast<size_t>(c)];
                    expected[static_cast<size_t>(dstBase + c)] = static_cast<uint8_t>(QuantValue(q));
                }
            } else {
                const int64_t dstRow = idx + k;
                if (dstRow < 0 || dstRow >= varAxis) {
                    continue;
                }
                for (int64_t j = 0; j < outer; ++j) {
                    const int64_t srcBase = p * updPStride + (j * updAxis + k) * rowLen;
                    const int64_t dstBase = p * varPStride + (j * varAxis + dstRow) * rowLen;
                    for (int64_t c = 0; c < rowLen; ++c) {
                        float denom = testCase.reciprocal ? (1.0f / scales[static_cast<size_t>(c)])
                                                          : scales[static_cast<size_t>(c)];
                        float q = LoadNum(updHost.data(), srcBase + c, updType) / denom +
                                  zeros[static_cast<size_t>(c)];
                        expected[static_cast<size_t>(dstBase + c)] = static_cast<uint8_t>(QuantValue(q));
                    }
                }
            }
        }
    }

    int64_t bad = 0;
    for (int64_t i = 0; i < varNum; ++i) {
        const int8_t got = static_cast<int8_t>(outGm[i]);
        const int8_t exp = static_cast<int8_t>(expected[static_cast<size_t>(i)]);
        if (got != exp) {
            if (bad < 5) {
                cerr << testCase.name << ": mismatch idx=" << i << " got=" << int(got)
                     << " expect=" << int(exp) << endl;
            }
            ++bad;
        }
    }

    AscendC::GmFree(varGm);
    AscendC::GmFree(idxGm);
    AscendC::GmFree(updGm);
    AscendC::GmFree(scaleGm);
    if (zpGm != nullptr) AscendC::GmFree(zpGm);
    AscendC::GmFree(outGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingGm);
    return bad == 0;
}

static bool RunCase(const KernelCase& testCase)
{
    if (testCase.profile == Profile::FP16) {
        return RunTyped<half, int32_t, float, int32_t, 0>(testCase, NumType::F16,
                                                          NumType::F32, NumType::I32);
    }
    return RunTyped<bfloat16_t, int64_t, bfloat16_t, bfloat16_t, 1>(
        testCase, NumType::BF16, NumType::BF16, NumType::BF16);
}

} // namespace

class QuantUpdateScatterKernelTest : public testing::Test {};

TEST_F(QuantUpdateScatterKernelTest, fp16_all_scenarios)
{
    vector<KernelCase> cases = {
        {"fp16_sample", Profile::FP16, {1, 2, 32}, {1, 1, 32}, {0}, true, false},
        {"fp16_3d_block2", Profile::FP16, {3, 5, 17}, {3, 2, 17}, {0, 2, 1}, true, false},
        {"fp16_4d_outer", Profile::FP16, {2, 3, 4, 7}, {2, 3, 2, 7}, {1, 2}, true, false},
        {"fp16_5d_outer", Profile::FP16, {2, 3, 2, 4, 9}, {2, 3, 2, 2, 9}, {1, 2}, true, false},
        {"fp16_no_zp", Profile::FP16, {2, 4, 9}, {2, 1, 9}, {2, 0}, false, false},
        {"fp16_reciprocal", Profile::FP16, {1, 3, 17}, {1, 1, 17}, {1}, true, true},
        {"fp16_wide_row", Profile::FP16, {1, 3, 9000}, {1, 1, 9000}, {0}, true, false},
        {"fp16_rank2", Profile::FP16, {8, 13}, {4, 13}, {0, 4}, true, false},
        {"fp16_empty_updates", Profile::FP16, {2, 3, 16}, {0, 1, 16}, {}, false, false},
        {"fp16_partial_dim0", Profile::FP16, {3, 4, 16}, {1, 1, 16}, {0}, false, false},
        {"fp16_single", Profile::FP16, {1, 1, 1}, {1, 1, 1}, {0}, true, false},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(RunCase(c));
    }
}

TEST_F(QuantUpdateScatterKernelTest, bf16_all_scenarios)
{
    vector<KernelCase> cases = {
        {"bf16_sample", Profile::BF16, {2, 3, 32}, {2, 1, 32}, {0, 2}, true, false},
        {"bf16_4d_outer", Profile::BF16, {2, 3, 4, 6}, {2, 3, 2, 6}, {1, 2}, true, false},
        {"bf16_5d_outer", Profile::BF16, {2, 2, 3, 5, 10}, {2, 2, 3, 2, 10}, {1, 0}, true, false},
        {"bf16_no_zp", Profile::BF16, {3, 5, 13}, {3, 2, 13}, {0, 2, 1}, false, false},
        {"bf16_reciprocal", Profile::BF16, {1, 4, 11}, {1, 2, 11}, {1}, true, true},
        {"bf16_wide_row", Profile::BF16, {1, 2, 7000}, {1, 1, 7000}, {0}, true, false},
        {"bf16_rank2", Profile::BF16, {7, 10}, {3, 10}, {0, 4}, true, false},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(RunCase(c));
    }
}

/*!
 * \file test_strided_slice_grad.cpp
 * \brief StridedSliceGrad kernel UT (tikicpulib) with an independent golden.
 *
 * The kernel gathers y from dy.  The golden here is built in the opposite
 * direction: it walks dy and scatters every element into y (zeros elsewhere).
 * That cross-checks the index mapping, the zero fill, negative strides,
 * begin/end clipping and all masks.  Comparison is on logical values.
 */

#include "strided_slice_grad_tiling.h"
#include "../../../op_kernel/strided_slice_grad.cpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

namespace {

enum class ValueType { F16 = 0, F32 = 1, BF16 = 2 };

struct KernelCase {
    string name;
    bool int64Index = false;
    ValueType type = ValueType::F16;
    vector<int64_t> yShape;
    vector<int64_t> begin;
    vector<int64_t> end;
    vector<int64_t> strides;
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
};

constexpr uint32_t CORE_NUM = 4;
constexpr int64_t UB_SIZE = 196608;
constexpr int64_t RESERVED_UB = 16384;

static int64_t CeilDivI(int64_t a, int64_t b) { return b > 0 ? (a + b - 1) / b : 0; }

static int64_t Product(const vector<int64_t>& shape)
{
    int64_t value = 1;
    for (int64_t d : shape) {
        value *= d;
    }
    return value;
}

static int32_t PopCount(int64_t v)
{
    int32_t c = 0;
    while (v != 0) {
        c += static_cast<int32_t>(v & 1);
        v >>= 1;
    }
    return c;
}

static size_t ElemBytes(ValueType t) { return t == ValueType::F32 ? 4U : 2U; }

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

static void StoreValue(uint8_t* base, int64_t index, ValueType t, float value)
{
    uint8_t* p = base + index * static_cast<int64_t>(ElemBytes(t));
    if (t == ValueType::F32) {
        memcpy(p, &value, sizeof(float));
        return;
    }
    const uint16_t bits = t == ValueType::F16 ? FloatToHalfBits(value) : FloatToBF16Bits(value);
    memcpy(p, &bits, sizeof(uint16_t));
}

static float LoadValue(const uint8_t* base, int64_t index, ValueType t)
{
    const uint8_t* p = base + index * static_cast<int64_t>(ElemBytes(t));
    if (t == ValueType::F32) {
        float v;
        memcpy(&v, p, sizeof(float));
        return v;
    }
    uint16_t bits;
    memcpy(&bits, p, sizeof(uint16_t));
    return t == ValueType::F16 ? HalfBitsToFloat(bits) : BF16BitsToFloat(bits);
}

static int64_t AlignIndex(int64_t index, int64_t dim, int64_t step)
{
    if (index < 0) {
        index += dim;
    }
    if (step > 0) {
        index = std::min<int64_t>(std::max<int64_t>(index, 0), dim);
    } else {
        index = std::min<int64_t>(std::max<int64_t>(index, -1), dim - 1);
    }
    return index;
}

struct RefDim {
    bool isShrink = false;
    bool isImplicit = false;
    int64_t start = 0;
    int64_t step = 1;
    int64_t count = 1;
    int64_t fixed = 0;
    int32_t dyAxis = -1;
};

struct RefPlan {
    vector<int64_t> dyShape;
    vector<RefDim> dims;
    int64_t totalY = 0;
    int64_t totalDy = 1;
};

/*! Independent reference plan built from the TF strided-slice semantics. */
static RefPlan BuildPlan(const KernelCase& c)
{
    const int64_t rank = static_cast<int64_t>(c.yShape.size());
    const int64_t specLen = static_cast<int64_t>(c.begin.size());
    RefPlan plan;
    plan.dims.resize(static_cast<size_t>(rank));

    const int32_t newAxisCount = PopCount(c.newAxisMask);
    int64_t ellipsisLen = 0;
    bool hasEllipsis = false;
    for (int64_t s = 0; s < specLen; ++s) {
        if (((c.ellipsisMask >> s) & 1) != 0) {
            hasEllipsis = true;
        }
    }
    if (hasEllipsis) {
        ellipsisLen = rank - (specLen - 1 - newAxisCount);
    }

    int64_t yDim = 0;
    for (int64_t s = 0; s < specLen; ++s) {
        if (((c.newAxisMask >> s) & 1) != 0) {
            plan.dyShape.push_back(1);
            continue;
        }
        if (((c.ellipsisMask >> s) & 1) != 0) {
            for (int64_t k = 0; k < ellipsisLen; ++k) {
                RefDim& d = plan.dims[static_cast<size_t>(yDim)];
                d.isImplicit = true;
                d.dyAxis = static_cast<int32_t>(plan.dyShape.size());
                plan.dyShape.push_back(c.yShape[static_cast<size_t>(yDim)]);
                ++yDim;
            }
            continue;
        }
        RefDim& d = plan.dims[static_cast<size_t>(yDim)];
        const int64_t dim = c.yShape[static_cast<size_t>(yDim)];
        if (((c.shrinkAxisMask >> s) & 1) != 0) {
            d.isShrink = true;
            int64_t index = c.begin[static_cast<size_t>(s)];
            if (index < 0) {
                index += dim;
            }
            d.fixed = index;
            ++yDim;
            continue;
        }
        const int64_t step = c.strides[static_cast<size_t>(s)];
        const bool beginIgnored = ((c.beginMask >> s) & 1) != 0;
        const bool endIgnored = ((c.endMask >> s) & 1) != 0;
        const int64_t start = beginIgnored ? (step < 0 ? dim - 1 : 0)
                                           : AlignIndex(c.begin[static_cast<size_t>(s)], dim, step);
        const int64_t stop = endIgnored ? (step < 0 ? -1 : dim)
                                        : AlignIndex(c.end[static_cast<size_t>(s)], dim, step);
        int64_t count = 0;
        if (step > 0) {
            count = start < stop ? (stop - start + step - 1) / step : 0;
        } else {
            count = start > stop ? (start - stop + (-step) - 1) / (-step) : 0;
        }
        d.start = start;
        d.step = step;
        d.count = count;
        d.dyAxis = static_cast<int32_t>(plan.dyShape.size());
        plan.dyShape.push_back(count);
        ++yDim;
    }
    while (yDim < rank) {
        RefDim& d = plan.dims[static_cast<size_t>(yDim)];
        d.isImplicit = true;
        d.dyAxis = static_cast<int32_t>(plan.dyShape.size());
        plan.dyShape.push_back(c.yShape[static_cast<size_t>(yDim)]);
        ++yDim;
    }

    plan.totalY = Product(c.yShape);
    plan.totalDy = Product(plan.dyShape);
    return plan;
}

static StridedSliceGradTilingData MakeTiling(const KernelCase& c, const RefPlan& plan,
                                             int64_t* blockDim)
{
    StridedSliceGradTilingData data;
    data.rank = static_cast<int32_t>(c.yShape.size());
    data.dyRank = static_cast<int32_t>(plan.dyShape.size());
    data.specLen = static_cast<int32_t>(c.begin.size());
    for (int32_t d = 0; d < data.rank; ++d) {
        data.yDims[d] = c.yShape[static_cast<size_t>(d)];
    }
    for (int32_t d = 0; d < data.dyRank; ++d) {
        data.dyDims[d] = plan.dyShape[static_cast<size_t>(d)];
    }
    int64_t stride = 1;
    for (int32_t d = data.rank - 1; d >= 0; --d) {
        data.yStrides[d] = stride;
        stride *= data.yDims[d];
    }
    stride = 1;
    for (int32_t d = data.dyRank - 1; d >= 0; --d) {
        data.dyStrides[d] = stride;
        stride *= data.dyDims[d];
    }
    data.totalY = plan.totalY;
    data.totalDy = plan.totalDy;

    for (int32_t d = 0; d < data.rank; ++d) {
        const RefDim& rd = plan.dims[static_cast<size_t>(d)];
        data.inputMode[d] = rd.isShrink ? 1 : 0;
        data.inputDyAxis[d] = rd.dyAxis;
    }
    // inputSpecIdx is only needed for BeginSliceParams; the kernel re-reads it,
    // so derive it from the plan by re-walking the spec order.
    {
        const int64_t rank = static_cast<int64_t>(c.yShape.size());
        const int64_t specLen = static_cast<int64_t>(c.begin.size());
        int32_t newAxisCount = PopCount(c.newAxisMask);
        int64_t ellipsisLen = 0;
        bool hasEllipsis = false;
        for (int64_t s = 0; s < specLen; ++s) {
            if (((c.ellipsisMask >> s) & 1) != 0) {
                hasEllipsis = true;
            }
        }
        if (hasEllipsis) {
            ellipsisLen = rank - (specLen - 1 - newAxisCount);
        }
        int64_t yDim = 0;
        for (int64_t s = 0; s < specLen; ++s) {
            if (((c.newAxisMask >> s) & 1) != 0) {
                continue;
            }
            if (((c.ellipsisMask >> s) & 1) != 0) {
                for (int64_t k = 0; k < ellipsisLen; ++k) {
                    data.inputSpecIdx[yDim++] = -1;
                }
                continue;
            }
            data.inputSpecIdx[yDim++] = static_cast<int32_t>(s);
        }
        while (yDim < rank) {
            data.inputSpecIdx[yDim++] = -1;
        }
    }

    data.beginMask = c.beginMask;
    data.endMask = c.endMask;
    data.ellipsisMask = c.ellipsisMask;
    data.newAxisMask = c.newAxisMask;
    data.shrinkAxisMask = c.shrinkAxisMask;

    const int64_t elemBytes = static_cast<int64_t>(ElemBytes(c.type));
    const int64_t alignElems = 32 / elemBytes;
    int64_t ubFactor = std::max<int64_t>(1, (UB_SIZE - RESERVED_UB) / (2 * elemBytes));
    ubFactor = ubFactor / alignElems * alignElems;
    // Cap the tile so every case exercises several tiles per block and a tail
    // that is not 32-byte aligned.
    ubFactor = std::min<int64_t>(ubFactor, 64);
    ubFactor = std::min<int64_t>(ubFactor, std::max<int64_t>(1, plan.totalY));
    int64_t yBlock = CeilDivI(CeilDivI(std::max<int64_t>(1, plan.totalY), CORE_NUM), alignElems) * alignElems;
    yBlock = std::max<int64_t>(1, yBlock);
    *blockDim = std::min<int64_t>(CORE_NUM, CeilDivI(std::max<int64_t>(1, plan.totalY), yBlock));
    if (*blockDim < 1) {
        *blockDim = 1;
    }
    data.yBlockFactor = yBlock;
    data.ubFactor = ubFactor;
    return data;
}

static vector<float> ScatterGolden(const KernelCase& c, const RefPlan& plan, const uint8_t* dy)
{
    vector<float> golden(static_cast<size_t>(plan.totalY), 0.0f);
    vector<int64_t> coords(static_cast<size_t>(plan.dyShape.size()), 0);
    for (int64_t flat = 0; flat < plan.totalDy; ++flat) {
        int64_t remain = flat;
        for (size_t a = 0; a < plan.dyShape.size(); ++a) {
            int64_t stride = 1;
            for (size_t b = a + 1; b < plan.dyShape.size(); ++b) {
                stride *= plan.dyShape[b];
            }
            coords[a] = plan.dyShape[a] == 0 ? 0 : remain / stride;
            if (plan.dyShape[a] != 0) {
                remain %= stride;
            }
        }

        int64_t yOffset = 0;
        for (size_t d = 0; d < plan.dims.size(); ++d) {
            const RefDim& rd = plan.dims[d];
            int64_t coord = 0;
            if (rd.isShrink) {
                coord = rd.fixed;
            } else {
                coord = rd.start + coords[static_cast<size_t>(rd.dyAxis)] * rd.step;
            }
            int64_t stride = 1;
            for (size_t b = d + 1; b < c.yShape.size(); ++b) {
                stride *= c.yShape[b];
            }
            yOffset += coord * stride;
        }
        if (yOffset >= 0 && yOffset < plan.totalY) {
            golden[static_cast<size_t>(yOffset)] = LoadValue(dy, flat, c.type);
        }
    }
    return golden;
}

template <uint32_t KEY>
static bool RunKernel(const KernelCase& c, const RefPlan& plan, uint32_t blockDim,
                      uint8_t* beginGm, uint8_t* endGm, uint8_t* stridesGm, uint8_t* dyGm,
                      uint8_t* yGm, uint8_t* workspace, uint8_t* tilingGm)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(KEY);
    ICPU_RUN_KF((strided_slice_grad<KEY>), blockDim, nullptr, beginGm, endGm, stridesGm, dyGm,
                yGm, workspace, tilingGm);
    (void)c;
    (void)plan;
    return true;
}

static bool RunCase(const KernelCase& c)
{
    const RefPlan plan = BuildPlan(c);
    int64_t blockDim = 1;
    StridedSliceGradTilingData data = MakeTiling(c, plan, &blockDim);

    const size_t elemBytes = ElemBytes(c.type);
    const size_t idxBytes = c.int64Index ? sizeof(int64_t) : sizeof(int32_t);
    const int64_t totalY = plan.totalY;
    const int64_t totalDy = plan.totalDy;
    const int64_t specLen = static_cast<int64_t>(c.begin.size());

    vector<uint8_t> beginHost(static_cast<size_t>(specLen) * idxBytes);
    vector<uint8_t> endHost(static_cast<size_t>(specLen) * idxBytes);
    vector<uint8_t> stridesHost(static_cast<size_t>(specLen) * idxBytes);
    vector<uint8_t> dyHost(static_cast<size_t>(totalDy) * elemBytes);
    vector<uint8_t> yHost(static_cast<size_t>(totalY) * elemBytes, 0xAB);

    for (int64_t i = 0; i < specLen; ++i) {
        if (c.int64Index) {
            const int64_t b = c.begin[static_cast<size_t>(i)];
            const int64_t e = c.end[static_cast<size_t>(i)];
            const int64_t s = c.strides[static_cast<size_t>(i)];
            memcpy(beginHost.data() + i * idxBytes, &b, sizeof(int64_t));
            memcpy(endHost.data() + i * idxBytes, &e, sizeof(int64_t));
            memcpy(stridesHost.data() + i * idxBytes, &s, sizeof(int64_t));
        } else {
            const int32_t b = static_cast<int32_t>(c.begin[static_cast<size_t>(i)]);
            const int32_t e = static_cast<int32_t>(c.end[static_cast<size_t>(i)]);
            const int32_t s = static_cast<int32_t>(c.strides[static_cast<size_t>(i)]);
            memcpy(beginHost.data() + i * idxBytes, &b, sizeof(int32_t));
            memcpy(endHost.data() + i * idxBytes, &e, sizeof(int32_t));
            memcpy(stridesHost.data() + i * idxBytes, &s, sizeof(int32_t));
        }
    }
    for (int64_t i = 0; i < totalDy; ++i) {
        const float v = std::sin(static_cast<double>(i) * 0.37 + 0.11) * 3.5f;
        StoreValue(dyHost.data(), i, c.type, v);
    }
    const vector<float> golden = ScatterGolden(c, plan, dyHost.data());

    const size_t dyBytes = std::max<size_t>(1, dyHost.size());
    const size_t yBytes = std::max<size_t>(1, yHost.size());
    uint8_t* beginGm = static_cast<uint8_t*>(AscendC::GmAlloc(std::max<size_t>(1, beginHost.size())));
    uint8_t* endGm = static_cast<uint8_t*>(AscendC::GmAlloc(std::max<size_t>(1, endHost.size())));
    uint8_t* stridesGm = static_cast<uint8_t*>(AscendC::GmAlloc(std::max<size_t>(1, stridesHost.size())));
    uint8_t* dyGm = static_cast<uint8_t*>(AscendC::GmAlloc(dyBytes));
    uint8_t* yGm = static_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tilingGm = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(StridedSliceGradTilingData)));

    memcpy(beginGm, beginHost.data(), beginHost.size());
    memcpy(endGm, endHost.data(), endHost.size());
    memcpy(stridesGm, stridesHost.data(), stridesHost.size());
    if (!dyHost.empty()) {
        memcpy(dyGm, dyHost.data(), dyHost.size());
    }
    memcpy(yGm, yHost.data(), yHost.size());
    memcpy(tilingGm, &data, sizeof(data));

    const uint32_t key = static_cast<uint32_t>((c.int64Index ? 3 : 0) +
                                               (c.type == ValueType::F16 ? 0 : (c.type == ValueType::F32 ? 1 : 2)));
    bool launched = false;
    switch (key) {
        case 0: launched = RunKernel<0>(c, plan, static_cast<uint32_t>(blockDim), beginGm, endGm, stridesGm, dyGm, yGm, workspace, tilingGm); break;
        case 1: launched = RunKernel<1>(c, plan, static_cast<uint32_t>(blockDim), beginGm, endGm, stridesGm, dyGm, yGm, workspace, tilingGm); break;
        case 2: launched = RunKernel<2>(c, plan, static_cast<uint32_t>(blockDim), beginGm, endGm, stridesGm, dyGm, yGm, workspace, tilingGm); break;
        case 3: launched = RunKernel<3>(c, plan, static_cast<uint32_t>(blockDim), beginGm, endGm, stridesGm, dyGm, yGm, workspace, tilingGm); break;
        case 4: launched = RunKernel<4>(c, plan, static_cast<uint32_t>(blockDim), beginGm, endGm, stridesGm, dyGm, yGm, workspace, tilingGm); break;
        default: launched = RunKernel<5>(c, plan, static_cast<uint32_t>(blockDim), beginGm, endGm, stridesGm, dyGm, yGm, workspace, tilingGm); break;
    }

    const double tol = c.type == ValueType::F32 ? 1e-4 : (c.type == ValueType::F16 ? 1e-3 : 1e-2);
    int64_t bad = 0;
    if (launched) {
        for (int64_t i = 0; i < totalY; ++i) {
            const float got = LoadValue(yGm, i, c.type);
            const float expect = golden[static_cast<size_t>(i)];
            if (std::fabs(static_cast<double>(got) - static_cast<double>(expect)) > tol) {
                if (bad < 5) {
                    cerr << c.name << ": mismatch idx=" << i << " got=" << got
                         << " expect=" << expect << endl;
                }
                ++bad;
            }
        }
    }

    AscendC::GmFree(beginGm);
    AscendC::GmFree(endGm);
    AscendC::GmFree(stridesGm);
    AscendC::GmFree(dyGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingGm);
    return launched && bad == 0;
}

} // namespace

class StridedSliceGradKernelTest : public testing::Test {};

TEST_F(StridedSliceGradKernelTest, fp16_all_scenarios)
{
    const vector<KernelCase> cases = {
        {"fp16_basic_4d", false, ValueType::F16, {2, 4, 5, 6}, {0, 1, 0, 0}, {2, 4, 5, 6}, {1, 1, 1, 1}},
        {"fp16_neg_strides", false, ValueType::F16, {3, 4, 5}, {2, -1, 4}, {-4, -5, -6}, {-1, -2, -1}},
        {"fp16_ellipsis", false, ValueType::F16, {2, 3, 4, 5}, {0, 1, 0}, {2, 3, 4}, {1, 1, 1}, 0, 0, 0x2, 0, 0},
        {"fp16_masks", false, ValueType::F16, {3, 4, 5}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, 0x5, 0x2, 0, 0, 0},
        {"fp16_empty_slice", false, ValueType::F16, {4, 5}, {2}, {2}, {1}},
        {"fp16_rank1_stride3", false, ValueType::F16, {10}, {1}, {9}, {3}},
        {"fp16_wide_row", false, ValueType::F16, {2, 1, 300}, {0, 0, 3}, {2, 1, 290}, {1, 1, 1}},
        {"fp16_rank5_step2", false, ValueType::F16, {2, 3, 2, 4, 5}, {0, 0, 0, 1, 0}, {2, 3, 2, 4, 5}, {1, 1, 1, 2, 1}},
        {"fp16_i64", true, ValueType::F16, {5, 6}, {0, 1}, {5, 6}, {1, 2}},
        {"fp16_i64_neg", true, ValueType::F16, {2, 6}, {1, 5}, {-3, -7}, {-1, -2}},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(RunCase(c));
    }
}

TEST_F(StridedSliceGradKernelTest, fp32_all_scenarios)
{
    const vector<KernelCase> cases = {
        {"fp32_basic_4d", false, ValueType::F32, {2, 4, 5, 6}, {0, 1, 0, 0}, {2, 4, 5, 6}, {1, 1, 1, 1}},
        {"fp32_new_axis", false, ValueType::F32, {3, 4}, {1, 0, 0}, {3, 1, 2}, {1, 1, 1}, 0, 0, 0, 0x2, 0},
        {"fp32_new_axis_first", false, ValueType::F32, {4, 5}, {0, 1, 0}, {0, 3, 5}, {1, 1, 1}, 0, 0, 0, 0x1, 0},
        {"fp32_shrink", false, ValueType::F32, {4, 5, 6}, {0, 2, 0}, {4, 3, 6}, {2, 1, 3}, 0, 0, 0, 0, 0x2},
        {"fp32_shrink_first", false, ValueType::F32, {3, 4, 5}, {1}, {2}, {1}, 0, 0, 0, 0, 0x1},
        {"fp32_ellipsis_first", false, ValueType::F32, {2, 3, 4}, {0, 1}, {0, 3}, {1, 1}, 0, 0, 0x1, 0, 0},
        {"fp32_empty_y", false, ValueType::F32, {0, 5}, {0, 0}, {0, 5}, {1, 1}},
        {"fp32_wide", false, ValueType::F32, {3, 2, 129}, {0, 1, 2}, {3, 2, 128}, {1, 1, 1}},
        {"fp32_i64_neg", true, ValueType::F32, {2, 6}, {1, 5}, {-3, -7}, {-1, -2}},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(RunCase(c));
    }
}

TEST_F(StridedSliceGradKernelTest, bf16_all_scenarios)
{
    const vector<KernelCase> cases = {
        {"bf16_basic_4d", false, ValueType::BF16, {2, 4, 5, 6}, {0, 1, 0, 0}, {2, 4, 5, 6}, {1, 1, 1, 1}},
        {"bf16_ellipsis_shrink", false, ValueType::BF16, {2, 3, 4, 5}, {0, 0, 0, 1}, {2, 0, 4, 2}, {2, 1, 1, 1}, 0, 0, 0x2, 0, 0x8},
        {"bf16_new_and_shrink", false, ValueType::BF16, {3, 4}, {1, 0, 2}, {3, 1, 3}, {1, 1, 1}, 0, 0, 0, 0x2, 0x4},
        {"bf16_neg_strides", false, ValueType::BF16, {3, 4, 5}, {2, -1, 4}, {-4, -5, -6}, {-1, -2, -1}},
        {"bf16_i64", true, ValueType::BF16, {4, 7}, {1, 2}, {4, 7}, {1, 2}},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(RunCase(c));
    }
}
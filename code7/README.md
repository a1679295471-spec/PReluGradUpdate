# PReluGradUpdate Ascend C 算子工程（code7）

基于昇腾内置 TBE 版 `PReluGradUpdate` 的 Ascend C 实现。工程结构沿用赛题官方模板
（`07PReluGradUpdate_problem_2329_template.zip`），目标环境 **CANN 9.0.0 /
ascend910b**，算子类型 `PReluGradUpdate`，包名 `p_relu_grad_update_custom`。

## 计算语义

`grads`、`features` 形状相同（rank >= 2，记为 `[N, C, ...]`），`weights` 为 `[C]`，
按通道广播为 `[1, C, 1, ...]`（与内置 TBE 的 `broadcast_weight_shape` 一致，不是右对齐）：

```text
dx     = features > 0 ? grads : grads * weights
update = features > 0 ? 0     : features * grads
```

`dx`、`update` 都与 `features` 同形，`update` 不做通道求和。该语义与内置
`opp/.../ops_legacy/dynamic/prelu_grad_update.py` 的 `after_v200` 分支（910B 走这里）
一致：

```text
zero_one = (features <= 0)
dx       = (1 + zero_one * (weights - 1)) * grads
update   = zero_one * features * grads
```

## 数据类型与格式

| 项 | 支持 |
| --- | --- |
| dtype | `float16`、`float32`、`bfloat16`（与官方原型 `DataType` 列表完全一致） |
| format | `ND`（`UnknownShapeFormat` 同为 `ND`，`AutoContiguous`） |
| rank | >= 2，任意合法维度组合（2D/3D/4D/5D/6D 均已覆盖） |
| 空 tensor | 支持（`totalNum == 0` 直接返回） |

内置 TBE 原型没有 int 类型注册，因此本工程不注册 int；CANN Judge 的 “int: 0”
阈值对本算子不适用。

## 实现方案

**Host tiling**（`op_host/p_relu_grad_update_tiling.cpp`）

- 校验 rank、`grads/features` 形状一致、`weights` 元素数等于 `C`（接受 `[C]` 或 `[1, C]`）、
  三个输入 dtype 一致。
- 根据 UB 大小计算 `ubFactor`；UB 预算 = 4 个队列 × 2 缓冲 × dtype + 5 个 fp32 临时
  缓冲，另留 8KB 余量。
- 两种调度模式：
  - `pattern`：权重周期 `C * innerSize` 对齐到 32B 后能和一整块 tile 一起放进 UB 时，
    kernel 每核只构造一次权重周期，再在 UB 内倍增复制。2D `[N, C]`（`innerSize == 1`）
    也走这里。
  - `channel`：周期放不下时，每个 tile 严格限制在单个通道内，权重退化为标量。
- 多核：按 `blockIdx * blockFactor` 切分；pattern 模式下 `blockFactor` 对齐到权重周期，
  保证每个 tile 的全局偏移都落在周期起点（权重相位恒为 0）。

**Kernel**（`op_kernel/p_relu_grad_update.h`）

- 所有输入/输出都用 `DataCopyPad`，支持任意元素数（不要求 32B 整除）。
- fp16/bf16 先 `Cast` 到 fp32 计算，结果再用 `CAST_RINT`（RNE）舍回转出类型；fp32 直接算。
- 用 `Sign + Maxs` 得到精确的 0/1 指示值 `p`，再以
  `coeff = p + (1 - p) * w` 完成分支：`p == 1` 时 `coeff` 精确为 1，`p == 0` 时精确为 `w`。
- `update = f*g - p*f*g`：`p == 1` 时严格为 0，`p == 0` 时严格为 `f*g`。
- 每个 tile 内的向量链路：`Sign -> Maxs -> Muls/Adds/Mul/Add -> Mul/Sub -> Mul`。

## 目录

```text
code7/
├── build.sh                       # -e / -u / -j / --make_clean
├── CMakeLists.txt                 # find_package(ASC) + npu_op_package(RUN)
├── op_host/                       # 原型、shape/dtype 推导、tiling（未新增文件）
├── op_kernel/                     # 入口、kernel 类、tiling data/key（未新增文件）
├── examples/                      # aclnn 调用示例 + run.sh
├── samples/                       # 官方首个测试点 case.json、reference.py、sample.py
└── tests/
    ├── ut/                        # host tiling UT + tikicpulib kernel UT
    └── selfcheck/                 # 无 CANN 的静态检查与随机差分验证
```

## 构建与运行

先把 CANN 9.0.0 环境变量加载好：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 路径以实际安装为准
echo "$ASCEND_HOME_PATH"
```

```bash
cd code7
bash build.sh -j8          # 只编译 RUN 包
bash build.sh -j8 -e       # 编译 + 安装到 build/packages + 跑第一个测试点（需要 NPU）
bash build.sh -j8 -u       # host tiling UT + tikicpulib kernel UT（CPU，不需要 NPU）
bash build.sh --make_clean # 清理
```

`-e` 的流程：cmake 配置 → `all/binary/package/install` → 用
`examples/run.sh` 生成 `samples/build` 数据 → 运行 `aclnnPReluGradUpdate` →
`sample.py verify` 用 `np.isclose`（逻辑值比较，rtol=atol=1e-4）比较 `dx`/`update`。
可通过 `DEVICE_ID=0` 指定卡号。

`-u` 依赖本地 GoogleTest 和 nlohmann/json（模板不自动下载）：

```bash
GTEST_SOURCE_DIR=/path/to/googletest-1.14.0 \
NLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include \
bash build.sh -j8 -u
```

其中 `NLOHMANN_JSON_INCLUDE_DIR` 下应有 `nlohmann/json.hpp`。

## 自验证覆盖

| 层次 | 覆盖内容 |
| --- | --- |
| host tiling UT（12 例） | fp16/fp32/bf16；2D/3D/4D；`innerSize == 1`；奇数 inner；pattern/channel 两种模式；单元素边界；tilingKey 0/1/2；workspace = 0 |
| kernel UT（13 组场景） | 真实 kernel 在 tikicpulib 上运行；fp16/fp32/bf16；2D~4D；period 模式、channel 模式、跨 block 边界；奇数 inner；单元素；输入先按 dtype 量化，golden 也按输出 dtype 做 RNE 量化后按阈值比较 |
| 随机差分（`fuzz_check.py`） | 293 组随机/边界 shape：rank 2~6、`C` 1~4096、inner 最大 8191，pattern 222 例 + channel 65 例 + 空 tensor；逐元素校验覆盖恰好一次、pattern 相位为 0、channel tile 不跨通道，并与 `samples/reference.py` 数值对齐 |
| `-e` 首点 | 官方 `[4, 6, 8]` float32 用例，`np.isclose` 比较 |

## 无 CANN / 无 NPU 时的本地检查

这两条在 Windows 本机即可运行（第一条只依赖标准库，第二条需要 numpy）：

```bash
python3 tests/selfcheck/static_check.py
python3 tests/selfcheck/fuzz_check.py
```

`static_check.py` 还会校验官方模板要求的目录结构，并把 host UT 期望表与 tiling 公式
逐项核对。它们不能替代 `build.sh -e/-u`，但能在没有 NPU 的机器上先排除调度/数值设计问题。

## CANN 版本说明

- 工程按 **CANN 9.0.0** 的 ops-nn CMake 模板编写：`find_package(ASC)`、
  `npu_op_package`、`npu_op_kernel_sources`、`ASCENDC_TPL_ARGS_DECL`。
- kernel 只使用基础 Ascend C 向量 API（`Cast/Sign/Maxs/Mul/Add/Sub/Muls/Adds/DataCopyPad`），
  没有使用 9.1 专有接口。
- CANN 8.5 的模板不含上述 CMake 宏，若在 8.5 环境 cmake 阶段报找不到 `ASC` 或
  `npu_op_kernel_sources`，请改用 9.0.0 镜像；本工程未做 8.5 兼容分支。
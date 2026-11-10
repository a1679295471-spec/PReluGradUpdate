# StridedSliceGrad（Ascend C，CANN 9.0.0）

对标昇腾内置 TBE 版 `StridedSliceGrad` 的 Ascend C 实现。输入 `dy` 是切片结果的梯度，
输出 `y` 是原始形状的梯度：落在切片内的位置写入对应的 `dy` 元素，其余位置写 0。

## 算子原型

| 名称 | 方向 | dtype | 说明 |
| --- | --- | --- | --- |
| `shape` | 输入 | int32 / int64 | 1-D，长度 = 输出 rank，值为 `y` 的每个维度 |
| `begin` | 输入 | int32 / int64 | 1-D，长度 = spec 数 |
| `end` | 输入 | int32 / int64 | 1-D，长度 = spec 数 |
| `strides` | 输入 | int32 / int64 | 1-D，长度 = spec 数，元素不能为 0 |
| `dy` | 输入 | float16 / float32 / bfloat16 | 切片结果的梯度 |
| `y` | 输出 | 同 `dy` | 原始形状的梯度 |
| `begin_mask` / `end_mask` / `ellipsis_mask` / `new_axis_mask` / `shrink_axis_mask` | 属性 | int | 默认 0 |

`shape` 与 `begin/end/strides` 的 dtype 相互独立（对应 TF 的 `Tshape` / `Index`
两个 attr），共有 `2 × 2 × 3 = 12` 组 dtype profile；kernel 只依赖
`begin/end/strides` 的 dtype 和 `dy` 的 dtype，因此 tiling key 仍然是 6 个。
`dy/y` 支持 `float16 / float32 / bfloat16`，数据格式 `ND`，计算单元 `ascend910b`。
`begin/end/strides` 允许短于 rank（缺省维度按整维切片），也允许配合 `ellipsis_mask` 展开。

## 语义要点

- 负索引先 `+dim` 归一化；正步长时 begin/end 截断到 `[0, dim]`，负步长时截断到 `[-1, dim-1]`。
- `begin_mask` 位为 1 时起点取整维起点（负步长取 `dim-1`）；`end_mask` 为 1 时终点取整维终点（负步长取 `-1`）。
- `ellipsis_mask` 最多一个 bit，展开为未被其它 spec 覆盖的维度，按整维切片。
- `new_axis_mask` 在切片结果中插入长度 1 的轴：不消耗 `y` 的维度，但会多出一个长度为 1 的 `dy` 轴。
- `shrink_axis_mask` 用 `begin[i]` 选定单个下标：该维在切片结果中消失，`dy` 不消耗对应轴。
- 切片之外（以及空切片时）的 `y` 元素恒为 0。

## 工程结构

```
code9/
├── build.sh                     # -e：编译并跑第一个测试点；-u：跑 UT
├── op_host/
│   ├── strided_slice_grad_def.cpp        # 6 组 dtype profile + 5 个 mask 属性
│   ├── strided_slice_grad_tiling.cpp     # 结构映射、block/UB 切分、tilingKey
│   └── strided_slice_grad_infershape.cpp # 由 shape 张量值推导 y 形状
├── op_kernel/
│   ├── strided_slice_grad.h              # kernel 实现（gather 方式）
│   ├── strided_slice_grad.cpp            # 6 个 tilingKey 的 dtype 分发
│   ├── strided_slice_grad_tiling_data.h  # 纯 POD tiling 结构
│   └── strided_slice_grad_tiling_key.h
├── examples/                    # aclnn 调用首个测试点
├── samples/                     # reference.py（纯 NumPy golden）+ 多场景数据生成
└── tests/
    ├── ut/                      # op_host tiling UT + op_kernel CPU UT
    └── selfcheck/               # 静态检查、随机语义对拍
```

## 编译与运行

```bash
source /path/to/cann-9.0.0/set_env.sh

bash build.sh -j8          # 只编译 RUN 包
bash build.sh -j8 -e       # 编译 + 跑第一个测试点并做逻辑值比对
NLOHMANN_JSON_INCLUDE_DIR=$HOME/nlohmann_json/include \
GTEST_SOURCE_DIR=/usr/src/googletest \
bash build.sh -j8 -u       # host tiling UT + kernel CPU UT
```

`build.sh` 显式传 `-DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}` 与
`-DASC_DIR=${ASCEND_HOME_PATH}/lib64/cmake`，不依赖 `find_package(ASC)` 的默认搜索；
UT 默认 `Release` 并把 `CMAKE_DEBUG_POSTFIX` 置空，避免 `libgtestd.a` 链接不上。

## 实现说明

- **输出侧 gather**：每个核只写自己负责的一段连续 `y`，切片内写 `dy`、切片外写 0。
  不存在多核先清零再 scatter 的竞争，也没有额外的 workspace。
- **不使用 Cast**：`float16/bf16` 用 `uint16_t` 原始位、`float32` 用 `uint32_t` 原始位搬运，
  完全绕开 bisheng 对 bf16 Cast 的后端限制，也避免了 fp32→int8 之类的跨级转换。
- **5 参数 `DataCopyExtParams`**：`{blockCount, blockLen, srcStride, dstStride, rsv}`，符合 CANN 9.0.0。
- **不依赖 begin/end/strides 的 host 值**：tiling 只做结构映射；具体 start/step/count 由 kernel
  在 device 上用 `GetValue` 读取后计算，因此这几个输入是动态值时也能正确工作。
  （`shape` 声明为 `ValueDepend`，因为形状推导必须读它的值。）

## 自验证

```bash
cd samples && python3 reference.py                    # golden 自检
python3 tests/selfcheck/static_check.py               # API/结构静态检查
python3 tests/selfcheck/fuzz_check.py                 # gather vs scatter 随机对拍
python3 tests/selfcheck/kernel_model_check.py         # kernel 算法 vs golden
```

`tests/selfcheck/*.py` 需要 `PYTHONPATH` 指向 `samples/` 才能 import `reference`：

```bash
PYTHONPATH=samples python3 tests/selfcheck/kernel_model_check.py
```

## 验证状态（重要）

**本工程没有在真实 CANN 9.0.0 + Ascend 910B 环境编译或运行过。** 当前只做了：

- 本地静态检查（结构、API 形态、`DataCopyExtParams` 参数个数、build 参数、无硬编码版本路径等）；
- 纯 NumPy golden 自检；
- gather / scatter 两种独立实现的 400 组随机对拍；
- kernel 算法模型与 golden 的 60 组对拍。

因此下面的点需要在真实环境优先验证。

## 高风险点（按优先级）

1. `gert::InferShapeContext::GetInputTensor(0)` / `TilingContext::GetInputTensor(0)` 在 9.0.0 的签名，
   以及 `ValueDepend(REQUIRED)` 是否让 `shape` 在 infershape/tiling 阶段拿到 host 数据。
   若接口不可用，infershape 会退化为「rank 已知、维度未知」，需要改用其它取值方式。
2. `LocalTensor<uint16_t>` / `LocalTensor<uint32_t>` 的 `SetValue` 与 `DataCopyPad` 是否被 9.0.0 的
   类型检查接受（绕开 bf16 Cast 的代价）。
3. `DataCopyPad` 对「32B 对齐起始地址 + 非 32B 对齐长度」的支持；tiling 已保证每个 tile 起点
   按 32B 对齐，但尾块长度是任意值。
4. `pipe_barrier(PIPE_ALL)` 在 kernel 中的可用性与开销。
5. tiling key 宏：`ASCENDC_TPL_UINT_DECL(schMode, 3, ...)` 的位宽与 6 个取值，以及
   `GET_TPL_TILING_KEY` 生成值是否与 kernel 的 `if constexpr` 分支一致（0..5）。
6. 与内置 TBE 算子在 `new_axis_mask` / `shrink_axis_mask` 组合下的细节是否完全一致；
   本实现按 TF `strided_slice` 语义处理，建议用内置算子对拍一次。
7. 性能：当前按元素标量 gather，正确性优先；大 shape 下吞吐不高，若平台有性能门槛需要改成
   连续段 `DataCopyPad` 的向量化版本。
8. `shape` 是 `ValueDepend` 输入。infershape 会优先使用调用方已经填好的输出 shape
   （aclnn 单算子模式），只有在输出维度未知时才去读 `shape` 张量的数据；如果真实环境里
   `-e` 在 infershape 阶段崩溃，说明该版本的 aclnn 把 `shape` 当作 device 张量且仍要读值，
   此时把 `examples/sample_support.h` 里 `shape` 输入改成 host 内存即可（kernel 本身不读它）。
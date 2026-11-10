# QuantUpdateScatter Ascend C 算子

目标环境：CANN 9.0.0，Ascend 910B。工程包含 `op_host`、`op_kernel`、`examples`、`tests`、`samples`，支持 `bash build.sh -e` 和 `bash build.sh -u`。

## 语义

算子只支持内置接口对应的 `reduce=update`、`axis=-2`、`quant_axis=-1` 组合：

```text
for p, idx in enumerate(indices):
    for k in range(updates.shape[axis]):
        var_out[p, ..., idx + k, :] =
            int8_rne(clip(updates[p, ..., k, :] / scale + zero_point))
```

- `quant_zero_points` 是可选输入；缺省时等价于 0。
- 未被写入的 `var_out` 元素保持 `var` 原值。
- 量化只在 fp32 中做除法/加法，使用 `CAST_RINT` 就近取偶，再裁剪到 `[-128, 127]`。
- `reciprocal_scale=true` 时先把 scale 取倒数，再执行除法。

## 数据类型

| 输入/输出 | dtype |
|---|---|
| var / var_out | int8 |
| indices | int32（fp16 profile）/ int64（bf16 profile） |
| updates | float16 / bfloat16 |
| quant_scales | float32（fp16 profile）/ bfloat16（bf16 profile） |
| quant_zero_points（可选） | int32（fp16 profile）/ bfloat16（bf16 profile） |

所有输入输出格式均为 ND。

## 构建与测试

```bash
source /path/to/cann-9.0.0/set_env.sh

# 编译、打包并跑通第一个 NPU 测试点
DEVICE_ID=0 bash build.sh -e

# host tiling UT + kernel CPU UT
NLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include \
GTEST_SOURCE_DIR=/path/to/googletest \
bash build.sh -u
```

`build.sh` 显式传递：

```text
-DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
-DASC_DIR=${ASCEND_HOME_PATH}/lib64/cmake
```

因此不依赖 `find_package(ASC)` 的默认搜索顺序。环境里同时存在 CANN 8.5.0 时，也不会误选旧版本头文件。

UT 的 GoogleTest 优先从 `GTEST_SOURCE_DIR` 源码编译；缺少时回退到系统 `GTest` 包。UT 使用 `Release`（除非调用者显式指定），并将 `CMAKE_DEBUG_POSTFIX` 置空，避免生成 `libgtestd.a` 与链接目标 `gtest` 不一致。`nlohmann/json.hpp` 缺失时会给出明确报错。

## 测试覆盖

- host tiling：两种 dtype profile、rank 2~5、宽行分块、无 zero_points、空 updates，以及 axis/quant_axis/长度/维度等非法场景。
- kernel CPU UT：fp16/bf16、int32/int64 indices、有无 zero_points、reciprocal_scale、rank 2~5、外层维度、非 32B 对齐行、宽行跨列 tile、极小尺寸和空 updates。
- 输出按解析后的 int8 逻辑值精确比较（阈值 0），不是原始二进制字节比较。

## 离线检查

没有 CANN 的机器可以运行：

```bash
python3 tests/selfcheck/static_check.py
python3 tests/selfcheck/fuzz_check.py --random 180
```

这两个脚本只检查工程结构、CANN 9.0.0 API 写法和调度模型与参考实现的一致性，**不能替代**真实 CANN 编译与 NPU 执行。

## 当前验证状态

本工程在无 CANN/NPU 的 Windows 主机上完成；已运行 Python 静态检查和调度模型差分检查，但尚未在真实 CANN 9.0.0 + Ascend 910B 环境编译或执行。首次上机时建议优先验证 kernel CPU UT、bf16 Cast、int64 indices 和宽行 `DataCopyPad` 路径。

# Kimi K3 Prefill KDA 适配报告

## 1. 变更目标

将 Kimi K3 prefill 阶段的 KDA 执行方式从：

```text
QKV 投影 → causal_conv1d → flash_kda
```

切换为：

```text
kda_prolog → flash_kda_dev
```

参考实现为 `k31_prefile_exmaple.py` 中的 `optimized_prefill()`。

## 2. 代码变更

当前提交：

```text
55d290def
```

远程分支：

```text
origin/releases/v0.30.0
```

主要文件：

- `vllm_ascend/ops/kimi_prefill_fused_kda.py`
- `vllm_ascend/ops/kimi_kda.py`

新增算子加载：

```python
ops.kda_prolog
ops.kda_prolog_metadata
ops.flash_kda_dev
ops.flash_kda_dev_metadata
```

新增算子加载和调用封装：

```python
run_prefill_kda_prolog()
```

运行时接入点为 `AscendKimiK3DeltaAttention._try_prefill_kda_prolog()`，由 `_kda_forward_impl()` 的纯 prefill 分支尝试触发。

## 3. 新路径触发条件

只有同时满足以下条件时才会进入 `kda_prolog + flash_kda_dev`：

- 纯 prefill；
- 没有 decode token；
- 没有 speculative decode token；
- 当前 DSL 提供四个新算子；
- 加载期已经生成 `_prefill_kda_weights`，其中包含按 Prolog ABI 切分并准备好的 NZ 权重；
- Prolog 输入权重均为 BF16；
- recurrent state copy 已完成初始化；
- prefill metadata 和 Conv metadata 完整；
- 当前不处于 `torch.compiler` 编译阶段。

当前仓库还没有生成 `_prefill_kda_weights` 的加载期逻辑，因此生产运行会安全地回退到原有 `causal_conv1d + flash_kda` 或 AscendC KDA 路径。这样可以避免把合并 checkpoint 权重、FP32 Conv 权重或二维 block table 误传给 Prolog。

## 4. Review 修复内容

已修复 review 中的运行安全问题：

- 移除不存在的 `self.f_a_proj`、`self.b_proj` 直接访问；
- 增加 Prolog 权重 bundle 门控，未完成 ABI 准备时不进入新路径；
- `cache_indices` 归一化为一维 `int32`；
- `initial_state_mode` 转换为 `int32`；
- `keep_meta` 同步压缩 `cu_seqlens`；
- 修正输出 gate 的 token 维切片，从 `g2[:, :T]` 改为 `g2[:T]`；
- 移除 module 上缓存 hidden states 的逻辑，避免显存驻留和 compile 副作用。

## 5. 数据流等价性

### 原始实现

```text
hidden_states
    ├── in_proj_qkvgfab
    ├── f_a_proj → f_b_proj
    ├── b_proj
    └── conv1d
             ↓
       q / k / v / gate / beta
             ↓
          flash_kda
             ↓
       output + recurrent_state
```

### 优化实现

```text
hidden_states
    ↓
kda_prolog
    ├── QKV projection
    ├── decay projection
    ├── beta projection
    ├── causal convolution
    ├── Q/K normalization
    ├── gate prefix calculation
    └── beta activation
             ↓
       q / k / v / prefix / beta
             ↓
       flash_kda_dev
             ↓
       output + recurrent_state
```

两条路径使用相同的 recurrent state cache、`prefill_state_indices`、`prefill_has_initial_state`、`cu_seqlens`、Conv cache、`A_log`、`dt_bias` 和 gate lower bound。优化路径完成后通过原有 `state_copy.scatter()` 写回 recurrent state。

## 6. 等价性结论

| 检查项 | 结论 |
|---|---|
| QKV 计算语义 | 设计上等价，由 `kda_prolog` 完成 |
| causal convolution | 设计上等价，由 `kda_prolog` 完成 |
| Q/K normalization | 由 `kda_prolog` / `flash_kda_dev` 完成 |
| gate prefix | 由 `kda_prolog` 输出 |
| beta 激活 | 由新组合算子完成 |
| recurrent state 输入 | 使用相同 state copy 逻辑 |
| recurrent state 输出 | 使用相同 scatter 逻辑 |
| 输出 layout | 目标为 `[1, T, H, D]` |
| 当前实际运行路径 | 仍是原有 FlashKDA/AscendC fallback |
| Prolog 路径数值等价性 | 尚未完成 Ascend NPU 实测 |
| bitwise 等价性 | 不保证 |
| 误差等价性 | 需要 NPU 实测确认 |

由于融合算子改变了计算顺序，即使数学语义一致，也可能存在 BF16 舍入误差。建议使用：

```python
torch.testing.assert_close(
    optimized.float(),
    reference.float(),
    atol=5e-3,
    rtol=5e-3,
)
```

不建议进行 bitwise 比较。

## 7. 已完成验证

已通过：

```bash
python3 -m py_compile \
  vllm_ascend/ops/kimi_kda.py \
  vllm_ascend/ops/kimi_prefill_fused_kda.py

ruff check \
  vllm_ascend/ops/kimi_kda.py \
  vllm_ascend/ops/kimi_prefill_fused_kda.py

git diff --check
```

## 8. 待完成工作和验证

功能性工作：

- 在加载期切分 `in_proj_qkvgfab` 的 QKV、decay A、beta 部分；
- 转置并转换为 Prolog 要求的 BF16 FRACTAL_NZ；
- 准备独立的 decay B 和 packed Conv 权重；
- 为每个 KDA 层建立可复用的 Prolog/Dev metadata context；
- 将 Prolog 分发上移到投影前，避免重复投影。

验证工作：

当前尚未完成 Ascend NPU 实测，包括：

- 原始路径与优化路径的输出误差对比；
- recurrent final state 对比；
- Conv history cache 对比；
- 多 batch 和变长序列验证；
- 混合 prefill/decode 验证；
- 长序列性能对比；
- FlashKDA 与 FlashKDADev 吞吐对比。

因此当前结论属于代码路径和数据契约层面的等价性，尚未达到数值回归验证级别。

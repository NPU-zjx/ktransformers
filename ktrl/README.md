# KTRL

KTRL 在 KT 原生推理和微调能力上实现同步在线 GRPO。`ktrl run` 由同一个控制器驱动 KT-SGLang rollout、奖励和优势计算、FSDP2 Actor 更新及权重发布。

## 多轮如何自动接续

1. [在线控制器](trainer/online_native.py) 从 base 策略 `v0` 启动 KT-SGLang TP8，从训练 Parquet 按 seed 打乱行号，每轮取下一批 8 道题，每题现场生成 4 答。每个选中的数据行只消费一次。数据本身的重复题按原始行保留。
2. [批次构造](trainer/online_batch.py) 直接读取服务返回的回答 token ID，计算数学奖励和组内 GRPO 优势。固定回答回放属于独立的 Actor 验证入口。
3. 控制器等 rollout 排空并释放其显存，再把当轮批次交给 [FSDP2 Actor](workers/engine/fsdp/actor_update.py)。Actor 在更新前计算 old logp，按 GAS4 完成反向、CPU/GPU 联合梯度裁剪和一次 AdamW 更新。常驻模式在同一次 Trainer 调用中保留八个 rank、模型、优化器和调度器，等待下一批。
4. Actor 每步导出模型权重。控制器恢复 rollout，通过[权重发布](weights/serving.py)将 CPU 专家直接加载到 CPU，并由原生 TP 加载器更新 GPU 参数。发布核对权重清单和八个 rank 的版本，重捕获 CUDA Graph。下一批用已发布的策略生成。整个批次优势均为零时记录并跳过优化器更新，策略版本保持不变。

训练和推理共用逐层专家放置。路由专家留在 CPU。常驻模式每步磁盘导出约 70.86 GiB，用于推理发布。多轮运行在确认新版发布后仅移除本次运行被替代的权重分片，保留发布清单、日志和压缩的原始生成证据。当前常驻模式没有进程退出后的续训入口。

## 入口与环境

从仓库根目录运行配置预览（Python 3.10+、PyYAML，模型和数据路径按实际环境配置）。

```bash
python3 -B -m ktrl run --config examples/ktrl/qwen36_35b_grpo.yaml --dry-run
```

当前使用 Qwen3.6-35B-A3B、八卡、8192 回答 token 上限、GAS4 和 [35B 配方](../examples/ktrl/qwen36_35b_grpo.yaml)。`trainer.steps` 或 `--steps` 指定要消费的批次数。全量 13,376 行对应 1,672 批。两步以上默认启用常驻 Actor。配方旁须有 `precision-gate.json` 和它引用的验收证据，门槛会核对配方和源码哈希。环境依赖原生 KT Actor 和 KT-SGLang Python 环境、SGLang 子模块、模型及数据挂载。宿主限制和路径见 [online_native.py](trainer/online_native.py)。

```bash
python3 -B -m ktrl run --config "$RECIPE" --profiling light \
  --resident-actor --output-dir "$KTRL_RUN_ROOT/rl-<fresh-run-name>"
```

## 证据与验收

2026-11-09，`rl-resident-a04` 控制器退出 0，两轮在线优化器更新为 `[1, 2]`。第二轮生成用 `v1`，八个 Actor rank 与优化器对象跨轮保持一致，第二轮反向前 204 个参数和优化器状态抽样位置逐值一致。最终用 `v2` 生成 32 token，八个 rollout rank 均有实际 CUDA Graph 回放。两轮端到端约 73 分 32 秒。

上述两轮是 KTRL1.0.1 的历史验收，证据位于 005 上的 `rl-resident-a04`。当前运行的 `accepted.json` 记录实际优化器更新数，`data-cursor.json` 记录消耗批数，`prompt-provenance.json` 记录数据哈希和原始行号顺序。每轮另保留阶段耗时、Actor 跨轮检查、发布证明、回放及显存采样。

训推 logp 偏差单独记录。默认与 GPU 对照一致，关闭重要性校正。可在 `algorithm.rollout_correction` 显式设置 `rollout_is: token` 和 `rollout_is_threshold: 2.0`，开启单独验证的 token TIS 选项。离线回归入口见 [tests/ktrl](../tests/ktrl/README.md)。

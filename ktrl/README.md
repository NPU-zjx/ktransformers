# KTRL

KTRL 在 KT 原生推理和微调能力上实现同步在线 GRPO。`ktrl run` 由同一个控制器驱动 KT-SGLang rollout、奖励和优势计算、FSDP2 Actor 更新及权重发布。005 已验收常驻 Actor 的两个完整在线训练步。

## 两轮如何自动接续

1. [在线控制器](trainer/online_native.py) 从 base 策略 `v0` 启动 KT-SGLang TP8，每轮按 8 题 × 4 答请求现场生成。两轮复用同一组题目，每轮回答均由当轮策略重新生成。
2. [批次构造](trainer/online_batch.py) 直接读取服务返回的回答 token ID，计算数学奖励和组内 GRPO 优势。固定回答回放属于独立的 Actor 验证入口。
3. 控制器等 rollout 排空并释放其显存，再把当轮批次交给 [FSDP2 Actor](workers/engine/fsdp/actor_update.py)。Actor 在更新前计算 old logp，按 GAS4 完成反向、CPU/GPU 联合梯度裁剪和一次 AdamW 更新。常驻模式在同一次 Trainer 调用中保留八个 rank、模型、优化器和调度器，等待下一批。
4. Actor 每步仅导出模型权重。控制器恢复 rollout，通过[权重发布](weights/serving.py)将 CPU 专家直接加载到 CPU，并由原生 TP 加载器更新 GPU 参数。发布核对权重清单和八个 rank 的版本，重捕获 CUDA Graph。第二轮用 `v1` 生成完整训练批次并执行 step 2，发布 `v2` 后再做一次生成验证。

训练和推理共用逐层专家放置。已验收配方的路由专家全部留在 CPU，发布时 GPU 中转字节为 0。常驻模式每步磁盘导出约 70.86 GiB，用于推理发布。需要在进程退出后续训时，需另行保存完整训练 checkpoint。

## 入口与环境

从仓库根目录运行配置预览（Python 3.10+、PyYAML，模型和数据路径按实际环境配置）。

```bash
python3 -B -m ktrl run --config examples/ktrl/qwen36_35b_grpo.yaml --dry-run
```

005 验收使用 Qwen3.6-35B-A3B、八卡、8192 回答 token 上限、GAS4 和 [35B 配方](../examples/ktrl/qwen36_35b_grpo.yaml)。以下是该实验环境的入口。`RECIPE` 指向通过验收的配方，旁边须有 `precision-gate.json` 和它引用的已有证据。门槛会核对配方、源码和证据哈希。环境还依赖原生 KT Actor 和 KT-SGLang Python 环境、SGLang 子模块、模型及数据挂载。宿主限制和环境路径见 [online_native.py](trainer/online_native.py)。

```bash
export KTRL_RUN_ROOT=/raid/data/zjx/kt-baseline/checks/ktrl-review-005-20261108
RECIPE="$KTRL_RUN_ROOT/resident-a04/recipe.yaml"
python3 -B -m ktrl run --config "$RECIPE" --steps 2 --profiling light \
  --resident-actor --output-dir "$KTRL_RUN_ROOT/rl-<fresh-run-name>"
```

## 005 验收结果

2026-11-09，`rl-resident-a04` 控制器退出 0，两轮在线优化器更新为 `[1, 2]`。第二轮生成用 `v1`，八个 Actor rank 与优化器对象跨轮保持一致，第二轮反向前 204 个参数和优化器状态抽样位置逐值一致。最终用 `v2` 生成 32 token，八个 rollout rank 均有实际 CUDA Graph 回放。两轮端到端约 73 分 32 秒。

证据位于 005 上的 `rl-resident-a04`。根目录 `accepted.json` 记录最终验收，`round-*/accepted.json` 和 `round-*/stages.jsonl` 记录逐轮版本与耗时，`round-*/actor/` 保存优化器和跨轮检查，`service/` 保存逐 rank 发布和回放证据，`resources.jsonl` 保存显存采样。

当前验收范围是两轮工程闭环。训推 logp 偏差仍单独记录，学习效果和长期数据迭代仍待验证。离线回归入口见 [tests/ktrl](../tests/ktrl/README.md)。

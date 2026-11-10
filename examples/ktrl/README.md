# KTRL 配方

KTRL 已实测跑通多轮在线 RL，没有两轮上限。当前正式验收进度和恢复证据见[运行快照](../../ktrl/docs/multi-round-status.md)；历史两轮测试不代表当前功能限制。

[qwen36_35b_grpo.yaml](qwen36_35b_grpo.yaml) 配置 Qwen3.6-35B-A3B、KT-SGLang TP8、HF/FSDP2 Actor 八 rank、每批 8 题 × 4 答、GAS4 和 8192 回答 token 上限。Actor 和 rollout 的路由专家均在 CPU。

在仓库根目录预览配方。将模型和数据路径设为实际环境的路径。

```bash
python3 -B -m ktrl run --config examples/ktrl/qwen36_35b_grpo.yaml --dry-run
```

`trainer.steps` 是从训练集消费的批数，CLI 的 `--steps` 可以覆盖它。全量 13,376 行用 1,672 批，optimizer_schedule_steps 应至少为批数乘以每批优化器更新次数。在线命令、运行环境及验收证据见 [KTRL README](../../ktrl/README.md)。

actor.ppo_mini_batch_size 默认等于 prompts_per_step，ppo_epochs 默认1。若8题×4答、mini=4、epochs=2，则每轮4次优化器更新，8rank下GAS应为2；若mini=8、epochs=2，则每轮2次更新，GAS仍为4。不要把GAS当作优化器更新次数。

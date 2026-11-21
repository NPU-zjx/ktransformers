# KTRL 配方

KTRL 已实测跑通多轮在线 RL，没有两轮上限。当前正式验收进度和恢复证据见[运行快照](../../ktrl/docs/multi-round-status.md)；历史两轮测试不代表当前功能限制。

[qwen36_35b_grpo.yaml](qwen36_35b_grpo.yaml) 配置 Qwen3.6-35B-A3B、KT-SGLang TP8、HF/FSDP2 Actor 八 rank、每批 8 题 × 4 答、GAS4 和 8192 回答 token 上限。Actor 和 rollout 的路由专家均在 CPU。

在仓库根目录预览配方。将模型和数据路径设为实际环境的路径。

```bash
python3 -B -m ktrl run --config examples/ktrl/qwen36_35b_grpo.yaml --dry-run
```

`trainer.steps` 是从训练集消费的批数，CLI 的 `--steps` 可以覆盖它。全量 13,376 行用 1,672 批，optimizer_schedule_steps 应至少为批数乘以每批优化器更新次数。在线命令、运行环境及验收证据见 [KTRL README](../../ktrl/README.md)。

actor.ppo_mini_batch_size 默认等于 prompts_per_step，ppo_epochs 默认1。若8题×4答、mini=4、epochs=2，则每轮4次优化器更新，8rank下GAS应为2；若mini=8、epochs=2，则每轮2次更新，GAS仍为4。不要把GAS当作优化器更新次数。

## 第五阶段（4090 / HotpotQA / Qwen3-30B-A3B）新配方

- [qwen3_30b_hotpotqa_grpo.yaml](qwen3_30b_hotpotqa_grpo.yaml) 面向 `Qwen3-30B-A3B`（`Qwen3MoeForCausalLM`）+ `HotpotQA (distractor)`（`data_source=hotpotqa`），冻结 `max_prompt_tokens=2560`、`max_response_tokens=512`、`overlong_buffer_tokens=0`、每批 `8` 题 × `4` 答、`ppo_mini_batch_size=4`、`ppo_epochs=1`、`GAS=2`（每轮 `2` 次优化器更新）。
- **当前验收状态**：截至 `kt-rl` 当前提交，仅完成了离线单元测试（55/55）、HotpotQA 预处理与 SFT `labels != -100` 长度审计（`scripts/ktrl/audit_sft_lengths.py`）以及 `--dry-run` 配方解析；**30B 模型在 4090 上的真实在线 RL（含 FSDP2 包装与显存/发布验证）尚待 Phase 3 实测验收**，本分支推送不代表正式实验已完成。

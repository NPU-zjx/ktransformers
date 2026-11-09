# KTRL 配方

[qwen36_35b_grpo.yaml](qwen36_35b_grpo.yaml) 配置 Qwen3.6-35B-A3B、KT-SGLang TP8、HF/FSDP2 Actor 八 rank、每批 8 题 × 4 答、GAS4 和 8192 回答 token 上限。Actor 和 rollout 的路由专家均在 CPU。

在仓库根目录预览配方。将模型和数据路径设为实际环境的路径。

```bash
python3 -B -m ktrl run --config examples/ktrl/qwen36_35b_grpo.yaml --dry-run
```

`trainer.steps` 是从训练集消费的批数，CLI 的 `--steps` 可以覆盖它。全量 13,376 行用 1,672 批，另将 `optimizer_schedule_steps` 设为至少 1,672。在线命令、运行环境及验收证据见 [KTRL README](../../ktrl/README.md)。

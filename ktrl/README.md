# KTRL

KTRL 在 KT 已有的推理和微调能力上组织 RL。后续同步 GRPO 由 KTRL 调用 rollout 生成回答，计算奖励和优势，调 Actor 更新，再将新策略发布给 rollout。veRL 仓库用于阅读目录组织和接口设计。配置、请求、调度及权重发布均属于 KTRL。

## M0 配置预览

在仓库根目录执行以下命令。环境需要 Python 3.10+ 和 PyYAML。

```bash
./bin/ktrl run --config examples/ktrl/qwen36_35b_grpo.yaml --dry-run
```

也可以使用 `python3 -B -m ktrl run --config examples/ktrl/qwen36_35b_grpo.yaml --dry-run`。CLI 输出 JSON，读取 YAML 和 HF 的 `config.json`，检查模型目录、后端源码入口和已指定的数据路径。相对路径按 YAML 所在目录解析。此入口只解析配置；模型加载、服务、训练和 profiler 的执行属于后续里程碑。

| 位置 | 职责 |
|---|---|
| `configs/recipe.py` | 解析 KTRL 配方，检查角色、输入、步数和诊断窗口。 |
| [rollout/sglang](workers/rollout/sglang/README.md) | 将 KTRL 生成配置映射为 KT-SGLang 参数。后续接收来自 KTRL 的 prompt 和采样设置，返回生成结果。 |
| [engine/fsdp](workers/engine/fsdp/README.md) | 解析 HF/FSDP2 Actor 和 KT SFT 参数。后续接收 KTRL 训练批次并更新专家和 GPU 参数。 |
| `weights/` | 后续组织 Actor 导出、分片映射、权重发布和各 rank 确认。 |
| [examples/ktrl](../examples/ktrl/README.md)、[tests/ktrl](../tests/ktrl/README.md) | 35B 配置示例与离线回归测试。 |

`expert_placement` 是 Actor 与 rollout 共同使用的专家放置策略。每个 MoE 层的同一 expert ID 在两端分配到相同设备类型；旧配置字段相互冲突时，两个入口均在启动服务前报错。当前 Full Actor 选用的 KT 路径要求路由专家驻留 CPU，默认每层 ID 0–255 在 CPU。推理禁用动态专家迁移和临时整层 GPU prefill。这样发布时可以按相同归属发送更新。进程间仍需同步新版参数，后续按方向统计专家权重、激活及梯度字节，并验收 CPU 专家权重不逐层往返 GPU。[KT-FT 图 1](../papers/kt-ft/ktransformers-finetune.pdf)

## M1 单请求 rollout

在 qjh005 上使用新的输出目录运行 `./bin/ktrl rollout --config examples/ktrl/qwen36_35b_grpo.yaml --output-dir /raid/data/zjx/kt-baseline/results/<新运行名>`。这个命令启动专用 KT-SGLang 容器，跳过 HTTP 热身，通过 KTRL 提交一条生成请求，收集各 TP rank 的 Graph replay，然后停止本次容器。它只解析模型和 rollout 配置。结果见 `request.json`、`response.json`、`phase/`、`exit.json` 和 `rollout-summary.json`。

## M2 固定批次 Actor 更新

`trainer/fixed_batch.py` 从已有回答和奖励构造同题多答批次，`trainer/grpo.py` 计算组内优势和按有效回答 token 聚合的策略损失。Actor 用初始模型重算旧概率，再调用 KT 原生微调路径。固定批次、两卡单层对照、八卡更新和独立 checkpoint 恢复分别有单独入口与输出目录。[qjh005 运行方法](../examples/ktrl/README.md)

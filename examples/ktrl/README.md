# KTRL 配方

[qwen36_35b_grpo.yaml](qwen36_35b_grpo.yaml) 对应 qjh005 常驻 Actor 两轮验收的配方。在 KT 仓库根目录预览：

```bash
./bin/ktrl run --config examples/ktrl/qwen36_35b_grpo.yaml --dry-run
```

该配方选择 KT-SGLang TP8、HF/FSDP2 + KT Full Actor、8 个训练 rank 和 2 次计划更新。每步 8 题 × 4 答、回答上限 8192 token、GAS4。CPU 专家使用 BF16 计算，非专家参数分片为 FP32。`expert_placement.gpu_experts_per_layer: 0` 使两端的路由专家全部在 CPU。在线执行与验收门槛见 [KTRL README](../../ktrl/README.md)。以下 M2、M3 命令是独立组件验证入口。

## M2 固定批次 Actor 验证

在 qjh005 的 KT 仓库根目录，`m2_qjh005.sh` 使用独立的 Actor 环境和该机的模型、数据挂载。各命令要求新的运行目录名。下面的标签用于说明依赖关系。

```bash
bash examples/ktrl/m2_qjh005.sh prepare batch1
bash examples/ktrl/m2_qjh005.sh layer layer1
bash examples/ktrl/m2_qjh005.sh update step1 batch1 layer1
bash examples/ktrl/m2_qjh005.sh restore reload1 batch1 step1
```

`prepare` 在 CPU 上核对 32 题 × 4 答的奖励、组内优势和有效 token。`layer` 在两张卡上核对一层 KT 专家与 BF16 参考的前向及梯度。`update` 在八张卡上用完整模型重放固定批次，每个 rank 累积 16 个微批，原生 Trainer 执行一次 AdamW 更新。`restore` 新建八个进程，从这一步的 checkpoint 读取参数、优化器和 RNG，核对固定输入的 logits 后退出。

证据位于 `/raid/data/zjx/kt-baseline/results/ktrl-m2-r01/`。`optimizer-started.json` 在调用优化器前独占创建，同一实验只允许一次有效更新。逐 rank 的 `events-rank*.jsonl` 记录已完成的反向。`update-rank*.json` 保存 CPU 专家及 GPU 参数抽样的梯度、前后值和 Adam 状态。

## M3 固定 checkpoint 热发布

在 qjh005 的仓库根目录执行 `python3 -B -m ktrl.workers.rollout.sglang.publication ktrl-m3-<新目录名>`。该验证入口启动专用 v0 服务，在同一服务中发布 M2 的 `update-a03/trainer/checkpoint-1`，保存同一输入在两个版本上的完整末 token logits、逐 rank 发布记录、Graph 回放计数和重复发布结果，随后关闭专用服务。

`ktrl/weights/checkpoint.py` 将 KT Full 专家文件按层读取到 CPU，交给原生 CPU 加载器。`ktrl/weights/serving.py` 在这个专用服务中接入权重更新，GPU 参数复用模型 TP 加载逻辑，重捕获 Graph 后返回发布确认，由 SGLang 清空缓存。中途异常会终止工作进程，保留现场供恢复。正式验收还需要单独记录故障注入和跨后端 logits 的容差检查。

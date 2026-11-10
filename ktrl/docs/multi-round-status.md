# 多轮在线 RL 正式运行证据快照

快照时间：2026-11-10 15:30，北京时间。本文记录实测进度，不是实时状态页。

KTRL 已跑通多轮自动在线 RL，没有两轮上限。当前运行累计 19 个有更新的 RL 轮次、38 次优化器更新；21 个数据批次已消费，含 2 个优势全零的跳过批次。第 22 个数据批次正在使用 v38 现场生成，将进入第 20 个有更新的轮次（若优势非零）。

## 配置与执行

- 模型：Qwen3.6-35B-A3B；KT-SGLang TP8，KT + FSDP2 Actor 8 rank。
- 每批 8 题 × 4 答，输出上限 8192，GAS4，ppo_mini_batch_size=8，ppo_epochs=2；每个有更新的批次执行两次优化器更新。
- old logp 在整批更新前由 Actor 重算并固定，所有 PPO epoch 共用；每轮现场生成新回答、计算奖励和优势。
- Actor 与优化器常驻；CPU 路由专家保持 CPU 放置，发布不经 GPU 中转。
- 从旧运行完整 checkpoint-20 恢复，继续完成第 11–19 个有更新的轮次。每 10 个有更新的轮次保存完整 checkpoint；快照时 checkpoint-40 尚未产生。
- 全量计划为 13,376 行 / 1,672 个数据批次；有更新的轮数可能因零优势跳过而少于批数。全量尚未结束。

## 可复核结果

`data-cursor.json`：batches_consumed=21，updates=19，optimizer_steps=38。

| 轮次 | 现场生成策略 | 本轮优化器次数 | 已发布策略 |
|---|---|---|---|
| 18 | v34 | 2 | v36 |
| 19 | v36 | 2 | v38 |
| 20（进行中） | v38 | 尚未更新 | 尚未发布 |

证据来自实际运行目录，不以程序中设计的 accepted.json 格式替代实测。随本文保存的 JSON 快照包含原始进度、18/19 轮验收记录、20 轮输入配置与阶段、运行来源记录及各源文件 SHA256。

## 溯源

服务器：qjh005。源码：`/raid/data/zjx/kt-baseline/ktransformers`。

结果：`/raid/data/zjx/kt-baseline/results/ktrl-formal-local-20261110-resume20-a01`。

恢复源：`/raid/data/zjx/kt-baseline/results/ktrl-formal-local-20261109-a01/round-10/actor/trainer/checkpoint-20`。

这些路径是实验溯源信息，不是运行要求。其他用户通过配方配置模型、数据与输出目录。

当前运行使用 KTRL1.0.3 提交加未提交的权重发布同步修复；准确 HEAD、Python 源码摘要与脏工作树标记见随附 JSON 的 source.json。不能将这次实测直接等同于未带修复的公开提交。此前 v22 发布遇到 NCCL 等待超时，修复发布阶段 CPU rendezvous 后从 checkpoint-20 恢复，旧记录保留。

## 验收边界

已验证多轮执行、优化器状态接续、完整 checkpoint 恢复以及更新权重用于后续现场生成。尚不宣称无缺陷、收敛、全量训练完成或训推数值一致。当前长跑每个 epoch 只有一个 minibatch，不独立覆盖所有多 minibatch 配方。历史两轮结果是早期验收记录，不是当前能力上限。

[原始记录快照](multi-round-status.json) · [KTRL 使用说明](../README.md)

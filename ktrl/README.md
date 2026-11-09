# KTRL

KTRL 在 KT 原生推理和微调能力上实现同步在线 GRPO。`ktrl run` 由同一个控制器驱动 KT-SGLang rollout、奖励和优势计算、FSDP2 Actor 更新及权重发布。

## 多轮如何自动接续

1. [在线控制器](trainer/online_native.py) 从 base 策略 `v0` 启动 KT-SGLang，从配置指定的训练 Parquet 按 seed 打乱行号，每轮按配方取下一批题并现场生成回答。每个选中的数据行只消费一次。数据本身的重复题按原始行保留。
2. [批次构造](trainer/online_batch.py) 直接读取服务返回的回答 token ID，计算数学奖励和组内 GRPO 优势。固定回答回放属于独立的 Actor 验证入口。
3. 控制器等 rollout 排空并释放其显存，再把当轮批次交给 [FSDP2 Actor](workers/engine/fsdp/actor_update.py)。Actor 默认在整批更新前一次性计算并固定所有回答的 old logp，然后按 ppo_mini_batch_size 和 ppo_epochs 循环；每个 minibatch 按 GAS 完成反向、CPU/GPU 联合梯度裁剪和一次 AdamW 更新。所有 epoch 共享同一份只读 old logp。常驻模式在同一次 Trainer 调用中保留模型、优化器和调度器。
4. Actor 完成本轮所有 minibatch/epoch 后导出一次模型权重。控制器恢复 rollout，通过[权重发布](weights/serving.py)将 CPU 专家直接加载到 CPU，并由原生 TP 加载器更新 GPU 参数。发布核对权重清单和每个 TP rank 的版本，重捕获 CUDA Graph。下一批用已发布的策略生成。整个批次优势均为零时记录并跳过优化器更新，策略版本保持不变。

训练和推理共用逐层专家放置。路由专家留在 CPU。常驻模式每步磁盘导出约 70.86 GiB，用于推理发布。多轮运行在确认新版发布后仅移除本次运行被替代的权重分片，保留发布清单、日志和压缩的原始生成证据。trainer.save_freq（0 表示关闭）指定每隔多少个已更新的 RL 轮次保存完整状态；其余轮次只导出模型。--resume-from 指向已成功发布、带 ktrl-resume.json 的完整 checkpoint，恢复时使用新的输出目录。模型、数据、优化器/PPO 配置必须匹配，trainer.steps 表示整个数据计划的批数。trainer.keep_full_checkpoints 默认保留本次运行最新 1 个完整 checkpoint，只在较新完整状态成功发布后删除过期二进制；JSON、日志保留。旧运行及恢复来源目录不自动删除。保存时需预留新旧完整状态同时存在的磁盘空间。此恢复路径新增，真实GPU验收进行中。

## 入口与环境

在单台 Linux GPU 主机上，按仓库的 KT 安装说明准备支持 KT 的 PyTorch、Transformers、Accelerate 和原生 SGLang 环境，安装 PyYAML 与 PyArrow，并初始化锁定的 `third_party/sglang` 子模块。当前 Actor 支持 KT 的 Qwen3.5 MoE 架构入口和 CPU 路由专家。

复制[示例配方](../examples/ktrl/qwen36_35b_grpo.yaml)，替换 `actor_rollout_ref.model.path` 和 `data.train_file`。模型目录需有 `config.json` 和本地权重。Parquet 需有聊天格式的 `prompt`、`data_source=math_dapo` 及 `reward_model.ground_truth`。路径支持绝对路径或相对配方目录。

```bash
python3 -B -m ktrl run --config ./my-recipe.yaml --dry-run
```

`trainer.steps` 或 `--steps` 指定要消费的批次数。每个 PPO minibatch 的回答数须等于 Actor rank 数 × GAS（每 rank 每微批一条回答）。actor.ppo_mini_batch_size 按 prompt 数配置，与 veRL 当前同步入口一致，实际乘以 responses_per_prompt；必须整除 prompts_per_step。每轮优化器次数 = prompts_per_step / ppo_mini_batch_size × ppo_epochs。trainer.optimizer_schedule_steps 需覆盖所有优化器更新；常数学习率调度器按 veRL 的轮末节奏推进。TP、Actor rank 数、题数、每题回答数和长度上限均从配方读取。两步以上默认启用常驻 Actor。

`runtime.python` 默认使用当前解释器。`runtime.actor_pythonpath`、`runtime.rollout_pythonpath` 和 `runtime.library_paths` 可追加用户自己的依赖目录。`runtime.cuda_visible_devices` 可指定 GPU，缺省继承环境。`runtime.export_min_free_gib` 可设定导出前的剩余磁盘下限。实验室可显式设置 `runtime.require_precision_gate: true`，用配方旁的验收文件锁定源码与证据。公共配方默认关闭此门槛。

```bash
python3 -B -m ktrl run --config ./my-recipe.yaml --output-dir ./runs/rl-a01
```

## 证据与验收

2026-11-09，`rl-resident-a04` 控制器退出 0，两轮在线优化器更新为 `[1, 2]`。第二轮生成用 `v1`，八个 Actor rank 与优化器对象跨轮保持一致，第二轮反向前 204 个参数和优化器状态抽样位置逐值一致。最终用 `v2` 生成 32 token，八个 rollout rank 均有实际 CUDA Graph 回放。两轮端到端约 73 分 32 秒。

上述两轮是 KTRL1.0.1 的历史验收。新运行的 `accepted.json` 记录实际优化器更新数，`data-cursor.json` 记录消耗批数，`prompt-provenance.json` 记录数据哈希和原始行号顺序。每轮另保留阶段耗时、Actor 跨轮检查、发布证明、回放及显存采样。

训推 logp 偏差单独记录。`algorithm.rollout_correction.bypass_mode` 默认 `false`，由 Actor 重算 old logp，与 GPU 对照一致。设为 `true` 时直接使用 KT-SGLang 已记录的逐 token logp，不使用 Actor 计算的概率作为 old；当前审计模式仍额外执行整批 Actor 前向以记录训推差异，因此不宣称这条模式已经节省前向开销。GRPO 的策略比率直接比较当前 Actor 与 rollout。`rollout_is: token` 是 Actor 重算模式的独立校正选项，不能和 bypass 同时开启。两个可选路径仍待 GPU 验收。运行 `python3 -B -m ktrl.trainer.metrics "$RUN_DIR"` 可汇总已完成批次的指标。[离线回归](../tests/ktrl/README.md)

## 当前变更验收状态

minibatch/ppo_epochs、整批冻结 old logp 和完整恢复为本轮新增功能；真实验证结果未完成前，不以历史两轮结果替代新代码验收。source.json 保存实际仓库路径、HEAD、工作树状态和 Python 源码摘要，launch.json 保存各子进程环境。运行中源码变化会阻止后续轮次，避免混用版本。配置中未实现的 ref/critic=true 会直接报错；不宣称支持参考模型 KL、奖励模型、多机或异步训练。

metrics.jsonl 自动记录已完成/跳过批次的奖励、回答正确率、长度、零信号比例、loss、梯度和训推差异。actor/optimizer-steps/ 保留每次优化器更新的证据，events-rank*.jsonl 记录各 minibatch 的 ratio 与实际 clip 比例。ratio=1 或 clip 比例0不是报错，不能为了触发clip伪造旧概率或修改奖励。

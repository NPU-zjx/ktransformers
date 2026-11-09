# KT-SGLang rollout

调用方向是 KTRL → rollout → KT 仓库随附的 SGLang。M0 的 `config.py` 接收 KTRL YAML 中的 rollout 配置，解析与原生 `sglang.launch_server` 对应的 `server_args`。后续由 KTRL 提交 prompt token 和采样参数，并消费回答 token、长度、结束原因以及按需计算的 log-prob。[KT 原生参数](../../../../third_party/sglang/python/sglang/srt/server_args.py)

KTRL 从同一个 `expert_placement` 生成 SGLang 的 `kt_num_gpu_experts` 和 Actor 的专家归属；均匀放置的 ID 划分与当前 fork 的 `generate_uniform_masks` 一致。生成侧关闭动态迁移和临时全层 GPU prefill。[KT 专家实现](../../../../third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py)

`smoke.py` 先用 `/v1/models` 核对服务身份，再打开并关闭一个空控制会话，启动 SGLang 的消息接收循环。随后用 `/server_info` 检查参数，只对固定的第 1 条 prompt 调用一次 `/generate`。验收核对请求 ID、完整输出 token ID、长度与结束原因。`qjh005.py` 管理这次容器的创建、启动和收尾。worker 计数在真实 `CudaGraphRunner.replay()` 返回后增加；验证窗口同时覆盖 prefill、decode 和每个 rank。权重发布、休眠/恢复、正式吞吐测量属于后续工作。

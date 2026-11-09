# KT-SGLang rollout

KTRL 的 [在线控制器](../../../trainer/online_native.py) 启动仓库随附的 KT-SGLang，按当前策略版本提交 prompt token 和采样参数，读取回答 token、结束原因和逐 token logp。每批生成结束后排空服务，休眠以释放显存；Actor 更新后恢复服务、发布权重并确认 CUDA Graph 回放。[KT 原生参数](../../../../third_party/sglang/python/sglang/srt/server_args.py)

KTRL 从同一个 `expert_placement` 生成 SGLang 的 `kt_num_gpu_experts` 和 Actor 的专家归属；均匀放置的 ID 划分与当前 fork 的 `generate_uniform_masks` 一致。生成侧关闭动态迁移和临时全层 GPU prefill。[KT 专家实现](../../../../third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py)

`ktrl rollout` 是独立的单请求诊断。在线训练从 `ktrl run` 进入，见 [KTRL README](../../../README.md)。

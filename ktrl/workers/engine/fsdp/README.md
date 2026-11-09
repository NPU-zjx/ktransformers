# HF/FSDP2 + KT Actor

KTRL 将训练批次交给 Actor，Actor 复用 HF、PyTorch FSDP2 和 KT 已有的微调能力。`config.py` 解析 FSDP2、rank 数、训练精度及专家放置。路由专家由 KT 在 CPU 上管理，attention、router 和共享路径等参数由 GPU 侧 FSDP2 管理。[KT 模型加载](../../../../kt-kernel/python/sft/wrapper.py)

Actor 和 rollout 使用同一个逐层专家放置。Actor 复用 KT SFT 的 `KTMoEFunction.backward` 和已绑定的专家优化器梯度路径；多 rank 的输入梯度与路由权重梯度按原路径返回。KTRL 连接 GRPO loss，核验有效 token 归一化、累积、裁剪和 owner 更新。发布阶段只传新参数，并按 CPU 专家和 GPU 参数的归属分别记传输字节。真实驻留和数值由 M2 验收。[KT 参数接口](../../../../kt-kernel/python/sft/lora.py) · [KT autograd](../../../../kt-kernel/python/sft/autograd.py)

[actor_update.py](actor_update.py) 把 GRPO 的 token loss 交给支持 KT 的 Transformers Trainer。常驻模式在每次更新后导出模型权重并等待控制器发来新批次。进入下一次反向前逐 rank 核验参数和 Adam 状态接续。固定批和单层对照是独立的诊断入口。

# HF/FSDP2 + KT Actor

KTRL 将训练批次交给 Actor，Actor 复用 HF、PyTorch FSDP2 和 KT 已有的微调能力。M0 的 `config.py` 解析 `strategy: fsdp2`、world size、Full BF16 训练和 KT 专家放置。当前 Full 配方使用 KT CPU 路由专家，attention、router、共享路径等非路由参数由 GPU 侧 FSDP2 管理。[KT 模型加载](../../../../kt-kernel/python/sft/wrapper.py)

Actor 和 rollout 使用同一个逐层专家放置。Actor 复用 KT SFT 的 `KTMoEFunction.backward` 和已绑定的专家优化器梯度路径；多 rank 的输入梯度与路由权重梯度按原路径返回。KTRL 连接 GRPO loss，核验有效 token 归一化、累积、裁剪和 owner 更新。发布阶段只传新参数，并按 CPU 专家和 GPU 参数的归属分别记传输字节。真实驻留和数值由 M2 验收。[KT 参数接口](../../../../kt-kernel/python/sft/lora.py) · [KT autograd](../../../../kt-kernel/python/sft/autograd.py)

M2 通过 [actor_update.py](actor_update.py) 加载完整 BF16 Actor，并把 GRPO 的 token loss 交给支持 KT 的 Transformers Trainer。优化器构建、梯度裁剪、专家更新和缓存刷新走 KT 微调路径。[layer_check.py](layer_check.py) 用可复算的一层先核对多 rank 变长路由和专家梯度。[restore.py](restore.py) 在新进程中验证 checkpoint，把参数和 Adam 的两个矩逐点与更新记录对照。

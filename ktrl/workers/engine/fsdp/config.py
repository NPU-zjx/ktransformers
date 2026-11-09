"""Resolve the HF/FSDP2 actor's planned GPU shards and KT CPU expert owner."""

from ktrl.configs.fields import fields, integer, require


def plan_actor(config, placement):
    label = "actor_rollout_ref.actor"
    fields(config, label, (
        "strategy", "world_size", "training_scope", "kt_num_gpu_experts",
        "kt_num_threads", "kt_threadpool_count", "nonexpert_parameter_dtype",
    ))
    strategy = require(config.get("strategy", "fsdp2"), f"{label}.strategy", ("fsdp2",))
    scope = require(config.get("training_scope", "full"), f"{label}.training_scope", ("full",))
    pools = integer(config.get("kt_threadpool_count", 1), f"{label}.kt_threadpool_count")
    return {
        "backend": "HF/FSDP2 + KT SFT",
        "strategy": strategy,
        "world_size": integer(config.get("world_size", 1), f"{label}.world_size"),
        "training_scope": scope,
        "nonexpert_parameter_dtype": require(config.get('nonexpert_parameter_dtype','float32'),
            f'{label}.nonexpert_parameter_dtype',('float32',)),
        'mixed_precision_policy':{'param_dtype':'bfloat16','reduce_dtype':'float32'},
        'cpu_expert_optimizer':{'name':'KTRL MasterAdamW','master_dtype':'float32',
            'moment_dtype':'float32','compute_copy_dtype':'bfloat16','gradient_buffer_dtype':'bfloat16','device':'cpu'},
        "fsdp_api": "torch.distributed.fsdp.fully_shard",
        "kt_loader": "kt_kernel.sft.wrapper.load_kt_model",
        "native_source": "kt-kernel/python/sft/wrapper.py",
        "kt_config": {
            "kt_backend": "AMXBF16",
            "kt_expert_weight_format": "bf16",
            "kt_train_mode": scope,
            "kt_num_gpu_experts": placement["gpu"]["count"],
            "kt_num_threads": integer(config.get("kt_num_threads", 48), f"{label}.kt_num_threads"),
            "kt_threadpool_count": pools,
            "kt_tp_enabled": pools > 1,
        },
        "expert_placement": placement,
        "ownership_target": {
            "cpu": "KT routed expert weights, gradients and optimizer state on the rank-0 owner",
            "gpu": "FSDP2-managed non-expert parameters including attention, router and shared path",
        },
    }

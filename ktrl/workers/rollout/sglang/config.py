"""Resolve KT-SGLang server settings without importing its CUDA runtime."""

from ktrl.configs.fields import fields, integer, path_from, require


def plan_rollout(config, model, recipe_dir, placement):
    label = "actor_rollout_ref.rollout"
    fields(config, label, (
        "name", "tp_size", "kt_method", "kt_weight_path", "kt_num_gpu_experts",
        "kt_expert_placement_strategy", "kt_cpuinfer", "kt_threadpool_count", "mem_fraction_static",
    ))
    require(config.get("name", "sglang"), f"{label}.name", ("sglang",))
    weights = path_from(config.get("kt_weight_path", model["path"]), f"{label}.kt_weight_path", recipe_dir)
    if not weights.is_dir():
        raise ValueError(f"{label}.kt_weight_path: missing directory {weights}")
    mem_fraction = config.get("mem_fraction_static", 0.35)
    if type(mem_fraction) not in (int, float) or not 0 < mem_fraction <= 1:
        raise ValueError(f"{label}.mem_fraction_static must be in (0, 1]")
    args = {
        "model_path": model["path"],
        "dtype": model["dtype"],
        "tp_size": integer(config.get("tp_size", 1), f"{label}.tp_size"),
        "kt_weight_path": str(weights),
        "kt_method": require(config.get("kt_method", "BF16"), f"{label}.kt_method", ("BF16",)),
        "kt_num_gpu_experts": placement["gpu"]["count"],
        "kt_expert_placement_strategy": placement["policy"],
        "kt_enable_dynamic_expert_update": False,
        "kt_gpu_prefill_token_threshold": 0,
        "kt_cpuinfer": integer(config.get("kt_cpuinfer", 32), f"{label}.kt_cpuinfer"),
        "kt_threadpool_count": integer(config.get("kt_threadpool_count", 2), f"{label}.kt_threadpool_count"),
        "disable_cuda_graph": False,
        "mem_fraction_static": mem_fraction,
    }
    return {
        "backend": "KT-SGLang",
        "native_entrypoint": "sglang.launch_server",
        "native_source": "third_party/sglang/python/sglang/launch_server.py",
        "server_args": args,
        "expert_placement": placement,
    }

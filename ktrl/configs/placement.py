"""One routed-expert placement contract for training and generation."""

from .fields import fields, integer, require, uniform_placement


def shared_placement(recipe, roles, model):
    label = "expert_placement"
    common = fields(recipe.get(label, {}), label, ("strategy", "gpu_experts_per_layer"))
    require(common.get("strategy", "uniform"), f"{label}.strategy", ("uniform",))
    total = model["routed_experts_per_moe_layer"]
    budgets = {}
    if "gpu_experts_per_layer" in common:
        key = f"{label}.gpu_experts_per_layer"
        budgets[key] = integer(common["gpu_experts_per_layer"], key, 0, total)

    # Earlier recipes used backend-specific fields. They can assert the shared
    # policy, but must never override it for just one role.
    for role in ("actor", "rollout"):
        options = roles.get(role, {})
        path = f"actor_rollout_ref.{role}"
        if not isinstance(options, dict):
            raise ValueError(f"{path} must be a mapping")
        if "kt_num_gpu_experts" in options:
            key = f"{path}.kt_num_gpu_experts"
            budgets[key] = integer(options["kt_num_gpu_experts"], key, 0, total)
        if "kt_expert_placement_strategy" in options:
            require(options["kt_expert_placement_strategy"], f"{path}.kt_expert_placement_strategy", ("uniform",))
    if len(set(budgets.values())) > 1:
        raise ValueError(f"Actor and rollout require the same expert placement; conflicting values: {budgets}")
    gpu = next(iter(budgets.values()), 0)
    if gpu:
        raise ValueError(f"Phase 1 Full Actor requires 0 GPU routed experts per layer for both roles; got {budgets}")
    return uniform_placement(total, gpu)

"""Resolve an offline KTRL plan from YAML and the model's config.json."""

import json
import shutil
from pathlib import Path

import yaml

from ktrl.workers.rollout.sglang.config import plan_rollout

from .fields import fields, integer, path_from, require
from .placement import shared_placement


REPO = Path(__file__).resolve().parents[2]


class RecipeLoader(yaml.SafeLoader):
    """Reject ambiguous YAML keys before they can replace a placement setting."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key, _ in node.value:
            name = self.construct_object(key)
            if not isinstance(name, str) or name in seen:
                raise ValueError(f"line {key.start_mark.line + 1}: invalid or duplicate key {name!r}")
            seen.add(name)
        return super().construct_mapping(node, deep=deep)


def read_model(config, recipe_dir):
    label = "actor_rollout_ref.model"
    fields(config, label, ("path", "dtype"))
    path = path_from(config.get("path"), f"{label}.path", recipe_dir)
    metadata = path / "config.json"
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{metadata}: {error}") from error
    architecture = "Qwen3_5MoeForConditionalGeneration"
    if not isinstance(raw, dict) or not isinstance(raw.get("architectures"), list) or architecture not in raw["architectures"]:
        raise ValueError(f"{metadata}: M0 expects the {architecture} architecture")
    text = raw.get("text_config")
    if not isinstance(text, dict):
        raise ValueError(f"{metadata}: missing text_config")
    count = integer(text.get("num_experts"), "text_config.num_experts")
    return {
        "path": str(path),
        "metadata": str(metadata),
        "architecture": architecture,
        "dtype": require(config.get("dtype", "bfloat16"), f"{label}.dtype", ("bfloat16",)),
        "text_layers": integer(text.get("num_hidden_layers"), "text_config.num_hidden_layers"),
        "routed_experts_per_moe_layer": count,
        "experts_per_token": integer(text.get("num_experts_per_tok"), "text_config.num_experts_per_tok", 1, count),
    }


def optional_role(config, label):
    fields(config, label, ("enabled", "strategy"))
    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError(f"{label}.enabled must be true or false")
    strategy = require(config.get("strategy", "fsdp2"), f"{label}.strategy", ("fsdp2",))
    if label == "critic" and enabled:
        raise ValueError("the Phase 1 GRPO recipe uses no critic")
    return {"enabled": enabled, "strategy": strategy}


def plan_inputs(config, recipe_dir):
    fields(config, "data", ("train_file", "prompts_per_step", "responses_per_prompt"))
    train_file = config.get("train_file")
    if train_file is not None:
        train_file = path_from(train_file, "data.train_file", recipe_dir)
        if not train_file.is_file():
            raise ValueError(f"data.train_file: missing file {train_file}")
    prompts = integer(config.get("prompts_per_step", 32), "data.prompts_per_step")
    answers = integer(config.get("responses_per_prompt", 4), "data.responses_per_prompt", 2)
    return {"train_file": str(train_file) if train_file else None, "prompts_per_step": prompts,
            "responses_per_prompt": answers, "responses_per_update": prompts * answers}


def profile_windows(preset):
    windows = {"lightweight": "formal rollout, actor update, weight publication and resource sampling"}
    if preset == "full":
        windows.update({
            "torch": "separate short prefill/decode and actor microbatch windows",
            "nsys": "separate bounded generation, actor and weight-publication timelines",
            "ncu": "one selected GPU hotspot per bounded diagnostic",
        })
    return {"preset": preset, "planned_windows": windows}


def _read_recipe(config_path):
    path = Path(config_path).resolve()
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=RecipeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: {error}") from error
    cfg = fields(raw, "recipe", ("algorithm", "actor_rollout_ref", "expert_placement", "critic", "data", "trainer", "profiling"))
    group = fields(cfg.get("actor_rollout_ref"), "actor_rollout_ref", ("model", "rollout", "actor", "ref"))
    return path, cfg, group


def load_rollout_config(config_path):
    """Select the inference role without requiring an actor or training dataset."""
    path, cfg, group = _read_recipe(config_path)
    model = read_model(group.get("model"), path.parent)
    placement = shared_placement(cfg, group, model)
    return plan_rollout(group.get("rollout", {}), model, path.parent, placement)


def load_plan(config_path, *, steps=None, profiling=None):
    from ktrl.workers.engine.fsdp.config import plan_actor

    path, cfg, group = _read_recipe(config_path)
    algorithm = fields(cfg.get("algorithm", {}), "algorithm", ("adv_estimator",))
    estimator = require(algorithm.get("adv_estimator", "grpo"), "algorithm.adv_estimator", ("grpo",))
    model = read_model(group.get("model"), path.parent)
    placement = shared_placement(cfg, group, model)
    rollout = plan_rollout(group.get("rollout", {}), model, path.parent, placement)
    actor = plan_actor(group.get("actor", {}), placement)
    for backend in (actor, rollout):
        if not (REPO / backend["native_source"]).is_file():
            raise ValueError(f"missing KT backend source: {REPO / backend['native_source']}")
    ref = optional_role(group.get("ref", {}), "actor_rollout_ref.ref")
    critic = optional_role(cfg.get("critic", {}), "critic")
    trainer = fields(cfg.get("trainer", {}), "trainer", ("steps",))
    num_steps = integer(trainer.get("steps", 1) if steps is None else steps, "trainer.steps", 1, 2)
    profile = fields(cfg.get("profiling", {}), "profiling", ("preset",))
    preset = require(profile.get("preset", "full") if profiling is None else profiling,
                     "profiling.preset", ("light", "full"))
    return {
        "mode": "dry-run",
        "orchestrator": "KTRL",
        "recipe": str(path),
        "algorithm": estimator,
        "model": model,
        "expert_placement": placement,
        "active_roles": ["rollout", "actor"] + (["ref"] if ref["enabled"] else []),
        "rollout": rollout,
        "actor": actor,
        "ref": ref,
        "critic": critic,
        "inputs": plan_inputs(cfg.get("data", {}), path.parent),
        "max_updates": num_steps,
        "profiling": profile_windows(preset),
        "resources": {
            "planned_gpu_ranks": {"rollout": rollout["server_args"]["tp_size"], "actor": actor["world_size"]},
            "role_runtimes": "separate processes",
            "checkout_disk_free_bytes": shutil.disk_usage(REPO).free,
        },
    }

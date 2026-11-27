"""Resolve an offline KTRL plan from YAML and the model's config.json."""

import json
import math
import shutil
import os
import sys
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



ARCHITECTURE_SPECS = {
    "Qwen3_5MoeForConditionalGeneration": {
        "auto_model_cls": "AutoModelForImageTextToText",
        "wrap_classes": ["Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock"],
        "expert_module_names": ("Qwen3_5MoeExperts",),
        "needs_vlm_conv3d_patch": True,
    },
    "Qwen3MoeForCausalLM": {
        "auto_model_cls": "AutoModelForCausalLM",
        "wrap_classes": ["Qwen3MoeDecoderLayer"],
        "expert_module_names": ("Qwen3MoeExperts",),
        "needs_vlm_conv3d_patch": False,
    },
}


def architecture_spec(architecture: str) -> dict:
    if architecture not in ARCHITECTURE_SPECS:
        raise ValueError(f"Unsupported KTRL architecture: {architecture!r}")
    spec = ARCHITECTURE_SPECS[architecture]
    return {
        "auto_model_cls": spec["auto_model_cls"],
        "wrap_classes": list(spec["wrap_classes"]),
        "expert_module_names": tuple(spec["expert_module_names"]),
        "needs_vlm_conv3d_patch": spec["needs_vlm_conv3d_patch"],
    }


def read_model(config, recipe_dir):
    label = "actor_rollout_ref.model"
    fields(config, label, ("path", "dtype"))
    path = path_from(config.get("path"), f"{label}.path", recipe_dir)
    metadata = path / "config.json"
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{metadata}: {error}") from error
    supported = ("Qwen3_5MoeForConditionalGeneration", "Qwen3MoeForCausalLM")
    archs = raw.get("architectures") if isinstance(raw, dict) else None
    matched = [a for a in supported if isinstance(archs, list) and a in archs]
    if not matched:
        raise ValueError(f"{metadata}: KTRL currently requires one of {supported}")
    architecture = matched[0]
    if architecture == "Qwen3_5MoeForConditionalGeneration":
        text = raw.get("text_config")
        if not isinstance(text, dict):
            raise ValueError(f"{metadata}: missing text_config")
        prefix = "text_config."
    else:
        text = raw
        prefix = ""
    count = integer(text.get("num_experts"), f"{prefix}num_experts")
    return {
        "path": str(path),
        "metadata": str(metadata),
        "architecture": architecture,
        "dtype": require(config.get("dtype", "bfloat16"), f"{label}.dtype", ("bfloat16",)),
        "text_layers": integer(text.get("num_hidden_layers"), f"{prefix}num_hidden_layers"),
        "routed_experts_per_moe_layer": count,
        "experts_per_token": integer(text.get("num_experts_per_tok"), f"{prefix}num_experts_per_tok", 1, count),
    }


def optional_role(config, label):
    fields(config, label, ("enabled", "strategy"))
    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError(f"{label}.enabled must be true or false")
    strategy = require(config.get("strategy", "fsdp2"), f"{label}.strategy", ("fsdp2",))
    if enabled:
        raise ValueError(f"{label}.enabled=true is unsupported by KTRL online GRPO")
    return {"enabled": enabled, "strategy": strategy}


def plan_inputs(config, recipe_dir):
    fields(config, "data", ("train_file", "validation_file", "prompts_per_step", "responses_per_prompt", "max_response_tokens", "max_prompt_tokens", "context_length", "seed", "overlong_buffer_tokens"))
    train_file = config.get("train_file")
    if train_file is not None:
        train_file = path_from(train_file, "data.train_file", recipe_dir)
        if not train_file.is_file():
            raise ValueError(f"data.train_file: missing file {train_file}")
    validation_file = config.get("validation_file")
    if validation_file is not None:
        validation_file = path_from(validation_file, "data.validation_file", recipe_dir)
        if not validation_file.is_file():
            raise ValueError(f"data.validation_file: missing file {validation_file}")
    prompts = integer(config.get("prompts_per_step", 8), "data.prompts_per_step")
    answers = integer(config.get("responses_per_prompt", 4), "data.responses_per_prompt", 2)
    response = integer(config.get('max_response_tokens',8192),'data.max_response_tokens')
    prompt = integer(config.get('max_prompt_tokens',1024),'data.max_prompt_tokens')
    context = integer(config.get('context_length',9216),'data.context_length',prompt+response)
    return {"train_file": str(train_file) if train_file else None,
            "validation_file": str(validation_file) if validation_file else None, "prompts_per_step": prompts,
            "responses_per_prompt": answers, "responses_per_update": prompts * answers,
            'max_response_tokens':response,'max_prompt_tokens':prompt,'context_length':context,
            'seed':integer(config.get('seed',42),'data.seed',0),
            'overlong_buffer_tokens':integer(config.get('overlong_buffer_tokens',0),'data.overlong_buffer_tokens',0,response)}


def plan_reward(config, recipe_dir):
    fields(config, "reward", ("name", "database_root", "timeout_seconds", "max_rows",
                              "max_result_bytes", "float_abs_tol", "float_rel_tol"))
    name = require(config.get("name", "auto"), "reward.name", ("auto", "sql_execution"))
    if name == "auto":
        if set(config) - {"name"}:
            raise ValueError("SQL reward options require reward.name=sql_execution")
        return {"name": name}
    root = path_from(config.get("database_root"), "reward.database_root", recipe_dir)
    if not root.is_dir():
        raise ValueError(f"reward.database_root: missing directory {root}")
    def number(key, default, positive=False):
        value = config.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"reward.{key} must be finite and {'positive' if positive else 'nonnegative'}")
        return value
    timeout = number("timeout_seconds", 2.0, True)
    if timeout > 60:
        raise ValueError("reward.timeout_seconds must be <= 60")
    return {"name": name, "database_root": str(root), "timeout_seconds": timeout,
            "max_rows": integer(config.get("max_rows", 100000), "reward.max_rows", 1, 1000000),
            "max_result_bytes": integer(config.get("max_result_bytes", 8388608), "reward.max_result_bytes", 1, 134217728),
            "float_abs_tol": number("float_abs_tol", 1e-6), "float_rel_tol": number("float_rel_tol", 1e-6)}


def profile_windows(preset):
    windows = {"lightweight": "formal rollout, actor update, weight publication and resource sampling"}
    if preset == "full":
        windows.update({
            "torch": "separate short prefill/decode and actor microbatch windows",
            "nsys": "separate bounded generation, actor and weight-publication timelines",
            "ncu": "one selected GPU hotspot per bounded diagnostic",
        })
    return {"preset": preset, "planned_windows": windows}


def plan_runtime(config, recipe_dir):
    fields(config, "runtime", ("python", "actor_pythonpath", "rollout_pythonpath", "library_paths",
                               "cuda_visible_devices", "require_precision_gate", "export_min_free_gib",
                               "final_probe_tokens", "actor_cuda_allocator_conf"))
    interpreter = config.get("python", sys.executable)
    if not isinstance(interpreter, str) or not interpreter.strip():
        raise ValueError("runtime.python must be a nonempty path")
    # Keep the venv entry point, which is often a symlink to a base interpreter.
    python = Path(os.path.abspath(recipe_dir / Path(interpreter).expanduser()))
    if not python.is_file():
        raise ValueError(f"runtime.python: missing interpreter {python}")
    paths = {}
    for key in ("actor_pythonpath", "rollout_pythonpath", "library_paths"):
        values = config.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"runtime.{key} must be a list of directories")
        directories = [path_from(value, f"runtime.{key}", recipe_dir) for value in values]
        if any(not directory.is_dir() for directory in directories):
            raise ValueError(f"runtime.{key} contains a missing directory")
        paths[key] = [str(directory) for directory in directories]
    inherited_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = config.get("cuda_visible_devices", "" if inherited_devices == "-1" else inherited_devices)
    if devices is not None and devices != "":
        parts = devices.split(",") if isinstance(devices, str) else []
        if (not parts or len(set(parts)) != len(parts)
                or any(not part.isascii() or not (part.isdecimal() or part.startswith("GPU-"))
                       for part in parts)):
            raise ValueError("runtime.cuda_visible_devices must be comma-separated GPU indexes or UUIDs")
    gate = config.get("require_precision_gate", False)
    if type(gate) is not bool:
        raise ValueError("runtime.require_precision_gate must be true or false")
    actor_allocator = config.get("actor_cuda_allocator_conf")
    if actor_allocator is not None and (not isinstance(actor_allocator, str) or not actor_allocator.strip()):
        raise ValueError("runtime.actor_cuda_allocator_conf must be a nonempty string")
    return dict(actor_cuda_allocator_conf=actor_allocator,
                python=str(python), cuda_visible_devices=devices, require_precision_gate=gate,
                final_probe_tokens=integer(config.get("final_probe_tokens", 32), "runtime.final_probe_tokens", 2),
                export_min_free_gib=integer(config.get("export_min_free_gib", 0),
                                            "runtime.export_min_free_gib", 0), **paths)


def _read_recipe(config_path):
    path = Path(config_path).resolve()
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=RecipeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: {error}") from error
    cfg = fields(raw, "recipe", ("algorithm", "actor_rollout_ref", "expert_placement", "critic", "data", "trainer", "profiling", "runtime", "reward"))
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
    algorithm = fields(cfg.get("algorithm", {}), "algorithm", ("adv_estimator", "rollout_correction"))
    estimator = require(algorithm.get("adv_estimator", "grpo"), "algorithm.adv_estimator", ("grpo",))
    correction = fields(algorithm.get("rollout_correction", {}), "algorithm.rollout_correction",
                        ("bypass_mode", "rollout_is", "rollout_is_threshold"))
    rollout_is = require(correction.get("rollout_is"), "algorithm.rollout_correction.rollout_is", (None, "token"))
    bypass = correction.get("bypass_mode", False)
    if type(bypass) is not bool:
        raise ValueError("algorithm.rollout_correction.bypass_mode must be true or false")
    if bypass and rollout_is is not None:
        raise ValueError("PPO-clip bypass_mode already uses the rollout ratio; disable rollout_is")
    threshold = correction.get("rollout_is_threshold", 2.0)
    if type(threshold) not in (int, float) or not 0 < threshold < float("inf"):
        raise ValueError("rollout_is_threshold must be finite and positive")
    model = read_model(group.get("model"), path.parent)
    placement = shared_placement(cfg, group, model)
    rollout = plan_rollout(group.get("rollout", {}), model, path.parent, placement)
    actor = plan_actor(group.get("actor", {}), placement)
    for backend in (actor, rollout):
        if not (REPO / backend["native_source"]).is_file():
            raise ValueError(f"missing KT backend source: {REPO / backend['native_source']}")
    ref = optional_role(group.get("ref", {}), "actor_rollout_ref.ref")
    critic = optional_role(cfg.get("critic", {}), "critic")
    trainer = fields(cfg.get("trainer", {}), "trainer", ("steps", "gradient_accumulation_steps", "optimizer_schedule_steps", "save_freq", "keep_full_checkpoints"))
    num_steps = integer(trainer.get("steps", 1) if steps is None else steps, "trainer.steps")
    inputs = plan_inputs(cfg.get('data', {}),path.parent)
    mini_prompts = actor["ppo_mini_batch_size"] or inputs["prompts_per_step"]
    if inputs["prompts_per_step"] % mini_prompts:
        raise ValueError("prompts_per_step must be divisible by actor.ppo_mini_batch_size")
    mini_responses = mini_prompts * inputs["responses_per_prompt"]
    if mini_responses % actor["world_size"]:
        raise ValueError("PPO minibatch responses must be divisible by Actor world_size")
    expected_gas = mini_responses // actor["world_size"]
    gas = integer(trainer.get("gradient_accumulation_steps", expected_gas), "trainer.gradient_accumulation_steps")
    if gas != expected_gas:
        raise ValueError("GAS must equal ppo_mini_batch_size * responses_per_prompt / world_size (microbatch=1)")
    actor["ppo_mini_batch_size"] = mini_prompts
    actor["minibatch_responses"] = mini_responses
    actor["minibatches_per_epoch"] = inputs["prompts_per_step"] // mini_prompts
    actor["optimizer_steps_per_rollout"] = actor["minibatches_per_epoch"] * actor["ppo_epochs"]
    profile = fields(cfg.get("profiling", {}), "profiling", ("preset",))
    preset = require(profile.get("preset", "full") if profiling is None else profiling,
                     "profiling.preset", ("light", "full"))
    runtime = plan_runtime(cfg.get("runtime", {}), path.parent)
    return {
        "mode": "dry-run",
        "orchestrator": "KTRL",
        "recipe": str(path),
        "algorithm": estimator,
        "reward": plan_reward(cfg.get("reward", {}), path.parent),
        "rollout_correction": {"bypass_mode": bypass, "rollout_is": rollout_is, "rollout_is_threshold": threshold},
        "runtime": runtime,
        "model": model,
        "expert_placement": placement,
        "active_roles": ["rollout", "actor"],
        "rollout": rollout,
        "actor": actor,
        "ref": ref,
        "critic": critic,
        "inputs": inputs,
        "gradient_accumulation_steps": gas,
        "optimizer_schedule_steps": integer(trainer.get('optimizer_schedule_steps',max(418, num_steps * actor['optimizer_steps_per_rollout'])),
                                            'trainer.optimizer_schedule_steps',num_steps * actor['optimizer_steps_per_rollout']),
        "keep_full_checkpoints": integer(trainer.get("keep_full_checkpoints", 1), "trainer.keep_full_checkpoints"),
        "save_freq": integer(trainer.get("save_freq", 0), "trainer.save_freq", 0),
        "max_updates": num_steps,
        "profiling": profile_windows(preset),
        "resources": {
            "planned_gpu_ranks": {"rollout": rollout["server_args"]["tp_size"], "actor": actor["world_size"]},
            "role_runtimes": "separate processes",
            "checkout_disk_free_bytes": shutil.disk_usage(REPO).free,
        },
    }

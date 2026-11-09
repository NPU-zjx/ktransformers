"""Published boundary: model, optimizer and dataset cursor agree."""
import json
from pathlib import Path

def contract(plan):
    return {key: plan[key] for key in ("model", "expert_placement", "actor", "inputs",
                                      "rollout_correction", "gradient_accumulation_steps",
                                      "optimizer_schedule_steps")}

def load_resume(checkpoint, plan):
    checkpoint = Path(checkpoint).resolve()
    for name in ("kt_optimizer.index.json", "scheduler.pt", "trainer_state.json", "ktrl-resume.json"):
        if not (checkpoint/name).is_file():
            raise ValueError(f"Full published resume checkpoint missing {name}: {checkpoint}")
    data = json.loads((checkpoint/"ktrl-resume.json").read_text())
    if data["contract"] != contract(plan):
        raise ValueError("Resume model/data/optimizer/PPO configuration differs from checkpoint")
    state = json.loads((checkpoint/"trainer_state.json").read_text())
    if state["global_step"] != data["optimizer_steps"]:
        raise ValueError("Resume optimizer step differs from data cursor")
    if data["optimizer_steps"] != data["completed_rounds"] * plan["actor"]["optimizer_steps_per_rollout"]:
        raise ValueError("Resume must be at a complete PPO rollout boundary")
    if not 0 < data["batches_consumed"] < plan["max_updates"]:
        raise ValueError("Resume cursor must leave unconsumed dataset batches")
    return data


def retire_old_full_checkpoints(root, keep):
    """Only retire this run's obsolete binary states after a newer durable save."""
    root = Path(root).resolve()
    candidates = sorted(root.glob("round-*/actor/trainer/checkpoint-*/ktrl-resume.json"),
                        key=lambda p: json.loads(p.read_text())["optimizer_steps"])
    for marker in candidates[:-keep]:
        checkpoint = marker.parent
        if checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(root):
            raise ValueError("Unsafe checkpoint retention path")
        binaries = [p for p in checkpoint.rglob("*") if p.suffix in (".pt", ".bin", ".safetensors") and p.is_file()]
        if any(p.is_symlink() or not p.resolve().is_relative_to(checkpoint.resolve()) for p in binaries):
            raise ValueError("Unsafe checkpoint binary path")
        # Publish the retired marker first: it must never be offered for resume.
        marker.rename(checkpoint/"ktrl-resume-retired.json")
        for p in binaries:
            p.unlink()
        (checkpoint/"retired-files.json").write_text(json.dumps([str(p.relative_to(checkpoint)) for p in binaries]))

"""Read completed online batches and emit one metrics JSON row per batch."""

import argparse
import json
from pathlib import Path
from statistics import fmean


def read_json(path):
    return json.loads(path.read_bytes())


def batch_metrics(out, path):
    batch = read_json(path / "round.json")
    audit = read_json(path / "batch" / "batch-audit.json")
    rewards = read_json(path / "batch" / "rewards.json")
    token_counts = {row["uid"]: row["response_tokens"] for row in rewards}
    skipped = (path / "skipped.json").is_file()
    row = {
        "dataset_batch": batch["dataset_batch"],
        "generated_policy": batch["policy_version"],
        "old_logp_source": audit.get("old_logp_source", "actor"),
        "optimizer_step": None,
        "skipped": skipped,
        "mean_reward": fmean(item["reward"] for item in rewards),
        "answer_accuracy": fmean(item["score"] > 0 for item in rewards),
        "mean_response_tokens": fmean(token_counts.values()),
        "at_output_limit_fraction": fmean(n == batch["max_response_tokens"] for n in token_counts.values()),
        "zero_advantage_groups": audit["zero_advantage_groups"],
        "nonzero_advantage_responses": audit["nonzero_advantage_responses"],
    }
    if skipped:
        return row
    actor = path / "actor"
    accepted = read_json(actor / "accepted.json")
    plan = read_json(out / "resolved-plan.json")
    losses = []
    losses_by_step = {}
    ratios = []
    logp_diffs = {}
    for events in sorted(actor.glob("events-rank*.jsonl")):
        with events.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["phase"] == "backward_start":
                    losses.append(event["loss"])
                    losses_by_step.setdefault(event.get("optimizer_step", accepted.get("start_step", accepted["end_step"]-1)), []).append(event["loss"])
                elif event["phase"] == "ppo_ratio":
                    ratios.append(event)
                elif event["phase"] == "actor_rollout_logprob_difference":
                    logp_diffs[event["uid"]] = event["mean_abs"]
    if len(losses) != len(rewards) * accepted.get("ppo_epochs", 1):
        raise ValueError(f"Loss evidence does not cover every response in {path}")
    world = plan["actor"]["world_size"]
    gas = plan["gradient_accumulation_steps"]
    row.update(
        optimizer_step=accepted["end_step"],
        loss=sum(losses) / (world * gas * accepted.get("updates", 1)),
        optimizer_steps=accepted.get("updates", 1),
        loss_by_optimizer_step={str(k+1):sum(v)/(world*gas) for k,v in losses_by_step.items()},
        ratio_nonunit_tokens=sum(e["nonunit_tokens"] for e in ratios),
        ratio_clip_fraction=(sum(e["clip_fraction"]*e["tokens"] for e in ratios)/sum(e["tokens"] for e in ratios) if ratios else None),
        grad_norm=read_json(actor / "preoptimizer-rank0.json")["grad_norm_preclip"],
        token_mean_abs_actor_rollout_logp_diff=(
            sum(logp_diffs[uid] * n for uid, n in token_counts.items()) / sum(token_counts.values())
            if logp_diffs.keys() == token_counts.keys() else None
        ),
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    out = parser.parse_args().run_dir
    batches = [
        p for pattern, marker in (("round-*", "accepted.json"), ("skipped-batch-*", "skipped.json"))
        for p in out.glob(pattern) if (p / marker).is_file()
    ]
    for row in sorted((batch_metrics(out, p) for p in batches), key=lambda row: row["dataset_batch"]):
        print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()

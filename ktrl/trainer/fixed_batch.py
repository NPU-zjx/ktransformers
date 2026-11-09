"""Freeze text-policy replay inputs and independently check recorded GRPO advantages."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .grpo import outcome_advantages


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(model_path, parquet, rollout, reference, folder):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    historical = json.loads(reference.read_bytes())
    groups = [tuple(r["prompt"]) for r in historical["rows"]]
    rewards = [r["reward"] for r in historical["rows"]]
    historical_adv = outcome_advantages(groups, rewards, 4)
    error = 0.0
    valid = 0
    for row, adv in zip(historical["rows"], historical_adv):
        assert len(row["response"]) == len(row["old_log_probs"]) == len(row["advantages"])
        error = max(error, max(abs(a - adv) for a in row["advantages"]))
        valid += len(row["response"])
    assert len(set(groups)) == 8 and valid == historical["valid_tokens"] and error < 2e-6

    records = [json.loads(line) for line in rollout.read_text().splitlines()]
    groups = [r["input"] for r in records]
    scores = [r["score"] for r in records]
    assert len(records) == len({r["uid"] for r in records}) == 128 and len(set(groups)) == 32
    advantages = outcome_advantages(groups, scores, 4)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompts = {}
    wanted = set(groups)
    for messages in pq.read_table(parquet, columns=["prompt"]).column("prompt").to_pylist():
        if len(messages) != 1 or messages[0]["role"] != "user":
            continue
        display = "user\n" + messages[0]["content"] + "\nassistant\n<think>\n"
        if display not in wanted or display in prompts:
            continue
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        if tokenizer.decode(encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False) != display:
            raise ValueError("Dataset chat template does not reproduce a recorded prompt")
        prompts[display] = list(encoded)
    if set(prompts) != wanted:
        raise ValueError(f"Recovered only {len(prompts)}/32 original prompts from the training dataset")
    rows = []
    by_group = defaultdict(list)
    for source, adv in zip(records, advantages):
        response = tokenizer.encode(source["output"], add_special_tokens=False)
        if tokenizer.decode(response, clean_up_tokenization_spaces=False) != source["output"]:
            raise ValueError(f"Response text did not round-trip: {source['uid']}")
        prompt = prompts[source["input"]]
        assert 0 < len(prompt) <= 1024 and 0 < len(response) <= 20480
        group = hashlib.sha256(source["input"].encode()).hexdigest()
        rows.append(dict(uid=source["uid"], group=group, prompt=prompt, response=response,
                         reward=source["score"], advantage=adv))
        by_group[group].append((source["score"], adv))
    assert all(abs(math.fsum(a for _, a in vals)) < 1e-5 for vals in by_group.values())
    mixed = next(vals for vals in by_group.values() if len({s for s, _ in vals}) > 1)
    equal = next(vals for vals in by_group.values() if len({s for s, _ in vals}) == 1)
    assert all(a == 0 for _, a in equal)
    path = folder / "fixed-rows.json"
    with path.open("x") as f:
        json.dump(rows, f, separators=(",", ":"), allow_nan=False)
    summary = dict(
        source_files={str(p): digest(p) for p in (parquet, rollout, reference)},
        fixed_sha256=digest(path), records=len(rows), questions=len(by_group),
        group_size_histogram=dict(Counter(len(vals) for vals in by_group.values())),
        valid_response_tokens=sum(len(r["response"]) for r in rows),
        max_sequence=max(len(r["response"]) + len(r["prompt"]) for r in rows),
        nonzero_advantage_responses=sum(r["advantage"] != 0 for r in rows),
        mixed_group=mixed, constant_group=equal,
        historical_8x4=dict(valid_tokens=valid, max_abs_error=error),
        provenance="32x4 recorded rollout texts and rewards, tokenizer template verified. Response IDs reconstructed from recorded text. No EOS inferred. Old-policy probabilities must be recomputed from the initial base actor.",
    )
    (folder / "batch-audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    for key in ("model", "parquet", "rollout", "reference", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    prepare(args.model, args.parquet, args.rollout, args.reference, args.out)

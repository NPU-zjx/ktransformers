"""Read-only acceptance of a real run; no fabricated data or model calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def verify(root, minimum_rounds=2):
    root = root.resolve()
    accepted = read(root/"accepted.json")
    assert read(root/"completion.json")["status"] == "complete"
    assert accepted["real_online_updates"] >= minimum_rounds
    plan = read(root/"resolved-plan.json")
    updates_per_round = plan["actor"]["optimizer_steps_per_rollout"]
    ranks = plan["actor"]["world_size"]
    epochs = plan["actor"]["ppo_epochs"]
    records = []
    previous_resident = None
    for folder in sorted(root.glob("round-*")):
        if not (folder/"accepted.json").is_file():
            continue
        actor = folder/"actor"
        proof = read(actor/"accepted.json")
        assert proof["updates"] == updates_per_round
        assert proof["end_step"] - proof["start_step"] == updates_per_round
        assert read(folder/"accepted.json")["generated_policy"] == f'v{proof["start_step"]}'
        assert read(folder/"accepted.json")["published_policy"] == f'v{proof["end_step"]}'
        live = read(folder/"rollout-policy-proof.json")
        assert live["version"] == f'v{proof["start_step"]}'
        assert all(counts.get(live["version"],0)>0 for counts in live["rank_replays"])
        rows = read(folder/"batch/fixed-rows.json")
        counts = Counter()
        ratio_by_step = defaultdict(list)
        frozen = []
        residents = []
        for rank in range(ranks):
            freeze = read(actor/f"old-logp-frozen-rank{rank}.json")
            assert freeze["optimizer_step"] == proof["start_step"]
            assert freeze["responses"] == len(rows)
            frozen.append(freeze["hashes"])
            residents.append(read(actor/f"resident-rank{rank}.json"))
            for line in (actor/f"events-rank{rank}.jsonl").read_text().splitlines():
                event = json.loads(line)
                if event["phase"] == "backward_start":
                    counts[event["uid"]] += 1
                if event["phase"] == "ppo_ratio":
                    ratio_by_step[event["optimizer_step"]+1].append(event)
            for step in range(proof["start_step"]+1,proof["end_step"]+1):
                update = read(actor/f"optimizer-steps/step-{step}/update-rank{rank}.json")
                assert all(x["step"] == step for x in update)
        assert all(x == frozen[0] for x in frozen)
        for uid,digest in frozen[0].items():
            assert hashlib.sha256((actor/"old_logp"/f"{uid}.pt").read_bytes()).hexdigest() == digest
        assert counts == Counter({row["uid"]:epochs for row in rows})
        assert len(ratio_by_step) == updates_per_round
        if updates_per_round > 1:
            assert sum(e["nonunit_tokens"] for step,es in ratio_by_step.items()
                       if step > proof["start_step"]+1 for e in es) > 0
        if previous_resident:
            for rank,(before,after) in enumerate(zip(previous_resident,residents)):
                resumed = read(actor/f"resume-rank{rank}.json")
                assert before["pid"] == after["pid"] == resumed["pid"]
                assert before["optimizer_id"] == after["optimizer_id"]
                assert resumed["exact_parameter_and_moment_positions"] > 0
        previous_resident = residents
        summary=[]
        for step,es in sorted(ratio_by_step.items()):
            tokens=sum(e["tokens"] for e in es)
            summary.append(dict(optimizer_step=step,
                ratio_min=min(e["ratio_min"] for e in es),ratio_max=max(e["ratio_max"] for e in es),
                nonunit_tokens=sum(e["nonunit_tokens"] for e in es),
                clip_fraction=sum(e["clip_fraction"]*e["tokens"] for e in es)/tokens))
        records.append(dict(round=folder.name,actor=proof,ratio=summary,
            frozen_old_logp_verified=True,
            source_response_sha256=read(folder/"batch/batch-audit.json")["fixed_sha256"]))
    return dict(accepted=True,source=read(root/"source.json"),run=accepted,rounds=records)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--minimum-rounds",type=int,default=2)
    args=p.parse_args()
    print(json.dumps(verify(args.run,args.minimum_rounds),indent=2,allow_nan=False))

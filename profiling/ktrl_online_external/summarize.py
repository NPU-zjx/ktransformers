"""Validate and summarize the external traces from one completed online round."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median, quantiles


def read(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def percentile(values):
    return {"p50": median(values), "p95": quantiles(values, n=20, method="inclusive")[18]}


def summarize_trace(path):
    rows = read(path)["traceEvents"]
    groups = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    for row in rows:
        category = row.get("cat", "")
        if row.get("ph") != "X" or "dur" not in row:
            continue
        if category not in {"cpu_op", "kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime"}:
            continue
        group = groups[(category, row["name"])]
        group["calls"] += 1
        group["total_us"] += row["dur"]
    categories = {category for category, _ in groups}
    assert {"cpu_op", "kernel"} <= categories, f"Missing CPU or CUDA activity in {path}"
    return [{"category": category, "name": name, **value,
             "mean_us": value["total_us"] / value["calls"]}
            for (category, name), value in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, help="New output file; never overwrite an earlier summary")
    args = parser.parse_args()
    root = args.root
    run = root / "rl-online-profile01"
    round_dir = run / "round-01"
    acceptance = read(round_dir / "accepted.json")
    actor = read(round_dir / "actor/accepted.json")
    assert acceptance["updates"] == actor["updates"] == 1
    assert acceptance["generated_policy"] == "v0" and acceptance["published_policy"] == "v1"
    stages = {r["phase"]: r for r in read_lines(round_dir / "stages.jsonl")}
    assert all(row["status"] == "complete" for row in stages.values())
    reports = root / "profiles"
    output = {"source": str(run), "stages": stages, "torch": {}, "missing": [],
              "rollout_tps": acceptance["output_tokens"] / stages["live_rollout_v0"]["seconds"]}
    for label in ("rollout-extend-tp0", "rollout-decode-tp0", "actor-microbatch-rank0"):
        traces = list(reports.glob(f"{label}-*.json"))
        if len(traces) != 1:
            output["missing"].append(f"{label}: {len(traces)} traces")
            continue
        try:
            output["torch"][label] = summarize_trace(traces[0])
        except (AssertionError, OSError, ValueError, KeyError) as error:
            output["missing"].append(f"{label}: {error}")

    requests = {path.name.removesuffix("-request.json") for path in
                (round_dir / "rollout").glob("*-request.json")}
    timing_rows = [row for p in reports.glob("scheduler-latency-*.jsonl") for row in read_lines(p)]
    by_request = {}
    conflicting = set()
    fields = ("rid", "first", "queued", "finished", "tokens")
    for row in timing_rows:
        previous = by_request.setdefault(row['rid'], row)
        if any(previous[field] != row[field] for field in fields):
            conflicting.add(row['rid'])
    output['scheduler_raw_records'] = len(timing_rows)
    output['scheduler_unique_requests'] = len(by_request)
    output['scheduler_conflicting_requests'] = sorted(conflicting)
    # The scheduler may observe a finished request twice. Deduplicate only
    # identical semantic timestamps and token counts, never conflicting rows.
    timing_rows = [row for rid, row in by_request.items() if rid not in conflicting]
    valid = [row for row in timing_rows if row["rid"] in requests and
             0 < row["queued"] <= row["first"] <= row["finished"] and row["tokens"] > 1]
    if len(valid) == 32 and {row["rid"] for row in valid} == requests:
        output["scheduler_latency"] = {
            "requests": 32,
            "queue_to_first_token_ms": percentile([(r["first"] - r["queued"]) * 1000 for r in valid]),
            "first_to_last_ms_per_subsequent_token": percentile(
                [(r["finished"] - r["first"]) * 1000 / (r["tokens"] - 1) for r in valid]),
            "scope": "server scheduler; excludes HTTP transport and tokenizer ingress",
        }
    else:
        output["missing"].append(f"scheduler latency coverage: {len(valid)}/32")
    output["profiler_errors"] = [row for path in reports.glob("profiler-errors-*.jsonl")
                                 for row in read_lines(path)]
    with (args.output or root / "profile-summary.json").open("x") as stream:
        json.dump(output, stream, indent=2)


if __name__ == "__main__":
    main()

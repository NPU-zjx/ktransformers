"""Summarize the real CPU calls and CUDA kernel events by vLLM phase and rank."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
import sys

from analysis.analyze_requests import percentile


def write(path, rows):
    with path.open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run):
    folder = run / "phase"
    out = run / "derived"
    out.mkdir(exist_ok=True)
    stats = defaultdict(list)
    coverage = defaultdict(set)
    shapes = []
    for path in sorted(folder.glob("*.trace.json.gz")):
        _, rank, phase, segment = path.name.split(".")[0].split("-")
        with gzip.open(path, "rt") as trace:
            events = json.load(trace)["traceEvents"]
        for event in events:
            cat = event.get("cat")
            if event.get("ph") == "X" and cat in {"cpu_op", "user_annotation", "kernel"}:
                stats[rank, phase, cat, event["name"]].append(event["dur"])
                coverage[rank, phase].add(cat)
        with (folder / f"TP-{rank}-{phase}-{segment}.csv").open() as src:
            for row in csv.DictReader(src):
                shapes.append(dict(rank=rank, phase=phase, segment=segment, **row))
    rows = []
    for (rank, phase, category, name), durations in sorted(stats.items()):
        rows.append(dict(rank=rank, phase=phase, category=category, name=name,
                         calls=len(durations), total_us=sum(durations),
                         avg_us=sum(durations)/len(durations), p50_us=percentile(durations, .5),
                         p95_us=percentile(durations, .95)))
    write(out / "torch-calls.csv", rows)
    write(out / "torch-shapes.csv", shapes)
    for rank in map(str, range(8)):
        for phase in ("EXTEND", "DECODE"):
            if not {"cpu_op", "kernel"}.issubset(coverage[rank, phase]):
                raise ValueError(f"Missing CPU/GPU events, rank {rank} phase {phase}")


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))

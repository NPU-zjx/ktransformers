"""Export phase-level memory, timing, and the process/device sampling summaries."""
import csv
import json
import sys
from pathlib import Path

from analysis import analyze_resources
from analysis.analyze_phases import union_ns


def export(run):
    output = run/"derived"
    analyze_resources.analyze(run, output)
    traces = [json.loads(path.read_text()) for path in sorted((run/"phase").glob("phase-TP*.json"))]
    intervals = []
    phase_intervals = {}
    rows = []
    for trace in traces:
        anchor = (trace["lower"] + trace["upper"]) // 2
        for event in trace["rows"]:
            if event["context_tokens"] + event["generation_tokens"] == 0:
                continue
            span = tuple(anchor + round(event[key]*1e6) for key in ("stream_start_ms", "stream_end_ms"))
            intervals.append(span)
            for field in ("context_tokens", "generation_tokens"):
                if event[field]:
                    phase_intervals.setdefault(field, []).append(span)
            rows.append(dict(rank=trace["rank"], **event))
    with (output/"phase-events.csv").open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    union = union_ns(intervals)/1e9
    wall = json.loads((run/"analysis/client-summary.json").read_text())["rollout_seconds"]
    coverage = dict(compute_window_union_seconds=union, client_seconds=wall,
                    phase_overlap_seconds=sum(union_ns(v)/1e9 for v in phase_intervals.values()) - union,
                    client_seconds_outside_compute_union=wall-union)
    (output/"phase-overlap.json").write_text(json.dumps(coverage, indent=2))
    samples = [json.loads(line) for line in (run/"telemetry/samples.jsonl").read_text().splitlines()]
    memory = []
    for trace in traces:
        rank = trace["rank"]
        base = trace["epoch_ns"] - trace["clock_ns"] + (trace["lower"] + trace["upper"])//2
        for label, field in (("prefill", "context_tokens"), ("decode", "generation_tokens")):
            selected = [e for e in trace["rows"] if e[field]]
            spans = [(base+e["stream_start_ms"]*1e6, base+e["stream_end_ms"]*1e6) for e in selected]
            observed = [s for s in samples if any(a <= s["time_unix"]*1e9 <= b for a,b in spans)]
            values = [int(s["gpus"][rank]["memory.used"]*2**20) for s in observed]
            memory.append(dict(rank=rank, phase=label, samples=len(values),
                               device_last_bytes=values[-1] if values else None,
                               device_peak_bytes=max(values) if values else None))
    with (output/"phase-device-memory.csv").open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(memory[0]))
        writer.writeheader()
        writer.writerows(memory)


if __name__ == "__main__":
    export(Path(sys.argv[1]))

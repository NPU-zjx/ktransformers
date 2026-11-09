"""Bounded diagnostic replay of the formal 32 prompts, separate from reference TPS."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

from analysis.analyze_operators import analyze
from client import inputs
from torch_profile import call, wait_ready


def capture(mode, run, container, url):
    wait_ready(run, container)
    info = call(url, "/server_info")
    if info["disable_cuda_graph"] or info["tp_size"] != 8 or any(info.get(k) for k in
            ("lora_paths", "enable_lora", "kt_expert_lora_path", "kt_lora_path", "kt_composite_lora_name")):
        raise RuntimeError("Expected base TP8 with CUDA Graph enabled")
    rows = inputs()
    folder = run / "capture"
    folder.mkdir(exist_ok=False)
    (folder / "server-info.json").write_text(json.dumps(info, indent=2))
    sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 32, "ignore_eos": False}
    profile = {"output_dir": "/output/capture"}
    if mode == "torch":
        profile.update(activities=["CPU", "GPU"], num_steps=8, profile_by_stage=True, record_shapes=True, with_stack=False)
    else:
        profile.update(activities=["CUDA_PROFILER"])
    (folder / "window.json").write_text(json.dumps(dict(profile=profile, sampling=sampling,
        requests=len(rows), input_tokens=sum(len(r["prompt"]) for r in rows)), indent=2))
    call(url, "/flush_cache", {})
    call(url, "/start_profile", profile)
    try:
        raw = call(url, "/generate", {"input_ids": [r["prompt"] for r in rows], "sampling_params": sampling})
        (folder / "response.json").write_text(json.dumps(raw))
        if len(raw) != 32 or not all(r["meta_info"]["finish_reason"] for r in raw):
            raise RuntimeError("Incomplete batch response")
    finally:
        if mode != "torch":
            call(url, "/stop_profile", {})
    seen = {}
    for path in folder.glob("phase-*.json"):
        trace = json.loads(path.read_text())
        seen.setdefault(trace["rank"], []).extend(trace["rows"])
    if set(seen) != set(range(8)):
        raise RuntimeError("Missing TP workers")
    counts = {}
    for rank, events in seen.items():
        pre = [r for r in events if r["phase"] == "EXTEND"]
        dec = [r for r in events if r["phase"] == "DECODE"]
        if not pre or not dec or not all(r["graph_replayed"] for r in dec):
            raise RuntimeError(f"Incomplete phase/replay coverage on TP{rank}")
        counts[rank] = {"prefill_batches": len(pre), "prefill_tokens": sum(r["tokens"] for r in pre),
                        "decode_batches": len(dec), "largest_decode_batch": max(r["tokens"] for r in dec)}
    if len({tuple(v.values()) for v in counts.values()}) != 1:
        raise RuntimeError("Rank counts disagree")
    if mode != "torch" and counts[0]["prefill_tokens"] != 5712:
        raise RuntimeError("Incomplete prefill token coverage")
    (folder / "coverage.json").write_text(json.dumps(counts, indent=2))
    if mode == "torch":
        analyze(folder)
    else:
        report = run / ("rollout.nsys-rep" if mode == "nsys" else "hotspot.ncu-rep")
        for _ in range(180):
            if report.is_file() and report.stat().st_size:
                return
            time.sleep(1)
        raise TimeoutError(report)


def validate_ncu(run):
    with (run / "raw-metrics.csv").open() as f:
        headers, units, values = list(csv.reader(f))
    data = dict(zip(headers, values))
    metrics = ["gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
               "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "sm__warps_active.avg.pct_of_peak_sustained_active"]
    selected = {k: float(data[k].replace(",", "")) for k in metrics}
    if not all(math.isfinite(v) for v in selected.values()) or selected[metrics[0]] <= 0 or "FusedAddRMSNormKernel" not in data["Kernel Name"] or "kt_extend" not in str(values):
        raise RuntimeError("Invalid NCU capture")
    (run / "selected-metrics.json").write_text(json.dumps({k: {"value": v, "unit": units[headers.index(k)]}
                                                           for k, v in selected.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("torch", "nsys", "ncu"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--container")
    parser.add_argument("--url")
    parser.add_argument("--validate-ncu", type=Path)
    args = parser.parse_args()
    if args.validate_ncu:
        validate_ncu(args.validate_ncu)
    else:
        capture(args.mode, args.run_dir, args.container, args.url)

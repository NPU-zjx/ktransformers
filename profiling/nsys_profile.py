"""Open a bounded cudaProfilerApi window around one base request."""
import argparse
import json
from pathlib import Path
import time

from client import inputs
from torch_profile import call, wait_ready


def capture(run, container, url):
    wait_ready(run, container)
    info = call(url, "/server_info")
    if info["tp_size"] != 8 or info["disable_cuda_graph"] or any(info.get(k) for k in
            ("enable_lora", "lora_paths", "kt_expert_lora_path", "kt_lora_path", "kt_composite_lora_name")):
        raise RuntimeError("Expected base TP8 with decode CUDA Graph enabled")
    folder = run / "request"
    folder.mkdir(exist_ok=False)
    prompt = inputs()[0]["prompt"]
    if len(prompt) != 127:
        raise ValueError("Expected the fixed step23 diagnostic input")
    (folder / "server-info.json").write_text(json.dumps(info, indent=2))
    call(url, "/flush_cache", {})
    call(url, "/start_profile", {"activities": ["CUDA_PROFILER"], "output_dir": "/output/request"})
    try:
        response = call(url, "/generate", {"input_ids": prompt, "sampling_params": {
            "temperature": 0, "max_new_tokens": 32, "ignore_eos": False}})
        (folder / "response.json").write_text(json.dumps(response))
    finally:
        call(url, "/stop_profile", {})
    count = response["meta_info"]["completion_tokens"]
    if not response["meta_info"].get("finish_reason") or not 2 <= count <= 32:
        raise RuntimeError("The short diagnostic request did not complete a usable decode window")
    traces = [json.loads(p.read_text()) for p in folder.glob("phase-*.json")]
    if len(traces) != 8 or {t["rank"] for t in traces} != set(range(8)):
        raise RuntimeError("Missing TP workers")
    for trace in traces:
        prefill = [r for r in trace["rows"] if r["phase"] == "EXTEND"]
        decode = [r for r in trace["rows"] if r["phase"] == "DECODE"]
        if sum(r["tokens"] for r in prefill) != len(prompt) or not decode or len(decode) < count - 1 or not all(r["graph_replayed"] for r in decode):
            raise RuntimeError(f"Missing phase coverage or Graph replay at rank {trace['rank']}")
    report = run / "rollout.nsys-rep"
    for _ in range(120):
        if report.is_file() and report.stat().st_size:
            return
        time.sleep(1)
    raise TimeoutError(f"Missing Nsight report; see {run}/server.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--container", required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    capture(args.run_dir, args.container, args.url)

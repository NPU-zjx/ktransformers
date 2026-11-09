"""Capture one base request using SGLang's native stage-specific Torch profiler."""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from client import inputs
from analysis.analyze_operators import analyze


def call(url, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        text = response.read().decode()
    try:
        return json.loads(text)
    except ValueError:
        return text


def wait_ready(run, name):
    cid = (run / "container.id").read_text().strip()
    deadline = time.monotonic() + 1800
    docker = ["bash", "/raid/data/zjx/gpu-baseline/scripts/docker/client.sh"]
    while time.monotonic() < deadline:
        info = json.loads(subprocess.check_output(docker + ["inspect", name], text=True))[0]
        if info["Id"] != cid or not info["State"]["Running"]:
            raise RuntimeError(f"Diagnostic container stopped or changed; see {run}/server.log")
        log = run / "server.log"
        if log.exists() and "The server is fired up and ready to roll!" in log.read_text(errors="replace"):
            return
        time.sleep(5)
    raise TimeoutError("Diagnostic service readiness timeout")


def verify_window(folder):
    observed = set()
    for path in folder.glob("phase-*.json"):
        trace = json.loads(path.read_text())
        rows = trace["rows"]
        phases = {row["phase"] for row in rows}
        if phases == {"EXTEND"} and len(rows) == 1 and rows[0]["tokens"] == 127:
            phase = "EXTEND"
        elif phases == {"DECODE"} and len(rows) == 8 and all(r["graph_replayed"] for r in rows):
            phase = "DECODE"
        else:
            raise ValueError(f"Unexpected batch coverage or missing graph replay: {path.name}")
        key = (trace["rank"], phase)
        if key in observed:
            raise ValueError(f"Duplicate rank/phase: {key}")
        observed.add(key)
    if observed != {(rank, stage) for rank in range(8) for stage in ("EXTEND", "DECODE")}:
        raise ValueError(f"Incomplete TP8 coverage: {observed}")


def capture(run, name, url):
    wait_ready(run, name)
    info = call(url, "/server_info")
    if info["tp_size"] != 8 or info["disable_cuda_graph"] or any(info.get(k) for k in
            ("enable_lora", "lora_paths", "kt_expert_lora_path", "kt_lora_path", "kt_composite_lora_name")):
        raise RuntimeError("Expected base TP8 with CUDA Graph enabled")
    prompt = inputs()[0]["prompt"]
    if len(prompt) != 127:
        raise ValueError("Diagnostic input differs from step23")
    folder = run / "torch-diagnostic"
    folder.mkdir(exist_ok=False)
    (folder / "server-info.json").write_text(json.dumps(info, indent=2))
    call(url, "/flush_cache", {})
    profile = dict(output_dir="/output/torch-diagnostic", activities=["CPU", "GPU"],
                   num_steps=8, profile_by_stage=True, record_shapes=True, with_stack=False)
    (folder / "profile-config.json").write_text(json.dumps(profile, indent=2))
    call(url, "/start_profile", profile)
    response = call(url, "/generate", dict(input_ids=prompt, sampling_params={
        "temperature": 0, "max_new_tokens": 32, "ignore_eos": False}))
    (folder / "response.json").write_text(json.dumps(response))
    # Native stage profiling auto-stops on the ninth decode batch.
    verify_window(folder)
    analyze(folder)
    print(f"Operator tables: {folder / 'analysis'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--container", required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    capture(args.run_dir.resolve(), args.container, args.url.rstrip("/"))

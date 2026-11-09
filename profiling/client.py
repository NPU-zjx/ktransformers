"""Validate the base service, then measure at most one 32-request rollout."""
import concurrent.futures
import contextlib
import fcntl
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

from analysis import analyze_phases, analyze_requests, analyze_resources

HERE = Path(__file__).resolve().parent
RUN = Path("/raid/data/zjx/kt-baseline/results/base-r01")
INPUT_SHA = "64f543ad9d067508e9eab2d07a783c91f37f9f836913c4dcf9e217613db472a7"


def inputs():
    raw = (HERE / "data/jiaheng-prompts32.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise ValueError("Input file differs from the step23 baseline")
    rows = json.loads(raw)["rows"]
    if len(rows) != 32 or sum(len(r["prompt"]) for r in rows) != 5712:
        raise ValueError("Expected 32 requests and 5712 prompt tokens")
    return rows


def service_id():
    docker = ["bash", "/raid/data/zjx/gpu-baseline/scripts/docker/client.sh"]
    info = json.loads(subprocess.check_output(docker + ["inspect", CONTAINER], text=True))[0]
    mount_ok = any(m["Destination"] == "/output" and Path(m["Source"]).resolve() == RUN.resolve()
                   for m in info["Mounts"])
    if not info["State"]["Running"] or not mount_ok:
        raise RuntimeError(f"Expected running {CONTAINER} with {RUN} mounted at /output")
    return info["Id"]


def write(name, data):
    path = RUN / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def call(path, payload=None):
    req = urllib.request.Request(URL + path, headers={"Content-Type": "application/json"},
                                 data=None if payload is None else json.dumps(payload).encode())
    with urllib.request.urlopen(req, timeout=300) as response:
        raw = response.read().decode()
    try:
        return json.loads(raw)
    except ValueError:
        return raw


@contextlib.contextmanager
def capture(folder):
    call("/start_profile", {"output_dir": "/output/" + folder, "activities": ["KT_PHASE"],
                           "record_shapes": False, "with_stack": False})
    try:
        yield
    finally:
        call("/stop_profile", {})


def warmup(row):
    call("/flush_cache", {})
    with capture("warmup-phase"):
        result = call("/generate", {"input_ids": row["prompt"], "sampling_params": {
            "temperature": 0, "max_new_tokens": 64, "ignore_eos": False}})
    write("warmup-response.json", result)
    count = result["meta_info"]["completion_tokens"]
    if count < 2:
        raise RuntimeError("Warmup did not exercise decode")
    analyze_phases.analyze(RUN / "warmup-phase", len(row["prompt"]), count - 1, RUN / "warmup-analysis")
    call("/flush_cache", {})


def generate(index, row, barrier):
    req = urllib.request.Request(URL + "/generate", headers={"Content-Type": "application/json"},
        data=json.dumps({"input_ids": row["prompt"], "sampling_params": SAMPLING, "stream": True}).encode())
    barrier.wait(timeout=60)
    start = time.perf_counter_ns()
    events = []
    previous = 0
    final = None
    with urllib.request.urlopen(req, timeout=14400) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            final = json.loads(body)
            now = time.perf_counter_ns()
            if "error" in final:
                raise RuntimeError(final["error"])
            count = int(final["meta_info"]["completion_tokens"])
            if count < previous:
                raise ValueError("Token count decreased")
            if count > previous:
                events.append([now, count, count - previous])
            previous = count
    end = time.perf_counter_ns()
    if not final or not final["meta_info"]["finish_reason"] or not previous:
        raise RuntimeError("Incomplete generation")
    result = dict(index=index, uid=row["uid"], prompt_tokens=len(row["prompt"]), output_tokens=previous,
                  start_ns=start, end_ns=end, events_ns_tokens_delta=events, final=final)
    write(f"baseline-request-{index:02d}.json", result)
    return result


def rollout():
    rows = inputs()
    cid = service_id()
    for filename in ("state.json", "launch.json"):
        write(filename, {"run_id": "base-r01", "container_id": cid})
    deadline = time.monotonic() + 1800
    while True:
        if service_id() != cid:
            raise RuntimeError("Container changed during startup")
        log = RUN / "server.log"
        if log.exists() and "The server is fired up and ready to roll!" in log.read_text(errors="replace"):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Service startup timed out; see {log}")
        time.sleep(5)
    info = call("/server_info")
    expected = {"disable_cuda_graph": False, "tp_size": 8}
    mismatch = {k: (v, info.get(k)) for k, v in expected.items() if info.get(k) != v}
    if mismatch or any(info.get(k) for k in ("enable_lora", "lora_paths", "kt_expert_lora_path", "kt_lora_path", "kt_composite_lora_name")):
        raise RuntimeError(f"Expected resolved base configuration; differences: {mismatch}")
    write("server-info.json", info)
    write("request-settings.json", {"sampling": SAMPLING, "prompts_sha256": INPUT_SHA, "url": URL})
    warmup(rows[0])
    write("current-phase.json", {"phase": "idle_before_rollout", "time_unix": time.time()})
    sampler = subprocess.Popen([sys.executable, str(HERE / "resource_monitor.py"), "--run-dir", str(RUN), "--interval", "2"])
    try:
        for _ in range(60):
            path = RUN / "telemetry/samples.jsonl"
            if path.exists() and path.read_text().strip():
                first = json.loads(path.read_text().splitlines()[0])
                if first["errors"]:
                    raise RuntimeError(first["errors"])
                break
            if sampler.poll() is not None:
                raise RuntimeError("Sampler exited before first sample")
            time.sleep(1)
        else:
            raise TimeoutError("No initial memory sample")
        barrier = threading.Barrier(len(rows))
        with capture("formal-phase"):
            with (RUN / "formal-ledger.json").open("x") as f:
                json.dump({"formal_step": 1, "requests": 32, "status": "submitted"}, f)
            write("current-phase.json", {"phase": "formal_rollout_1", "time_unix": time.time()})
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
                futures = [pool.submit(generate, i, row, barrier) for i, row in enumerate(rows)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            write("current-phase.json", {"phase": "idle_after_rollout", "time_unix": time.time()})
        total = sum(r["output_tokens"] for r in results)
        analyze_requests.analyze(RUN)
        analyze_phases.analyze(RUN / "formal-phase", sum(r["prompt_tokens"] for r in results),
                               total - len(results), RUN / "analysis")
        write("formal-ledger.json", {"formal_step": 1, "requests": len(results), "status": "complete", "output_tokens": total})
    except BaseException as error:
        if (RUN / "formal-ledger.json").exists():
            write("formal-ledger.json", {"formal_step": 1, "status": "failed", "error": repr(error)})
        raise
    finally:
        write("current-phase.json", {"phase": "idle_after_rollout", "time_unix": time.time()})
        write("completion.json", {"time_unix": time.time()})
        sampler.wait(timeout=60)
    analyze_resources.analyze(RUN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--url", default="http://127.0.0.1:35439")
    parser.add_argument("--container", default="zjx-kt-base-r01")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    if Path("/raid/data/zjx/kt-baseline/results") not in RUN.parents:
        parser.error("--run-dir must be inside /raid/data/zjx/kt-baseline/results")
    URL, CONTAINER = args.url.rstrip("/"), args.container
    SAMPLING = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                "top_p": args.top_p, "top_k": args.top_k, "ignore_eos": False}
    with (RUN / "client.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / "formal-ledger.json").exists():
            raise SystemExit("One-round limit: a formal submission already exists")
        rollout()

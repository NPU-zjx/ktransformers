"""One native online update, then a full v1 rollout with the resident Actor."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time


CGROUP = Path("/sys/fs/cgroup")


class MeasuredV1(Exception):
    pass


def read_json(path):
    return json.loads(path.read_text())


def bytes_fields(path):
    return {parts[0].rstrip(":"): int(parts[1]) * 1024
            for line in path.read_text().splitlines()
            if len(parts := line.split()) == 3 and parts[2] == "kB"}


def assert_private_cgroup():
    assert Path("/proc/self/cgroup").read_text().strip() == "0::/", "Need a private container cgroup v2 namespace"
    assert str(os.getpid()) in (CGROUP / "cgroup.procs").read_text().split()
    assert int((CGROUP / "memory.current").read_text()) > 0


class Recorder:
    def __init__(self, root):
        self.root = root
        self.stage = None
        self.roles = {}
        self.stop = threading.Event()
        self.failures = []

    def container_sample(self):
        stat = dict(line.split() for line in (CGROUP / "memory.stat").read_text().splitlines())
        host = bytes_fields(Path("/proc/meminfo"))
        return dict(current_bytes=int((CGROUP / "memory.current").read_text()),
                    memory_stat={k: int(stat[k]) for k in ("anon", "file", "shmem", "kernel") if k in stat},
                    host_used_bytes=host["MemTotal"] - host["MemAvailable"])

    def processes_sample(self):
        parents = {}
        for p in (CGROUP / "cgroup.procs").read_text().split():
            pid = int(p)
            try:
                parents[pid] = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
            except (FileNotFoundError, ProcessLookupError):
                continue
        roots = self.roles.copy()
        totals = {role: 0 for role in roots}
        members = {role: [] for role in roots}
        errors = {}
        for pid in parents:
            parent = pid
            while parent in parents:
                role = next((r for r, root in roots.items() if parent == root), None)
                if role is not None:
                    try:
                        pss = bytes_fields(Path(f"/proc/{pid}/smaps_rollup"))["Pss"]
                    except (FileNotFoundError, ProcessLookupError):
                        break
                    except (OSError, KeyError) as exc:
                        errors[pid] = repr(exc)
                        break
                    totals[role] += pss
                    members[role].append(pid)
                    break
                parent = parents[parent]
        return dict(pss_bytes=totals, pids=members, errors=errors)

    def collect(self, name, period, sample):
        try:
            with (self.root / f"{name}.jsonl").open("x") as log:
                while not self.stop.is_set():
                    began = time.monotonic()
                    stage, start = self.stage, time.time_ns()
                    result = sample()
                    log.write(json.dumps(dict(stage=stage, start_ns=start,
                                              end_ns=time.time_ns(), **result)) + "\n")
                    log.flush()
                    self.stop.wait(max(0, period - (time.monotonic() - began)))
        except Exception as exc:
            self.failures.append(f"{name}: {exc!r}")

    @contextmanager
    def recording(self):
        jobs = [threading.Thread(target=self.collect, args=args, daemon=True) for args in (
            ("container-memory", 2, self.container_sample), ("process-memory", 30, self.processes_sample))]
        for job in jobs:
            job.start()
        try:
            yield
        finally:
            self.stop.set()
            for job in jobs:
                job.join(timeout=30)
            assert not any(job.is_alive() for job in jobs), "Sampler did not finish"
            assert not any(error.startswith("container-memory:") for error in self.failures), self.failures


def summarize(root, out):
    container = list(map(json.loads, (root / "container-memory.jsonl").read_text().splitlines()))
    processes = list(map(json.loads, (root / "process-memory.jsonl").read_text().splitlines()))
    phases = {}
    for label, stage_file, name in (
        ("actor_update", out / "round-01/stages.jsonl", "actor_update"),
        ("rollout_v1", out / "round-02/stages.jsonl", "live_rollout_v1"),
    ):
        stage, = [r for r in map(json.loads, stage_file.read_text().splitlines()) if r["phase"] == name]
        assert stage["status"] == "complete"
        begin, end = stage["start_unix_ns"], stage["end_unix_ns"]
        rows = [r for r in container if begin <= r["start_ns"] <= r["end_ns"] <= end]
        assert len(rows) >= 30 and all(r["current_bytes"] > 0 for r in rows)
        interval = statistics.median((b["start_ns"] - a["start_ns"]) / 1e9 for a, b in zip(rows, rows[1:]))
        assert 1.8 <= interval <= 2.3
        peak = max(rows, key=lambda r: r["current_bytes"])
        candidates = [r for r in processes if begin <= r["start_ns"] <= r["end_ns"] <= end
                      and not r["errors"] and all(r["pss_bytes"].get(k, 0) > 0 for k in ("actor", "rollout"))]
        nearest = min(candidates, key=lambda r: abs(r["start_ns"] - peak["start_ns"]), default=None)
        separation_s = abs(nearest["start_ns"] - peak["start_ns"]) / 1e9 if nearest else None
        phases[label] = dict(stage=stage, samples=len(rows), median_interval_s=interval,
            container_peak_bytes=peak["current_bytes"], peak_sample=peak,
            host_peak_bytes=max(r["host_used_bytes"] for r in rows),
            nearest_process_pss=nearest, process_pss_offset_s=separation_s)
    return phases


def smoke(recorder):
    command = [sys.executable, "-c", "import time; data = bytearray(8 << 20); time.sleep(15)"]
    with subprocess.Popen(command) as actor, subprocess.Popen(command) as rollout:
        recorder.roles.update(actor=actor.pid, rollout=rollout.pid)
        try:
            time.sleep(0.5)
            with recorder.recording():
                time.sleep(4.2)
            sample = json.loads((recorder.root / "process-memory.jsonl").read_text().splitlines()[0])
            assert sample["pids"] == {"actor": [actor.pid], "rollout": [rollout.pid]}, sample
            assert all(value >= 8 << 20 for value in sample["pss_bytes"].values()), sample
        finally:
            actor.terminate()
            rollout.terminate()
    samples = list(map(json.loads, (recorder.root / "container-memory.jsonl").read_text().splitlines()))
    assert len(samples) >= 3 and all(r["current_bytes"] > 0 for r in samples)
    print("Private cgroup v2 memory.current and separate Actor/rollout PSS passed", flush=True)

def main():
    source, recipe, root = (Path(x).resolve() for x in sys.argv[1:])
    assert_private_cgroup()
    recorder = Recorder(root)
    if os.environ.get("KTRL_MEMORY_PROBE") == "1":
        smoke(recorder)
        return

    os.environ["KTRL_RUN_ROOT"] = str(root)
    sys.path.insert(0, str(source))
    from ktrl.trainer import online_native as native
    from ktrl.trainer.precision_gate import verify

    gate = verify(recipe)
    native.guard_gpu()
    out = root / "rl-memory"
    original_phase, original_service, original_actor = native.phase, native.service, native.resident_worker

    @contextmanager
    def phase(folder, name):
        if folder == out / "round-02" and name == "reward_advantage":
            proof = read_json(folder / "rollout-policy-proof.json")
            assert proof["version"] == "v1" and len(proof["rank_replays"]) == 8
            assert all(r.get("v1", 0) > 0 for r in proof["rank_replays"])
            for rank in range(8):
                actor = read_json(out / f"round-01/actor/resident-rank{rank}.json")
                assert actor["step"] == 1 and Path(f"/proc/{actor['pid']}").is_dir()
            raise MeasuredV1
        previous = recorder.stage
        recorder.stage = name
        try:
            with original_phase(folder, name):
                yield
        finally:
            recorder.stage = previous

    @contextmanager
    def service(*args, **kwargs):
        with original_service(*args, **kwargs) as pair:
            recorder.roles["rollout"] = pair[1]
            yield pair

    @contextmanager
    def actor_worker(*args, **kwargs):
        with original_actor(*args, **kwargs) as process:
            recorder.roles["actor"] = process.pid
            yield process

    native.phase, native.service, native.resident_worker = phase, service, actor_worker
    with recorder.recording():
        try:
            native.run(recipe, out, steps=2, resident_actor=True)
        except MeasuredV1:
            pass
        else:
            raise RuntimeError("Expected to stop after v1 rollout")
    native.guard_gpu()
    assert read_json(out / "round-01/accepted.json")["updates"] == 1
    assert not (out / "round-02/optimizer-started.json").exists()
    assert not (out / "round-02/actor-ready.json").exists()
    native.save(root / "memory-accepted.json", dict(verified=True, optimizer_updates=1,
        source_sha256=gate["source_sha256"], measured_policy="v1",
        stop_reason="intentionally stopped after v1 rollout before scoring or second update",
        sampler_failures=recorder.failures, phases=summarize(root, out)))


if __name__ == "__main__":
    main()

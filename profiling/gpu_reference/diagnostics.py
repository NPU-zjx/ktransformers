"""Run each diagnostic exactly once after the verified formal rollout has exited."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path("/raid/data/zjx/kt-baseline/results/gpu-verl-r01")
NSYS = "/opt/nvidia/nsight-systems/2024.6.2/bin/nsys"
NCU = "/usr/local/cuda-13.0/bin/ncu"


def read(path):
    return json.loads(path.read_text())


def collect(mode, run):
    if mode == "torch":
        env = dict(os.environ, PYTHONPATH=str(HERE.parent))
        subprocess.run([sys.executable, str(HERE/"torch_tables.py"), str(run)], env=env, check=True)
    elif mode == "nsys":
        subprocess.run([NSYS, "export", "--type", "sqlite", "--output", str(run/"rollout.sqlite"),
                        str(run/"rollout.nsys-rep")], check=True)
        subprocess.run([sys.executable, str(HERE.parent/"analysis/analyze_nsys.py"),
                        str(run/"rollout.sqlite"), "--gpu-reference", "--output", str(run/"derived")], check=True)
    else:
        for args, filename in [(["--csv", "--page", "raw"], "raw-metrics.csv"),
                               (["--page", "details"], "details.txt")]:
            with (run/filename).open("w") as stream:
                subprocess.run([NCU, "--import", str(run/"hotspot.ncu-rep"), *args], stdout=stream, check=True)
        if "Kernel Name" not in (run/"raw-metrics.csv").read_text():
            raise ValueError("NCU has no kernel metric rows")


def main(formal, attempt):
    while not (formal/"completion.json").exists():
        time.sleep(10)
    if read(formal/"completion.json")["exit_code"] or read(formal/"ledger.json")["status"] != "complete":
        raise RuntimeError("Formal rollout failed; diagnostic queue stopped")
    status = {}
    for mode in ("torch", "nsys", "ncu"):
        run = ROOT/f"{mode}-{attempt}"
        try:
            subprocess.run(["bash", str(HERE/"launch.sh"), mode, attempt], check=True)
            collect(mode, run)
            status[mode] = "collected"
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            status[mode] = repr(exc)
        (ROOT/f"diagnostics-{attempt}.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("formal", type=Path)
    parser.add_argument("attempt")
    args = parser.parse_args()
    main(args.formal, args.attempt)

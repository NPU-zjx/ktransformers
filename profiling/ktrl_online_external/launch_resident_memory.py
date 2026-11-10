"""Launch a bounded, dedicated container using the accepted native runtime."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


BASE = Path("/raid/data/zjx/kt-baseline")
RESULTS = BASE / "results"
REVIEW = BASE / "checks/ktrl-review-005-20261108/resident-a04"
ENTRY = BASE / "ktransformers/profiling/ktrl_online_external/measure_resident_memory.py"
PYTHON = "/mnt/qjh007/jiahengdai/venvs/kt-py31115/bin/python"


def gpus_idle():
    jobs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"], text=True)
    return not jobs.strip()


def same_process(pid, start_ticks):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19] == start_ticks
    except FileNotFoundError:
        return False


def wait_for_slot(pid, start_ticks):
    deadline = time.monotonic() + 60 * 60
    print(f"Waiting up to 60 minutes for PID {pid} and for all eight GPUs to become idle", flush=True)
    while time.monotonic() < deadline:
        if not same_process(pid, start_ticks) and gpus_idle():
            time.sleep(10)
            if gpus_idle():
                return
        time.sleep(20)
    raise SystemExit("GPU wait expired. No container created.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=("mc-smoke01", "mc-a01"))
    parser.add_argument("--after", nargs=2, metavar=("PID", "START_TICKS"))
    args = parser.parse_args()
    smoke = args.name == "mc-smoke01"
    root = RESULTS / args.name
    command = json.loads((RESULTS / "ns-a03/container-command.json").read_text())
    docker = command[:2]
    entrypoint = command.index("--entrypoint")
    image = command[entrypoint + 2]
    command = command[:entrypoint]
    command[command.index("--name") + 1] = args.name
    command[-1] = f"type=bind,src={root},dst={root}"
    command += ["--cgroupns", "private"]
    if smoke:
        gpu_arg = command.index("--gpus")
        del command[gpu_arg:gpu_arg + 2]
        command += ["--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "KTRL_MEMORY_PROBE=1"]
    else:
        if args.after:
            wait_for_slot(*args.after)
        if not gpus_idle():
            raise SystemExit("GPUs occupied. No container created.")
        space = shutil.disk_usage(RESULTS)
        if space.free < 120 << 30:
            raise SystemExit("Need at least 120 GiB free on /raid. No container created.")
    command += ["--entrypoint", "/bin/bash", image, "-c", 'umask 0002; exec "$@"',
                "memory-probe", PYTHON, "-B", "-u", str(ENTRY), str(REVIEW / "source"),
                str(REVIEW / "recipe.yaml"), str(root)]
    root.mkdir(mode=0o2770)
    (root / "container-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (root / "driver.log").open("x") as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=90 if smoke else 85 * 60).returncode
        except subprocess.TimeoutExpired:
            subprocess.run([*docker, "stop", "-t", "30", args.name],
                           stdout=log, stderr=subprocess.STDOUT, timeout=60)
            code = 124
    (root / "container-exit.json").write_text(json.dumps({"exit_code": code}) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()

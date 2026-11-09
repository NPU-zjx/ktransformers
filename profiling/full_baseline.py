"""Run one formal rollout followed by three isolated, bounded diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = Path("/raid/data/zjx/kt-baseline/results/base-all-r01")


def run():
    ROOT.mkdir(exist_ok=False)
    env = dict(os.environ, KT_FULL_BASELINE="1")
    source = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    (ROOT / "source-sha256.json").write_text(json.dumps(source, indent=2))
    state = {"started_unix": time.time(), "pid": os.getpid(), "stages": {}}
    def save():
        tmp = ROOT / "suite.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(ROOT / "suite.json")
    for mode in ("rollout", "torch", "nsys", "ncu"):
        prefix = "" if mode == "rollout" else mode + "-"
        stage = state["stages"][mode] = {"status": "running", "started_unix": time.time()}
        save()
        try:
            with (ROOT / f"{mode}-console.log").open("x") as log:
                def invoke(action):
                    subprocess.run(["bash", str(HERE / "base.sh"), action], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                invoke(prefix + "serve")
                try:
                    invoke("rollout" if not prefix else prefix + "group-profile")
                finally:
                    invoke(prefix + "stop")
            stage["status"] = "complete"
        except Exception as exc:
            stage.update(status="failed", error=str(exc))
        stage["finished_unix"] = time.time()
        save()
        if mode == "rollout" and stage["status"] != "complete":
            break
    state["complete"] = len(state["stages"]) == 4 and all(s["status"] == "complete" for s in state["stages"].values())
    save()
    return state["complete"]


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

"""Run a dedicated KT-SGLang service using the qjh005 inference environment."""

from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import uuid

from ktrl.configs.recipe import REPO


RESULTS = Path("/raid/data/zjx/kt-baseline/results")
DOCKER = ["bash", "/raid/data/zjx/gpu-baseline/scripts/docker/client.sh"]
IMAGE = "verl-ci-cn-beijing.cr.volces.com/verlai/verl@sha256:aaf5271484842d12a40de5e92ea0ce5beb3b478f6c7988e330baebb170d383c2"
MOUNT = Path("/mnt/qjh007/jiahengdai")
RUNTIME = Path("/raid/data/zjx/jiaheng-repro/reference/full-engineering") / MOUNT.relative_to("/")
CUDNN = MOUNT / "qwen36-native-smoke-20260930/sglang-kt-migration-20261102-p01/service-deps-cu12-cudnn916/nvidia/cudnn/lib"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def state(cid):
    return json.loads(subprocess.check_output(DOCKER + ["inspect", cid], text=True))[0]


def command(rollout, out, name, port):
    args = rollout["server_args"]
    devices = ",".join(str(i) for i in range(args["tp_size"]))
    create = DOCKER + [
        "create", "--name", name, "--label", f"ktrl.run={name}", "--cidfile", str(out / "container.id"),
        "--runtime=nvidia", "--network", "host", "--read-only", "--no-healthcheck", "--user", "0:0",
        "--ipc", "private", "--shm-size", "16g", "--memory", "1024g", "--memory-swap", "1024g",
        "--tmpfs", "/tmp:rw,size=16g", "-w", "/repo", "--entrypoint", "/bin/bash",
        "-v", f"{REPO}:/repo:ro", "-v", f"{RUNTIME}:{MOUNT}:ro", "-v", f"{out}:/output:rw",
    ]
    for path in sorted({args["model_path"], args["kt_weight_path"]}):
        create.extend(["-v", f"{path}:{path}:ro"])
    environment = {
        "NVIDIA_VISIBLE_DEVICES": devices, "CUDA_VISIBLE_DEVICES": devices,
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "HOME": "/output/home", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false", "LD_LIBRARY_PATH": str(CUDNN),
        "FLASHINFER_WORKSPACE_BASE": "/output/cache/flashinfer", "TORCHINDUCTOR_CACHE_DIR": "/output/cache/inductor",
        "KT_PHASE_HOOKS": "1", "KT_TORCH_DIAGNOSTIC": "0", "SGLANG_PROFILE_V2": "0",
        "PYTHONPATH": "/repo/profiling/worker:/repo/third_party/sglang/python",
    }
    for key, value in environment.items():
        create.extend(["-e", f"{key}={value}"])
    create.extend([IMAGE, "-c", 'exec "$@" >/output/server.log 2>&1', "--",
                   str(MOUNT / "venvs/kt-py31115/bin/python"), "-m", rollout["native_entrypoint"]])
    for key, value in args.items():
        if isinstance(value, bool):
            if key == "disable_cuda_graph" and value:
                raise ValueError("M1 requires CUDA Graph replay")
            if value:
                create.append("--" + key.replace("_", "-"))
            continue
        create.extend(["--" + key.replace("_", "-"), str(value)])
    return create + [
        "--host", "127.0.0.1", "--port", str(port), "--served-model-name", name,
        "--context-length", "9216", "--mem-fraction-static", "0.35", "--max-running-requests", "32",
        "--chunked-prefill-size", "1024", "--cuda-graph-max-bs", "32", "--disable-radix-cache",
        "--disable-shared-experts-fusion", "--random-seed", "42", "--skip-server-warmup",
    ]


@contextmanager
def service(rollout, out):
    import fcntl

    with (RESULTS / ".ktrl-rollout.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip()
        if active:
            raise RuntimeError(f"GPU in use; active compute PIDs: {active}")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        name = "zjx-ktrl-rollout-" + uuid.uuid4().hex[:12]
        create = command(rollout, out, name, port)
        save(out / "launch.json", {"name": name, "port": port, "argv": create})
        (out / "home").mkdir()
        cidfile = out / "container.id"
        try:
            subprocess.run(create, check=True, stdout=subprocess.DEVNULL)
            cid = cidfile.read_text().strip()
            subprocess.run(DOCKER + ["start", cid], check=True, stdout=subprocess.DEVNULL)
            print(f"Loading KT-SGLang in {name} ({cid[:12]})", flush=True)
            yield cid, name, f"http://127.0.0.1:{port}"
        finally:
            if cidfile.exists():
                cid = cidfile.read_text().strip()
                info = state(cid)
                if info["Id"] != cid or info["Config"]["Labels"].get("ktrl.run") != name:
                    raise RuntimeError("Container identity changed")
                if info["State"]["Running"]:
                    subprocess.run(DOCKER + ["stop", "--time", "90", cid], check=True, stdout=subprocess.DEVNULL)
                final = state(cid)["State"]
                save(out / "exit.json", {"container_id": cid, "name": name, **final})
                if final["Running"]:
                    raise RuntimeError("Dedicated container did not stop")

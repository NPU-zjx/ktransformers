"""Launch one native online round with a separate profiling overlay."""
import argparse
from datetime import datetime
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys


DATA_ROOT = Path("/raid/data/zjx/kt-baseline/results")
HOME_ROOT = Path("/home/zjx")
# One diagnostic exports one ~70.86 GiB model, no optimizer checkpoint.
# Reserve 20 GiB for reports and >49 GiB of safety headroom.
MIN_FREE_GIB = 140


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--root", type=Path,
                        help="experiment directory (default: new timestamped directory on /raid)")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mode", choices=("torch", "nsys", "ncu"), default="torch")
    parser.add_argument("--ncu-roles", action="store_true",
                        help="Keep online controller uninstrumented; wrap its native GPU workers with NCU")
    parser.add_argument("--ncu-actor-only", action="store_true",
                        help="Use CUDA Torch for rollout and NCU for Actor in a complete online cycle")
    args = parser.parse_args()
    if args.ncu_actor_only and (args.mode != 'ncu' or not args.ncu_roles):
        parser.error('--ncu-actor-only requires --mode ncu --ncu-roles')
    root = (args.root or DATA_ROOT / f"ktrl-online-profile-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    if not any(base in root.parents for base in (DATA_ROOT, HOME_ROOT)):
        parser.error("--root must be under /raid/data/zjx/kt-baseline/results or /home/zjx")
    storage = next(path for path in (root, *root.parents) if path.is_dir())
    free_gib = shutil.disk_usage(storage).free / 2**30
    if free_gib < MIN_FREE_GIB:
        parser.error(f"{storage} has {free_gib:.1f} GiB free; need {MIN_FREE_GIB} GiB for {root}")
    out = root / "rl-online-profile01"
    # Native SGLang uses Unix-domain sockets below this directory (107 bytes).
    ipc_example = str(out / "service/tmp/tmp12345678")
    if len(os.fsencode(ipc_example)) > 107:
        parser.error(f"Output path would exceed ZMQ IPC limit: {ipc_example}")
    assert not out.exists()
    os.environ["KTRL_RUN_ROOT"] = str(root)
    source = args.source.resolve(strict=True)
    sys.path.insert(0, str(source))
    from ktrl.trainer import online_native as native
    from ktrl.trainer.precision_gate import verify

    gate = verify(args.recipe.resolve())
    assert native.ROOT == root and native.REPO == source
    native.guard_gpu()
    print("Native source gate, GPU availability and output budget verified", flush=True)
    if args.check:
        return

    native_environment = native.environment
    overlay = Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    profile = root / "profiles"
    profile.mkdir()

    def environment(folder, actor):
        env = native_environment(folder, actor)
        env["PYTHONPATH"] = f'{overlay}:{env["PYTHONPATH"]}'
        env["KTRL_EXTERNAL_PROFILE_DIR"] = str(profile)
        env["KTRL_EXTERNAL_PROFILE_ROLE"] = "actor" if actor else "rollout"
        env["KTRL_EXTERNAL_PROFILE_MODE"] = 'torch' if args.ncu_actor_only and not actor else args.mode
        return env

    # Only extend subprocess environment; native code still controls the run.
    native.environment = environment
    if args.ncu_roles:
        assert args.mode == 'ncu'
        options = json.loads((root/'ncu/options.json').read_text())
        original_popen = subprocess.Popen
        wrapped = set()

        def profiling_popen(argv, *positional, **kwargs):
            role = None
            if isinstance(argv, (list, tuple)):
                if 'sglang.launch_server' in argv:
                    role = 'rollout'
                elif 'ktrl.workers.engine.fsdp.actor_update' in argv:
                    role = 'actor'
            if role is None or (args.ncu_actor_only and role == 'rollout'):
                return original_popen(argv, *positional, **kwargs)
            assert role not in wrapped, 'Never repeat an online worker implicitly'
            wrapped.add(role)
            # Native PYTHONPATH belongs to Python workers, not the profiler itself.
            worker_env = dict(kwargs['env'])
            pythonpath = worker_env.pop('PYTHONPATH')
            ncu_binary = (root/'ncu/binary.txt').read_text().strip()
            assert Path(ncu_binary).is_file()
            command = [ncu_binary, *options,
                       '--export', str(root/'ncu'/f'{role}-core'),
                       '/usr/bin/env', 'PYTHONPATH=' + pythonpath, *argv]
            with (root/'ncu/worker-launches.jsonl').open('a') as log:
                log.write(json.dumps(dict(role=role, native_argv=argv,
                    profiler_argv=command, start_new_session=kwargs.get('start_new_session'))) + '\n')
            kwargs['env'] = worker_env
            return original_popen(command, *positional, **kwargs)

        subprocess.Popen = profiling_popen
    print(f"Running native online KTRL with source {gate['source_sha256']}", flush=True)
    try:
        native.run(args.recipe, out, steps=1, resident_actor=True)
    finally:
        if args.ncu_roles:
            subprocess.Popen = original_popen


if __name__ == "__main__":
    main()

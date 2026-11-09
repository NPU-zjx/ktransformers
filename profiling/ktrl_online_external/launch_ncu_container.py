"""Isolated NCU counter permissions; native files mounted read-only."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--recipe', type=Path, required=True)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--check', action='store_true')
parser.add_argument('--profiler', choices=('ncu', 'nsys'), default='ncu')
parser.add_argument('--resume-preflight', action='store_true')
parser.add_argument('--ncu-actor-only', action='store_true')
args = parser.parse_args()
if args.ncu_actor_only and args.profiler != 'ncu':
    parser.error('--ncu-actor-only requires --profiler ncu')
root = args.root.resolve()
base = Path('/raid/data/zjx/kt-baseline')
assert base / 'results' in root.parents
if args.resume_preflight:
    assert args.profiler == 'nsys'
    assert json.loads((root/'container-exit.json').read_text())['exit_code'] != 0
    assert not (root/'rl-online-profile01').exists(), 'Never resume/repeat a submitted online update'
    assert not (root/'resume-command.json').exists(), 'Single explicit preflight continuation'
else:
    assert not root.exists(), 'Never reuse a diagnostic directory'
assert shutil.disk_usage(base).free >= 140 * 2**30
assert args.source.is_dir() and args.recipe.is_file()
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                               '--format=csv,noheader,nounits'], text=True).strip()
assert not gpu, f'Another run owns GPUs: {gpu}'
image = ('verl-ci-cn-beijing.cr.volces.com/verlai/verl@sha256:'
         'aaf5271484842d12a40de5e92ea0ce5beb3b478f6c7988e330baebb170d383c2')
command = ['bash', '/raid/data/zjx/gpu-baseline/scripts/docker/client.sh', 'run',
           '--name', root.name + ('-resume' if args.resume_preflight else ''), '--hostname', 'qjh005', '--network', 'host',
           '--pid', 'host', '--ipc', 'host', '--gpus', 'all', '--cap-add', 'SYS_ADMIN',
           '--user', '0:2151', '--workdir', str(base),
           '--mount', f'type=bind,src={base},dst={base},readonly',
           '--mount', 'type=bind,src=/mnt/qjh007,dst=/mnt/qjh007,readonly',
           '--mount', 'type=bind,src=/home/zjx/Documents/ktrl-review-000-20261108,dst=/home/zjx/Documents/ktrl-review-000-20261108,readonly',
           '--mount', 'type=bind,src=/raid/data/zjx/gpu-baseline/data,dst=/raid/data/zjx/gpu-baseline/data,readonly',
           '--mount', f'type=bind,src={root},dst={root}',
           '--entrypoint', '/bin/bash', image, '-c',
           'umask 0002; exec bash "$@"', 'ktrl-ncu',
           str(Path(__file__).resolve().parent / ('run_nsys_container.sh' if args.profiler == 'nsys' else 'run_ncu.sh')),
           str(args.source.resolve()), str(args.recipe.resolve()), str(root)]
if args.ncu_actor_only:
    command.append('actor')
print(json.dumps(command, indent=2))
if not args.check:
    if not args.resume_preflight:
        root.mkdir()
    prefix = 'resume' if args.resume_preflight else 'container'
    (root / (prefix+'-command.json')).write_text(json.dumps(command, indent=2))
    with (root / (prefix+'.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (root / (prefix+'-exit.json')).write_text(json.dumps({'exit_code': result.returncode}))
    raise SystemExit(result.returncode)

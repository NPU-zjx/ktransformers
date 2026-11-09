"""Wait for the identified online run; validate NSys injection before new run."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--previous-root', type=Path, required=True)
parser.add_argument('--previous-pid', type=int, required=True)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--recipe', type=Path, required=True)
parser.add_argument('--output-root', type=Path, required=True)
parser.add_argument('--control-dir', type=Path, required=True)
args = parser.parse_args()
args.control_dir.mkdir(exist_ok=False)
here = Path(__file__).resolve().parent
nsys = '/opt/nvidia/nsight-systems/2024.6.2/bin/nsys'
python = '/mnt/qjh007/jiahengdai/venvs/kt-py31115/bin/python'


def state(phase, **fields):
    temp = args.control_dir / 'state.tmp'
    temp.write_text(json.dumps(dict(phase=phase, time=time.time(), **fields), indent=2))
    temp.replace(args.control_dir / 'state.json')


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return stat[19] if stat[0] != 'Z' else None
    except FileNotFoundError:
        return None


def run(command, name):
    with (args.control_dir / (name + '.log')).open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


try:
    original = identity(args.previous_pid)
    if original is not None:
        cmd = Path(f'/proc/{args.previous_pid}/cmdline').read_bytes()
        assert str(args.previous_root).encode() in cmd and b'launch.py' in cmd
    state('waiting_current_run', pid=args.previous_pid, start_ticks=original)
    deadline = time.monotonic() + 3 * 3600
    while original is not None and identity(args.previous_pid) == original:
        if time.monotonic() > deadline:
            raise TimeoutError('Previous run still active; no restart')
        time.sleep(10)
    accepted = args.previous_root / 'rl-online-profile01/accepted.json'
    proof = json.loads(accepted.read_text())
    # Native acceptance must exist, and the GPU must be released.
    assert proof['real_online_updates'] == 1 and proof['optimizer_steps'] == [1], proof
    assert proof['final_generation_policy'] == 'v1' and proof['resident_actor'] is True, proof
    gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True).strip()
    assert not gpu_pids, f'GPU still owned: {gpu_pids}'
    state('nsys_injection_smoke')
    run([nsys, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
         '--capture-range=cudaProfilerApi', '--capture-range-end=repeat:8',
         '--kill=none', '--wait=all', '--output=' + str(args.control_dir / 'smoke'),
         python, '-B', str(here / 'nsys_smoke.py')], 'smoke')
    reports = list(args.control_dir.glob('smoke*.nsys-rep'))
    assert reports, 'NSys produced no reports'
    labels = set()
    total_kernels = 0
    for report in reports:
        db = report.with_suffix('.sqlite')
        run([nsys, 'export', '--type=sqlite', '--output=' + str(db), str(report)],
            report.stem + '-export')
        with sqlite3.connect(db) as con:
            total_kernels += con.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
            labels.update(row[0] for row in con.execute('SELECT text FROM NVTX_EVENTS WHERE text IS NOT NULL'))
            columns = {row[1] for row in con.execute('PRAGMA table_info(NVTX_EVENTS)')}
            if 'textId' in columns:
                labels.update(row[0] for row in con.execute(
                    'SELECT s.value FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id'))
    assert total_kernels >= 2 and {'rollout-probe', 'actor-probe'} <= labels, (total_kernels, labels)
    state('online_nsys_running', smoke_kernels=total_kernels, labels=sorted(labels))
    run(['bash', str(here / 'run_nsys.sh'), str(args.source), str(args.recipe),
         str(args.output_root)], 'online-nsys')
    assert (args.output_root / 'rl-online-profile01/accepted.json').is_file()
    reports = list((args.output_root / 'nsys').glob('*.nsys-rep'))
    assert reports, 'Online cycle completed but NSys reports absent'
    state('online_finished_reports_pending_validation', reports=[str(p) for p in reports])
except BaseException as error:
    state('failed_no_retry', error=repr(error))
    raise

"""Continue to NCU after the identified NSys controller exits successfully."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--nsys-pid', type=int, required=True)
parser.add_argument('--nsys-control', type=Path, required=True)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--recipe', type=Path, required=True)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--control', type=Path, required=True)
args = parser.parse_args()
args.control.mkdir(exist_ok=False)


def identity():
    try:
        fields = Path(f'/proc/{args.nsys_pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def state(phase, **fields):
    temp = args.control / 'state.tmp'
    temp.write_text(json.dumps(dict(phase=phase, time=time.time(), **fields), indent=2))
    temp.replace(args.control / 'state.json')


try:
    original = identity()
    if original is not None:
        command = Path(f'/proc/{args.nsys_pid}/cmdline').read_bytes()
        assert b'continue_nsys.py' in command and str(args.nsys_control).encode() in command
    state('waiting_nsys', pid=args.nsys_pid, start_ticks=original)
    deadline = time.monotonic() + 4 * 3600
    while original is not None and identity() == original:
        if time.monotonic() >= deadline:
            raise TimeoutError('NSys controller still running; no new workload started')
        time.sleep(10)
    previous = json.loads((args.nsys_control / 'state.json').read_text())
    assert previous['phase'] == 'online_finished_reports_pending_validation', previous
    assert previous['reports'] and all(Path(p).is_file() for p in previous['reports'])
    state('ncu_smoke_then_online')
    launcher = Path(__file__).resolve().parent / 'launch_ncu_container.py'
    with (args.control / 'launch.log').open('x') as log:
        subprocess.run(['python3', '-B', str(launcher), '--source', str(args.source),
                        '--recipe', str(args.recipe), '--root', str(args.root)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = json.loads((args.root / 'rl-online-profile01/accepted.json').read_text())
    assert proof['real_online_updates'] == 1 and proof['optimizer_steps'] == [1], proof
    assert proof['final_generation_policy'] == 'v1', proof
    report = args.root / 'ncu/online-core.ncu-rep'
    assert report.is_file() and report.stat().st_size > 0
    state('online_finished_report_pending_validation', report=str(report))
except BaseException as error:
    state('failed_no_retry', error=repr(error))
    raise

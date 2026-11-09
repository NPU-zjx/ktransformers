"""Resolve first profiled microbatch boundaries from native, unmodified events."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
profiles = args.root / 'profiles'
windows = [json.loads(line) for p in profiles.glob('windows-*.jsonl')
           for line in p.read_text().splitlines()]
window, = [r for r in windows if r['stage'] == 'actor-microbatch-rank0']
events = [json.loads(line) for line in
          (args.root / 'rl-online-profile01/round-01/actor/events-rank0.jsonl').read_text().splitlines()]
start, end = window['start_ns'], window['end_ns']
old, = [r for r in events if r['phase'] == 'old_logp_start' and start <= r['time_unix_ns'] < end]
uid = old['uid']
forward, = [r for r in events if r['phase'] == 'actor_forward_start' and r.get('uid') == uid]
backward, = [r for r in events if r['phase'] == 'backward_start' and r.get('uid') == uid]
boundaries = [old['time_unix_ns'], forward['time_unix_ns'], backward['time_unix_ns'], end]
assert start <= boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3]
output = profiles / 'actor-phases'
output.mkdir(exist_ok=False)
trace = profiles / window['trace']
records = []
for i, phase in enumerate(('old_logp_and_audit', 'forward_and_loss', 'backward')):
    begin, finish = boundaries[i:i + 2]
    subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('analyze_trace.py')),
                    str(trace), str(output / (phase + '.json')),
                    '--start-ns', str(begin), '--end-ns', str(finish)], check=True)
    records.append(dict(phase=phase, start_unix_ns=begin, end_unix_ns=finish,
                        duration_s=(finish-begin)/1e9, uid=uid))
(output / 'boundaries.json').write_text(json.dumps(dict(
    records=records, source=str(trace),
    scope='First rank0 microbatch with Torch overhead; not formal timing. '
          'Old-logp range includes native audit/save. Forward includes loss. '
          'Only complete events within each host boundary are counted. '
          'GPU records classified by execution interval, not causal launch attribution. '
          'Optimizer is outside this trace.'), indent=2))

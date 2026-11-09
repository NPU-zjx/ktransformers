"""Export completed online reports without launching any model workload."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

from analyze_nsys_online import analyze

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--nsys', default='/opt/nvidia/nsight-systems/2024.6.2/bin/nsys')
args = parser.parse_args()
root = args.root.resolve()
accepted = json.loads((root / 'rl-online-profile01/accepted.json').read_text())
assert accepted['real_online_updates'] == 1
reports = sorted((root / 'nsys').glob('*.nsys-rep'))
assert reports
out = root / 'nsys-analysis'
out.mkdir(exist_ok=True)
manifest = []
phase_devices = {}
total_kernels = linked = 0
for report in reports:
    db = out / (report.stem + '.sqlite')
    if not db.exists():
        with (out / (report.stem + '-export.log')).open('x') as log:
            subprocess.run([args.nsys,
                'export', '--type=sqlite', '--output=' + str(db), str(report)],
                stdout=log, stderr=subprocess.STDOUT, check=True)
    with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as con:
        tables = {r[0] for r in con.execute('SELECT name FROM sqlite_master WHERE type="table"')}
    digest = hashlib.sha256()
    with report.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    item = dict(report=str(report), bytes=report.stat().st_size,
                sha256=digest.hexdigest())
    if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:
        item['status'] = 'no_kernel_table'
        manifest.append(item)
        continue
    result = analyze(db)
    (out / (report.stem + '.json')).write_text(json.dumps(result, indent=2))
    total_kernels += result['kernel_count']
    linked += result['api_linked_kernels']
    for row in result['phase_kernels']:
        if row['calls']:
            phase_devices.setdefault(row['phase'], set()).add(row['device'])
    item.update(status='parsed', kernels=result['kernel_count'],
                linked=result['api_linked_kernels'], devices=result['observed_devices'],
                markers=sorted({m['name'] for m in result['markers']}))
    manifest.append(item)
required = ['rollout-extend-tp0', 'rollout-decode-tp0',
            'actor_actor_forward_start_rank0', 'actor_backward_start_rank0',
            'actor-optimizer-rank0']
missing = {phase: sorted(set(range(8)) - phase_devices.get(phase, set()))
           for phase in required if phase_devices.get(phase, set()) != set(range(8))}
proof = dict(reports=manifest, kernels=total_kernels, linked=linked,
    phase_devices={k: sorted(v) for k,v in phase_devices.items()},
    missing_device_coverage=missing,
    kernel_api_and_device_coverage_passed=not missing and linked == total_kernels and total_kernels > 0,
    note='Inspect NSys diagnostics/warnings separately; API coverage is not proof of zero event loss.')
(out / 'coverage.json').write_text(json.dumps(proof, indent=2))
print(json.dumps({k:v for k,v in proof.items() if k not in ('reports','phase_devices')}, indent=2))

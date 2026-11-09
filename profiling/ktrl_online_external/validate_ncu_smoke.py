"""Require finite core counters for all four short diagnostic windows."""
import csv
import io
import json
import math
from pathlib import Path
import sys
from analyze_ncu_online import analyze

root = Path(sys.argv[1])
raw = (root / 'smoke.csv').read_text()
lines = raw.splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
rows = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
required = ['gpu__time_duration.sum', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
            'dram__throughput.avg.pct_of_peak_sustained_elapsed',
            'sm__warps_active.avg.pct_of_peak_sustained_active']
assert rows and 'Metric Name' not in rows[0], 'Expected raw wide-format NCU export'
assert set(required) <= rows[0].keys()
# First raw CSV data row contains units.
results = rows[1:]
assert len(results) >= 4, f'Need prefill, graph decode, forward, backward; got {len(results)}'
names = [r.get('Kernel Name', '') for r in results]
assert sum('RMSNormKernel' in name for name in names) >= 2, names
assert any('layer_norm_fwd_kernel' in n or 'layer_norm_gated_fwd_kernel' in n for n in names), names
assert any('layer_norm_bwd_kernel' in n or 'layer_norm_gated_bwd_kernel' in n for n in names), names
for row in results:
    values = [float(row[name].replace(',', '')) for name in required]
    assert all(math.isfinite(v) and v >= 0 for v in values), row
    assert values[0] > 0, row
proofs = [json.loads((root / (role + '-proof.json')).read_text()) for role in ('rollout', 'actor')]
assert all(p['optimizer_updates'] == 0 and len(p['stages']) == 2 for p in proofs)
assert proofs[0]['graph_replay'] is True
classified = analyze(root / 'smoke.csv', ['prefill', 'decode', 'forward', 'backward'])
assert classified['verified'], classified['missing_stages']
cpu_summaries = [json.loads(p.read_text()) for p in root.glob('*-cpu-rank0-*.summary.json')]
assert len(cpu_summaries) == 4 and all(p['activities'] == ['CPU'] for p in cpu_summaries)
assert all(p['record_shapes'] and p['operators'] for p in cpu_summaries)
assert any(row['input_shapes'] for p in cpu_summaries for row in p['operators'])
(root / 'validated.json').write_text(json.dumps(dict(
    core_counters_valid=True, records=len(results), proofs=proofs,
    observed_stages=classified['observed_stages'], cpu_only_torch_compatible=True,
    cuda_torch_activity=False,
    note='CPU-only traces checked. Kineto may warn that CUDA CUPTI subscription is unavailable while NCU owns it.'), indent=2))
print('NCU graph/forward/backward smoke counters valid; no model updates')

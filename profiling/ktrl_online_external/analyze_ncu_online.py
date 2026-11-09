"""Validate NCU CSV counters and classify kernels by recorded NVTX scope."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

CORE = ('gpu__time_duration.sum', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__warps_active.avg.pct_of_peak_sustained_active')
DETAIL = ('l1tex__t_sector_hit_rate.pct', 'lts__t_sector_hit_rate.pct',
          'launch__registers_per_thread', 'launch__shared_mem_per_block',
          'smsp__inst_executed.avg.per_cycle_active')


def analyze(path, required_stages, *, recover_partial=False):
    raw = path.read_bytes()
    lines = raw.decode('utf-8-sig').splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    units = next(reader)
    assert set(CORE) <= units.keys(), 'NCU core counters missing'
    missing_detail = sorted(set(DETAIL) - units.keys())
    stall_keys = [k for k in units if k and 'warps_issue_stalled_' in k]
    assert not missing_detail and stall_keys, ('Detailed counters missing', missing_detail, stall_keys)
    output = []
    rejected = []
    for row in reader:
        assert row.get('ID', '').isdigit(), f'Malformed data row: {row.get("ID")}'
        stack = ' '.join(str(v) for k, v in row.items() if k and 'Range:' in k)
        phase = 'unattributed'
        for label, name in (
            ('rollout-extend-tp0', 'prefill'), ('rollout-decode-tp0', 'decode'),
            ('actor_actor_forward_start_rank0', 'forward'), ('actor_backward_start_rank0', 'backward'),
            ('GRPO_forward', 'forward'), ('GRPO_backward', 'backward')):
            if label in stack:
                phase = name
                break
        metrics = {}
        invalid = []
        for key in (*CORE, *DETAIL, *stall_keys):
            raw_value = row[key]
            try:
                value = float(raw_value.replace(',', ''))
            except (AttributeError, ValueError):
                invalid.append(key)
                continue
            if not math.isfinite(value) or value < 0 or (key == CORE[0] and value == 0):
                invalid.append(key)
                continue
            metrics[key] = dict(value=value, unit=units[key])
        if invalid:
            assert recover_partial, (row['ID'], phase, invalid)
            rejected.append(dict(id=row['ID'], phase=phase, kernel=row['Kernel Name'],
                                 invalid_metrics=invalid))
            continue
        assert metrics[CORE[0]]['unit'] in {'ns', 'us', 'ms', 's'}, units[CORE[0]]
        assert all(metrics[k]['unit'] == '%' for k in CORE[1:])
        output.append(dict(id=row['ID'], pid=row['Process ID'], device=row['Device'],
            kernel=row['Kernel Name'], phase=phase, nvtx_stack=stack,
            grid=row['Grid Size'], block=row['Block Size'], metrics=metrics))
    assert output, 'No NCU kernel records'
    phases = {r['phase'] for r in output}
    missing = sorted(set(required_stages) - phases)
    return dict(source=str(path), sha256=hashlib.sha256(raw).hexdigest(), records=output,
        observed_stages=sorted(phases), required_stages=required_stages,
        missing_stages=missing, rejected_records=rejected,
        verified=not missing and not rejected and not recover_partial,
        scope='Selected kernel hardware counters only; not whole-model utilization.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--required-stages', nargs='+', default=['prefill', 'decode', 'forward', 'backward'])
    parser.add_argument('--recover-partial', action='store_true',
                        help='Preserve valid records from a failed run without passing validation')
    args = parser.parse_args()
    result = analyze(args.csv, args.required_stages, recover_partial=args.recover_partial)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ('verified', 'observed_stages', 'missing_stages')}))
    raise SystemExit(0 if result['verified'] else 1)

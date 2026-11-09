"""Formal-rollout resource samples, with explicit byte and sample-peak semantics."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def analyze(run, output=None):
    samples=[json.loads(line) for line in (run/'telemetry/samples.jsonl').read_text().splitlines()]
    selected=[s for s in samples if s.get('phase_hint','').startswith('formal_rollout_')]
    assert len(selected)>1
    assert all(not s['errors'] for s in selected)
    intervals=[b['time_unix']-a['time_unix'] for a,b in zip(selected,selected[1:])]
    peak=max(selected,key=lambda s:s['container']['memory_current_bytes'])
    rows=[]
    for s in selected:
        mem=s['container']['memory_stat']
        row=dict(time_unix=s['time_unix'],elapsed_seconds=s['time_unix']-selected[0]['time_unix'],
                 container_current_bytes=s['container']['memory_current_bytes'],
                 container_anon_bytes=mem['anon'],container_file_bytes=mem['file'],
                 container_shmem_bytes=mem['shmem'],host_available_bytes=s['host_memory_bytes']['MemAvailable'])
        for g in s['gpus']:
            i=int(g['index'])
            row[f'gpu{i}_used_bytes']=int(g['memory.used']*1024**2)
            row[f'gpu{i}_utilization_pct']=g['utilization.gpu']
        rows.append(row)
    summary=dict(formal_samples=len(selected),all_samples=len(samples),sample_errors=0,
        sample_interval_median_seconds=statistics.median(intervals),sample_interval_max_seconds=max(intervals),
        container_sample_peak_bytes=peak['container']['memory_current_bytes'],
        container_sample_peak_gib=peak['container']['memory_current_bytes']/1024**3,
        container_at_peak_memory_stat=peak['container']['memory_stat'],
        host_min_available_bytes=min(r['host_available_bytes'] for r in rows),
        memory_events_start=selected[0]['container']['memory_events'],
        memory_events_end=selected[-1]['container']['memory_events'],gpus=[])
    for i in range(8):
        summary['gpus'].append(dict(index=i,sample_peak_bytes=max(r[f'gpu{i}_used_bytes'] for r in rows),
            sample_peak_gib=max(r[f'gpu{i}_used_bytes'] for r in rows)/1024**3,
            mean_sample_utilization_pct=statistics.mean(r[f'gpu{i}_utilization_pct'] for r in rows)))
    summary['definitions']={'memory':'2-second external sample peak, not an absolute instantaneous peak; initialization excluded.',
        'cgroup':'shmem is included in file; do not add again. Host MemAvailable is a separate accounting scope.',
        'gpu':'Whole-device memory from nvidia-smi MiB converted to bytes; sampled utilization is not SM throughput.',
        'phase':'Filtered by driver formal_rollout marker; per-operation phase allocator boundaries are separate.'}
    output=output or run/'analysis';output.mkdir(exist_ok=True)
    (output/'resource-summary.json').write_text(json.dumps(summary,indent=2))
    with (output/'resource-samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);analyze(p.parse_args().run)

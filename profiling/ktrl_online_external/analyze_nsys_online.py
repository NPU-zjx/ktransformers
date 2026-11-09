"""Read-only NSys analysis for bounded KTRL windows; explicit coverage gaps."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return sum(b-a for a, b in merged)


def analyze(path):
    con = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    tables = {r[0] for r in con.execute('SELECT name FROM sqlite_master WHERE type="table"')}
    strings = dict(con.execute('SELECT id,value FROM StringIds'))
    markers = []
    for raw in con.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        r = dict(raw)
        name = r.get('text') or strings.get(r.get('textId'), '')
        if name.startswith(('rollout-extend-', 'rollout-decode-', 'actor-microbatch-',
                            'actor_old_logp_start_', 'actor_actor_forward_start_',
                            'actor_backward_start_', 'actor-optimizer-')):
            markers.append(dict(name=name, start=r['start'], end=r['end'],
                                pid=(r['globalTid'] >> 24) & 0xffffff))
    api = {}
    api_events = []
    for table in ('CUPTI_ACTIVITY_KIND_DRIVER', 'CUPTI_ACTIVITY_KIND_RUNTIME'):
        if table not in tables:
            continue
        for raw in con.execute('SELECT * FROM ' + table):
            r = dict(raw)
            r['source_table'] = table
            api_events.append(r)
            api[(r['globalTid'] & -16777216, r['correlationId'])] = r
    kernels = [dict(r) for r in con.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')]
    copies = ([dict(r) for r in con.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY')]
              if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else [])
    memset = ([dict(r) for r in con.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET')]
              if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else [])
    grouped = defaultdict(lambda: dict(calls=0, total_us=0.0, api_linked=0, graph_nodes=0))
    for r in kernels:
        name = strings[r['demangledName']]
        key = (r['deviceId'], (r['globalPid'] >> 24) & 0xffffff, name)
        row = grouped[key]
        row['calls'] += 1
        row['total_us'] += (r['end']-r['start']) / 1000
        row['api_linked'] += (r['globalPid'], r['correlationId']) in api
        row['graph_nodes'] += bool(r.get('graphNodeId'))
    devices = sorted({r['deviceId'] for r in kernels})
    copy_names = (dict((r[0], r[2]) for r in con.execute('SELECT * FROM ENUM_CUDA_MEMCPY_OPER'))
                  if 'ENUM_CUDA_MEMCPY_OPER' in tables else {})
    copy_groups = defaultdict(lambda: dict(calls=0, bytes=0, total_us=0.0))
    for r in copies:
        row = copy_groups[(r['deviceId'], r['copyKind'])]
        row['calls'] += 1
        row['bytes'] += r['bytes']
        row['total_us'] += (r['end'] - r['start']) / 1000
    api_groups = defaultdict(lambda: dict(calls=0, total_us=0.0))
    for r in api_events:
        row = api_groups[((r['globalTid'] >> 24) & 0xffffff, r['source_table'], strings[r['nameId']])]
        row['calls'] += 1
        row['total_us'] += (r['end'] - r['start']) / 1000
    windows = []
    phase_kernels = []
    # Each rank0 phase provides one shared host-time interval for device comparison.
    # Other ranks' annotations remain in markers for independent boundary checks.
    selected = [r for r in markers if r['name'].endswith(('tp0', 'rank0'))]
    for marker in selected:
        begin, end = marker['start'], marker['end']
        for device in devices:
            stage_groups = defaultdict(lambda: dict(calls=0, total_us=0.0,
                api_linked=0, graph_nodes=0, launch_to_start_us=0.0))
            for r in kernels:
                if r['deviceId'] != device or not (begin <= r['start'] and r['end'] <= end):
                    continue
                row = stage_groups[strings[r['demangledName']]]
                row['calls'] += 1
                row['total_us'] += (r['end']-r['start'])/1000
                row['graph_nodes'] += bool(r.get('graphNodeId'))
                launch = api.get((r['globalPid'], r['correlationId']))
                if launch is not None:
                    row['api_linked'] += 1
                    row['launch_to_start_us'] += (r['start']-launch['start'])/1000
            phase_kernels.extend(dict(phase=marker['name'], device=device, name=name,
                scope='fully_contained_in_rank0_host_range', **row,
                mean_kernel_us=row['total_us']/row['calls'],
                mean_api_start_to_kernel_start_us=(row['launch_to_start_us']/row['api_linked']
                    if row['api_linked'] else None)) for name,row in stage_groups.items())
            def clipped(rows):
                return [(max(begin, r['start']), min(end, r['end'])) for r in rows
                        if r['deviceId'] == device and r['start'] < end and r['end'] > begin]
            computation, communication = [], []
            for r in kernels:
                name = strings[r['demangledName']].lower()
                destination = communication if any(x in name for x in
                    ('nccl', 'allreduce', 'all_gather', 'cross_device_reduce')) else computation
                destination.append(r)
            compute = clipped(computation)
            comm = clipped(communication)
            copy = clipped(copies)
            fill = clipped(memset)
            active = union(compute + comm + copy + fill)
            required_actor_kernel_missing = (marker['name'].startswith(
                ('actor_actor_forward_start_', 'actor_backward_start_')) and not stage_groups)
            windows.append(dict(phase=marker['name'], device=device,
                scope='rank0_host_range_clipped_gpu_activity',
                kernel_coverage='missing' if required_actor_kernel_missing else 'observed',
                range_ms=(end-begin)/1e6, compute_union_ms=union(compute)/1e6,
                communication_union_ms=union(comm)/1e6, copy_union_ms=union(copy)/1e6,
                active_union_ms=active/1e6,
                no_observed_activity_ms=None if required_actor_kernel_missing else ((end-begin)-active)/1e6,
                compute_communication_overlap_ms=(union(compute)+union(comm)-union(compute+comm))/1e6))
    result = dict(database=str(path), observed_devices=devices, markers=markers,
                  kernel_count=len(kernels), api_linked_kernels=sum(r['api_linked'] for r in grouped.values()),
                  copy_events=len(copies), windows=windows,
                  copies=[dict(device=k[0], direction=copy_names.get(k[1], str(k[1])), **v,
                               bytes_per_summed_copy_second=v['bytes']/(v['total_us']/1e6) if v['total_us'] else None)
                          for k,v in copy_groups.items()],
                  cuda_api=[dict(pid=k[0], source_table=k[1], name=k[2], **v,
                                 mean_us=v['total_us']/v['calls']) for k,v in api_groups.items()],
                  kernels=[dict(device=k[0], pid=k[1], name=k[2], **v) for k, v in grouped.items()],
                  phase_kernels=phase_kernels,
                  limitations=['Activity gaps are not CPU expert time.',
                               'Named communication classification is heuristic.',
                               'Nested phase windows must not be summed.',
                               'API-start to kernel-start includes submission and queue delay, not pure scheduler latency.',
                               'Graph nodes can share one graph launch API; per-node launch delay is not additive.',
                               'Driver and runtime API times are reported separately and can overlap.',
                               'Missing devices are missing coverage, not idle devices.'])
    con.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(args.database)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ('observed_devices', 'kernel_count', 'api_linked_kernels')}))

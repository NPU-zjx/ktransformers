"""Offline event counts and same-thread exclusive CPU durations; no training."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def analyze(path, start_ns=None, end_ns=None):
    raw = path.read_bytes()
    trace = json.loads(raw)
    events = [e for e in trace['traceEvents'] if e.get('ph') == 'X'
              and e.get('cat') in {'cpu_op', 'kernel', 'gpu_memcpy', 'gpu_memset', 'cuda_runtime'}
              and isinstance(e.get('dur'), (int, float)) and e['dur'] >= 0]
    excluded_boundary_events = 0
    if start_ns is not None or end_ns is not None:
        assert start_ns is not None and end_ns is not None and start_ns < end_ns
        base = trace['baseTimeNanoseconds']
        start_us, end_us = (start_ns - base) / 1000, (end_ns - base) / 1000
        excluded_boundary_events = sum(
            e['ts'] < end_us and e['ts'] + e['dur'] > start_us and
            not (start_us <= e['ts'] and e['ts'] + e['dur'] <= end_us) for e in events)
        events = [e for e in events if start_us <= e['ts'] and e['ts'] + e['dur'] <= end_us]
    threads = defaultdict(list)
    for i, e in enumerate(events):
        if e['cat'] == 'cpu_op':
            threads[(e['pid'], e['tid'])].append(i)
    exclusive = {}
    invalid = set()
    for indices in threads.values():
        stack = []
        for i in sorted(indices, key=lambda i: (events[i]['ts'], -events[i]['dur'])):
            e = events[i]
            end = e['ts'] + e['dur']
            while stack and events[stack[-1]]['ts'] + events[stack[-1]]['dur'] <= e['ts']:
                stack.pop()
            exclusive[i] = e['dur']
            if stack:
                parent = stack[-1]
                pe = events[parent]
                if end > pe['ts'] + pe['dur']:
                    invalid.update(stack)
                    invalid.add(i)
                else:
                    exclusive[parent] -= e['dur']
            stack.append(i)
    groups = defaultdict(lambda: {'calls': 0, 'total_us': 0, 'self_cpu_us': 0,
                                  'self_cpu_valid': True})
    for i, e in enumerate(events):
        g = groups[(e['cat'], e['name'])]
        g['calls'] += 1
        g['total_us'] += e['dur']
        if e['cat'] != 'cpu_op' or i in invalid or exclusive.get(i, 0) < 0:
            g['self_cpu_valid'] = False
        else:
            g['self_cpu_us'] += exclusive[i]
    rows = []
    for (category, name), g in sorted(groups.items()):
        valid = g.pop('self_cpu_valid')
        if not valid:
            g['self_cpu_us'] = None
        rows.append(dict(category=category, name=name, **g,
                         mean_us=g['total_us'] / g['calls']))
    counts = {cat: sum(r['calls'] for r in rows if r['category'] == cat)
              for cat in sorted({r['category'] for r in rows})}
    return dict(source=str(path), sha256=hashlib.sha256(raw).hexdigest(), counts=counts,
                start_unix_ns=start_ns, end_unix_ns=end_ns,
                excluded_boundary_events=excluded_boundary_events,
                crossing_cpu_events=len(invalid), rows=rows,
                scope='Recorded rank/window only. CPU total is inclusive, not additive. '
                      'Self CPU excludes nested cpu_op events on the same thread. '
                      'Kernel totals are event sums, not GPU busy-time unions. '
                      'No CPU-to-kernel attribution inferred from names.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start-ns', type=int)
    parser.add_argument('--end-ns', type=int)
    args = parser.parse_args()
    result = analyze(args.trace, args.start_ns, args.end_ns)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))

"""Reconstruct requested-byte peak within retained history, backwards from snapshot.

This measures one PyTorch allocator, not whole-device physical VRAM. No fabricated
activation classification: report allocation stacks and retain unknown attribution.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path


def current_allocations(data, device):
    """Summarize live blocks at the snapshot itself, not an earlier peak."""
    states, requested, stacks, counts = Counter(), 0, Counter(), Counter()
    reserved = 0
    for segment in data['segments']:
        if segment['device'] != device:continue
        reserved += segment['total_size']
        for block in segment['blocks']:
            if block['state'] not in ('active_allocated','active_pending_free','inactive'):
                raise ValueError('Unknown allocator block state')
            states[block['state']] += block['size']
            if block['state'] != 'active_allocated':continue
            requested += block['requested_size']
            frames = block.get('frames',[])
            py = [f for f in frames if f.get('filename','').endswith('.py')]
            chosen = py[:8] if py else frames
            stack = tuple(f'{f.get("filename", "")}:{f.get("line", 0)}:{f.get("name", "")}' for f in chosen)
            stacks[stack] += block['requested_size']; counts[stack] += 1
    errors = []
    if sum(states.values()) != reserved:errors.append('Block sizes do not sum to segment sizes')
    if sum(stacks.values()) != requested:errors.append('Stack attribution does not sum to requested bytes')
    return dict(device=device,snapshot_requested_bytes=requested,
                snapshot_allocated_block_bytes=states['active_allocated'],
                snapshot_pending_free_block_bytes=states['active_pending_free'],
                snapshot_inactive_block_bytes=states['inactive'],snapshot_reserved_segment_bytes=reserved,
                unknown_allocation_stack_bytes=stacks.get((),0),consistency_errors=errors,
                top_stacks=[dict(bytes=v,allocations=counts[k],stack=list(k)) for k,v in stacks.most_common()],
                limitations=['Current active allocations only; this is not a historical peak.',
                             'Producer stacks do not identify Python reference owners.',
                             'Reserved virtual sleep pools and allocations outside PyTorch require separate physical accounting.'])


def reconstruct(data, device):
    live = {}
    segments = {s['address']:s['total_size'] for s in data['segments']
                if s['device']==device and 'address' in s and 'total_size' in s}
    reserved = sum(segments.values())
    final_reserved = reserved
    final_block_bytes = 0
    for seg in data['segments']:
        if seg['device'] != device:
            continue
        for block in seg['blocks']:
            if block['state'] == 'active_allocated':
                final_block_bytes += block['size']
                live[block['address']] = dict(size=block['requested_size'], frames=block['frames'], source='snapshot')
    traces = data['device_traces'][device]
    alloc_indices = defaultdict(list)
    for i, event in enumerate(traces):
        if event['action'] == 'alloc':
            alloc_indices[event['addr']].append(i)
    current = sum(b['size'] for b in live.values())
    final_bytes = current
    peak = current
    peak_blocks = dict(live)
    peak_reserved = reserved
    peak_event = len(traces)
    consistency_errors = []
    for i in range(len(traces)-1, -1, -1):
        event = traces[i]
        addr = event.get('addr')
        if event['action'] == 'alloc':
            block = live.pop(addr, None)
            if block is None or block['size'] != event['size']:
                consistency_errors.append(dict(index=i, action='alloc', address=addr, error='reverse allocation mismatch'))
            current -= event['size']
        elif event['action'] == 'free_requested':
            if addr in live:
                consistency_errors.append(dict(index=i, action='free_requested', error='address already live'))
            prior = alloc_indices.get(addr, [])
            j = bisect.bisect_left(prior, i)-1
            origin = traces[prior[j]] if j >= 0 else None
            live[addr] = dict(size=event['size'], frames=origin.get('frames', []) if origin else [],
                              source='recorded_alloc' if origin else 'allocation_before_retained_history')
            current += event['size']
        elif event['action'] == 'segment_alloc':
            size = segments.pop(addr, None)
            if size != event['size']:
                consistency_errors.append(dict(index=i, action='segment_alloc', error='reverse segment mismatch'))
            reserved -= event['size']
        elif event['action'] == 'segment_free':
            if addr in segments:
                consistency_errors.append(dict(index=i, action='segment_free', error='segment already present'))
            segments[addr] = event['size']
            reserved += event['size']
        elif event['action'] == 'segment_map':
            # Expandable segments reserve virtual address ranges, but these
            # events account only for physically mapped allocator pages.
            reserved -= event['size']
        elif event['action'] == 'segment_unmap':
            reserved += event['size']
        if reserved < 0:
            consistency_errors.append(dict(index=i, error='negative reserved bytes'))
        if current > peak:
            peak = current
            peak_event = i
            peak_blocks = dict(live)
            peak_reserved = reserved
    if sum(x['size'] for x in peak_blocks.values()) != peak:
        consistency_errors.append(dict(error='peak block sum mismatch'))
    stacks = Counter()
    stack_counts = Counter()
    unknown = 0
    no_python = 0
    top = []
    for addr, block in peak_blocks.items():
        python_frames = [f for f in block['frames'] if f.get('filename', '').endswith('.py')]
        if not python_frames:
            no_python += block['size']
        # Autograd worker allocations often have C++ frames only. Retain these:
        # missing Python frames does not mean the producer is unrecorded.
        chosen_frames = python_frames[:8] if python_frames else block['frames']
        stack = tuple(f'{f.get("filename", "")}:{f.get("line", 0)}:{f.get("name", "")}' for f in chosen_frames)
        if not stack:
            unknown += block['size']
        stacks[stack] += block['size']
        stack_counts[stack] += 1
        top.append(dict(address=addr, bytes=block['size'], stack=list(stack), source=block['source'],
                        frame_kind='python' if python_frames else ('cpp' if stack else 'missing')))
    return dict(device=device, retained_events=len(traces), snapshot_requested_bytes=final_bytes,
                snapshot_allocated_block_bytes=final_block_bytes,
                snapshot_reserved_segment_bytes=final_reserved,
                peak_reserved_segment_bytes=peak_reserved,
                reserved_replay_supported=True,
                reserved_constant_over_retained_history=not any(
                    e['action'] in ('segment_alloc','segment_free','segment_map','segment_unmap') for e in traces),
                retained_history_peak_requested_bytes=peak, peak_before_event=peak_event,
                peak_time_us=(traces[peak_event].get('time_us') if peak_event < len(traces) else None),
                peak_interval_start_us=(traces[peak_event-1].get('time_us') if peak_event else None),
                retained_start_us=(traces[0].get('time_us') if traces else None),
                retained_end_us=(traces[-1].get('time_us') if traces else None),
                unknown_allocation_stack_bytes=unknown, consistency_errors=consistency_errors,
                missing_python_stack_bytes=no_python,
                top_allocations=sorted(top, key=lambda x:x['bytes'], reverse=True)[:40],
                top_stacks=[dict(bytes=n, allocations=stack_counts[s], stack=list(s)) for s,n in stacks.most_common(30)],
                limitations=['Peak applies only to this retained history, not the whole step/run.',
                             'Trace sizes are requested bytes; allocator block rounding is excluded from replay.',
                             'Allocation stack identifies producer; it does not automatically label activation vs workspace.',
                             'Virtual/sleeping pools and external CUDA allocations require separate reconciliation.'])


def sample_window_bounds(data, device, start, end):
    """Bound allocator states throughout a complete system sampling interval.

NVML is collected inside this interval. Bounds avoid claiming that the start
timestamp is the exact GPU query time. Requires monotonically ordered epoch-us
trace timestamps and full coverage of the interval in retained history.
"""
    traces=data['device_traces'][device]
    times=[e.get('time_us') for e in traces]
    if not times or any(t is None for t in times) or any(a>b for a,b in zip(times,times[1:])):
        return dict(fully_covered=False, reason='missing or nonmonotonic trace timestamps')
    lo, hi=start*1e6, end*1e6
    if lo < times[0] or hi > times[-1] or hi < lo:
        return dict(fully_covered=False, reason='sampling interval outside retained trace window')
    requested=sum(b['requested_size'] for s in data['segments'] if s['device']==device
                  for b in s['blocks'] if b['state']=='active_allocated')
    reserved=sum(s['total_size'] for s in data['segments'] if s['device']==device)
    values=[]
    for i in range(len(traces)-1,0,-1):
        event=traces[i]
        if event['action']=='alloc': requested-=event['size']
        elif event['action']=='free_requested': requested+=event['size']
        elif event['action'] in ('segment_alloc','segment_map'): reserved-=event['size']
        elif event['action'] in ('segment_free','segment_unmap'): reserved+=event['size']
        if times[i-1] <= hi and times[i] >= lo:
            values.append((requested,reserved))
        if times[i-1] < lo: break
    if not values:
        return dict(fully_covered=False,reason='no overlapping allocator interval')
    return dict(fully_covered=True, start_unix=start, end_unix=end,
                requested_min_bytes=min(x[0] for x in values),requested_max_bytes=max(x[0] for x in values),
                reserved_min_bytes=min(x[1] for x in values),reserved_max_bytes=max(x[1] for x in values))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('snapshot')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out', required=True)
    p.add_argument('--at-snapshot', action='store_true')
    args = p.parse_args()
    # Only use trusted snapshots produced by this experiment.
    with Path(args.snapshot).open('rb') as stream:
        result = (current_allocations if args.at_snapshot else reconstruct)(pickle.load(stream), args.device)
    result['snapshot'] = args.snapshot
    Path(args.out).write_text(json.dumps(result, indent=2))
    keys = ('snapshot_requested_bytes','snapshot_reserved_segment_bytes','consistency_errors') if args.at_snapshot else ('retained_events','snapshot_requested_bytes','retained_history_peak_requested_bytes','consistency_errors')
    print(json.dumps({k:result[k] for k in keys}))
    raise SystemExit(1 if result['consistency_errors'] else 0)

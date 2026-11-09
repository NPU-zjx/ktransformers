"""TP-deduplicated phase throughput from CUDA stream interval unions.

CUDA event timestamps are anchored to a bounded host clock interval per worker.
Stream elapsed time includes queued work and CPU-expert waits, not just kernels.
"""
import argparse
import csv
import json
from pathlib import Path


def union_ns(intervals):
    merged=[]
    for start,end in sorted(intervals):
        assert end>=start, (start,end)
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return sum(b-a for a,b in merged)


def analyze(folder, input_tokens, accepted_decode_tokens, output):
    traces=[json.loads(p.read_text()) for p in folder.glob('phase-*.json')]
    assert len(traces)==8 and {t['rank'] for t in traces}==set(range(8))
    traces.sort(key=lambda t:t['rank'])
    baseline=[(r['phase'],r['tokens']) for r in traces[0]['rows']]
    assert all([(r['phase'],r['tokens']) for r in t['rows']]==baseline for t in traces)
    assert all(r['phase'] in ('EXTEND','DECODE') for t in traces for r in t['rows'])
    assert sum(r['tokens'] for r in traces[0]['rows'] if r['phase']=='EXTEND')==input_tokens
    assert all(r['graph_replayed'] for t in traces for r in t['rows'] if r['phase']=='DECODE')
    scheduled=sum(r['tokens'] for r in traces[0]['rows'] if r['phase']=='DECODE')
    assert scheduled>=accepted_decode_tokens
    all_rows=[dict(rank=t['rank'],anchor_lower_ns=t['lower'],anchor_upper_ns=t['upper'],**r) for t in traces for r in t['rows']]
    result={'ranks':8,'input_tokens':input_tokens,'accepted_decode_tokens':accepted_decode_tokens,
            'scheduled_decode_tokens':scheduled,'extra_scheduled_decode_tokens':scheduled-accepted_decode_tokens,
            'clock_anchor_max_bracket_ns':max(t['upper']-t['lower'] for t in traces),'phases':{}}
    all_mid=[]
    for phase,tokens in [('EXTEND',input_tokens),('DECODE',accepted_decode_tokens)]:
        rows=[r for r in all_rows if r['phase']==phase]
        bounds={}
        # Conservative independent-start/end anchor bounds for the cross-rank union.
        for key in ('lower','upper'):
            start_key='upper' if key=='lower' else 'lower'
            bounds[key]=union_ns([(r[f'anchor_{start_key}_ns']+int(r['stream_start_ms']*1e6),
                max(r[f'anchor_{start_key}_ns']+int(r['stream_start_ms']*1e6),r[f'completion_{key}_ns'])) for r in rows])/1e9
        mid=[((r['anchor_lower_ns']+r['anchor_upper_ns'])//2+int(r['stream_start_ms']*1e6),
              (r['completion_lower_ns']+r['completion_upper_ns'])//2) for r in rows]
        duration=union_ns(mid)/1e9;all_mid.extend(mid)
        result['phases'][phase]={'tokens':tokens,'seconds':duration,'tps':tokens/duration,
            'seconds_lower':bounds['lower'],'seconds_upper':bounds['upper'],
            'host_entry_to_completion_union_seconds':union_ns([(r['cpu_start_ns'],max(r['cpu_start_ns'],(r['completion_lower_ns']+r['completion_upper_ns'])//2)) for r in rows])/1e9,
            'batches_per_rank':len(rows)//8,'graph_replay_batches_all_ranks':sum(r['graph_replayed'] for r in rows)}
    result['phase_overlap_seconds']=sum(p['seconds'] for p in result['phases'].values())-union_ns(all_mid)/1e9
    result['allocator_memory']=[dict(rank=t['rank'],gpu_uuid=t['uuid'],
        window_peak_allocated_bytes=t['window_max_allocated_bytes'],window_peak_reserved_bytes=t['window_max_reserved_bytes'],
        **{f'{phase.lower()}_boundary_max_{kind}_bytes':max(r[f'{kind}_{when}_bytes'] for r in t['rows'] if r['phase']==phase for when in ('before','after'))
           for phase in ('EXTEND','DECODE') for kind in ('allocated','reserved')}) for t in traces]
    result['definitions']={'denominator':'Union across TP workers of current-stream CUDA start/end events around forward_batch_generation. Excludes work queued before the start event, scheduler output handling and network; includes CPU-expert and communication waits inside the stream interval. Not pure GPU kernel time. Host entry-to-completion residence time is separately reported and may overlap preceding phases.',
        'prefill_numerator':'Input token count, counted once on TP0, prefix cache disabled.',
        'decode_numerator':'Accepted output tokens minus one first token per request. Speculative/overlap discarded scheduled steps reported separately.',
        'memory':'Allocator before/after boundary maximum per phase; not an instantaneous per-phase peak. Whole-window allocator peak separately recorded.',
        'clock':'Bounds cover anchor bracket only; GPU/CPU clock drift over long windows has not been independently calibrated.'}
    output.mkdir(exist_ok=True,parents=True)
    (output/'phase-summary.json').write_text(json.dumps(result,indent=2))
    with (output/'phase-events.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('allocator_memory','definitions')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--input-tokens',type=int,required=True)
    p.add_argument('--decode-tokens',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();analyze(args.folder,args.input_tokens,args.decode_tokens,args.output)

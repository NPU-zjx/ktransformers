"""Derive client latency metrics from streaming events, never infer phase TPS."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def percentile(values, q):
    values = sorted(values)
    position = (len(values)-1)*q
    lo = int(position)
    hi = min(lo+1, len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def analyze(run):
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(run.glob('baseline-request-*.json'))]
    assert len(records) == 32
    rows = []
    for r in records:
        events = r['events_ns_tokens_delta']
        assert events and events[-1][1] == r['output_tokens']
        assert sum(e[2] for e in events) == r['output_tokens']
        first_exact = events[0][2] == 1
        rows.append(dict(index=r['index'], input_tokens=r['prompt_tokens'], output_tokens=r['output_tokens'],
                         response_seconds=(r['end_ns']-r['start_ns'])/1e9,
                         first_chunk_ms=(events[0][0]-r['start_ns'])/1e6,
                         ttft_ms=(events[0][0]-r['start_ns'])/1e6 if first_exact else None,
                         tpot_ms=(events[-1][0]-events[0][0])/1e6/(r['output_tokens']-1)
                         if first_exact and r['output_tokens']>1 else None,
                         stream_events=len(events), multi_token_chunks=sum(e[2]>1 for e in events)))
    elapsed=(max(r['end_ns'] for r in records)-min(r['start_ns'] for r in records))/1e9
    summary=dict(requests=len(rows),input_tokens=sum(r['input_tokens'] for r in rows),
                 output_tokens=sum(r['output_tokens'] for r in rows),rollout_seconds=elapsed)
    summary['overall_rollout_tps']=summary['output_tokens']/elapsed
    for metric in ('ttft_ms','tpot_ms','response_seconds'):
        values=[r[metric] for r in rows if r[metric] is not None]
        summary[metric]=dict(valid_requests=len(values),mean=statistics.mean(values),
                            p50=percentile(values,.5),p95=percentile(values,.95),p99=percentile(values,.99)) if values else None
    summary['definitions']={'ttft':'Client first single-token SSE arrival minus request start; null for multi-token first chunk.',
                            'tpot':'Last token-bearing SSE arrival minus first, divided by output_tokens - 1; per-request average, not inter-token percentile.',
                            'rollout':'Earliest request start through latest response completion; includes scheduling and network.',
                            'phase_tps':'Not inferred from client events.'}
    output=run/'analysis';output.mkdir(exist_ok=True)
    with (output/'request-metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'client-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    analyze(parser.parse_args().run)

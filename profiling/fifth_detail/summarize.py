"""Evidence-backed stage tables. Residuals remain explicitly unclassified."""
import argparse
import json
from pathlib import Path
import statistics
import sys


def records(path):
    with path.open() as f:
        return [json.loads(s) for s in f if s.strip()]


def percentile(values, q):
    values=sorted(values)
    if not values:return None
    index=(len(values)-1)*q;lo=int(index);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();root=a.root;out=root/'analysis';out.mkdir(exist_ok=True)
    resource=records(root/'resources/container.jsonl')
    stages=[]
    for path in sorted((root/'rl').glob('**/stages.jsonl')):
        for row in records(path):
            if row.get('status')!='complete':continue
            begin,end=row['start_unix_ns'],row['end_unix_ns']
            selected=[r for r in resource if begin<=r['start_ns']<=r['end_ns']<=end]
            item=dict(folder=str(path.parent),phase=row['phase'],seconds=(end-begin)/1e9,samples=len(selected))
            if selected:
                peak=max(selected,key=lambda r:int(r['memory.current']))
                stat={k:int(v) for k,v in (s.split() for s in peak['memory.stat'].splitlines())}
                item.update(container_peak_bytes=int(peak['memory.current']),peak_time_ns=peak['start_ns'],
                    peak_components_bytes={k:stat.get(k) for k in ('anon','file','kernel','shmem','file_dirty','file_writeback','slab','pagetables')},
                    peak_sample=peak)
            stages.append(item)
    (out/'stage-container.json').write_text(json.dumps(stages,indent=2))
    sys.path.insert(0,str(a.source/'profiling/analysis'))
    from analyze_phases import analyze
    rollouts=[]
    plan=json.loads((root/'rl/resolved-plan.json').read_text())
    for wave in sorted((root/'profiles/client').glob('wave*')):
        rows=[json.loads(p.read_text()) for p in wave.glob('*.json')]
        if not rows:continue
        idx=int(wave.name[4:])
        matches=[p.parent for p in (root/'rl').glob('*/round.json')
                 if json.loads(p.read_text())['dataset_batch']==idx]
        if len(matches)!=1:
            raise ValueError(f'Cannot uniquely map client wave {idx} to dataset batch')
        rnd=matches[0]
        if len(rows)!=plan['inputs']['responses_per_update']:
            continue  # The active rollout has not finished; never report partial throughput.
        # The stage includes native instrumentation control/flush overhead. Also
        # report client submission->completion independently, never conflate.
        wall=(max(r['end_ns'] for r in rows)-min(r['start_ns'] for r in rows))/1e9
        result=dict(wave=idx,requests=len(rows),input_tokens=sum(r['input_tokens'] for r in rows),
            output_tokens=sum(r['output_tokens'] for r in rows),client_wall_seconds=wall,
            client_output_tps=sum(r['output_tokens'] for r in rows)/wall,
            truncated_fraction=sum(r['finish_reason']['type']=='length' for r in rows)/len(rows))
        for metric in ('ttft_ms','tpot_ms','first_to_second_token_ms','steady_tpot_ms'):
            values=[r[metric] for r in rows if r.get(metric) is not None]
            result[metric]=dict(mean=statistics.fmean(values) if values else None,
                **{f'p{int(q*100)}':percentile(values,q) for q in (.5,.9,.95,.99)},max=max(values) if values else None)
        rollouts.append(result)
        traces=list((rnd/'rollout/phase').glob('phase-*.json'))
        world_size=plan['rollout']['server_args']['tp_size']
        if len(traces)!=world_size:
            raise ValueError(f'Wave {idx}: expected {world_size} phase traces, got {len(traces)}')
        # Count actual EXTEND work, not cached request tokens, for compute TPS.
        tp0 = next(json.loads(p.read_text()) for p in traces if json.loads(p.read_text())['rank']==0)
        computed = sum(r['tokens'] for r in tp0['rows'] if r['phase']=='EXTEND')
        result['computed_prefill_tokens'] = computed
        result['cached_tokens_reported'] = sum(r.get('cached_tokens') or 0 for r in rows)
        analyze(rnd/'rollout/phase',computed,
                result['output_tokens']-len(rows),out/f'rollout-{idx:02d}',world_size=world_size)
    (out/'rollout-summary.json').write_text(json.dumps(rollouts,indent=2))
    print(json.dumps(dict(stages=len(stages),rollouts=rollouts),indent=2))


if __name__=='__main__':main()

"""All-rank first-microbatch trace and native expert analysis, CPU-only."""
import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys


def union(values):
    merged=[]
    for start,end in sorted(values):
        if end<start:raise ValueError('negative duration')
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return sum(b-a for a,b in merged)


def kernel_kind(name):
    name=name.lower()
    if 'nccl' in name:return 'nccl_communication_or_wait'
    if any(s in name for s in ['gemm','matmul','mma','cutlass']):return 'matrix_compute'
    if 'attention' in name or 'flash' in name:return 'attention'
    return 'other_recorded_kernel'


def csv_file(path,records):
    if not records:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def analyze(path,profile,out):
    raw=path.read_bytes();source_hash=hashlib.sha256(raw).hexdigest();trace=json.loads(raw);del raw
    pid=int(path.name.split('-')[2].split('.')[0])
    timings=[json.loads(s) for s in (profile/f'timing-actor-pid{pid}.jsonl').read_text().splitlines()]
    first=min((r for r in timings if r['label']=='training_microbatch'),key=lambda r:r['start_ns'])
    rank=int(first['rank']);base=trace['baseTimeNanoseconds']
    phases=[dict(label='first_microbatch',start_ns=first['start_ns'],end_ns=first['end_ns'])]
    phases.extend(r for r in timings if r['label'] in ('policy_forward_loss','backward') and
                  first['start_ns']<=r['start_ns']<r['end_ns']<=first['end_ns'])
    complete=[e for e in trace['traceEvents'] if e.get('ph')=='X' and isinstance(e.get('dur'),(int,float)) and e['dur']>=0]
    categories=Counter(e.get('cat') for e in trace['traceEvents'])
    cpu_by_external={e.get('args',{}).get('External id'):e['name'] for e in complete if e.get('cat')=='cpu_op' and 'External id' in e.get('args',{})}
    phase_reports=[];flat=[]
    for phase in phases:
        begin,end=(phase['start_ns']-base)/1000,(phase['end_ns']-base)/1000
        selected=[e for e in complete if begin<=e['ts'] and e['ts']+e['dur']<=end]
        boundary_crossings=Counter(e.get('cat') for e in complete if e['ts']<end and e['ts']+e['dur']>begin and not(begin<=e['ts'] and e['ts']+e['dur']<=end))
        cpu=[e for e in selected if e.get('cat')=='cpu_op'];threads=defaultdict(list)
        for i,e in enumerate(cpu):threads[(e['pid'],e['tid'])].append(i)
        exclusive={i:e['dur'] for i,e in enumerate(cpu)};invalid=set()
        for indices in threads.values():
            stack=[]
            for i in sorted(indices,key=lambda i:(cpu[i]['ts'],-cpu[i]['dur'])):
                e=cpu[i];stop=e['ts']+e['dur']
                while stack and cpu[stack[-1]]['ts']+cpu[stack[-1]]['dur']<=e['ts']:stack.pop()
                if stack:
                    parent=stack[-1]
                    if stop>cpu[parent]['ts']+cpu[parent]['dur']+1e-3:invalid.update(stack);invalid.add(i)
                    else:exclusive[parent]-=e['dur']
                stack.append(i)
        groups=defaultdict(lambda:dict(calls=0,total_us=0.,self_us=0.,self_valid=True))
        for i,e in enumerate(cpu):
            group=groups[('cpu_op',e['name'])];group['calls']+=1;group['total_us']+=e['dur']
            if i in invalid or exclusive[i]<-1e-3:group['self_valid']=False
            else:group['self_us']+=max(exclusive[i],0)
        kernels=[e for e in selected if e.get('cat')=='kernel'];linked=0;gpu_groups=defaultdict(list)
        associated=Counter();transfer=Counter()
        for e in selected:
            cat=e.get('cat')
            if cat=='cpu_op':continue
            if cat in ('kernel','cuda_runtime','cuda_driver','gpu_memcpy','gpu_memset','user_annotation'):
                g=groups[(cat,e['name'])];g['calls']+=1;g['total_us']+=e['dur'];g['self_valid']=False
            if cat in ('gpu_memcpy','gpu_memset'):
                args=e.get('args',{});key=e['name'];transfer[key+'.calls']+=1;transfer[key+'.bytes']+=int(args.get('bytes',0))
            if cat=='kernel':
                gpu_groups[kernel_kind(e['name'])].append((e['ts'],e['ts']+e['dur']))
                name=cpu_by_external.get(e.get('args',{}).get('External id'))
                if name is not None:linked+=1;associated[name]+=e['dur']
        for (cat,name),g in groups.items():
            flat.append(dict(rank=rank,phase=phase['label'],category=cat,name=name,calls=g['calls'],
                total_us=g['total_us'],mean_us=g['total_us']/g['calls'],self_cpu_us=g['self_us'] if g['self_valid'] else None,
                source=path.name,sha256=source_hash))
        report=dict(rank=rank,label=phase['label'],wall_seconds=(end-begin)/1e6,
            counts=dict(Counter(e.get('cat') for e in selected)),boundary_crossings=dict(boundary_crossings),
            crossing_cpu_events=len(invalid),kernel_calls=len(kernels),kernels_linked_by_external_id=linked,
            kernel_interval_union_seconds=union([(e['ts'],e['ts']+e['dur']) for e in kernels])/1e6,
            kernel_groups={k:dict(calls=len(v),event_sum_seconds=sum(b-a for a,b in v)/1e6,interval_union_seconds=union(v)/1e6) for k,v in gpu_groups.items()},
            transfers=dict(transfer),largest_kernel_producers=[dict(name=k,gpu_event_sum_seconds=v/1e6) for k,v in associated.most_common(12)])
        phase_reports.append(report)
    csv_file(out/f'operators-rank{rank}.csv',flat)
    summary=dict(source=str(path),sha256=source_hash,rank=rank,all_categories=dict(categories),phases=phase_reports,
        limits=['First profiled microbatch only; not all optimizer steps or the entire RL round.',
                'Kernel duration includes NCCL polling/wait; interval union is not SM compute utilization.',
                'CPU inclusive totals and concurrent thread durations cannot be added to wall time.',
                'Only fully-contained events assigned to a phase; crossing events counted separately.',
                'External-id producer relation is recorded correlation, not exclusive ownership of all work.'])
    (out/f'trace-rank{rank}.json').write_text(json.dumps(summary,indent=2));return summary


def native(path,out):
    raw=path.read_bytes();data=json.loads(raw);records=[];groups=defaultdict(lambda:dict(calls=0,ns=0,bytes=0,layers=0))
    for layer,stats in data['layers'].items():
        for key,value in stats.items():
            if not key.endswith('.total_ns'):continue
            counter=key.removesuffix('.total_ns');calls=stats.get(counter+'.calls')
            if calls is None:raise ValueError(key)
            row=dict(layer=int(layer),counter=counter,calls=int(calls),total_ns=value,bytes=stats.get(counter+'.bytes',0),
                scope='summed_worker_thread_time' if '.worker_cpu.' in counter else 'native_inclusive_timer')
            records.append(row);g=groups[counter];g['calls']+=int(calls);g['ns']+=value;g['bytes']+=row['bytes'];g['layers']+=1
    csv_file(out/'native-experts-per-layer.csv',records)
    summary=dict(source=str(path),sha256=hashlib.sha256(raw).hexdigest(),enabled=data['enabled'],layers=len(data['layers']),
        counters=[dict(counter=k,seconds=v['ns']/1e9,calls=v['calls'],bytes=v['bytes'],layers=v['layers'],
            scope='summed_worker_thread_time' if '.worker_cpu.' in k else 'native_inclusive_timer') for k,v in sorted(groups.items())],
        limits=['Counters may be nested or parallel; never sum across categories as wall time.',
                'Worker CPU times sum across threads and may exceed stage wall time.',
                'Byte counters are transferred/processed volume, not resident RAM capacity.',
                'Counter enabled with zero calls means unobserved path, not a measured zero-cost kernel.'])
    (out/'native-experts-summary.json').write_text(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root;out=root/'analysis/trace-details';out.mkdir(exist_ok=True)
    try:os.nice(5)
    except OSError:pass
    result=[]
    for path in sorted((root/'profiles').glob('actor-microbatch-*.trace.json')):
        summary=analyze(path,root/'profiles',out);result.append(summary)
        print(json.dumps(dict(rank=summary['rank'],phases=[dict(label=r['label'],kernels=r['kernel_calls'],linked=r['kernels_linked_by_external_id']) for r in summary['phases']])),flush=True)
        gc.collect()
    kt=native(root/'rl/round-01/actor/native-expert-profile.json',out)
    (out/'all-ranks.json').write_text(json.dumps(dict(ranks=result,native=kt),indent=2))
    print('TRACE_ANALYSIS_COMPLETE',len(result),'ranks',kt['layers'],'native layers',flush=True)


if __name__=='__main__':main()

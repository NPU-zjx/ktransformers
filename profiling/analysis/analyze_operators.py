"""Parse real Torch operator exports; preserve shapes and rank before aggregation."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re


def write_csv(path,rows):
    assert rows
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(folder):
    detail=[];seen=set()
    sums=('calls','cpu_total_us','cpu_self_us','device_total_us','device_self_us',
          'cpu_net_memory_bytes','device_net_memory_bytes','device_self_net_memory_bytes')
    for path in sorted(folder.glob('operators-*.csv')):
        match=re.fullmatch(r'operators-TP(\d+)-(EXTEND|DECODE)-(\d+)\.csv',path.name)
        assert match, path.name
        rank,phase=int(match[1]),match[2]
        assert (rank,phase) not in seen, 'Repeated stage exports require separate windows'
        seen.add((rank,phase))
        with path.open(encoding='utf-8') as f:
            for row in csv.DictReader(f):
                detail.append(dict(rank=rank,phase=phase,operator=row['operator'],shapes=row['shapes'],
                    **{k:float(row[k]) for k in sums},source=path.name))
    assert seen=={(rank,phase) for rank in range(8) for phase in ('EXTEND','DECODE')},seen
    grouped={}
    for row in detail:
        key=(row['rank'],row['phase'],row['operator'])
        if key not in grouped:grouped[key]=dict(rank=key[0],phase=key[1],operator=key[2],**{k:0 for k in sums})
        for k in sums:grouped[key][k]+=row[k]
    cpu_counts=defaultdict(int);cpu_shape_counts=defaultdict(int);kernel_groups={};trace_coverage=[]
    for path in sorted(folder.glob('*.trace.json.gz')):
        match=re.search(r'TP-(\d+)-(EXTEND|DECODE)\.trace\.json\.gz$',path.name)
        assert match,path.name
        rank,phase=int(match[1]),match[2]
        trace=json.load(gzip.open(path,'rt'));events=trace['traceEvents']
        categories=defaultdict(int)
        for event in events:
            cat=event.get('cat','');categories[cat]+=1
            if cat in ('cpu_op','user_annotation') and event.get('ph')=='X':
                cpu_counts[(rank,phase,event['name'])]+=1
                shape=json.dumps(event.get('args',{}).get('Input Dims',[]))
                cpu_shape_counts[(rank,phase,event['name'],shape)]+=1
            if cat=='kernel' and event.get('ph')=='X':
                key=(rank,phase,event['name'])
                if key not in kernel_groups:kernel_groups[key]=dict(rank=rank,phase=phase,kernel=event['name'],calls=0,total_us=0)
                kernel_groups[key]['calls']+=1;kernel_groups[key]['total_us']+=event['dur']
        assert categories['cpu_op'] and categories['kernel'] and categories['user_annotation'],categories
        trace_coverage.append(dict(rank=rank,phase=phase,file=path.name,categories=dict(categories)))
    assert len(trace_coverage)==16 and {(r['rank'],r['phase']) for r in trace_coverage}==seen
    aggregate=list(grouped.values())
    for row in aggregate:
        row['profiler_event_count']=int(row['calls'])
        row['calls']=cpu_counts[(row['rank'],row['phase'],row['operator'])]
        row['cpu_avg_us']=row['cpu_total_us']/row['calls'] if row['calls'] else None
        row['device_avg_us']=row['device_total_us']/row['calls'] if row['calls'] else None
    shape_grouped={}
    for row in detail:
        key=(row['rank'],row['phase'],row['operator'],row['shapes'])
        if key not in shape_grouped:
            shape_grouped[key]=dict(row)
        else:
            for metric in sums:shape_grouped[key][metric]+=row[metric]
    detail=list(shape_grouped.values())
    for row in detail:
        row['profiler_event_count']=row.pop('calls')
        row['calls']=cpu_shape_counts[(row['rank'],row['phase'],row['operator'],json.dumps(json.loads(row['shapes'])))]
    shape_totals=defaultdict(int)
    for row in detail:shape_totals[(row['rank'],row['phase'],row['operator'])]+=row['calls']
    ambiguous={key for key,value in shape_totals.items() if value!=cpu_counts[key]}
    for row in detail:
        if (row['rank'],row['phase'],row['operator']) in ambiguous:row['calls']=None
    output=folder/'analysis';output.mkdir(exist_ok=True)
    write_csv(output/'operators-by-shape.csv',detail)
    write_csv(output/'operators-by-rank-phase.csv',aggregate)
    kernels=list(kernel_groups.values())
    for row in kernels:row['avg_us']=row['total_us']/row['calls']
    write_csv(output/'kernels-by-rank-phase.csv',kernels)
    (output/'trace-coverage.json').write_text(json.dumps(trace_coverage,indent=2))
    functions=[r for r in aggregate if r['operator'].startswith('KT_FUNC/')]
    assert functions, 'No key function markers captured'
    write_csv(output/'key-functions-by-rank-phase.csv',functions)
    paired=defaultdict(dict)
    for row in aggregate:
        if row['rank']==0:paired[row['operator']][row['phase']]=row
    comparison=[]
    for operator,phases in paired.items():
        row={'operator':operator,'rank':0}
        for phase in ('EXTEND','DECODE'):
            data=phases.get(phase)
            for metric in ('calls','cpu_total_us','cpu_self_us','cpu_avg_us','device_total_us','device_self_us','device_avg_us'):
                row[phase.lower()+'_'+metric]=data[metric] if data else None
        comparison.append(row)
    comparison.sort(key=lambda r:sum(r.get(p+'_cpu_self_us') or 0 for p in ('extend','decode')),reverse=True)
    write_csv(output/'rank0-prefill-decode-comparison.csv',comparison)
    acceptance={'rank_phase_exports':len(seen),'shape_rows':len(detail),'operator_rows':len(aggregate),
        'function_rows':len(functions),'units':'time microseconds, memory net bytes, calls integer',
        'scope':'Short diagnostic window; prefill/decode have different token and batch counts. CUDA Graph replay does not re-execute Python/ATen bodies for every graph node. Missing captured operators are not zero-cost operations.',
        'time_semantics':'CPU total includes child calls; CPU self excludes them. Do not sum nested totals into wall time. Device event time is not CPU wall time. Net allocation is not peak memory.',
        'device_event_validation':'All16 Chrome traces parsed; CPU operator, user annotation and real kernel events present in every rank/phase.',
        'call_count_correction':'calls counts CPU cpu_op/user_annotation spans from raw traces. profiler_event_count preserves Torch key_averages count, which merges same-name GPU annotations and is not a logical function call count.',
        'gpu_time_caution':'device_total/self columns preserve Torch API accounting, which can include GPU annotation ranges and nested activity. Use kernels-by-rank-phase.csv for actual kernel duration and calls; do not treat annotation-range GPU time as pure compute.'}
    acceptance['shape_call_count_ambiguous_groups']=len(ambiguous)
    acceptance['shape_call_count_note']='Tensor-list shape representation differs between Chrome and key_averages for some operators; those per-shape CPU counts are null. Aggregate per-rank/operator CPU call counts are directly counted and unaffected.'
    (output/'coverage.json').write_text(json.dumps(acceptance,indent=2));print(json.dumps(acceptance,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);analyze(p.parse_args().folder)

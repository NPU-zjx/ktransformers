"""Analyze the observed Nsight SQLite schema; use unions for GPU activity."""
import bisect
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("database",type=Path)
parser.add_argument("--output",type=Path)
parser.add_argument("--gpu-reference",action="store_true")
args=parser.parse_args()
root=args.database.resolve().parent
db=sqlite3.connect(args.database.resolve().as_uri()+"?mode=ro",uri=True);db.row_factory=sqlite3.Row
strings=dict(db.execute('select id,value from StringIds'))
out=args.output or root/'analysis';out.mkdir(parents=True,exist_ok=True)

def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)

def save(name,rows):
    if not rows:return
    with (out/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

nvtx=[dict(r) for r in db.execute('select * from NVTX_EVENTS where end is not null')]
stages=defaultdict(list);functions={}
for r in nvtx:
    name=r['text'] or strings.get(r['textId'],'')
    pid=r['globalTid'] & -16777216 if r['globalTid'] else None
    if name in ('kt_extend','kt_decode') or (args.gpu_reference and name=='kt_mixed'):
        stages[pid].append((r['start'],r['end'],name[3:].upper()))
    if name.startswith('KT_FUNC/') or (args.gpu_reference and name.startswith('vLLM_FUNC/')):
        key=(pid,name)
        if key not in functions:functions[key]=dict(global_pid=pid,function=name,calls=0,cpu_range_total_us=0)
        functions[key]['calls']+=1;functions[key]['cpu_range_total_us']+=(r['end']-r['start'])/1000
assert len(stages)==8
for pid in stages:stages[pid].sort()
starts={pid:[r[0] for r in rows] for pid,rows in stages.items()}
assert all(any(s=="EXTEND" for _,_,s in rows) and any(s=="DECODE" for _,_,s in rows) for rows in stages.values())
runtime={}
api={}
for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
    pid=r['globalTid'] & -16777216;runtime[(pid,r['correlationId'])]=dict(r)
    name=strings[r['nameId']]
    key=(pid,name)
    if key not in api:api[key]=dict(global_pid=pid,api=name,calls=0,total_us=0)
    api[key]['calls']+=1;api[key]['total_us']+=(r['end']-r['start'])/1000

def phase(pid,t):
    if pid not in starts or t is None:return 'OUTSIDE_MARKER'
    i=bisect.bisect_right(starts[pid],t)-1
    if i>=0 and stages[pid][i][0]<=t<=stages[pid][i][1]:return stages[pid][i][2]
    return 'OUTSIDE_MARKER'

kernels=[];groups={};gpu=defaultdict(lambda:defaultdict(list));lags=[]
phase_gpu=defaultdict(lambda:defaultdict(list))
for raw in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
    r=dict(raw);name=strings[r['demangledName']];pid=r['globalPid'];device=r['deviceId']
    launch=runtime.get((pid,r['correlationId']))
    p=phase(pid,launch['start'] if launch else None)
    category='communication_named_kernel' if any(x in name.lower() for x in ('nccl','cross_device_reduce','allreduce','all_gather')) else 'other_kernel'
    gpu[device][category].append((r['start'],r['end']));kernels.append(r)
    phase_gpu[(device,p)][category].append((r['start'],r['end']))
    key=(device,p,name)
    if key not in groups:groups[key]=dict(device=device,phase=p,kernel=name,category=category,calls=0,total_us=0)
    groups[key]['calls']+=1;groups[key]['total_us']+=(r['end']-r['start'])/1000
    if launch:
        lags.append(dict(device=device,phase=p,kernel=name,graph_node=bool(r['graphNodeId']),
            gpu_start_minus_api_end_us=(r['start']-launch['end'])/1000,
            cuda_api=strings[launch['nameId']]))
assert set(gpu)==set(range(8))
copy_names={r[0]:r[2] for r in db.execute('select * from ENUM_CUDA_MEMCPY_OPER')};copies={}
for r in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'):
    device=r['deviceId'];gpu[device]['copy'].append((r['start'],r['end']))
    launch=runtime.get((r['globalPid'],r['correlationId']))
    p=phase(r['globalPid'],launch['start'] if launch else None)
    phase_gpu[(device,p)]['copy'].append((r['start'],r['end']))
    key=(device,p,r['copyKind'])
    if key not in copies:copies[key]=dict(device=device,phase=p,direction=copy_names[r['copyKind']],calls=0,bytes=0,total_us=0)
    copies[key]['calls']+=1;copies[key]['bytes']+=r['bytes'];copies[key]['total_us']+=(r['end']-r['start'])/1000
for r in (db.execute('select * from CUPTI_ACTIVITY_KIND_MEMSET') if db.execute("select 1 from sqlite_master where name='CUPTI_ACTIVITY_KIND_MEMSET'").fetchone() else ()):
    gpu[r['deviceId']]['memset'].append((r['start'],r['end']))
intervals=[x for categories in gpu.values() for items in categories.values() for x in items]
start=min(a for a,b in intervals);end=max(b for a,b in intervals);window=end-start
summary=[]
for device,categories in sorted(gpu.items()):
    active=union([x for items in categories.values() for x in items])
    comm=union(categories['communication_named_kernel']);compute=union(categories['other_kernel'])
    kernel_union=union(categories['communication_named_kernel']+categories['other_kernel'])
    summary.append(dict(device=device,observed_activity_span_ms=window/1e6,
        gpu_active_union_ms=active/1e6,gpu_idle_within_span_ms=(window-active)/1e6,
        communication_named_kernel_union_ms=comm/1e6,other_kernel_union_ms=compute/1e6,
        compute_communication_overlap_ms=(comm+compute-kernel_union)/1e6,
        copy_union_ms=union(categories['copy'])/1e6))
for row in groups.values():row['avg_us']=row['total_us']/row['calls']
for row in copies.values():row['effective_transfer_gb_s']=row['bytes']/(row['total_us']*1000) if row['total_us'] else None
for row in functions.values():row['cpu_range_avg_us']=row['cpu_range_total_us']/row['calls']
save('kernels.csv',sorted(groups.values(),key=lambda r:r['total_us'],reverse=True));save('gpu-activity.csv',summary)
save('copies.csv',list(copies.values()));save('cuda-api.csv',list(api.values()))
save('function-ranges.csv',list(functions.values()));save('launch-timing.csv',lags)
waits=[]
for (device,p),categories in sorted(phase_gpu.items()):
    parts=[v for events in categories.values() for v in events]
    begin=min(a for a,b in parts);finish=max(b for a,b in parts)
    all_activity=[(max(a,begin),min(b,finish)) for events in gpu[device].values()
                  for a,b in events if a<finish and b>begin]
    waits.append(dict(device=device,phase=p,first_to_last_attributed_activity_ms=(finish-begin)/1e6,
        no_observed_gpu_activity_in_span_ms=(finish-begin-union(all_activity))/1e6,
        communication_kernel_union_ms=union(categories['communication_named_kernel'])/1e6,
        copy_union_ms=union(categories['copy'])/1e6))
save('phase-waits.csv',waits)
sync=defaultdict(list)
for (pid,_),r in runtime.items():
    name=strings[r['nameId']]
    if 'Synchronize' in name:
        sync[(pid,phase(pid,r['start']),name)].append((r['start'],r['end']))
save('cpu-sync.csv',[dict(global_pid=pid,phase=p,api=name,calls=len(intervals),
    cpu_sync_api_union_ms=union(intervals)/1e6) for (pid,p,name),intervals in sync.items()])
for device in range(8):
    for p in ('EXTEND','DECODE'):
        assert any(key[0:2]==(device,p) for key in groups),(device,p)
diagnostics=[dict(r) for r in db.execute('select * from DIAGNOSTIC_EVENT where severity>=2')]
coverage={'gpu_devices':8,'kernel_events':len(kernels),'matched_launch_events':len(lags),
    'phase_ranges':{str(pid):dict(extend=sum(r[2]=='EXTEND' for r in rows),decode=sum(r[2]=='DECODE' for r in rows),mixed=sum(r[2]=='MIXED' for r in rows)) for pid,rows in stages.items()},
    'warnings':sorted({r['text'] for r in diagnostics}),
    'definitions':{'idle':'Complement of observed kernel/memcpy/memset union within common first-to-last observed GPU activity span; not SM idleness and not full client wall time.',
        'communication':'Kernel-name classification includes NCCL and SGLang custom cross_device_reduce; duration may include peer/CPU waiting, not pure link transfer.',
        'latency':'GPU start minus correlated API return, can be negative. Graph nodes include dependency/earlier-node time and must not be called individual kernel launch overhead.',
        'copies':'bytes / summed copy duration in decimal GB/s; not peak PCIe/NVLink hardware bandwidth.',
        'warnings':'Driver13.0 with CUPTI12.8 warning and capture-range partial-event warnings retained. No CPU scheduler sampling; do not claim precise thread-state coverage.'}}
(out/'coverage.json').write_text(json.dumps(coverage,indent=2));(out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
print(json.dumps({'coverage':coverage,'gpu_activity':summary},indent=2))

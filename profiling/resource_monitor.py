"""Read-only host/container telemetry. Sampling peaks are not instantaneous peaks."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/raid/data/zjx/kt-baseline')
CLIENT = ['bash', '/raid/data/zjx/gpu-baseline/scripts/docker/client.sh']

def command(args):
    return subprocess.check_output(args, text=True, timeout=8).strip()

def counters(path):
    return {a: int(b) for a,b in (line.split() for line in path.read_text().splitlines())}


def process_identity(pid, device, cid, proc_root=Path('/proc')):
    proc=proc_root/str(pid)
    if cid not in (proc/'cgroup').read_text():
        return None
    status=(proc/'status').read_text()
    nspid=next([int(x) for x in line.split()[1:]] for line in status.splitlines()
               if line.startswith('NSpid:'))
    assert nspid[0]==int(pid)
    ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
    return dict(host_pid=int(pid),namespace_pids=nspid,gpu_uuid=device,start_ticks=ticks)


def required_run_identity(run):
    state = json.loads((run/'state.json').read_text())
    cid = state.get('container_id', '')
    if len(cid) != 64 or any(c not in '0123456789abcdef' for c in cid):
        raise ValueError('Run state must contain a full container ID before monitoring starts')
    launch = json.loads((run/'launch.json').read_text())
    if launch.get('container_id') != cid:
        raise ValueError('Run state and launch container IDs differ')
    return cid

def cpu_ticks():
    return {s[0]: list(map(int,s[1:9])) for line in Path('/proc/stat').read_text().splitlines()
            if (s:=line.split())[0].startswith('cpu')}

def cpu_percent(old, new):
    result={}
    for key, vals in new.items():
        if key not in old: continue
        delta=[v-o for v,o in zip(vals,old[key])]
        total=sum(delta)
        if total>0:
            result[key]={'busy_pct':100*(total-delta[3]-delta[4])/total,
                         'iowait_pct':100*delta[4]/total}
    return result

def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run-dir',type=Path); g.add_argument('--queue-dir',type=Path)
    p.add_argument('--interval',type=float,default=5)
    args=p.parse_args()
    target=(args.run_dir or args.queue_dir).resolve()
    assert ROOT/'results' in target.parents and target.is_dir()
    assert args.interval>=2
    if args.run_dir:
        required_run_identity(target)
    output=target/'telemetry'
    output.mkdir(exist_ok=False)
    shutil.copy2(__file__,output/'resource_monitor.py')
    started=time.time()
    meta={'started_unix':started,'interval_seconds':args.interval,'pid':os.getpid(),
          'coverage':'from attachment onward; earlier measurements cannot be reconstructed',
          'gpu_peak_kind':'sampled device memory; includes all processes on each device',
          'cpu_units':'container core equivalents; 1.0 = one logical CPU fully busy',
          'cpu_normalization':'host busy excludes idle and iowait; container quota_pct uses cpu.max',
          'phase_kind':'driver phase instrumentation when present; otherwise log hint, not kernel tracing',
          'memory_peak_kind':'cgroup memory.peak is kernel lifetime high-water if available; includes cache',
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2))
    summary={'samples':0,'errors':0,'cases':{},'started_unix':started}
    previous_cpu=cpu_ticks(); previous_container={}; identities={}
    log_offsets={}; hints={}
    with (output/'samples.jsonl').open('x') as log:
        while True:
            tick=time.monotonic(); now=time.time()
            sample={'time_unix':now,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'errors':[]}
            run=args.run_dir
            try:
                if args.queue_dir:
                    queue=json.loads((target/'state.json').read_text())
                    variant=queue.get('current_case')
                    run=target.with_name(target.name+'-'+variant) if variant else None
                sample['run']=str(run) if run else None
                current_cpu=cpu_ticks()
                sample['host_cpu']=cpu_percent(previous_cpu,current_cpu); previous_cpu=current_cpu
                sample['host_memory_bytes']={line.split(':')[0]:int(line.split()[1])*1024
                    for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ['MemTotal','MemAvailable','SwapTotal','SwapFree','Cached','Dirty']}
                sample['host_loadavg']=list(os.getloadavg())
                sample['disk_free_bytes']=shutil.disk_usage(ROOT).free
                fields=['index','memory.used','memory.total','utilization.gpu','utilization.memory','power.draw','temperature.gpu','clocks.sm','clocks.mem']
                lines=command(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits']).splitlines()
                sample['gpus']=[]
                for line in lines:
                    values=line.split(',')
                    row={}
                    for key,value in zip(fields,values):
                        try: row[key]=float(value.strip())
                        except ValueError: row[key]=None
                    sample['gpus'].append(row)
                sample['gpu_processes']=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']).splitlines()
                if run and (run/'state.json').exists():
                    state=json.loads((run/'state.json').read_text()); cid=state.get('container_id')
                    if cid:
                        inspect=json.loads(command(CLIENT+['inspect',cid]))[0]
                        assert inspect['Id']==cid
                        # Persist host/container PID evidence while workers are alive.
                        # After container exit /proc can no longer reconstruct this.
                        changed=False
                        for raw in sample['gpu_processes']:
                            host_pid,device,_=[x.strip() for x in raw.split(',')]
                            key=(int(host_pid),device)
                            if key in identities: continue
                            try:
                                identity=process_identity(host_pid,device,cid)
                            except FileNotFoundError:
                                sample['errors'].append('GPU process exited during identity capture: '+host_pid)
                                continue
                            if identity is not None:
                                identities[key]=identity; changed=True
                        if changed:
                            proof={'time_unix':now,'container_id':cid,
                                   'run_id':state.get('run_id',state.get('run_token')),
                                   'processes':list(identities.values())}
                            temp=run/'process-identity.tmp'
                            temp.write_text(json.dumps(proof,indent=2));temp.replace(run/'process-identity.json')
                        if not (run/'gpu-identity.json').exists():
                            query=['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits']
                            (run/'gpu-identity.json').write_text(json.dumps({'time_unix':now,'command':query,'csv':command(query)},indent=2))
                        sample['container_state']=inspect['State']['Status']
                        pid=inspect['State']['Pid']
                        if pid:
                            cgline=next(x for x in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if x.startswith('0::'))
                            cg=Path('/sys/fs/cgroup')/cgline[3:].lstrip('/')
                            stat=counters(cg/'cpu.stat'); wall=time.monotonic()
                            data={'cpu_stat':stat,'memory_current_bytes':int((cg/'memory.current').read_text()),
                                  'memory_stat':counters(cg/'memory.stat'), 'memory_events':counters(cg/'memory.events'),
                                  'io_stat_raw':(cg/'io.stat').read_text().splitlines()}
                            if (cg/'memory.peak').exists(): data['memory_peak_bytes']=int((cg/'memory.peak').read_text())
                            quota,period=(cg/'cpu.max').read_text().split()
                            data['cpu_quota_cores']=float(quota)/float(period) if quota!='max' else None
                            if cid in previous_container:
                                old_wall,old_stat=previous_container[cid]; dt=wall-old_wall
                                cores=(stat['usage_usec']-old_stat['usage_usec'])/1e6/dt
                                data['cpu_cores_used']=cores
                                if data['cpu_quota_cores']: data['cpu_quota_pct']=100*cores/data['cpu_quota_cores']
                                data['cpu_throttled_usec_delta']=stat.get('throttled_usec',0)-old_stat.get('throttled_usec',0)
                            previous_container[cid]=(wall,stat)
                            sample['container']=data
                    # Read new log bytes only. Never claim precise subphase attribution.
                    name=str(run); path=run/'train.log'
                    if path.exists():
                        with path.open('rb') as f:
                            f.seek(log_offsets.get(name,0)); tail=f.read().decode('utf-8',errors='replace'); log_offsets[name]=f.tell()
                        if 'all initialize finished, ready to fit' in tail:
                            hints[name]='validation' if 'external-eval-' in run.name else 'train_or_validation_or_save'
                        if 'Initial validation metrics:' in tail and 'external-eval-' in run.name: hints[name]='validation_finished_cleanup'
                    sample['phase_hint']=hints.get(name,'initialization_or_checkpoint_loading')
                    sample['phase_is_exact']=False
                    phase_path=run/'current-phase.json'
                    if phase_path.exists():
                        sample['driver_phase']=json.loads(phase_path.read_text())
                        sample['phase_hint']=sample['driver_phase']['phase']
                        sample['phase_is_exact']=True
                    case=summary['cases'].setdefault(run.name,{'samples':0,'gpu_memory_sample_peak_mib':{},'gpu_util_sample_max_pct':{},'phase_hints':{}})
                    case['samples']+=1
                    phase=sample['phase_hint']; case['phase_hints'][phase]=case['phase_hints'].get(phase,0)+1
                    for gpu in sample['gpus']:
                        key=str(int(gpu['index']))
                        for source,dest in [('memory.used','gpu_memory_sample_peak_mib'),('utilization.gpu','gpu_util_sample_max_pct')]:
                            if gpu[source] is not None: case[dest][key]=max(case[dest].get(key,0),gpu[source])
                    for source,dest in [('cpu_cores_used','container_cpu_sample_peak_cores'),('memory_current_bytes','container_memory_sample_peak_bytes'),('memory_peak_bytes','container_memory_kernel_peak_bytes')]:
                        v=sample.get('container',{}).get(source)
                        if v is not None: case[dest]=max(case.get(dest,0),v)
            except Exception as exc:
                sample['errors'].append(repr(exc)); summary['errors']+=1
            summary['samples']+=1; summary['last_sample_unix']=now
            sample['collection_seconds']=time.monotonic()-tick
            log.write(json.dumps(sample)+'\n'); log.flush()
            tmp=output/'summary.tmp'; tmp.write_text(json.dumps(summary,indent=2)); tmp.replace(output/'summary.json')
            if (target/'completion.json').exists(): break
            if now-started>180*3600:
                summary['stop_reason']='monitor_180h_limit'; break
            time.sleep(max(0,args.interval-(time.monotonic()-tick)))
    summary['finished_unix']=time.time()
    (output/'completion.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__': main()

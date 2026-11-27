"""Reconcile the completed owned run; never equate tensor capacity with RSS."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def union(intervals):
    result=[]
    for begin,end in sorted(intervals):
        if end<begin:raise ValueError('Negative interval')
        if result and begin<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([begin,end])
    return sum(end-begin for begin,end in result)


def integers(text):
    return {k:int(v) for k,v in (line.split() for line in text.splitlines())}


def gpu_rows(sample):
    result={}
    for line in sample['cards_csv'].splitlines():
        v=[x.strip() for x in line.split(',')]
        result[v[1]]=dict(index=int(v[0]),uuid=v[1],used_bytes=int(v[2])*2**20,total_bytes=int(v[3])*2**20)
    return result


def gpu_processes(sample):
    result=[]
    for line in sample['processes_csv'].splitlines():
        v=[x.strip() for x in line.split(',')]
        if len(v)==3 and v[0].isdigit() and v[2].isdigit():
            result.append(dict(pid=int(v[0]),uuid=v[1],used_bytes=int(v[2])*2**20))
    return result


def frame_category(frames):
    text=' '.join(str(f) for f in frames).lower()
    if not text:return 'unclassified.missing_stack'
    if 'foreach_reduce' in text or 'reduce_scatter' in text:return 'producer.fsdp_gradient_reduction'
    if 'foreach_all_gather' in text or 'all_gather' in text:return 'producer.fsdp_parameter_gather'
    if '_fsdp_param' in text:return 'producer.fsdp_parameter_materialization'
    if 'adamw' in text or 'adam.py' in text:return 'producer.optimizer'
    if 'computelogprobs' in text or 'tokenlogprobs' in text or 'logsumexp' in text:return 'producer.logprob_loss'
    if 'attention' in text or 'scaled_dot_product' in text:return 'producer.attention'
    if 'backward' in text:return 'producer.autograd_or_recomputation'
    if 'qwen3' in text or 'linear' in text or 'forward' in text:return 'producer.model_forward_or_recomputation'
    return 'unclassified.recorded_stack'


def snapshot_ledger(data,device,known):
    categories=Counter(); details=[]; active=reserved=pending=inactive=0
    spans=sorted((s['address'],s['address']+s['storage_bytes'],s['category'])
                 for s in known if s['device']==f'cuda:{device}')
    for seg in data['segments']:
        if seg['device']!=device:continue
        reserved+=seg['total_size']
        for block in seg['blocks']:
            state=block['state']
            if state=='inactive':inactive+=block['size'];continue
            if state=='active_pending_free':pending+=block['size'];continue
            if state!='active_allocated':raise ValueError(state)
            active+=block['size'];start=block['address'];requested=block['requested_size'];end=start+requested
            matched=0
            for lo,hi,category in spans:
                if hi<=start:continue
                if lo>=end:break
                n=max(0,min(hi,end)-max(lo,start))
                categories['tensor.'+category]+=n;matched+=n
            if matched>requested:raise ValueError('Overlapping deduplicated storages')
            category=frame_category(block.get('frames',[]))
            categories[category]+=requested-matched
            categories['allocator.block_rounding']+=block['size']-requested
            if requested-matched:
                details.append(dict(address=start,bytes=requested-matched,category=category,frames=block.get('frames',[])))
    assert sum(categories.values())==active
    assert active+pending+inactive==reserved
    return dict(active_block_bytes=active,reserved_bytes=reserved,pending_free_bytes=pending,
                inactive_bytes=inactive,active_bytes_by_category=dict(categories),
                largest_unmatched=sorted(details,key=lambda x:x['bytes'],reverse=True)[:12],
                scope='Snapshot state; producer classification is not a complete tensor owner/lifetime proof.')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--replay-module',type=Path)
    a=p.parse_args();root=a.root;profile=root/'profiles';out=root/'analysis/detailed';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text())
    plan=json.loads((root/'rl/resolved-plan.json').read_text())
    world=plan['actor']['world_size']
    boundaries=[r for path in profile.glob('boundaries-actor-*.jsonl') for r in rows(path)]
    roles={r['pid']:dict(role='actor',rank=int(r['rank'])) for r in boundaries}
    rollout_boundaries=[r for path in profile.glob('boundaries-rollout-*.jsonl') for r in rows(path)]
    roles.update({r['pid']:dict(role='rollout',rank=r['tp_rank']) for r in rollout_boundaries})
    maps={}
    for row in rows(root/'host-pid-map.jsonl'):
        for item in row['processes']:maps[item['host_pid']]=item['container_pid']
    nvml=rows(root/'resources/nvml.jsonl');cpu=rows(root/'resources/container.jsonl');processes=rows(root/'resources/processes.jsonl')
    stages=[]
    for path in sorted((root/'rl').glob('**/stages.jsonl')):
        for stage in rows(path):
            if stage.get('status')=='complete':
                stages.append(dict(path=str(path),**stage))
    # Separate export from training using owner-rank actual entry/return; peer
    # export wrappers start earlier while waiting for the owner CPU optimizer.
    rank0=[r for r in boundaries if int(r['rank'])==0]
    export_events=[r for path in profile.glob('timing-actor-*.jsonl') for r in rows(path)
                   if r['label']=='checkpoint_export' and int(r['rank'])==0]
    if len(export_events)==1:
        event=export_events[0]
        stages.append(dict(path='rank0 observer',phase='model_export_owner',status='complete',
                           start_unix_ns=event['start_ns'],end_unix_ns=event['end_ns']))
        actor=next(s for s in stages if s['phase']=='actor_update')
        stages.append(dict(path='actor update excluding owner export',phase='actor_before_owner_export',status='complete',
            start_unix_ns=actor['start_unix_ns'],end_unix_ns=event['start_ns']))
    results=[]
    for s in stages:
        begin,end=s['start_unix_ns'],s['end_unix_ns']
        cs=[r for r in cpu if begin<=r['start_ns']<=r['end_ns']<=end]
        gs=[r for r in nvml if begin<=r['start_ns']<=r['end_ns']<=end]
        item=dict(phase=s['phase'],path=s['path'],start_ns=begin,end_ns=end,seconds=(end-begin)/1e9,
                  container_samples=len(cs),gpu_samples=len(gs))
        if cs:
            peak=max(cs,key=lambda r:int(r['memory.current']));stat=integers(peak['memory.stat']);cur=int(peak['memory.current'])
            item['container_peak']=dict(time_ns=peak['start_ns'],current_bytes=cur,anon_bytes=stat['anon'],file_bytes=stat['file'],
                shmem_included_in_file_bytes=stat.get('shmem'),kernel_bytes=stat.get('kernel'),
                other_accounting_bytes=cur-stat['anon']-stat['file'],
                file_dirty_bytes=stat.get('file_dirty'),file_writeback_bytes=stat.get('file_writeback'))
            nearest=min(processes,key=lambda r:abs(r['start_ns']-peak['start_ns']))
            totals=Counter();process_roles=[];members={r['pid']:r for r in nearest['records']}
            for proc in nearest['records']:
                ancestor=proc['pid'];seen=set();role=None
                while ancestor in members and ancestor not in seen:
                    seen.add(ancestor)
                    if ancestor in roles:role=roles[ancestor]['role'];break
                    cmd=members[ancestor]['cmd']
                    if 'ktrl.workers.engine.fsdp.actor_update' in cmd:role='actor';break
                    if 'sglang.launch_server' in cmd:role='rollout';break
                    match=re.search(r'^PPid:\s*(\d+)',members[ancestor]['status'],re.M)
                    ancestor=int(match.group(1)) if match else 0
                role=role or 'controller_or_other_container_process'
                value=proc['smaps_rollup_bytes'];pss=value.get('Pss',0);totals[role]+=pss
                process_roles.append(dict(pid=proc['pid'],role=role,rank=roles.get(proc['pid'],{}).get('rank'),
                    pss_bytes=pss,uss_bytes=proc['uss_bytes'],rss_bytes=value.get('Rss'),
                    pss_anon_bytes=value.get('Pss_Anon'),pss_file_bytes=value.get('Pss_File'),pss_shmem_bytes=value.get('Pss_Shmem')))
            item['nearest_pss']=dict(sample_time_ns=nearest['start_ns'],offset_seconds=(nearest['start_ns']-peak['start_ns'])/1e9,
                bytes_by_role=dict(totals),processes=process_roles,errors=nearest['errors'])
        cards=defaultdict(list)
        for sample in gs:
            for uuid,card in gpu_rows(sample).items():cards[uuid].append((sample,card))
        peaks=[]
        for uuid,observations in sorted(cards.items(),key=lambda x:x[1][0][1]['index']):
            sample,card=max(observations,key=lambda x:x[1]['used_bytes']);owned=[]
            for proc in gpu_processes(sample):
                if proc['uuid']!=uuid:continue
                cp=maps.get(proc['pid']);owned.append(dict(**proc,container_pid=cp,**roles.get(cp,dict(role='unmapped'))))
            peaks.append(dict(**card,time_ns=sample['start_ns'],sample_end_ns=sample['end_ns'],processes=owned,
                card_minus_processes_bytes=card['used_bytes']-sum(r['used_bytes'] for r in owned)))
        item['gpu_peaks']=peaks;results.append(item)
    (out/'physical-stage-ledger.json').write_text(json.dumps(results,indent=2))
    # Rank-local same-boundary capacity. Do not subtract it from a distant PSS peak.
    state_rows=[]
    for r in boundaries:
        totals=r['inventory']['bytes_by_device_category'];gpu=sum(v for k,v in totals.items() if k.startswith('cuda:'))
        state_rows.append(dict(rank=int(r['rank']),pid=r['pid'],label=r['label'],step=r.get('optimizer_step'),time_ns=r['time_ns'],
            allocated=r['allocated'],reserved=r['reserved'],peak_allocated=r['peak_allocated'],peak_reserved=r['peak_reserved'],
            persistent_gpu_capacity_bytes=gpu,unmatched_live_bytes=r['allocated']-gpu,
            reserved_minus_allocated_bytes=r['reserved']-r['allocated'],inactive_split_bytes=r['inactive_split_bytes'],
            capacity=totals,errors=r['inventory']['errors']))
    (out/'tensor-boundary-ledger.json').write_text(json.dumps(state_rows,indent=2))
    timing=[r for path in profile.glob('timing-actor-*.jsonl') for r in rows(path)]
    timing += [dict(label='optimizer_native',**r) for path in profile.glob('optimizer-timing-actor-*.jsonl') for r in rows(path)]
    stats=[]
    for label in sorted({r['label'] for r in timing}):
        selected=[r for r in timing if r['label']==label];per_rank={}
        for rank in range(world):
            rr=[r for r in selected if int(r['rank'])==rank]
            per_rank[str(rank)]=dict(calls=len(rr),sum_seconds=sum(r['end_ns']-r['start_ns'] for r in rr)/1e9,
                intervals=[[r['start_ns'],r['end_ns']] for r in rr])
        stats.append(dict(label=label,rank_wall_seconds=per_rank,cross_rank_union_seconds=union([(r['start_ns'],r['end_ns']) for r in selected])/1e9,
                          note='Inclusive method wall time. Nested/parallel labels must not be summed.'))
    (out/'training-method-times.json').write_text(json.dumps(stats,indent=2))
    snaps=[];replay=[]
    if a.replay_module:
        sys.path.insert(0,str(a.replay_module.parent));from analyze_snapshot import reconstruct
    for path in sorted(profile.glob('snapshot-actor-*.pickle')):
        parts=path.stem.split('-');pid=int(parts[2]);ns=int(parts[3]);label='-'.join(parts[4:]);rank=roles[pid]['rank']
        candidates=[r for r in boundaries if r['pid']==pid and r['label']==label and r['time_ns']<=ns]
        if not candidates:raise ValueError(str(path))
        boundary=max(candidates,key=lambda r:r['time_ns'])
        data=pickle.loads(path.read_bytes());ledger=snapshot_ledger(data,boundary['device'],boundary['inventory']['storages'])
        row=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rank=rank,pid=pid,label=label,time_ns=ns,
                 boundary_to_snapshot_seconds=(ns-boundary['time_ns'])/1e9,**ledger)
        snaps.append(row)
        if a.replay_module:
            rep=reconstruct(data,boundary['device'])
            replay.append(dict(file=path.name,rank=rank,label=label,**rep))
    (out/'snapshot-state-ledger.json').write_text(json.dumps(snaps,indent=2))
    (out/'snapshot-retained-peaks.json').write_text(json.dumps(replay,indent=2))
    (out/'limitations.json').write_text(json.dumps(dict(
        run_status=manifest['status'],effective_rounds=1,optimizer_steps=2,
        baseline_container_bytes=int(cpu[0]['memory.current']),baseline_file_bytes=integers(cpu[0]['memory.stat'])['file'],
        missing=['Resident Actor next full rollout; this run has two v0 batches and one effective update round.',
            'Native CPU workspace and rollout expert packed-weight allocation stacks.',
            'External CUDA allocation ownership (NVML minus PyTorch is not automatically NCCL).',
            'Lightweight repeated profiling; diagnostic traces are not formal timing.',
            'Recorded producer stacks do not by themselves prove every activation owner.',
            'CPU tensor capacity is not a physically resident-memory attribution.',
            'Cgroup memory.peak is lifetime peak; CPU/GPU sampling may miss instantaneous spikes.']),indent=2))
    print(json.dumps(dict(stages=len(results),boundaries=len(state_rows),snapshots=len(snaps),replays=len(replay),
        replay_errors=sum(bool(r['consistency_errors']) for r in replay),out=str(out))))


if __name__=='__main__':main()

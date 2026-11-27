"""Fail-closed collector coverage. Completeness is separate from RL acceptance."""
import argparse
import json
import pickle
from pathlib import Path


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def audit(root):
    profile, rl = root/'profiles', root/'rl'
    checks = {}
    def check(name, passed, evidence, note=''):
        checks[name] = dict(status='pass' if passed else 'pending', evidence=evidence, note=note)
    manifests = root/'manifest.json'
    manifest = json.loads(manifests.read_text())
    check('native_run_complete', manifest['status']=='complete' and (rl/'accepted.json').exists(), [str(manifests)])
    status = root/'resources/sampler-status.json'
    check('sampler_complete', status.exists() and json.loads(status.read_text())['complete'], [str(status)])
    for name in ('container','processes','nvml'):
        f = root/f'resources/{name}.jsonl'
        records = lines(f) if f.exists() else []
        check('resource_'+name, len(records)>=2, [str(f)], f'{len(records)} samples')
    clients = sorted((profile/'client').glob('wave*'))
    check('two_live_rollouts', len(clients)>=2, [str(f) for f in clients])
    for wave in clients:
        paths = sorted(wave.glob('*.json'))
        rows = [json.loads(p.read_text()) for p in paths]
        check(wave.name+'_client_latency', bool(rows) and all(r['ttft_ms']>=0 and
            (r['output_tokens']==1 or r['tpot_ms'] is not None) for r in rows), [str(wave)],
            f'{len(rows)} requests; client coalesced chunks retained, not invented ITL')
    for rnd in sorted(rl.glob('round-*')):
        paths = sorted((rnd/'rollout/phase').glob('phase-*.json'))
        traces = [json.loads(p.read_text()) for p in paths]
        expected = json.loads((rl/'resolved-plan.json').read_text())['rollout']['server_args']['tp_size']
        ranks = {t['rank'] for t in traces}
        check(rnd.name+'_cuda_phases', ranks==set(range(expected)) and all(
            any(r['phase']=='EXTEND' for r in t['rows']) and
            any(r['phase']=='DECODE' and r['graph_replayed'] for r in t['rows']) for t in traces),
            [str(p) for p in paths])
    boundary_paths = sorted(profile.glob('boundaries-actor-*.jsonl'))
    boundaries = [r for p in boundary_paths for r in lines(p)]
    plan_path=rl/'resolved-plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else {}
    world=plan.get('actor',{}).get('world_size',8)
    for label in ('old_logp_end','optimizer_start','optimizer_end','export_start','export_end','bf16_accumulation'):
        rows = [r for r in boundaries if r['label']==label]
        ranks = {int(r['rank']) for r in rows}
        check('actor_'+label, ranks==set(range(world)) and all(not r['inventory']['errors'] for r in rows),
              [str(p) for p in boundary_paths], f'{len(rows)} boundaries / {len(ranks)} ranks')
    correct_devices=[]
    for path in sorted(profile.glob('allocator-actor-*.jsonl')):
        samples=lines(path)
        pid=samples[0]['pid'] if samples else None
        refs=[r for r in boundaries if r['pid']==pid]
        expected=refs[0]['gpu_uuid'] if refs else None
        valid=bool(samples and expected) and all(r.get('gpu_uuid')==expected for r in samples)
        correct_devices.append(valid)
    check('actor_sampler_device_mapping',len(correct_devices)==world and all(correct_devices),
        [str(p) for p in profile.glob('allocator-actor-*.jsonl')],
        'Every sampling thread must query its own rank GPU; old a02 ranks 1-7 queried device 0.')
    internal_ranks={int(r['rank']) for r in boundaries
        if any(a['role'].startswith('fsdp.') for s in r['inventory']['storages']
               for a in s['aliases'])}
    check('fsdp_hidden_compute_storage',internal_ranks==set(range(world)),[str(p) for p in boundary_paths],
        'Existing live compute copies and all-gather aliases must be inventoried by object address.')
    finals=[r for p in profile.glob('allocator-final-actor-*.jsonl') for r in lines(p)
            if r.get('rank') is not None and r.get('cuda_initialized', True)]
    problems=[]
    for row in finals:
        for metric in ('peak_allocated', 'peak_reserved', 'peak_requested_bytes'):
            target=profile/f"highwater-actor-{row['pid']}-lifetime-{metric}.pickle"
            try:
                with target.open('rb') as handle:
                    snapshot=pickle.load(handle)
                meta=snapshot['ktrl_highwater']
                if not snapshot['segments'] or meta[metric]!=row[metric] or meta['gpu_uuid']!=row['gpu_uuid']:
                    raise ValueError('snapshot disagrees with final allocator counters')
            except Exception as error:
                problems.append(f'{target.name}: {error}')
    check('durable_actor_highwater_snapshots',
        {int(r['rank']) for r in finals}==set(range(world)) and not problems,
        [str(p) for p in profile.glob('highwater-actor-*.pickle')],
        'Final allocator counters checked against retained snapshots. Historical peak replay remains separate. '
        + '; '.join(problems))
    accepted_path=rl/'accepted.json'
    accepted=json.loads(accepted_path.read_text()) if accepted_path.exists() else {}
    check('at_least_three_effective_rl_rounds',accepted.get('real_online_updates',0)>=3,[str(accepted_path)],
        'Skipped zero-advantage batches are excluded. Two PPO minibatches are not two RL rounds.')
    errors = sorted(profile.glob('errors-*.jsonl'))
    check('worker_errors_absent', not errors, [str(p) for p in errors])
    # These require separate evidence-backed analysis. They must never pass merely
    # because a trace, a snapshot or an accepted.json was written.
    for metric in ('tps_stage_union_verified','gpu_physical_tensor_reconciliation',
                   'cpu_anon_file_tensor_reconciliation','native_cpu_workspace_attribution',
                   'cuda_external_allocation_attribution','all_phase_time_breakdown',
                   'light_vs_diagnostic_overhead','actor_rollout_probability_abs_relative'):
        check(metric, False, [], 'Requires analysis / additional bounded diagnostics; not yet accepted')
    return dict(complete=all(r['status']=='pass' for r in checks.values()), checks=checks)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    result=audit(a.root)
    (a.root/'coverage.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['complete'] else 2)

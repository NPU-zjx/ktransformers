"""Real dedicated service publication, fail-closed injection and cold recovery; no updates."""
import argparse
import json
import time
from pathlib import Path
from ktrl.configs.recipe import load_plan
from ktrl.trainer.online_native import service,guard_gpu,resources,control_probe,drain_rollout
from ktrl.trainer.online import save
from .identity import build_manifest,validate_manifest


def run(config,checkpoint,prompts,out):
    out.mkdir()
    plan=load_plan(config,steps=1,profiling='light')
    if not (checkpoint/'publication-manifest.json').exists():build_manifest(checkpoint)
    step,digest=validate_manifest(checkpoint)
    version = f"v{step}"
    tp = plan["rollout"]["server_args"]["tp_size"]
    devices = plan["runtime"]["cuda_visible_devices"]
    layers = json.loads((checkpoint / "kt_full_experts.index.json").read_text())["layers"]
    seed = plan["inputs"]["seed"]
    question=json.loads(prompts.read_text())[0]
    probe=dict(input_ids=question['prompt'],return_logprob=True,
        sampling_params=dict(temperature=1.,top_p=1.,top_k=-1,sampling_seed=seed,max_new_tokens=16,ignore_eos=True))
    request=dict(model_path=str(checkpoint),load_format='safetensors',flush_cache=True,recapture_cuda_graph=True)

    def publish(client,folder):
        drain_rollout(client,folder,'before-publication')
        result=client.call('/update_weights_from_disk',request,timeout=1800)
        assert result.get('success'),result
        for rank in range(tp):
            proof=json.loads((folder/f'published-{version}-rank{rank}.json').read_text())
            assert proof['manifest_sha256']==digest and proof['actor_step']==step
        for layer in layers:
            assert json.loads((folder/f'cpu-verified-{version}-layer{layer}.json').read_text())['packed_native_forward_relative_l2']<=.06
        return result

    with resources(out):
        with service(config,out/'happy',plan) as (client,_):
            save(out/'base-probe.json',client.generate(probe))
            save(out/'publication.json',publish(client,out/'happy'))
            control_probe(client,out/'happy',question['prompt'],version,seed)
            save(out/'happy/seed-audit-active.json',dict(tag='first',expected_seed=seed))
            after=client.generate(probe);save(out/'v1-probe.json',after)
            drain_rollout(client,out/'happy','after-seed-first')
            save(out/'happy/seed-audit-active.json',dict(tag='second',expected_seed=seed))
            repeated=client.generate(probe)
            drain_rollout(client,out/'happy','after-seed-second')
            save(out/'v1-repeated-probe.json',repeated)
            def audit_runs(rank):
                path=out/f'happy/seed-audit-rank{rank}.jsonl'
                if not path.exists():return []
                runs=[]
                for line in path.read_text().splitlines():
                    try:row=json.loads(line)
                    except json.JSONDecodeError:continue
                    if row['positions']==[len(question['prompt'])-1]:runs.append([])
                    if runs:runs[-1].append(row)
                return runs
            deadline=time.monotonic()+30
            while not all(len(audit_runs(rank))==2 and len(audit_runs(rank)[-1])>=16 for rank in range(tp)):
                if time.monotonic()>deadline:raise RuntimeError('Missing service sampler seed evidence')
                time.sleep(.1)
            (out/'happy/seed-audit-active.json').unlink()
            seed_proofs=[]
            for rank in range(tp):
                runs=audit_runs(rank)
                first,second=runs[0][:16],runs[1][:16]
                rows=first+second
                assert all(x['seeds']==[seed] and x['same_probabilities_repeat_exact'] for x in rows)
                assert [x['positions'] for x in first]==[x['positions'] for x in second]
                equal_probs=[a['probabilities_sha256']==b['probabilities_sha256'] for a,b in zip(first,second)]
                assert all(a['output_ids']==b['output_ids'] for a,b,equal in zip(first,second,equal_probs) if equal)
                seed_proofs.append(dict(rank=rank,actual_seed=seed,same_distribution_repeat_exact=True,
                    equal_probability_steps=sum(equal_probs),steps=16))
            save(out/'seed-service-proof.json',dict(accepted=True,ranks=seed_proofs,
                same_generated_tokens=after['output_ids']==repeated['output_ids'],
                note='Seed and position must reach sampler; differing model probabilities may produce differing tokens'))
            duplicate=client.call('/update_weights_from_disk',request,timeout=1800)
            assert duplicate.get('success') and 'Already published' in duplicate.get('message','')
            save(out/'idempotent.json',duplicate)
            assert all(json.loads((out/f'happy/replays-rank{r}.json').read_text()).get(version,0)>0 for r in range(tp))
        guard_gpu(devices=devices)
        with service(config,out/'fault',plan) as (client,_):
            save(out/'fault/inject-publication-failure.json',dict(stage='after_gpu_load',rank=0))
            failed=False
            try:
                result=client.call('/update_weights_from_disk',request,timeout=1800)
                failed=not result.get('success')
            except Exception as exc:
                failed=True;save(out/'fault-client-error.json',dict(error=str(exc)))
            assert failed,'Injected publication unexpectedly succeeded'
            proof=json.loads((out/'fault/publication-failed-rank0.json').read_text())
            assert proof['exit_code']==81 and proof['failed_closed']
            assert not (out/f'fault/published-{version}-rank0.json').exists()
        guard_gpu(devices=devices)
        # Recover by restarting and reloading a complete known checkpoint, never
        # by resuming mixed weights or repeating the optimizer update.
        with service(config,out/'recovery',plan) as (client,_):
            save(out/'recovery-publication.json',publish(client,out/'recovery'))
            control_probe(client,out/'recovery',question['prompt'],version,seed)
            import torch
            first_logits=torch.load(out/f'happy/{version}-control-logits.pt',weights_only=True)
            restored_logits=torch.load(out/f'recovery/{version}-control-logits.pt',weights_only=True)
            assert torch.isfinite(restored_logits).all()
            for rank in range(tp):
                original=json.loads((out/f'happy/parameters-published-{version}-rank{rank}.json').read_text())
                restored=json.loads((out/f'recovery/parameters-published-{version}-rank{rank}.json').read_text())
                assert original==restored,'Cold recovery loaded different GPU parameter values'
            recovered=client.generate(probe);save(out/'recovered-probe.json',recovered)
            first=after['meta_info']['output_token_logprobs']
            second=recovered['meta_info']['output_token_logprobs']
            numerical=dict(exact_logits=torch.equal(first_logits,restored_logits),
                max_absolute_logit_difference=(first_logits-restored_logits).abs().max().item(),
                same_output_ids=recovered['output_ids']==after['output_ids'],same_output_logprobs=first==second,
                acceptance='Exact published parameter hashes and native CPU expert checks')
            save(out/'recovery-numerics.json',numerical)
        guard_gpu(devices=devices)
    save(out/'accepted.json',dict(accepted=True,optimizer_updates=0,cpu_layers_verified=len(layers),
        request_seed_verified_at_sampler=True,idempotent=True,failed_closed=True,cold_recovery_exact_weights=True,
        cold_recovery_exact_logits=numerical['exact_logits'],manifest_sha256=digest))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','checkpoint','prompts','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.config,a.checkpoint,a.prompts,a.out)

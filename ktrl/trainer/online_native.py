"""Run synchronous online RL in an isolated, explicitly selected host process tree."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import uuid

from ktrl.configs.recipe import REPO,load_plan,load_rollout_config
from ktrl.workers.rollout.sglang.smoke import SGLangClient,validate_response
from .online import save,phase,gpu_snapshot
from ktrl.weights.identity import build_manifest, sha256

EXPORT_FREE_BYTES = 80 * 1024**3

HOST=socket.gethostname().split('.')[0]
ROOT=Path(os.environ.get('KTRL_RUN_ROOT',
    '/raid/data/zjx/kt-baseline/checks/ktrl-review-005-20261108' if HOST=='qjh005'
    else '/home/zjx/Documents/ktrl-review-000-20261108')).resolve()
PYTHON=Path('/mnt/qjh007/jiahengdai/venvs/kt-py31115/bin/python')
OVERLAY=Path('/raid/data/zjx/kt-baseline/checks/ktrl-m2-r01' if HOST=='qjh005'
    else '/mnt/qjh005/zjx/kt-baseline/checks/ktrl-m2-r01')
CUDNN=Path('/mnt/qjh007/jiahengdai/qwen36-native-smoke-20260930/sglang-kt-migration-20261102-p01/service-deps-cu12-cudnn916/nvidia/cudnn/lib')


def environment(out,actor):
    (out/'tmp').mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.pop('DOCKER_HOST',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2' if actor else '4',
        MKL_NUM_THREADS='2' if actor else '4',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true',
        XDG_CACHE_HOME=str(out/'cache'),TRITON_CACHE_DIR=str(out/'cache/triton'),
        TMPDIR=str(out/'tmp'),CUDA_CACHE_PATH=str(out/'cache/cuda'),TORCH_EXTENSIONS_DIR=str(out/'cache/extensions'),
        TORCHINDUCTOR_CACHE_DIR=str(out/'cache/inductor'),FLASHINFER_WORKSPACE_BASE=str(out/'cache/flashinfer'),
        LD_LIBRARY_PATH=str(CUDNN)+':'+str(PYTHON.parent.parent/'lib/python3.11/site-packages/nvidia/cuda_runtime/lib'))
    if actor:
        env.update(PYTHONPATH=f'{OVERLAY}/site:{OVERLAY}/extras:{REPO}',KT_SFT_PROFILE='1')
    else:
        env.update(PYTHONPATH=f'{REPO}/ktrl/workers/rollout/sglang/publication_worker:{REPO}:{REPO}/profiling/worker:{REPO}/third_party/sglang/python',
            KTRL_PUBLICATION_OUTPUT=str(out),KTRL_ONLINE_PHASES='1',KTRL_SKIP_LOGITS_DUMP='1',
            KTRL_REQUEST_SEEDS='1',
            KT_PHASE_HOOKS='1',KT_TORCH_DIAGNOSTIC='0',SGLANG_PROFILE_V2='0')
    return env


def process_group(pid):
    return os.getpgid(pid)


def owned_descendant(pid, owner_pid):
    """Torch elastic can put each rank in a separate process group."""
    while pid > 1:
        if pid == owner_pid:
            return True
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return False
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
    return False


def guard_gpu(allowed=(), *, actor_pid=None):
    ids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    for value in ids.splitlines():
        try:
            pgid=process_group(int(value))
        except ProcessLookupError:
            continue
        if pgid not in allowed and not (actor_pid and owned_descendant(int(value), actor_pid)):
            raise RuntimeError(f'Unrelated GPU process {value}; no new workload started')


def run_worker(out,stage,args,*,gpu=False,allowed=()):
    folder=out/stage
    folder.mkdir()
    env=environment(folder,True)
    if not gpu:
        env['CUDA_VISIBLE_DEVICES']=''
    else:
        guard_gpu(allowed)
    argv=[str(PYTHON),'-B']+args
    from .precision_gate import source_digest
    save(folder/'launch.json',dict(argv=argv,environment=env_subset(env),source_sha256=source_digest()))
    with (folder/'run.log').open('x') as log:
        result=subprocess.run(argv,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    save(folder/'exit.json',dict(exit_code=result.returncode))
    if result.returncode:
        raise RuntimeError(f'{stage} exited {result.returncode}; see {folder}/run.log')


def env_subset(env):
    return {k:v for k,v in env.items() if k in ('PYTHONPATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','KT_SFT_PROFILE')
            or k.startswith('KTRL_')}


@contextmanager
def resident_worker(out, args, allowed):
    guard_gpu(allowed)
    out.mkdir()
    env = environment(out, True)
    argv = [str(PYTHON), '-B', *args]
    from .precision_gate import source_digest
    save(out/'launch.json', dict(argv=argv, environment=env_subset(env), source_sha256=source_digest()))
    with (out/'run.log').open('x') as log:
        proc = subprocess.Popen(argv, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    try:
        yield proc
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        save(out/'exit.json', dict(exit_code=proc.returncode))


def wait_actor_round(proc, proof, timeout=7200):
    deadline = time.monotonic() + timeout
    while not proof.is_file():
        if proc.poll() is not None:
            if proc.returncode == 0 and proof.is_file():
                break
            raise RuntimeError(f"Resident Actor exited {proc.returncode} before {proof}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Resident Actor did not produce {proof}")
        time.sleep(.2)


def drain_rollout(client,folder,label):
    deadline=time.monotonic()+120
    attempts=0
    while True:
        attempts+=1
        try:
            result=client.call('/flush_cache',{},timeout=30)
            save(folder/f'{label}-drained.json',dict(attempts=attempts,response=result,
                native_scheduler_confirmed_empty=True,time_unix_ns=time.time_ns()))
            return
        except urllib.error.HTTPError as exc:
            if exc.code!=400 or time.monotonic()>deadline:raise
            time.sleep(.1)


def control_probe(client,folder,prompt,version,seed=42):
    save(folder/'capture-next-prefill.json',dict(filename=f'{version}-control-logits.pt'))
    result=client.generate(dict(input_ids=prompt,sampling_params=dict(temperature=0,max_new_tokens=2,ignore_eos=True,sampling_seed=seed)))
    assert not (folder/'capture-next-prefill.json').exists()
    assert (folder/f'{version}-control-logits.pt').is_file()
    save(folder/f'{version}-control-response.json',result)
    drain_rollout(client,folder,version+'-control')


@contextmanager
def resources(out):
    stop=threading.Event()
    def collect():
        with (out/'resources.jsonl').open('x') as log:
            while not stop.is_set():
                record=dict(time_unix_ns=time.time_ns(),meminfo=Path('/proc/meminfo').read_text())
                for key,query in [('gpu','index,memory.used,memory.free'),('processes','pid,gpu_uuid,used_memory')]:
                    option='--query-gpu=' if key=='gpu' else '--query-compute-apps='
                    p=subprocess.run(['nvidia-smi',option+query,'--format=csv,noheader,nounits'],capture_output=True,text=True)
                    record[key]=p.stdout;record[key+'_returncode']=p.returncode
                log.write(json.dumps(record)+'\n');log.flush()
                stop.wait(5)
    thread=threading.Thread(target=collect,daemon=True);thread.start()
    try:yield
    finally:stop.set();thread.join(timeout=30)


@contextmanager
def service(config,folder,plan):
    guard_gpu()
    folder.mkdir()
    rollout=load_rollout_config(config)
    args=rollout['server_args']
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    name='ktrl-'+HOST+'-'+uuid.uuid4().hex[:12]
    argv=[str(PYTHON),'-m',rollout['native_entrypoint']]
    for key,value in args.items():
        if isinstance(value,bool):
            if value:argv+=['--'+key.replace('_','-')]
        else:argv+=['--'+key.replace('_','-'),str(value)]
    argv+=['--host','127.0.0.1','--port',str(port),'--served-model-name',name,
        '--context-length',str(plan['inputs']['context_length']),'--mem-fraction-static','0.35',
        '--max-running-requests','32','--chunked-prefill-size','1024','--cuda-graph-max-bs','32',
        '--disable-radix-cache','--disable-shared-experts-fusion','--random-seed',str(plan['inputs']['seed']),
        '--skip-server-warmup','--enable-memory-saver','--enable-weights-cpu-backup','--sampling-backend','pytorch']
    env=environment(folder,False)
    save(folder/'launch.json',dict(argv=argv,environment=env_subset(env),name=name,port=port))
    with (folder/'server.log').open('x') as log:
        proc=subprocess.Popen(argv,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(folder/'process.json',dict(pid=proc.pid,process_group=proc.pid))
    monitor_stop=threading.Event()
    monitor=threading.Thread(target=monitor_publication_failure,args=(folder,proc.pid,monitor_stop),daemon=True)
    monitor.start()
    client=SGLangClient(f'http://127.0.0.1:{port}')
    try:
        deadline=time.monotonic()+2400
        while True:
            if proc.poll() is not None:raise RuntimeError(f'Rollout startup exited {proc.returncode}')
            if time.monotonic()>deadline:raise TimeoutError('Rollout startup')
            try:models=client.call('/v1/models',timeout=8)
            except (urllib.error.URLError,TimeoutError,ConnectionError):
                time.sleep(3);continue
            assert name in [m['id'] for m in models['data']]
            break
        session='control-'+uuid.uuid4().hex
        assert client.call('/open_session',dict(session_id=session,capacity_of_str_len=1))==session
        client.call('/close_session',dict(session_id=session))
        info=client.call('/server_info');save(folder/'server-info.json',info)
        expected=dict(args,enable_memory_saver=True,context_length=plan['inputs']['context_length'],
            served_model_name=name,disable_cuda_graph=False)
        differences={k:(v,info.get(k)) for k,v in expected.items() if info.get(k)!=v}
        assert not differences,differences
        yield client,proc.pid
    finally:
        monitor_stop.set();monitor.join(timeout=2)
        # The HTTP parent may already have exited after a worker fault. Its
        # remaining worker process group still belongs to this service.
        try:
            os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=30)
        deadline=time.monotonic()+30
        while True:
            try:os.killpg(proc.pid,0)
            except ProcessLookupError:break
            if time.monotonic()>=deadline:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                break
            time.sleep(.5)
        save(folder/'exit.json',dict(pid=proc.pid,exit_code=proc.returncode))


def monitor_publication_failure(folder,pid,stop):
    """A failed TP worker may leave HTTP awaiting a reply; stop its owned server."""
    while not stop.wait(.2):
        failures=list(folder.glob('publication-failed-rank*.json'))
        if not failures:continue
        path=Path(f'/proc/{pid}/cmdline')
        if not path.exists():return
        command=path.read_bytes().split(b'\0')
        launch=json.loads((folder/'launch.json').read_text())
        assert launch['name'].encode() in command and os.getpgid(pid)==pid,'Service identity changed'
        save(folder/'failure-supervisor.json',dict(pid=pid,signal='SIGQUIT',
            reason='Publication worker failed; unblock HTTP and stop owned process tree',
            failures=[p.name for p in failures],time_unix_ns=time.time_ns()))
        os.kill(pid,signal.SIGQUIT)
        return


def generate(client,folder,questions,inputs,policy):
    folder.mkdir()
    target=folder/'phase';target.mkdir()
    client.call('/start_profile',dict(activities=['KT_PHASE'],output_dir=str(target),record_shapes=False,with_stack=False))
    def one(pair):
        wave,index=pair
        uid=f'm4-w{wave}-q{index:02d}'
        payload=dict(rid=uid,input_ids=questions[index]['prompt'],stream=False,return_logprob=True,
            sampling_params=dict(temperature=1.,top_p=1.,top_k=-1,max_new_tokens=inputs['max_response_tokens'],
                sampling_seed=inputs['seed']+questions[index]['dataset_row']*inputs['responses_per_prompt']+wave,
                ignore_eos=False))
        with (folder/(uid+'-request.json')).open('x') as f:json.dump(payload,f)
        start=time.time_ns()
        result=client.call('/generate',payload,timeout=18000)
        validate_response(payload,result)
        save(folder/(uid+'.json'),result)
        save(folder/(uid+'-timing.json'),dict(start_unix_ns=start,end_unix_ns=time.time_ns(),policy_version=policy))
        print(f'{policy} {uid}: {len(result["output_ids"])} tokens',flush=True)
        return len(result['output_ids'])
    try:
        with ThreadPoolExecutor(max_workers=32) as pool:
            tokens=sum(pool.map(one,[(w,i) for w in range(4) for i in range(8)]))
    finally:client.call('/stop_profile',{},timeout=120)
    from .phase_completion import wait_phase_traces
    traces=wait_phase_traces(target)
    save(folder/'phase-complete.json',dict(ranks=[t['rank'] for t in traces],
        decode_counts=[sum(row['phase']=='DECODE' for row in t['rows']) for t in traces],
        time_unix_ns=time.time_ns()))
    return tokens


def retire_model_export(root, checkpoint, step):
    """Remove only manifest-listed weight shards from a superseded run export."""
    expected = root / f"round-{step:02d}" / "actor" / "trainer" / f"checkpoint-{step}"
    if checkpoint.resolve() != expected or not checkpoint.is_relative_to(root):
        raise ValueError(f"Unexpected checkpoint path: {checkpoint}")
    manifest = json.loads((checkpoint / "publication-manifest.json").read_text())
    if manifest["version_id"] != step:
        raise ValueError(f"Unexpected checkpoint version at {checkpoint}")
    shards = [checkpoint / name for name in manifest["files"] if name.endswith(".safetensors")]
    if not shards or any(p.resolve().parent != checkpoint or p.is_symlink() for p in shards):
        raise ValueError(f"Unsafe publication shards at {checkpoint}")
    for shard in shards:
        shard.unlink()
    save(checkpoint.parent.parent / "export-retired.json",
         dict(step=step, manifest=str(checkpoint / "publication-manifest.json"),
              removed_files=[shard.name for shard in shards]))


def archive_rollout(current):
    """Keep lossless raw rollout evidence in a bounded-size per-batch archive."""
    rollout = current / "rollout"
    archive = current / "rollout.tar.gz"
    if rollout.resolve() != current.resolve() / "rollout" or rollout.is_symlink():
        raise ValueError(f"Unexpected rollout path: {rollout}")
    with tarfile.open(archive, "x:gz", compresslevel=1) as writer:
        writer.add(rollout, arcname="rollout")
    with tarfile.open(archive, "r:gz") as reader:
        expected = {p.relative_to(current).as_posix(): p.stat().st_size for p in rollout.rglob("*") if p.is_file()}
        actual = {item.name: item.size for item in reader if item.isfile()}
        if actual != expected:
            raise RuntimeError(f"Incomplete rollout archive: {archive}")
    save(current / "rollout-archive.json", dict(file=archive.name, sha256=sha256(archive), files=len(expected)))
    shutil.rmtree(rollout)


def run(config,out,*,steps=2,final_model_only=False,resident_actor=False):
    import fcntl
    assert HOST in ('qjh000','qjh005'),'This entrypoint is restricted to authorized hosts'
    out=out.resolve();config=config.resolve()
    from .precision_gate import digest, verify
    gate = verify(config)
    if resident_actor:
        validation = gate.get('resident_validation', {})
        if not validation or digest(validation['path']) != validation['sha256']:
            raise RuntimeError("Resident Actor requires verified native smoke evidence")
        proof = json.loads(Path(validation['path']).read_text())
        if proof.get('resident_updates', 0) < min(steps, 3) or proof.get('source_sha256') != gate['source_sha256']:
            raise RuntimeError("Resident Actor smoke did not verify this source")
    assert out.parent==ROOT and out.name.startswith('rl-')
    with (ROOT/'.rl.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard_gpu()
        out.mkdir()
        shutil.copyfile(config,out/'recipe.yaml')
        plan=load_plan(config,steps=steps,profiling='light')
        assert plan['inputs']['prompts_per_step']==8 and plan['inputs']['responses_per_prompt']==4
        assert plan['inputs']['responses_per_update']==32 and plan['gradient_accumulation_steps']==4
        assert plan['inputs']['max_response_tokens']==8192 and plan['actor']['world_size']==8
        if resident_actor and shutil.disk_usage(out).free < EXPORT_FREE_BYTES:
            raise RuntimeError("Insufficient free space for the first model export")
        save(out/'resolved-plan.json',plan)
        status='failed';checkpoint=None
        try:
            with resources(out):
                with phase(out,'prepare_prompts'):
                    inputs = plan["inputs"]
                    batch_size = inputs["prompts_per_step"]
                    run_worker(out,'prepare',['-m','ktrl.trainer.online_batch','prepare','--out',str(out),
                        '--model',plan['model']['path'],'--parquet',plan['inputs']['train_file'],
                        '--prompt-count',str(steps*batch_size),'--seed',str(inputs['seed']),
                        '--max-prompt-tokens',str(inputs['max_prompt_tokens'])])
                prompt_schedule=json.loads((out/'prompts.json').read_text())
                with service(config,out/'service',plan) as (client,pgid), ExitStack() as workers:
                    actor_proc = None
                    probe_prompt = prompt_schedule[0]["prompt"]
                    control_probe(client,out/'service',probe_prompt,'v0',plan['inputs']['seed'])
                    updates = 0
                    for batch_index in range(steps):
                        step = updates + 1
                        questions = prompt_schedule[batch_index*batch_size:(batch_index+1)*batch_size]
                        current=out/f'round-{step:02d}';current.mkdir()
                        save(current/'prompts.json', questions)
                        shutil.copyfile(config,current/'recipe.yaml')
                        policy=f'v{step-1}'
                        save(current/'round.json',dict(**plan['inputs'],policy_version=policy,
                            rollout_correction=plan["rollout_correction"],
                            dataset_batch=batch_index+1, dataset_rows=[q["dataset_row"] for q in questions],
                            actor_resume_checkpoint=str(checkpoint) if checkpoint and not resident_actor else None,
                            resident_actor=resident_actor))
                        if checkpoint:
                            for rank in range(8):
                                proof=json.loads((out/f'service/published-{policy}-rank{rank}.json').read_text())
                                assert proof['version']==str(checkpoint) and proof['actor_step']==step-1
                        with phase(current,f'live_rollout_{policy}'):
                            tokens=generate(client,current/'rollout',questions,plan['inputs'],policy)
                        replays=[json.loads((out/f'service/replays-rank{r}.json').read_text()) for r in range(8)]
                        assert all(r.get(policy,0)>0 for r in replays)
                        save(current/'rollout-policy-proof.json',dict(version=policy,rank_replays=replays,
                            checkpoint=str(checkpoint) if checkpoint else 'base'))
                        with phase(current,'reward_advantage'):
                            run_worker(current,'score',['-m','ktrl.trainer.online_batch','score','--out',str(current),'--model',plan['model']['path']])
                        batch_audit = json.loads((current/'batch'/'batch-audit.json').read_text())
                        if steps > 2:
                            archive_rollout(current)
                        if batch_audit["nonzero_advantage_responses"] == 0:
                            save(current/'skipped.json', dict(reason="all GRPO advantages are zero", policy=policy))
                            current.rename(out/f"skipped-batch-{batch_index+1:04d}")
                            save(out/'data-cursor.json', dict(batches_consumed=batch_index+1, updates=updates))
                            continue
                        if resident_actor and shutil.disk_usage(out).free < EXPORT_FREE_BYTES:
                            raise RuntimeError("Insufficient free space for the next model export")
                        with phase(current,'sleep_rollout'):
                            drain_rollout(client,current,'before-sleep')
                            save(current/'release.json',client.call('/release_memory_occupation',{},timeout=180))
                            snap=gpu_snapshot(current,'sleeping')
                            guard_gpu((pgid,), actor_pid=actor_proc.pid if actor_proc else None)
                            minimum_free = 65000 if actor_proc else 90000
                            assert min(float(s.split(',')[2]) for s in snap['devices'].splitlines()) >= minimum_free
                        args=['-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                            '--module','ktrl.workers.engine.fsdp.actor_update','--config',str(config),
                            '--batch-dir',str(current/'batch'),'--out',str(current/'actor')]
                        if checkpoint and not resident_actor:
                            args+=['--restore-from',str(checkpoint),'--resume-training']
                        model_only = resident_actor or (final_model_only and step == steps)
                        if model_only and not resident_actor:
                            args.append('--save-model-only')
                        with phase(current,'actor_update'):
                            if resident_actor:
                                if actor_proc is None:
                                    args += ['--resident-rounds', str(steps)]
                                    actor_proc = workers.enter_context(resident_worker(current/'actor',args,(pgid,)))
                                else:
                                    (current/'actor').mkdir()
                                    save(current/'actor-ready.json',dict(step=step))
                                wait_actor_round(actor_proc,current/'actor/accepted.json')
                                if step == steps and actor_proc.wait(timeout=120) != 0:
                                    raise RuntimeError(f"Resident Actor exited {actor_proc.returncode}")
                            else:
                                run_worker(current,'actor',args,gpu=True,allowed=(pgid,))
                        proof=json.loads((current/'actor/accepted.json').read_text())
                        assert proof['updates']==1 and proof['start_step']==step-1 and proof['end_step']==step
                        assert proof['checkpoint_kind']==('model_only' if model_only else 'full')
                        if step > 1:
                            for rank in range(8):
                                resumed=json.loads((current/f'actor/resume-rank{rank}.json').read_text())
                                assert resumed['restored_step']==resumed['scheduler_epoch']==step-1
                                assert resumed['exact_parameter_and_moment_positions']>0
                                if resident_actor:
                                    previous=json.loads((out/f'round-{step-1:02d}/actor/resident-rank{rank}.json').read_text())
                                    latest=json.loads((current/f'actor/resident-rank{rank}.json').read_text())
                                    assert resumed['resident'] and resumed['pid']==previous['pid']==latest['pid']
                                    assert previous['optimizer_id']==latest['optimizer_id']
                        previous_checkpoint = checkpoint
                        checkpoint=current/f'actor/trainer/checkpoint-{step}'
                        with phase(current,'publication_manifest'):
                            manifest_sha=build_manifest(checkpoint)
                        try:
                            with phase(current,'resume_publish'):
                                save(current/'resume.json',client.call('/resume_memory_occupation',{},timeout=180))
                                drain_rollout(client,current,'before-publication')
                                request=dict(model_path=str(checkpoint),load_format='safetensors',flush_cache=True,recapture_cuda_graph=True)
                                save(current/'publication-request.json',dict(**request,expected_version_id=step,expected_manifest_sha256=manifest_sha))
                                result=client.call('/update_weights_from_disk',request,timeout=1800)
                                save(current/'publication.json',result);assert result.get('success'),result
                                for rank in range(8):
                                    published=json.loads((out/f'service/published-v{step}-rank{rank}.json').read_text())
                                    assert published['version']==str(checkpoint) and published['actor_step']==step
                                    assert published['manifest_sha256']==manifest_sha
                                for layer in range(40):
                                    verified=json.loads((out/f'service/cpu-verified-v{step}-layer{layer}.json').read_text())
                                    assert verified['packed_native_forward_relative_l2']<=.06
                                control_probe(client,out/'service',probe_prompt,f'v{step}',plan['inputs']['seed'])
                                import torch
                                previous=torch.load(out/f'service/v{step-1}-control-logits.pt',weights_only=True)
                                latest=torch.load(out/f'service/v{step}-control-logits.pt',weights_only=True)
                                difference=(latest-previous).abs()
                                assert torch.isfinite(latest).all() and difference.max().item()>0
                                save(current/'same-input-publication-logits.json',dict(before=f'v{step-1}',after=f'v{step}',
                                    max_absolute_change=difference.max().item(),mean_absolute_change=difference.mean().item(),
                                    changed_elements=int((difference!=0).sum())))
                        except BaseException as exc:
                            save(current/'publication-failed.json',dict(time_unix_ns=time.time_ns(),policy_step=step,
                                checkpoint=str(checkpoint),error=str(exc),action='stop owned rollout; forbid next generation',
                                rank_failures=[json.loads(p.read_text()) for p in (out/'service').glob('publication-failed-rank*.json')]))
                            raise
                        save(current/'accepted.json',dict(updates=1,output_tokens=tokens,generated_policy=policy,published_policy=f'v{step}'))
                        updates = step
                        save(out/'data-cursor.json', dict(batches_consumed=batch_index+1, updates=updates))
                        if resident_actor and steps > 2 and previous_checkpoint is not None:
                            retire_model_export(out, previous_checkpoint, step-1)
                    if not updates:
                        raise RuntimeError("No batch had nonzero GRPO advantages")
                    if actor_proc is not None and updates < steps:
                        save(out/'actor-stop.json', dict(step=updates))
                        if actor_proc.wait(timeout=120) != 0:
                            raise RuntimeError(f"Resident Actor exited {actor_proc.returncode}")
                    probe=dict(input_ids=probe_prompt,sampling_params=dict(temperature=0,max_new_tokens=32,sampling_seed=plan['inputs']['seed']))
                    save(out/f'final-v{updates}-generation.json',client.generate(probe))
                    assert all(json.loads((out/f'service/replays-rank{r}.json').read_text()).get(f'v{updates}',0)>0 for r in range(8))
                guard_gpu()
                save(out/'accepted.json',dict(real_online_updates=updates, dataset_batches=steps, skipped_batches=steps-updates,
                    optimizer_steps=list(range(1,updates+1)), optimizer_state_verified_before_updates=list(range(2,updates+1)),
                    final_generation_policy=f'v{updates}',
                    final_checkpoint_kind='model_only' if final_model_only or resident_actor else 'full',
                    resident_actor=resident_actor,
                    numerical_parity='deferred; no claim of actor-rollout numerical agreement'))
                status='complete'
        finally:
            save(out/'completion.json',dict(status=status,time_unix_ns=time.time_ns()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.config,a.out)

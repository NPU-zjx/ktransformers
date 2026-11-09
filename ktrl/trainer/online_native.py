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

from ktrl.configs.recipe import REPO, load_plan, plan_runtime
from ktrl.workers.rollout.sglang.smoke import SGLangClient,validate_response,wait_ready
from .online import save,phase,gpu_snapshot
from ktrl.weights.identity import build_manifest, sha256

def environment(out, actor, runtime):
    (out/'tmp').mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.pop('DOCKER_HOST',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',
        OMP_NUM_THREADS='2' if actor else '4',
        MKL_NUM_THREADS='2' if actor else '4',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true',
        XDG_CACHE_HOME=str(out/'cache'),TRITON_CACHE_DIR=str(out/'cache/triton'),
        TMPDIR=str(out/'tmp'),CUDA_CACHE_PATH=str(out/'cache/cuda'),TORCH_EXTENSIONS_DIR=str(out/'cache/extensions'),
        TORCHINDUCTOR_CACHE_DIR=str(out/'cache/inductor'),FLASHINFER_WORKSPACE_BASE=str(out/'cache/flashinfer'))
    if runtime["cuda_visible_devices"] is not None:
        env["CUDA_VISIBLE_DEVICES"] = runtime["cuda_visible_devices"]
    if runtime["library_paths"]:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(runtime["library_paths"] +
                                               ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    inherited = [env["PYTHONPATH"]] if env.get("PYTHONPATH") else []
    if actor:
        env.update(PYTHONPATH=os.pathsep.join([str(REPO), *runtime["actor_pythonpath"], *inherited]),
                   KT_SFT_PROFILE='1')
    else:
        paths = [REPO / "ktrl/workers/rollout/sglang/publication_worker", REPO,
                 REPO / "profiling/worker", REPO / "third_party/sglang/python"]
        env.update(PYTHONPATH=os.pathsep.join([*(str(path) for path in paths),
                                              *runtime["rollout_pythonpath"], *inherited]),
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


def guard_gpu(allowed=(), *, actor_pid=None, devices=None):
    selector = [f"--id={devices}"] if devices else []
    ids=subprocess.check_output(['nvidia-smi',*selector,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    for value in ids.splitlines():
        try:
            pgid=process_group(int(value))
        except ProcessLookupError:
            continue
        if pgid not in allowed and not (actor_pid and owned_descendant(int(value), actor_pid)):
            raise RuntimeError(f'Unrelated GPU process {value}; no new workload started')


def run_worker(out,stage,args,*,runtime,gpu=False,allowed=()):
    folder=out/stage
    folder.mkdir()
    env=environment(folder,True,runtime)
    if not gpu:
        env['CUDA_VISIBLE_DEVICES']=''
    else:
        guard_gpu(allowed, devices=runtime["cuda_visible_devices"])
    argv=[runtime["python"],'-B']+args
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
def resident_worker(out, args, allowed, *, runtime=None):
    runtime = runtime or plan_runtime({}, Path.cwd())
    guard_gpu(allowed, devices=runtime["cuda_visible_devices"])
    out.mkdir()
    env = environment(out, True, runtime)
    argv = [runtime["python"], '-B', *args]
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
    runtime = plan["runtime"]
    guard_gpu(devices=runtime["cuda_visible_devices"])
    folder.mkdir()
    rollout=plan["rollout"]
    args=rollout['server_args']
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    name='ktrl-'+uuid.uuid4().hex[:12]
    argv=[runtime["python"],'-m',rollout['native_entrypoint']]
    for key,value in args.items():
        if isinstance(value,bool):
            if value:argv+=['--'+key.replace('_','-')]
        else:argv+=['--'+key.replace('_','-'),str(value)]
    batch = plan["inputs"]["responses_per_update"]
    argv+=['--host','127.0.0.1','--port',str(port),'--served-model-name',name,
        '--context-length',str(plan['inputs']['context_length']),
        '--max-running-requests',str(batch),'--chunked-prefill-size',str(plan["inputs"]["max_prompt_tokens"]),
        '--cuda-graph-max-bs',str(batch),
        '--disable-radix-cache','--disable-shared-experts-fusion','--random-seed',str(plan['inputs']['seed']),
        '--skip-server-warmup','--enable-memory-saver','--enable-weights-cpu-backup','--sampling-backend','pytorch']
    env=environment(folder,False,runtime)
    save(folder/'launch.json',dict(argv=argv,environment=env_subset(env),name=name,port=port))
    with (folder/'server.log').open('x') as log:
        proc=subprocess.Popen(argv,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(folder/'process.json',dict(pid=proc.pid,process_group=proc.pid))
    monitor_stop=threading.Event()
    monitor=threading.Thread(target=monitor_publication_failure,args=(folder,proc.pid,monitor_stop),daemon=True)
    monitor.start()
    client=SGLangClient(f'http://127.0.0.1:{port}')
    try:
        expected=dict(args,enable_memory_saver=True,context_length=plan['inputs']['context_length'],
            served_model_name=name,disable_cuda_graph=False)
        wait_ready(client, lambda: proc.poll() is None, expected, folder)
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


def generate(client,folder,questions,inputs,policy,tp_size):
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
        validate_response(payload,result,minimum_tokens=1)
        save(folder/(uid+'.json'),result)
        save(folder/(uid+'-timing.json'),dict(start_unix_ns=start,end_unix_ns=time.time_ns(),policy_version=policy))
        print(f'{policy} {uid}: {len(result["output_ids"])} tokens',flush=True)
        return len(result['output_ids'])
    try:
        answers = inputs["responses_per_prompt"]
        with ThreadPoolExecutor(max_workers=inputs["responses_per_update"]) as pool:
            tokens=sum(pool.map(one,[(w,i) for w in range(answers) for i in range(len(questions))]))
    except BaseException:
        try:
            client.call('/stop_profile',{},timeout=120)
        except Exception as cleanup_error:
            save(folder/'stop-profile-failed.json',dict(error=str(cleanup_error)))
        raise
    else:
        client.call('/stop_profile',{},timeout=120)
    from .phase_completion import wait_phase_traces
    traces=wait_phase_traces(target, world_size=tp_size)
    save(folder/'phase-complete.json',dict(ranks=[t['rank'] for t in traces],
        decode_counts=[sum(row['phase']=='DECODE' for row in t['rows']) for t in traces],
        time_unix_ns=time.time_ns()))
    return tokens


def retire_model_export(root, checkpoint, step, optimizer_step=None):
    """Remove only manifest-listed weight shards from a superseded run export."""
    optimizer_step = step if optimizer_step is None else optimizer_step
    expected = root / f"round-{step:02d}" / "actor" / "trainer" / f"checkpoint-{optimizer_step}"
    if checkpoint.resolve() != expected or not checkpoint.is_relative_to(root):
        raise ValueError(f"Unexpected checkpoint path: {checkpoint}")
    manifest = json.loads((checkpoint / "publication-manifest.json").read_text())
    if manifest["version_id"] != optimizer_step:
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


def run(config,out,*,steps=2,final_model_only=False,resident_actor=False,resume_from=None):
    import fcntl
    out=out.resolve();config=config.resolve()
    plan=load_plan(config,steps=steps,profiling='light')
    from .resume import load_resume, contract, retire_old_full_checkpoints
    resume = load_resume(resume_from, plan) if resume_from else None
    runtime = plan["runtime"]
    devices = runtime["cuda_visible_devices"]
    actor_world = plan["actor"]["world_size"]
    updates_per_round = plan["actor"]["optimizer_steps_per_rollout"]
    rollout_tp = plan["rollout"]["server_args"]["tp_size"]
    if devices is not None and (len(devices.split(",")) if devices else 0) < max(actor_world, rollout_tp):
        raise ValueError("Selected CUDA devices do not cover Actor world size and rollout TP")
    if not plan["inputs"]["train_file"]:
        raise ValueError("Online execution requires data.train_file")
    if plan["inputs"]["max_response_tokens"] < 2:
        raise ValueError("Online rollout verification requires max_response_tokens >= 2")
    if runtime["require_precision_gate"]:
        from .precision_gate import digest, verify
        gate = verify(config)
        if resident_actor:
            validation = gate.get('resident_validation', {})
            if not validation or digest(validation['path']) != validation['sha256']:
                raise RuntimeError("Resident Actor requires verified native smoke evidence")
            proof = json.loads(Path(validation['path']).read_text())
            if proof.get('resident_updates', 0) < min(steps, 3) or proof.get('source_sha256') != gate['source_sha256']:
                raise RuntimeError("Resident Actor smoke did not verify this source")
    export_free = runtime["export_min_free_gib"] * 1024**3
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent/'.rl.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard_gpu(devices=devices)
        out.mkdir()
        shutil.copyfile(config,out/'recipe.yaml')
        if resident_actor and shutil.disk_usage(out).free < export_free:
            raise RuntimeError("Insufficient free space for the first model export")
        save(out/'resolved-plan.json',plan)
        from .precision_gate import source_digest
        launch_digest = source_digest()
        save(out/"source.json",dict(repo=str(REPO.resolve()),source_sha256=launch_digest,
             head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
             dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=REPO,text=True))))
        if os.environ.get("KTRL_EXPECTED_SOURCE") and os.environ["KTRL_EXPECTED_SOURCE"] != launch_digest:
            raise RuntimeError("Source differs from the GPU-validated version")
        status='failed';checkpoint=resume_from.resolve() if resume_from else None
        try:
            with resources(out):
                with phase(out,'prepare_prompts'):
                    inputs = plan["inputs"]
                    batch_size = inputs["prompts_per_step"]
                    run_worker(out,'prepare',['-m','ktrl.trainer.online_batch','prepare','--out',str(out),
                        '--model',plan['model']['path'],'--parquet',plan['inputs']['train_file'],
                        '--prompt-count',str(steps*batch_size),'--seed',str(inputs['seed']),
                        '--max-prompt-tokens',str(inputs['max_prompt_tokens'])], runtime=runtime)
                prompt_schedule=json.loads((out/'prompts.json').read_text())
                with service(config,out/'service',plan) as (client,pgid), ExitStack() as workers:
                    actor_proc = None
                    probe_prompt = prompt_schedule[0]["prompt"]
                    control_probe(client,out/'service',probe_prompt,'v0',plan['inputs']['seed'])
                    updates = resume["completed_rounds"] if resume else 0
                    if resume:
                        request = dict(model_path=str(checkpoint), load_format="safetensors",
                                       flush_cache=True, recapture_cuda_graph=True)
                        result = client.call("/update_weights_from_disk", request, timeout=1800)
                        if not result.get("success"):
                            raise RuntimeError(f"Resume publication failed: {result}")
                        version = resume["optimizer_steps"]
                        for rank in range(rollout_tp):
                            published = json.loads((out/f"service/published-v{version}-rank{rank}.json").read_text())
                            assert published["actor_step"] == version and published["version"] == str(checkpoint)
                        control_probe(client,out/"service",probe_prompt,f"v{version}",inputs["seed"])
                        save(out/"resumed.json",resume)
                    for batch_index in range(resume["batches_consumed"] if resume else 0, steps):
                        if source_digest() != launch_digest:
                            raise RuntimeError("Source changed during training; refusing a mixed-code run")
                        step = updates + 1
                        actor_start = updates * updates_per_round
                        actor_end = step * updates_per_round
                        questions = prompt_schedule[batch_index*batch_size:(batch_index+1)*batch_size]
                        current=out/f'round-{step:02d}';current.mkdir()
                        save(current/'prompts.json', questions)
                        shutil.copyfile(config,current/'recipe.yaml')
                        policy=f'v{actor_start}'
                        save(current/'round.json',dict(**plan['inputs'],policy_version=policy,
                            rollout_correction=plan["rollout_correction"],
                            dataset_batch=batch_index+1, dataset_rows=[q["dataset_row"] for q in questions],
                            actor_resume_checkpoint=str(checkpoint) if checkpoint and not resident_actor else None,
                            resident_actor=resident_actor))
                        if checkpoint:
                            for rank in range(rollout_tp):
                                proof=json.loads((out/f'service/published-{policy}-rank{rank}.json').read_text())
                                assert proof['version']==str(checkpoint) and proof['actor_step']==actor_start
                        with phase(current,f'live_rollout_{policy}'):
                            tokens=generate(client,current/'rollout',questions,plan['inputs'],policy,rollout_tp)
                        replays=[json.loads((out/f'service/replays-rank{r}.json').read_text()) for r in range(rollout_tp)]
                        assert all(r.get(policy,0)>0 for r in replays)
                        save(current/'rollout-policy-proof.json',dict(version=policy,rank_replays=replays,
                            checkpoint=str(checkpoint) if checkpoint else 'base'))
                        with phase(current,'reward_advantage'):
                            run_worker(current,'score',['-m','ktrl.trainer.online_batch','score','--out',str(current),'--model',plan['model']['path']],runtime=runtime)
                        batch_audit = json.loads((current/'batch'/'batch-audit.json').read_text())
                        if steps > 2:
                            archive_rollout(current)
                        if batch_audit["nonzero_advantage_responses"] == 0:
                            save(current/'skipped.json', dict(reason="all GRPO advantages are zero", policy=policy))
                            from .metrics import batch_metrics
                            with (out/"metrics.jsonl").open("a") as log:
                                log.write(json.dumps(batch_metrics(out, current), allow_nan=False)+"\n")
                            current.rename(out/f"skipped-batch-{batch_index+1:04d}")
                            save(out/'data-cursor.json', dict(batches_consumed=batch_index+1, updates=updates, optimizer_steps=updates*updates_per_round))
                            continue
                        if resident_actor and shutil.disk_usage(out).free < export_free:
                            raise RuntimeError("Insufficient free space for the next model export")
                        with phase(current,'sleep_rollout'):
                            drain_rollout(client,current,'before-sleep')
                            save(current/'release.json',client.call('/release_memory_occupation',{},timeout=180))
                            gpu_snapshot(current,'sleeping')
                            guard_gpu((pgid,), actor_pid=actor_proc.pid if actor_proc else None, devices=devices)
                        args=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={actor_world}',
                            '--module','ktrl.workers.engine.fsdp.actor_update','--config',str(config),
                            '--batch-dir',str(current/'batch'),'--out',str(current/'actor')]
                        if checkpoint and (not resident_actor or actor_proc is None):
                            args+=['--restore-from',str(checkpoint),'--resume-training']
                        model_only = ((not (plan['save_freq'] and step % plan['save_freq'] == 0)) if resident_actor
                                      else (final_model_only and step == steps))
                        if model_only and not resident_actor:
                            args.append('--save-model-only')
                        with phase(current,'actor_update'):
                            if resident_actor:
                                if actor_proc is None:
                                    args += ['--resident-rounds', str(steps)]
                                    actor_proc = workers.enter_context(resident_worker(current/'actor',args,(pgid,),runtime=runtime))
                                else:
                                    (current/'actor').mkdir()
                                    save(current/'actor-ready.json',dict(step=step))
                                wait_actor_round(actor_proc,current/'actor/accepted.json')
                                if step == steps and actor_proc.wait(timeout=120) != 0:
                                    raise RuntimeError(f"Resident Actor exited {actor_proc.returncode}")
                            else:
                                run_worker(current,'actor',args,runtime=runtime,gpu=True,allowed=(pgid,))
                        proof=json.loads((current/'actor/accepted.json').read_text())
                        assert proof['updates']==updates_per_round and proof['start_step']==actor_start and proof['end_step']==actor_end
                        assert proof['checkpoint_kind']==('model_only' if model_only else 'full')
                        if step > 1:
                            for rank in range(actor_world):
                                resumed=json.loads((current/f'actor/resume-rank{rank}.json').read_text())
                                assert resumed['restored_step']==actor_start and resumed['scheduler_epoch']==step-1
                                assert resumed['exact_parameter_and_moment_positions']>0
                                if resident_actor:
                                    previous_path = out/f'round-{step-1:02d}/actor/resident-rank{rank}.json'
                                    if not previous_path.exists() and resume and step == resume['completed_rounds']+1:
                                        assert resumed['restored_step'] == resume['optimizer_steps']
                                        continue
                                    previous=json.loads(previous_path.read_text())
                                    latest=json.loads((current/f'actor/resident-rank{rank}.json').read_text())
                                    assert resumed['resident'] and resumed['pid']==previous['pid']==latest['pid']
                                    assert previous['optimizer_id']==latest['optimizer_id']
                        previous_checkpoint = checkpoint
                        checkpoint=current/f'actor/trainer/checkpoint-{actor_end}'
                        with phase(current,'publication_manifest'):
                            manifest_sha=build_manifest(checkpoint)
                        try:
                            with phase(current,'resume_publish'):
                                save(current/'resume.json',client.call('/resume_memory_occupation',{},timeout=180))
                                drain_rollout(client,current,'before-publication')
                                request=dict(model_path=str(checkpoint),load_format='safetensors',flush_cache=True,recapture_cuda_graph=True)
                                save(current/'publication-request.json',dict(**request,expected_version_id=actor_end,expected_manifest_sha256=manifest_sha))
                                result=client.call('/update_weights_from_disk',request,timeout=1800)
                                save(current/'publication.json',result);assert result.get('success'),result
                                for rank in range(rollout_tp):
                                    published=json.loads((out/f'service/published-v{actor_end}-rank{rank}.json').read_text())
                                    assert published['version']==str(checkpoint) and published['actor_step']==actor_end
                                    assert published['manifest_sha256']==manifest_sha
                                layers = json.loads((checkpoint/'kt_full_experts.index.json').read_text())["layers"]
                                for layer in layers:
                                    verified=json.loads((out/f'service/cpu-verified-v{actor_end}-layer{layer}.json').read_text())
                                    assert verified['packed_native_forward_relative_l2']<=.06
                                control_probe(client,out/'service',probe_prompt,f'v{actor_end}',plan['inputs']['seed'])
                                import torch
                                previous=torch.load(out/f'service/v{actor_start}-control-logits.pt',weights_only=True)
                                latest=torch.load(out/f'service/v{actor_end}-control-logits.pt',weights_only=True)
                                difference=(latest-previous).abs()
                                assert torch.isfinite(latest).all() and difference.max().item()>0
                                save(current/'same-input-publication-logits.json',dict(before=f'v{actor_start}',after=f'v{actor_end}',
                                    max_absolute_change=difference.max().item(),mean_absolute_change=difference.mean().item(),
                                    changed_elements=int((difference!=0).sum())))
                        except BaseException as exc:
                            save(current/'publication-failed.json',dict(time_unix_ns=time.time_ns(),policy_step=step,
                                checkpoint=str(checkpoint),error=str(exc),action='stop owned rollout; forbid next generation',
                                rank_failures=[json.loads(p.read_text()) for p in (out/'service').glob('publication-failed-rank*.json')]))
                            raise
                        save(current/'accepted.json',dict(updates=updates_per_round,output_tokens=tokens,generated_policy=policy,published_policy=f'v{actor_end}'))
                        from .metrics import batch_metrics
                        with (out/"metrics.jsonl").open("a") as log:
                            log.write(json.dumps(batch_metrics(out, current), allow_nan=False)+"\n")
                        updates = step
                        if not model_only:
                            save(checkpoint/"ktrl-resume.json", dict(contract=contract(plan),
                                 completed_rounds=updates, optimizer_steps=actor_end,
                                 batches_consumed=batch_index+1, source_run=str(out)))
                            retire_old_full_checkpoints(out, plan["keep_full_checkpoints"])
                        save(out/'data-cursor.json', dict(batches_consumed=batch_index+1, updates=updates, optimizer_steps=updates*updates_per_round))
                        if resident_actor and previous_checkpoint is not None and previous_checkpoint.is_relative_to(out) and not (previous_checkpoint/"ktrl-resume.json").exists() and not (previous_checkpoint/"ktrl-resume-retired.json").exists():
                            retire_model_export(out, previous_checkpoint, step-1, actor_start)
                    if not updates:
                        raise RuntimeError("No batch had nonzero GRPO advantages")
                    if actor_proc is not None and updates < steps:
                        save(out/'actor-stop.json', dict(step=updates))
                        if actor_proc.wait(timeout=120) != 0:
                            raise RuntimeError(f"Resident Actor exited {actor_proc.returncode}")
                    probe_tokens = min(runtime["final_probe_tokens"], inputs["max_response_tokens"])
                    probe=dict(input_ids=probe_prompt,sampling_params=dict(
                        temperature=0,max_new_tokens=probe_tokens,sampling_seed=inputs['seed']))
                    save(out/f'final-v{updates*updates_per_round}-generation.json',client.generate(probe))
                    assert all(json.loads((out/f'service/replays-rank{r}.json').read_text()).get(f'v{updates*updates_per_round}',0)>0 for r in range(rollout_tp))
                guard_gpu(devices=devices)
                save(out/'accepted.json',dict(real_online_updates=updates, dataset_batches=steps, skipped_batches=steps-updates,
                    optimizer_steps=list(range(1,updates*updates_per_round+1)), optimizer_state_verified_before_updates=[i*updates_per_round+1 for i in range(1,updates)],
                    final_generation_policy=f'v{updates*updates_per_round}',
                    final_checkpoint_kind=proof['checkpoint_kind'],
                    resident_actor=resident_actor,
                    numerical_parity='deferred; no claim of actor-rollout numerical agreement'))
                status='complete'
        finally:
            save(out/'completion.json',dict(status=status,time_unix_ns=time.time_ns()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.config,a.out)

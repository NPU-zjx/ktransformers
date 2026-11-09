"""qjh005 synchronous, one-update RL driver. No veRL process or imports."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ktrl.configs.recipe import REPO, load_plan, load_rollout_config
from ktrl.workers.rollout.sglang import qjh005 as host
from ktrl.workers.rollout.sglang.smoke import SGLangClient, wait_ready, validate_response


MODEL = '/mnt/qjh007/models/Qwen3.6-35B-A3B'
ENV = Path('/raid/data/zjx/kt-baseline/checks/ktrl-m2-r01')
DATA = Path('/raid/data/zjx/gpu-baseline/data/full20k-20261025/train-13376.parquet')
REFERENCE = host.RESULTS/'ktrl-m2-r01/prepare-a01/fixed-rows.json'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    tmp.replace(path)


@contextmanager
def phase(out, name):
    start = time.time_ns()
    for directory in [out, out/'rollout', out/'actor']:
        if directory.exists():
            save(directory/'current-phase.json', dict(phase=name, start_unix_ns=start))
    print(f'{time.strftime("%F %T")} phase={name} started', flush=True)
    status = 'failed'
    try:
        yield
        status = 'complete'
    finally:
        end = time.time_ns()
        with (out/'stages.jsonl').open('a') as log:
            log.write(json.dumps(dict(phase=name,start_unix_ns=start,end_unix_ns=end,
                                     seconds=(end-start)/1e9,status=status))+'\n')
        print(f'{time.strftime("%F %T")} phase={name} {status} seconds={(end-start)/1e9:.3f}',flush=True)


def gpu_snapshot(out, label):
    result = dict(time_unix_ns=time.time_ns(),
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True),
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True))
    save(out/(label+'-gpu.json'),result)
    return result


def monitor(folder, cid):
    launch = json.loads((folder/'launch.json').read_text())
    launch['container_id'] = cid
    save(folder/'launch.json',launch)
    save(folder/'state.json', dict(container_id=cid))
    log = (folder/'monitor.log').open('w')
    process = subprocess.Popen([sys.executable,str(REPO/'profiling/resource_monitor.py'),
                                '--run-dir',str(folder),'--interval','5'], stdout=log,stderr=subprocess.STDOUT)
    log.close()
    return process


def stop_monitor(folder, process):
    save(folder/'completion.json',dict(time_unix_ns=time.time_ns()))
    process.wait(timeout=30)
    if process.returncode:
        raise RuntimeError(f'Memory monitor failed: {folder}')


def worker(out, stage, module_args, gpu=False, rollout_cid=None):
    folder = out/stage
    folder.mkdir()
    if gpu:
        # Only the exact sleeping rollout container is allowed to coexist.
        snap = gpu_snapshot(out,'before-actor')
        for line in snap['processes'].splitlines():
            pid = int(line.split(',')[0])
            if not rollout_cid or rollout_cid not in Path(f'/proc/{pid}/cgroup').read_text():
                raise RuntimeError(f'Unrelated GPU process {pid}; actor not started')
        assert min(float(line.split(',')[2]) for line in snap['devices'].splitlines()) >= 90000, 'Insufficient released GPU memory'
        assert not (out/'optimizer-started.json').exists(), 'One-update ledger already exists'
    name = 'zjx-'+out.name+'-'+stage
    python = str(host.MOUNT/'venvs/kt-py31115/bin/python')
    cmd = host.DOCKER+['create','--name',name,'--label','ktrl.run='+out.name,'--cidfile',str(folder/'container.id'),
        '--runtime=nvidia','--network','host','--read-only','--ipc','private','--shm-size','32g',
        '--memory','1550g','--memory-swap','1550g','--tmpfs','/tmp:rw,exec,size=24g','--user','0:0',
        '-v',f'{REPO}:/repo:ro','-v',f'{ENV}:/env:ro','-v',f'{out}:/out:rw',
        '-v',f'{host.RUNTIME}:{host.MOUNT}:ro','-v',f'{MODEL}:{MODEL}:ro',
        '-v',f'{DATA}:/data/train.parquet:ro','-v',f'{DATA}:{DATA}:ro',
        '-v',f'{REFERENCE}:/reference.json:ro','-w','/repo']
    environment = dict(NVIDIA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' if gpu else 'none',
        NVIDIA_DRIVER_CAPABILITIES='compute,utility',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HOME='/out/cache/home',XDG_CACHE_HOME='/out/cache/xdg',
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true',KT_SFT_PROFILE='1',
        PYTHONPATH='/env/site:/env/extras:/repo',LD_LIBRARY_PATH=str(host.CUDNN),
        TRITON_CACHE_DIR='/out/cache/triton',TORCHINDUCTOR_CACHE_DIR='/out/cache/inductor')
    for key,value in environment.items():
        cmd += ['-e',key+'='+value]
    cmd += ['--entrypoint',python,host.IMAGE,'-B']+module_args
    save(folder/'launch.json',dict(argv=cmd))
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
    cid = (folder/'container.id').read_text().strip()
    mon = monitor(folder,cid)
    try:
        with (folder/'train.log').open('w') as log:
            subprocess.run(host.DOCKER+['start','-a',cid],stdout=log,stderr=subprocess.STDOUT,check=True)
        state = host.state(cid)['State']
        save(folder/'exit.json',state)
        if state['Running'] or state['ExitCode'] != 0 or state['OOMKilled']:
            raise RuntimeError(f'{stage} failed: {state}')
    finally:
        if host.state(cid)['State']['Running']:
            subprocess.run(host.DOCKER+['stop','--time','90',cid],check=True)
        stop_monitor(folder,mon)


def rollout_wave(client, folder, questions, wave):
    target = folder/f'phase-wave{wave}'
    target.mkdir()
    client.call('/start_profile',dict(activities=['KT_PHASE'],output_dir=f'/output/{target.name}',record_shapes=False,with_stack=False))
    def generate(index):
        uid = f'm4-w{wave}-q{index:02d}'
        payload = dict(rid=uid,input_ids=questions[index]['prompt'],stream=False,
            sampling_params=dict(temperature=1.0,top_p=1.0,top_k=-1,max_new_tokens=20480,ignore_eos=False))
        with (folder/(uid+'-request.json')).open('x') as file:
            json.dump(payload,file)
        start = time.time_ns()
        result = client.call('/generate',payload,timeout=18000)
        validate_response(payload,result)
        save(folder/(uid+'.json'),result)
        save(folder/(uid+'-timing.json'),dict(start_unix_ns=start,end_unix_ns=time.time_ns()))
        print(f'Completed {uid}: {len(result["output_ids"])} tokens',flush=True)
        return len(result['output_ids'])
    try:
        with ThreadPoolExecutor(max_workers=32) as pool:
            tokens = sum(pool.map(generate,range(32)))
    finally:
        client.call('/stop_profile',{},timeout=120)
    traces = [json.loads(p.read_text()) for p in target.glob('phase-*.json')]
    assert {t['rank'] for t in traces} == set(range(8))
    for trace in traces:
        decode = [r for r in trace['rows'] if r['phase']=='DECODE']
        assert decode and all(r['graph_replayed'] for r in decode)
    return tokens


def run(config, out, *, preflight_only=False):
    out = Path(out).resolve()
    if out.parent != host.RESULTS or not out.name.startswith('ktrl-m4-'):
        raise ValueError('Choose a fresh results/ktrl-m4-* directory')
    out.mkdir()
    (out/'recipe.yaml').write_bytes(Path(config).read_bytes())
    plan = load_plan(config,steps=1,profiling='light')
    assert plan['model']['path'] == MODEL and plan['inputs']['responses_per_update'] == 128
    assert plan['actor']['world_size'] == 8 and not plan['ref']['enabled']
    plan['mode'] = 'execute-one-step'
    save(out/'resolved-plan.json',plan)
    save(out/'execution-recipe.json',dict(prompts=32,responses_per_prompt=4,max_response_tokens=20480,
        initial_policy='base (same start as M2), new one-update ledger',sampling_temperature=1.,top_p=1.,
        rollout_seed=20261024,concurrent_requests=32,optimizer_schedule_steps=418,
        CPU_experts_per_layer=256,GPU_experts_per_layer=0,numerical_parity='deferred',
        preflight_only=preflight_only))
    with phase(out,'prepare_prompts'):
        worker(out,'prepare',['-m','ktrl.trainer.online_batch','prepare','--out','/out','--model',MODEL,
                             '--parquet','/data/train.parquet','--reference','/reference.json'])
    questions = json.loads((out/'prompts.json').read_text())
    rollout = load_rollout_config(config)
    folder = out/'rollout'
    folder.mkdir()
    original = host.command
    checkpoint = out/'actor/trainer/checkpoint-1'
    def command(*args):
        cmd = original(*args)
        idx = cmd.index(host.IMAGE)
        cmd[idx:idx] = ['-v',f'{out}:{out}:ro','-e','KTRL_PUBLICATION_OUTPUT=/output',
            '-e','LD_LIBRARY_PATH='+str(host.CUDNN)+':'+str(host.MOUNT/'venvs/kt-py31115/lib/python3.11/site-packages/nvidia/cuda_runtime/lib'),
            '-e','KTRL_ONLINE_PHASES=1','-e','KTRL_SKIP_LOGITS_DUMP=1',
            '-e','PYTHONPATH=/repo/ktrl/workers/rollout/sglang/publication_worker:/repo:/repo/profiling/worker:/repo/third_party/sglang/python']
        cmd[cmd.index('--context-length')+1] = '21504'
        cmd[cmd.index('--random-seed')+1] = '20261024'
        return cmd+['--enable-memory-saver','--enable-weights-cpu-backup']
    host.command = command
    try:
        with host.service(rollout,folder) as (cid,name,url):
            mon = monitor(folder,cid)
            try:
                client = SGLangClient(url)
                with phase(out,'rollout_initialization'):
                    wait_ready(client,cid,dict(rollout['server_args'],served_model_name=name,
                               skip_server_warmup=True,enable_memory_saver=True,context_length=21504),folder)
                probe = dict(input_ids=questions[0]['prompt'],sampling_params=dict(temperature=0,max_new_tokens=2,ignore_eos=True))
                if preflight_only:
                    with phase(out,'sleep_preflight'):
                        before = client.generate(probe)
                        save(out/'before-sleep.json',before)
                        gpu_snapshot(out,'before-sleep')
                        save(out/'release.json',client.call('/release_memory_occupation',{},timeout=180))
                        snap = gpu_snapshot(out,'sleeping')
                        assert min(float(l.split(',')[2]) for l in snap['devices'].splitlines()) >= 90000
                        save(out/'resume.json',client.call('/resume_memory_occupation',{},timeout=180))
                        after = client.generate(probe)
                        save(out/'after-resume.json',after)
                        assert before['output_ids'] == after['output_ids']
                    save(out/'preflight-accepted.json',dict(sleep_resume=True,update_steps=0))
                    return
                with phase(out,'online_rollout_v0'):
                    tokens = sum(rollout_wave(client,folder,questions,wave) for wave in range(4))
                with phase(out,'reward_and_advantage'):
                    worker(out,'score',['-m','ktrl.trainer.online_batch','score','--out','/out','--model',MODEL])
                with phase(out,'rollout_sleep'):
                    save(out/'release.json',client.call('/release_memory_occupation',{},timeout=180))
                    gpu_snapshot(out,'sleeping')
                with phase(out,'actor_update_and_checkpoint'):
                    worker(out,'actor',['-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                        '--module','ktrl.workers.engine.fsdp.actor_update','--config','/out/recipe.yaml',
                        '--batch-dir','/out/batch','--out','/out/actor'],gpu=True,rollout_cid=cid)
                    assert json.loads((out/'actor/accepted.json').read_text())['updates'] == 1
                with phase(out,'rollout_resume'):
                    save(out/'resume.json',client.call('/resume_memory_occupation',{},timeout=180))
                with phase(out,'publish_v1'):
                    request = dict(model_path=str(checkpoint),load_format='safetensors',flush_cache=True,recapture_cuda_graph=True)
                    result = client.call('/update_weights_from_disk',request,timeout=1800)
                    save(out/'publication.json',result)
                    assert result.get('success'), result
                with phase(out,'next_generation_v1'):
                    probe['sampling_params']['max_new_tokens'] = 32
                    probe['sampling_params']['ignore_eos'] = False
                    save(out/'next-input.json',probe)
                    save(out/'next-response.json',client.generate(probe))
                    ranks = [json.loads((folder/f'replays-rank{r}.json').read_text()) for r in range(8)]
                    assert all(r.get('v1',0)>0 for r in ranks)
                    assert len(list(folder.glob('published-rank*.json'))) == 8
                save(out/'accepted.json',dict(updates=1,live_responses=128,output_tokens=tokens,
                    next_generation_policy='v1',all_ranks_v1_graph_replay=True,numerical_parity='deferred'))
            finally:
                stop_monitor(folder,mon)
    finally:
        host.command = original
        gpu_snapshot(out,'final')
    save(out/'completion.json',dict(status='complete',time_unix_ns=time.time_ns()))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    a = p.parse_args()
    run(a.config,a.out,preflight_only=a.preflight_only)

"""Run the native online trainer with an external observation overlay."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from observe import Sampler
from stream_metrics import arrival_metrics


def stream_request(client, payload, out):
    request = urllib.request.Request(client.url+'/generate',
        json.dumps(dict(payload, stream=True)).encode(),
        headers={'Content-Type':'application/json','Accept':'text/event-stream'})
    start = time.perf_counter_ns()
    events, ids, final, done = [], [], None, False
    with client.opener.open(request, timeout=18000) as response:
        if response.headers.get_content_type() != 'text/event-stream':
            raise ValueError('Expected SSE; TTFT cannot be inferred from non-streaming latency')
        for line in response:
            now = time.perf_counter_ns()
            if not line.startswith(b'data:'): continue
            raw = line[5:].strip()
            if raw == b'[DONE]': done = True; break
            row = json.loads(raw)
            if 'error' in row: raise RuntimeError(row)
            new_ids = row.get('output_ids', [])
            if new_ids[:len(ids)] != ids: raise ValueError('Non-cumulative SSE token IDs')
            if len(new_ids) > len(ids):
                events.append(dict(time_ns=now, total_tokens=len(new_ids), delta=len(new_ids)-len(ids)))
            ids, final = new_ids, row
    if not done or final is None or not ids: raise ValueError('Incomplete streaming output')
    if len(ids) != final['meta_info']['completion_tokens']: raise ValueError('Final token count mismatch')
    first, last = events[0]['time_ns'], events[-1]['time_ns']
    second = next((e for e in events if e['total_tokens'] >= 2), None)
    steady_tokens = len(ids) - second['total_tokens'] if second is not None else 0
    record = dict(rid=payload['rid'], start_ns=start, first_token_ns=first, last_token_ns=last,
        end_ns=time.perf_counter_ns(), output_tokens=len(ids), input_tokens=len(payload['input_ids']),
        ttft_ms=(first-start)/1e6, tpot_ms=(last-first)/1e6/(len(ids)-1) if len(ids)>1 else None,
        # Keep E2E TPOT intact: prefill blocking is a real user-visible delay.
        **arrival_metrics(events),
        cached_tokens=final['meta_info'].get('cached_tokens'),
        first_chunk_tokens=events[0]['delta'], events=events,
        coalesced_chunks=sum(e['delta']>1 for e in events), finish_reason=final['meta_info']['finish_reason'])
    out.mkdir(parents=True, exist_ok=True)
    (out/(payload['rid']+'.json')).write_text(json.dumps(record))
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--mode', choices=['light','diagnostic'], default='light')
    p.add_argument('--check', action='store_true')
    p.add_argument('--cpu-native-allocations', action='store_true')
    p.add_argument('--nsys-actor', action='store_true')
    args = p.parse_args()
    source, config, out = args.source.resolve(), args.config.resolve(), args.out.resolve()
    if out.exists(): raise FileExistsError(out)
    sys.path.insert(0, str(source))
    from ktrl.trainer import online_native as native
    native.guard_gpu()
    plan = native.load_plan(config, steps=args.steps, profiling='light')
    if plan['save_freq']:
        raise ValueError('This performance recipe must set save_freq=0; no new full optimizer checkpoint budgeted')
    import shutil
    if shutil.disk_usage(out.parent).free < 140*2**30:
        raise RuntimeError('Need at least 140 GiB for model publications and profiling data')
    if args.check:
        print(json.dumps(dict(ready=True, plan=plan))); return
    out.mkdir()
    profile = out/'profiles'; profile.mkdir()
    overlay = Path(__file__).resolve().parent
    if (args.cpu_native_allocations or args.nsys_actor) and args.mode != 'diagnostic':
        raise ValueError('Heavy native profilers are diagnostic-only')
    original_environment, original_phase, original_load = native.environment, native.phase, native.load_plan
    def load(*a, **k):
        result = original_load(*a, **k)
        result['rollout']['server_args']['stream_interval'] = 1
        return result
    native.load_plan = load
    def environment(folder, actor, runtime):
        env = original_environment(folder, actor, runtime)
        env['PYTHONPATH'] = str(overlay)+os.pathsep+env.get('PYTHONPATH','')
        env.update(KTRL_DETAIL_PROFILE=str(profile), KTRL_DETAIL_ROLE='actor' if actor else 'rollout',
                   KTRL_DETAIL_MODE=args.mode)
        if actor and args.cpu_native_allocations:
            env['KTRL_NATIVE_MEMORY']='1'
            env['PYTHONPATH'] += os.pathsep+'/workspace/tools'
        if actor and args.nsys_actor:
            env['KTRL_NSYS_CAPTURE']='1'
        # Existing HOME cache can be root-owned. TVM-FFI retries PermissionError
        # indefinitely as a lock conflict, so isolate JIT artifacts for this run.
        env['TVM_FFI_CACHE_DIR'] = str(out/'jit-cache'/'tvm-ffi')
        return env
    native.environment = environment
    original_resident = native.resident_worker
    @contextmanager
    def resident(folder, worker_args, allowed, **kw):
        worker_args=list(worker_args)
        if args.cpu_native_allocations or args.nsys_actor:
            index=worker_args.index('ktrl.workers.engine.fsdp.actor_update')
            worker_args[index]='profiled_actor'
        with original_resident(folder,worker_args,allowed,**kw) as process:
            yield process
    native.resident_worker=resident
    original_popen=subprocess.Popen
    def popen(argv,*a,**kw):
        if (args.nsys_actor and isinstance(argv,list) and 'torch.distributed.run' in argv
                and any(x in argv for x in ('profiled_actor','ktrl.workers.engine.fsdp.actor_update'))):
            argv=['/opt/nsys/bin/nsys','profile','--trace=cuda,nvtx,osrt',
                  '--sample=none','--cpuctxsw=none','--cuda-memory-usage=true',
                  '--capture-range=cudaProfilerApi','--capture-range-end=stop','--kill=none',
                  '--output='+str(profile/'actor-nsys')]+argv
            (out/'nsys-launch.json').write_text(json.dumps(argv))
        return original_popen(argv,*a,**kw)
    if args.nsys_actor:subprocess.Popen=popen
    @contextmanager
    def phase(folder, name):
        with (out/'external-stages.jsonl').open('a') as f:
            f.write(json.dumps(dict(event='begin', phase=name, folder=str(folder), time_ns=time.time_ns()))+'\n')
        try:
            with original_phase(folder, name): yield
        finally:
            with (out/'external-stages.jsonl').open('a') as f:
                f.write(json.dumps(dict(event='end', phase=name, folder=str(folder), time_ns=time.time_ns()))+'\n')
    native.phase = phase
    original_call = native.SGLangClient.call
    def call(client, path, data=None, *, timeout=120):
        if path == '/generate' and data and data.get('rid','').startswith('m4-w'):
            active = profile/'client'
            # Native round reuses request IDs, so use a per-rollout folder counter.
            return stream_request(client, data, active/f'wave{COUNTER[0]}')
        return original_call(client, path, data, timeout=timeout)
    native.SGLangClient.call = call
    COUNTER = [0]
    original_generate = native.generate
    def generate(*a, **k):
        COUNTER[0] += 1
        return original_generate(*a, **k)
    native.generate = generate
    manifest = dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        source=str(source), config=str(config), mode=args.mode, steps=args.steps,
        overlay_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in overlay.glob('*.py')},
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        timing_scope='native online RL; SSE transport observation; GPU stream durations include dependencies',
        status='running', started_ns=time.time_ns())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'source.diff').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=source))
    try:
        with Sampler(out/'resources'):
            native.run(config, out/'rl', steps=args.steps, resident_actor=True)
        manifest['status']='complete'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error)); raise
    finally:
        manifest['ended_ns']=time.time_ns()
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__': main()

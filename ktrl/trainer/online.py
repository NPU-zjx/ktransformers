"""Shared online stage evidence; execution delegates to the guarded native runner."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import time

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


def run(config,out,*,steps=2,final_model_only=False,resident_actor=False,resume_from=None):
    from .online_native import run as execute
    return execute(config,out,steps=steps,final_model_only=final_model_only,resident_actor=resident_actor,resume_from=resume_from)

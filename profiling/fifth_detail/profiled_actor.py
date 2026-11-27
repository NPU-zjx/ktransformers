"""Diagnostic entrypoint; delegate unchanged Actor implementation with native allocation tracking."""
import os,runpy
from pathlib import Path

root=Path(os.environ['KTRL_DETAIL_PROFILE'])
rank=int(os.environ.get('RANK','0'))
if os.environ.get('KTRL_NSYS_CAPTURE')=='1':
    import torch
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.cuda.cudart().cudaProfilerStart()
if os.environ.get('KTRL_NATIVE_MEMORY')=='1':
    import memray
    with memray.Tracker(str(root/f'cpu-native-rank{rank}-pid{os.getpid()}.bin'),native_traces=True):
        runpy.run_module('ktrl.workers.engine.fsdp.actor_update',run_name='__main__')
else:
    runpy.run_module('ktrl.workers.engine.fsdp.actor_update',run_name='__main__')

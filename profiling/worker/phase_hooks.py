import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

window=None

def begin(scheduler, mode):
    global window
    import torch
    assert window is None
    torch.cuda.synchronize()
    anchor=torch.cuda.Event(enable_timing=True)
    lower=time.perf_counter_ns();anchor.record();anchor.synchronize();upper=time.perf_counter_ns()
    torch.cuda.reset_peak_memory_stats()
    folder=Path(scheduler.torch_profiler_output_dir);folder.mkdir(parents=True,exist_ok=True)
    window=dict(rank=scheduler.tp_rank,pid=os.getpid(),window_id=upper,mode=mode,
                output=folder,anchor=anchor,lower=lower,upper=upper,pending=[],rows=[],replays=0,
                epoch_ns=time.time_ns(),clock_ns=time.perf_counter_ns(),
                device=torch.cuda.current_device(),uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid))

def drain(wait=False):
    w=window;pending=[]
    for row,start,end in w['pending']:
        if wait:end.synchronize()
        if not end.query():pending.append((row,start,end));continue
        row['stream_start_ms']=w['anchor'].elapsed_time(start)
        row['stream_end_ms']=w['anchor'].elapsed_time(end)
        row['stream_ms']=start.elapsed_time(end)
        row['completion_lower_ns']=w['lower']+int(row['stream_end_ms']*1e6)
        row['completion_upper_ns']=w['upper']+int(row['stream_end_ms']*1e6)
        w['rows'].append(row)
    w['pending']=pending

def end():
    global window
    if window is None:return
    import torch
    drain(True);w=window
    out={k:v for k,v in w.items() if k not in ('output','anchor','pending')}
    out.update(window_max_allocated_bytes=torch.cuda.max_memory_allocated(),
               window_max_reserved_bytes=torch.cuda.max_memory_reserved())
    path=w['output']/f"phase-TP{w['rank']}-PID{w['pid']}-W{w['window_id']}.json"
    path.write_text(json.dumps(out))
    window=None

def decorate_batch(fn):
    @functools.wraps(fn)
    def measured(self,model_worker_batch,*args,**kwargs):
        if window is None:return fn(self,model_worker_batch,*args,**kwargs)
        import torch
        w=window;drain()
        assert len(w['pending'])<4096 and len(w['rows'])<65536, 'Timing capacity exceeded'
        batch=model_worker_batch or kwargs.get('forward_batch') or (args[0] if args else None)
        assert batch is not None
        name=batch.forward_mode.name
        row=dict(index=len(w['rows'])+len(w['pending']),phase=name,
                 tokens=int(batch.input_ids.numel()),cpu_start_ns=time.perf_counter_ns(),
                 allocated_before_bytes=torch.cuda.memory_allocated(),reserved_before_bytes=torch.cuda.memory_reserved())
        start,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();replays=w['replays']
        result=fn(self,model_worker_batch,*args,**kwargs)
        end_event.record()
        row.update(cpu_return_ns=time.perf_counter_ns(),graph_replayed=w['replays']>replays,
                   graph_eligible=bool(getattr(result,'can_run_cuda_graph',False)),
                   delayed_sampling=bool(getattr(result,'delay_sample_func',None)),
                   allocated_after_bytes=torch.cuda.memory_allocated(),reserved_after_bytes=torch.cuda.memory_reserved())
        w['pending'].append((row,start,end_event))
        return result
    return measured

def patch_module(module):
    name=module.__name__
    if name.endswith('tp_worker'):
        cls=module.TpModelWorker
        cls.forward_batch_generation=decorate_batch(cls.forward_batch_generation)
    elif name.endswith('cuda_graph_runner'):
        cls=module.CudaGraphRunner;original=cls.replay
        @functools.wraps(original)
        def replay(self,*a,**kw):
            result=original(self,*a,**kw)
            if window is not None:window['replays']+=1
            return result
        cls.replay=replay
    else:
        cls=module.SchedulerProfilerMixin;start=cls.start_profile;stop=cls.stop_profile
        @functools.wraps(start)
        def start_profile(self,stage=None):
            if self.profiler_activities==['KT_PHASE']:
                begin(self,'KT_PHASE');self.profile_in_progress=True
                return module.ProfileReqOutput(success=True,message='KT phase window started')
            return start(self,stage)
        @functools.wraps(stop)
        def stop_profile(self,stage=None):
            if self.profiler_activities==['KT_PHASE']:
                end();self.profile_in_progress=False
                return module.ProfileReqOutput(success=True,message='KT phase window stopped')
            return stop(self,stage)
        cls.start_profile=start_profile;cls.stop_profile=stop_profile

class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
    def exec_module(self,module):
        self.original.exec_module(module);patch_module(module)

class Finder(importlib.abc.MetaPathFinder):
    names={'sglang.srt.managers.tp_worker','sglang.srt.model_executor.cuda_graph_runner',
           'sglang.srt.managers.scheduler_profiler_mixin'}
    def find_spec(self,fullname,path,target=None):
        if fullname not in self.names:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is not None:spec.loader=Loader(spec.loader)
        return spec

def install():
    assert not Finder.names.intersection(sys.modules), 'Hooks installed after worker modules loaded'
    sys.meta_path.insert(0,Finder())

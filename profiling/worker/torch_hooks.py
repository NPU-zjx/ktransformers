"""Worker-side Torch profiler exports and module markers for the base diagnostic."""
import csv
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
        diagnostic=w['mode']!='KT_PHASE'
        if diagnostic:torch.cuda.nvtx.range_push('kt_'+name.lower())
        try:
            if diagnostic:
                with torch.profiler.record_function('KT/'+name):result=fn(self,model_worker_batch,*args,**kwargs)
            else:result=fn(self,model_worker_batch,*args,**kwargs)
        finally:
            if diagnostic:torch.cuda.nvtx.range_pop()
        end_event.record()
        row.update(cpu_return_ns=time.perf_counter_ns(),graph_replayed=w['replays']>replays,
                   graph_eligible=bool(getattr(result,'can_run_cuda_graph',False)),
                   delayed_sampling=bool(getattr(result,'delay_sample_func',None)),
                   allocated_after_bytes=torch.cuda.memory_allocated(),reserved_after_bytes=torch.cuda.memory_reserved())
        w['pending'].append((row,start,end_event))
        return result
    return measured

def export_ops(prof,folder,rank,stage):
    rows=[]
    for e in prof.key_averages(group_by_input_shape=True):
        rows.append(dict(operator=e.key,calls=e.count,cpu_total_us=e.cpu_time_total,cpu_self_us=e.self_cpu_time_total,
                         cpu_avg_us=e.cpu_time_total/max(1,e.count),device_total_us=e.device_time_total,
                         device_self_us=e.self_device_time_total,device_avg_us=e.device_time_total/max(1,e.count),
                         cpu_net_memory_bytes=e.cpu_memory_usage,device_net_memory_bytes=e.device_memory_usage,
                         device_self_net_memory_bytes=e.self_device_memory_usage,shapes=json.dumps(e.input_shapes)))
    if rows:
        with (Path(folder)/f'operators-TP{rank}-{stage}-{time.time_ns()}.csv').open('w',newline='') as output:
            writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def mark_method(cls,name):
    original=getattr(cls,name,None)
    if original is None:return
    @functools.wraps(original)
    def marked(self,*args,**kwargs):
        if window is None or window['mode']=='KT_PHASE':return original(self,*args,**kwargs)
        import torch
        label='KT_FUNC/'+cls.__name__+'.'+name
        torch.cuda.nvtx.range_push(label)
        try:
            with torch.profiler.record_function(label):return original(self,*args,**kwargs)
        finally:torch.cuda.nvtx.range_pop()
    setattr(cls,name,marked)

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
    elif name.endswith('model_runner'):
        for method in ('forward','sample'):mark_method(module.ModelRunner,method)
    elif name.endswith('kt_ep_wrapper'):
        for method in ('apply','_submit_cpu_forward','_sync_cpu_forward'):mark_method(module.KTEPWrapperMethod,method)
    else:
        cls=module.SchedulerProfilerMixin;start=cls.start_profile;stop=cls.stop_profile
        @functools.wraps(start)
        def start_profile(self,stage=None):
            if self.profiler_activities==['KT_PHASE']:
                begin(self,'KT_PHASE');self.profile_in_progress=True
                return module.ProfileReqOutput(success=True,message='KT phase window started')
            # Enable allocator events only during the original Torch diagnostic profile.
            import torch
            factory=torch.profiler.profile
            def configured(*a,**kw):
                kw['profile_memory']=True
                return factory(*a,**kw)
            torch.profiler.profile=configured
            try:result=start(self,stage)
            finally:torch.profiler.profile=factory
            begin(self,'diagnostic')
            return result
        @functools.wraps(stop)
        def stop_profile(self,stage=None):
            if self.profiler_activities==['KT_PHASE']:
                end();self.profile_in_progress=False
                return module.ProfileReqOutput(success=True,message='KT phase window stopped')
            prof=self.torch_profiler
            end();result=stop(self,stage)
            if prof is not None:export_ops(prof,self.torch_profiler_output_dir,self.tp_rank,getattr(stage,'name','all'))
            return result
        cls.start_profile=start_profile;cls.stop_profile=stop_profile

class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
    def exec_module(self,module):
        self.original.exec_module(module);patch_module(module)

class Finder(importlib.abc.MetaPathFinder):
    names={'sglang.srt.managers.tp_worker','sglang.srt.model_executor.cuda_graph_runner',
           'sglang.srt.managers.scheduler_profiler_mixin','sglang.srt.model_executor.model_runner',
           'sglang.srt.layers.moe.kt_ep_wrapper'}
    def find_spec(self,fullname,path,target=None):
        if fullname not in self.names:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is not None:spec.loader=Loader(spec.loader)
        return spec

def install():
    assert not Finder.names.intersection(sys.modules), 'Hooks installed after worker modules loaded'
    sys.meta_path.insert(0,Finder())

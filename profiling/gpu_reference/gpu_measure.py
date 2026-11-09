"""vLLM worker instrumentation installed through veRL's collective RPC."""
import functools
import json
import os
from pathlib import Path
import time

window = None
replays = 0


def install(worker):
    import torch
    from vllm.v1.utils import compute_iteration_details
    global replays
    if not getattr(worker,"gpu_reference_installed",False):
        replay=torch.cuda.CUDAGraph.replay
        @functools.wraps(replay)
        def counted(graph,*a,**kw):
            global replays
            result=replay(graph,*a,**kw)
            if window is not None: replays+=1
            return result
        torch.cuda.CUDAGraph.replay=counted
        for method in ("execute_model","sample_tokens"):
            original=getattr(worker,method)
            def wrap(fn,name):
                @functools.wraps(fn)
                def measured(*a,**kw):
                    if window is None: return fn(*a,**kw)
                    w=window
                    if name=="execute_model":
                        schedule=a[0] if a else kw["scheduler_output"]
                        details=compute_iteration_details(schedule)
                        pre,dec=details.num_ctx_tokens,details.num_generation_tokens
                        phase="MIXED" if pre and dec else "EXTEND" if pre else "DECODE" if dec else "IDLE"
                        row=dict(index=len(w["rows"]),phase=phase,context_tokens=pre,generation_tokens=dec,
                                 tokens=pre+dec,cpu_start_ns=time.perf_counter_ns(),allocated_before_bytes=torch.cuda.memory_allocated(),
                                 reserved_before_bytes=torch.cuda.memory_reserved())
                        w["rows"].append(row)
                        advance_torch(w,phase)
                    else:
                        if not w["rows"]:return fn(*a,**kw)
                        row=w["rows"][-1];phase=row["phase"]
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    diag=w["mode"]!="formal"
                    if diag:
                        torch.cuda.nvtx.range_push("kt_"+phase.lower())
                        torch.cuda.nvtx.range_push("vLLM_FUNC/"+name)
                    before=replays
                    try:
                        start.record()
                        if w.get("prof") is not None:
                            with torch.profiler.record_function("vLLM/"+name): result=fn(*a,**kw)
                        else: result=fn(*a,**kw)
                        end.record()
                    finally:
                        if diag:
                            torch.cuda.nvtx.range_pop()
                            torch.cuda.nvtx.range_pop()
                    row.setdefault("_start",start)
                    row["_end"]=end
                    row.update(cpu_return_ns=time.perf_counter_ns(),allocated_after_bytes=torch.cuda.memory_allocated(),
                        reserved_after_bytes=torch.cuda.memory_reserved(),graph_replayed=row.get("graph_replayed",False) or replays>before)
                    return result
                return measured
            setattr(worker,method,wrap(original,method))
        worker.gpu_reference_installed=True
    config=worker.vllm_config
    return dict(rank=worker.rank,pid=os.getpid(),compilation=str(config.compilation_config),
                scheduler=str(config.scheduler_config),cache=str(config.cache_config),parallel=str(config.parallel_config))


def stop_torch(w):
    if w.get("prof") is None:return
    import csv
    prof=w["prof"];phase=w["active_phase"];w["prof"]=None
    prof.stop()
    tag=f"TP-{w['rank']}-{phase}-{w['segment']}"
    prof.export_chrome_trace(str(w["folder"]/(tag+".trace.json.gz")))
    with (w["folder"]/(tag+".csv")).open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["operator","shapes","profiler_event_count","cpu_total_us","cpu_self_us","device_total_us","device_self_us","cpu_net_memory_bytes","device_net_memory_bytes","device_self_net_memory_bytes","supported_flops"])
        for e in prof.key_averages(group_by_input_shape=True):
            writer.writerow([e.key,json.dumps(e.input_shapes),e.count,e.cpu_time_total,e.self_cpu_time_total,e.device_time_total,e.self_device_time_total,e.cpu_memory_usage,e.device_memory_usage,e.self_device_memory_usage,e.flops or None])


def advance_torch(w,phase):
    if w["mode"]!="torch":return
    import torch
    count=w["counts"].get(phase,0);w["counts"][phase]=count+1
    active=phase if phase in ("EXTEND","MIXED","DECODE") and count<8 else None
    if active!=w.get("active_phase"):
        stop_torch(w);w["active_phase"]=active
        if active:
            w["segment"]+=1
            w["prof"]=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,profile_memory=True,with_flops=True,with_stack=False)
            w["prof"].start()


def begin(worker,folder,mode):
    import torch
    global window,replays
    if window is not None:raise RuntimeError("Capture already active")
    torch.cuda.synchronize()
    anchor=torch.cuda.Event(enable_timing=True)
    lower=time.perf_counter_ns();anchor.record();anchor.synchronize();upper=time.perf_counter_ns()
    torch.cuda.reset_peak_memory_stats()
    dest=Path(folder);dest.mkdir(parents=True,exist_ok=True)
    replays=0
    window=dict(rank=worker.rank,pid=os.getpid(),device=torch.cuda.current_device(),
        uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
        folder=dest,mode=mode,anchor=anchor,lower=lower,upper=upper,epoch_ns=time.time_ns(),
        clock_ns=time.perf_counter_ns(),rows=[],counts={},segment=0)
    if mode in ("nsys","ncu") and worker.rank==0:torch.cuda.cudart().cudaProfilerStart()
    return worker.rank


def finish(worker):
    import torch
    global window
    w=window
    if w is None:raise RuntimeError("No capture")
    torch.cuda.synchronize()
    stop_torch(w)
    if w["mode"] in ("nsys","ncu") and worker.rank==0:torch.cuda.cudart().cudaProfilerStop()
    for r in w["rows"]:
        start=r.pop("_start");end=r.pop("_end")
        r["stream_start_ms"]=w["anchor"].elapsed_time(start);r["stream_end_ms"]=w["anchor"].elapsed_time(end)
        r["stream_ms"]=start.elapsed_time(end)
        r["completion_lower_ns"]=w["lower"]+int(r["stream_end_ms"]*1e6)
        r["completion_upper_ns"]=w["upper"]+int(r["stream_end_ms"]*1e6)
    dest=w["folder"];data={k:v for k,v in w.items() if k not in ("folder","anchor","prof")}
    data.update(replays=replays,window_max_allocated_bytes=torch.cuda.max_memory_allocated(),
        window_max_reserved_bytes=torch.cuda.max_memory_reserved())
    (dest/f"phase-TP{worker.rank}.json").write_text(json.dumps(data))
    window=None
    return {"rank":worker.rank,"batches":len(w["rows"]),"replays":replays}

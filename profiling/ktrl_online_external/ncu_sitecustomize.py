"""External observation of native Actor and rollout subprocesses."""
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys
import time


def install(profile_dir, role):
    original = next((candidate for entry in sys.path
                     if (candidate := Path(entry) / "sitecustomize.py").is_file()
                     and candidate.resolve() != Path(__file__).resolve()), None)
    if original:
        runpy.run_path(str(original))
    completed = set()
    requests = {}
    mode = os.environ.get("KTRL_EXTERNAL_PROFILE_MODE", "torch")
    rank = int(os.environ.get("RANK", "0"))
    function_window = None
    function_done = set()

    def start_nsys(label):
        import torch
        # One owner controls the process-tree capture; all ranks annotate work.
        if rank == 0:
            torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(label)
        return time.time_ns()

    def stop_nsys(label, started):
        import torch
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        if rank == 0:
            torch.cuda.cudart().cudaProfilerStop()
        emit("windows", dict(stage=label, start_ns=started, end_ns=time.time_ns(),
                             profiler=mode, rank=rank))

    def emit(name, row):
        with (profile_dir / f"{name}-{os.getpid()}.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")

    def capture(label, function, *args, **kwargs):
        if label in completed:
            return function(*args, **kwargs)
        completed.add(label)
        import torch

        if mode in {"nsys", "ncu"}:
            from cpu_observer import CPUWindow
            cpu = CPUWindow(profile_dir, label, rank)
            # Decode is required. A failed replay must fail acceptance, not silently
            # convert the requested NCU window into CPU-only observation.
            hardware_window = True
            if hardware_window:
                started = start_nsys(label)
            else:
                started = time.time_ns()
                torch.cuda.nvtx.range_push(label)
            try:
                return function(*args, **kwargs)
            finally:
                if hardware_window:
                    stop_nsys(label, started)
                else:
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                    emit('windows', dict(stage=label, rank=rank, start_ns=started,
                        end_ns=time.time_ns(), profiler='torch_cpu',
                        ncu_status='unavailable_full_graph_node_replay_failed_in_a08'))
                try:
                    cpu.finish()
                except Exception as error:
                    emit('profiler-errors', dict(stage=label, profiler='torch_cpu', error=repr(error)))

        try:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False)
            profiler.start()
        except Exception as error:
            emit("profiler-errors", {"stage": label, "error": repr(error)})
            return function(*args, **kwargs)
        start_ns = time.time_ns()
        try:
            return function(*args, **kwargs)
        finally:
            end_ns = time.time_ns()
            try:
                profiler.stop()
                path = profile_dir / f"{label}-{os.getpid()}.json"
                profiler.export_chrome_trace(str(path))
                emit("windows", {"stage": label, "start_ns": start_ns, "end_ns": end_ns,
                                 "trace": path.name})
            except Exception as error:
                emit("profiler-errors", {"stage": label, "error": repr(error)})

    def patch(module):
        if role == 'rollout' and module.__name__ in extra_targets:
            cls, methods = ((module.ModelRunner, ('forward', 'sample'))
                if module.__name__.endswith('model_runner') else
                (module.KTEPWrapperMethod, ('apply', '_submit_cpu_forward', '_sync_cpu_forward')))
            for name in methods:
                original_method = getattr(cls, name, None)
                if original_method is None:
                    continue

                def wrap(original, label):
                    @functools.wraps(original)
                    def timed(self, *args, **kwargs):
                        if function_window is None:
                            return original(self, *args, **kwargs)
                        started = time.perf_counter_ns()
                        try:
                            return original(self, *args, **kwargs)
                        finally:
                            row = function_window['functions'].setdefault(label, {'calls': 0, 'total_ns': 0})
                            row['calls'] += 1
                            row['total_ns'] += time.perf_counter_ns() - started
                    return timed
                setattr(cls, name, wrap(original_method, cls.__name__ + '.' + name))
            return
        if role == "actor":
            trainer = module.Trainer
            step = trainer.training_step

            @functools.wraps(step)
            def training_step(self, *args, **kwargs):
                if type(self).__name__ == "GRPOTrainer":
                    label = f"actor-microbatch-rank{rank}"
                    if mode not in {"nsys", "ncu"} or label in completed:
                        return capture(label, step, self, *args, **kwargs)
                    import torch
                    original_event = self.evidence.event
                    active_phase = []

                    def observe_event(phase, **fields):
                        if phase in {"old_logp_start", "actor_forward_start", "backward_start"}:
                            if active_phase:
                                torch.cuda.nvtx.range_pop()
                            active_phase[:] = [phase]
                            torch.cuda.nvtx.range_push(f"actor_{phase}_rank{rank}")
                        return original_event(phase, **fields)

                    self.evidence.event = observe_event

                    def measured_step():
                        try:
                            return step(self, *args, **kwargs)
                        finally:
                            if active_phase:
                                torch.cuda.nvtx.range_pop()
                                active_phase.clear()

                    try:
                        return capture(label, measured_step)
                    finally:
                        self.evidence.event = original_event
                return step(self, *args, **kwargs)

            trainer.training_step = training_step
            if mode == "ncu":
                import torch.optim.optimizer as optimizer_module
                from cpu_observer import CPUWindow
                cpu_optimizers = {}

                def before_cpu_optimizer(optimizer, args, kwargs):
                    if 'cpu-optimizer' not in completed:
                        completed.add('cpu-optimizer')
                        cpu_optimizers[id(optimizer)] = CPUWindow(profile_dir, 'actor-optimizer', rank)

                def after_cpu_optimizer(optimizer, args, kwargs):
                    cpu = cpu_optimizers.pop(id(optimizer), None)
                    if cpu is not None:
                        try:
                            cpu.finish()
                        except Exception as error:
                            emit('profiler-errors', dict(stage='actor-optimizer', profiler='torch_cpu', error=repr(error)))

                optimizer_module.register_optimizer_step_pre_hook(before_cpu_optimizer)
                optimizer_module.register_optimizer_step_post_hook(after_cpu_optimizer)
            if mode == "nsys":
                import torch.optim.optimizer as optimizer_module
                active = {}

                def before_optimizer(optimizer, args, kwargs):
                    if "actor-optimizer" not in completed:
                        completed.add("actor-optimizer")
                        active[id(optimizer)] = start_nsys(f"actor-optimizer-rank{rank}")

                def after_optimizer(optimizer, args, kwargs):
                    started = active.pop(id(optimizer), None)
                    if started is not None:
                        stop_nsys(f"actor-optimizer-rank{rank}", started)

                optimizer_module.register_optimizer_step_pre_hook(before_optimizer)
                optimizer_module.register_optimizer_step_post_hook(after_optimizer)
            return

        scheduler = module.Scheduler
        run_batch = scheduler.run_batch
        process_batch = scheduler.process_batch_result

        @functools.wraps(run_batch)
        def rollout_batch(self, batch, *args, **kwargs):
            nonlocal rank, function_window
            rank = self.tp_rank
            window = getattr(sys.modules.get("phase_hooks"), "window", None)
            stage = batch.forward_mode.name
            if window is not None and stage in ("EXTEND", "DECODE"):
                label = f"rollout-{stage.lower()}-tp{self.tp_rank}"
                if label in completed and stage not in function_done:
                    function_done.add(stage)
                    function_window = dict(stage=stage, rank=rank, functions={},
                        requests=len(batch.reqs), start_ns=time.time_ns(),
                        scope='inclusive_cpu_functions_next_batch_after_ncu_window')
                    try:
                        return run_batch(self, batch, *args, **kwargs)
                    finally:
                        function_window['end_ns'] = time.time_ns()
                        emit('function-timing', function_window)
                        function_window = None
                return capture(label, run_batch, self, batch, *args, **kwargs)
            return run_batch(self, batch, *args, **kwargs)

        @functools.wraps(process_batch)
        def observe(self, batch, *args, **kwargs):
            reqs = tuple(batch.reqs) if self.tp_rank == 0 else ()
            result = process_batch(self, batch, *args, **kwargs)
            for req in reqs:
                if not req.rid.startswith("m4-w") or not req.output_ids:
                    continue
                stats = req.time_stats
                first = requests.setdefault(req.rid, stats.prefill_finished_time)
                if req.finished():
                    emit("scheduler-latency", {"rid": req.rid, "first": first,
                         "queued": stats.wait_queue_entry_time, "finished": stats.completion_time,
                         "observed_finished": time.perf_counter(), "tokens": len(req.output_ids)})
                    requests.pop(req.rid, None)
            return result

        scheduler.run_batch = rollout_batch
        scheduler.process_batch_result = observe

    class Loader(importlib.abc.Loader):
        def __init__(self, loader):
            self.loader = loader

        def create_module(self, spec):
            return self.loader.create_module(spec)

        def exec_module(self, module):
            self.loader.exec_module(module)
            patch(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != target_name and not (role == 'rollout' and fullname in extra_targets):
                return None
            # Preserve publication/compatibility loaders already installed by
            # native sitecustomize. Observers must not bypass their patches.
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    spec.loader = Loader(spec.loader)
                    return spec
            return None

    if role == "rollout" or os.environ.get("RANK") == "0" or mode in {"nsys", "ncu"}:
        target_name = ("sglang.srt.managers.scheduler" if role == "rollout" else "transformers.trainer")
        extra_targets = {'sglang.srt.model_executor.model_runner', 'sglang.srt.layers.moe.kt_ep_wrapper'}
        sys.meta_path.insert(0, Finder())
        # Native publication sitecustomize can import ModelRunner before this
        # observer is installed. Preserve that module and wrap its existing class.
        for name in ({target_name} | extra_targets if role == 'rollout' else {target_name}):
            loaded = sys.modules.get(name)
            if loaded is not None:
                patch(loaded)


if "KTRL_EXTERNAL_PROFILE_DIR" in os.environ:
    install(Path(os.environ["KTRL_EXTERNAL_PROFILE_DIR"]), os.environ["KTRL_EXTERNAL_PROFILE_ROLE"])

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
            started = start_nsys(label)
            try:
                return function(*args, **kwargs)
            finally:
                stop_nsys(label, started)

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
            window = getattr(sys.modules.get("phase_hooks"), "window", None)
            stage = batch.forward_mode.name
            if self.tp_rank == 0 and window is not None and stage in ("EXTEND", "DECODE"):
                return capture(f"rollout-{stage.lower()}-tp0", run_batch, self, batch, *args, **kwargs)
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
            if fullname != target_name:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                spec.loader = Loader(spec.loader)
            return spec

    if role == "rollout" or os.environ.get("RANK") == "0" or mode == "nsys":
        target_name = ("sglang.srt.managers.scheduler" if role == "rollout" else "transformers.trainer")
        sys.meta_path.insert(0, Finder())


if "KTRL_EXTERNAL_PROFILE_DIR" in os.environ:
    install(Path(os.environ["KTRL_EXTERNAL_PROFILE_DIR"]), os.environ["KTRL_EXTERNAL_PROFILE_ROLE"])

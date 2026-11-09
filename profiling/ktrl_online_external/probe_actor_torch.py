"""Collect one real Actor microbatch per rank, then exit before any optimizer step."""
import json
import os
from pathlib import Path
import sys


class MicrobatchComplete(Exception):
    pass


def main():
    config, batch, folder = map(Path, sys.argv[1:])
    import torch
    import torch.distributed as dist
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    from ktrl.workers.engine.fsdp import actor_update

    rank = int(os.environ["LOCAL_RANK"])
    native_event = actor_update.Evidence.event
    native_step = actor_update.GRPOTrainer.training_step
    optimizer_path = folder / f"optimizer-hook-rank{rank}.json"

    def optimizer_guard(*_):
        raise RuntimeError("Torch microbatch probe must never execute optimizer.step")

    def event(evidence, phase, **fields):
        if phase == "optimizer_start":
            optimizer_guard()
        return native_event(evidence, phase, **fields)

    def training_step(self, *args, **kwargs):
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            native_step(self, *args, **kwargs)
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(folder / f"actor-microbatch-rank{rank}.json"))
        optimizer_path.write_text(json.dumps(dict(rank=rank, guard_registered=True, optimizer_updates=0)))
        dist.barrier()
        raise MicrobatchComplete

    actor_update.Evidence.event = event
    actor_update.GRPOTrainer.training_step = training_step
    guard = register_optimizer_step_pre_hook(optimizer_guard)
    try:
        actor_update.main(config, batch, folder, resident_rounds=1)
        raise RuntimeError("The Torch probe never reached the stop boundary")
    except MicrobatchComplete:
        pass
    finally:
        guard.remove()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

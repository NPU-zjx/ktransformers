"""Short NSys acceptance probe using the accepted Actor input. No backward or update."""
import json
import os
from pathlib import Path
import sys


class ForwardProbeComplete(Exception):
    pass


def main():
    config, batch, folder = map(Path, sys.argv[1:])
    import torch
    import torch.distributed as dist
    from ktrl.workers.engine.fsdp import actor_update

    rank = int(os.environ["LOCAL_RANK"])
    original = actor_update.Evidence.event

    def event(evidence, phase, **fields):
        if phase == "optimizer_start":
            raise RuntimeError("Forward-only probe reached the optimizer guard")
        if phase == "actor_forward_start":
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                torch.cuda.cudart().cudaProfilerStart()
            dist.barrier()
            torch.cuda.nvtx.range_push(f"actor-forward-probe-rank{rank}")
        elif phase == "backward_start":
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            dist.barrier()
            if rank == 0:
                torch.cuda.cudart().cudaProfilerStop()
            dist.barrier()
            raise ForwardProbeComplete
        return original(evidence, phase, **fields)

    actor_update.Evidence.event = event
    try:
        actor_update.main(config, batch, folder, resident_rounds=1)
        raise RuntimeError("The forward probe never reached the stop boundary")
    except ForwardProbeComplete:
        (folder / f"forward-probe-rank{rank}.json").write_text(
            json.dumps(dict(rank=rank, forward_complete=True, optimizer_updates=0)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

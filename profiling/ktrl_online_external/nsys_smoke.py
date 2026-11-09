"""Two sequential GPU subprocesses; test profiling injection, no model/update."""
import subprocess
import sys

if len(sys.argv) == 1:
    for label in ('rollout-probe', 'actor-probe'):
        subprocess.run([sys.executable, '-B', __file__, label], check=True)
else:
    import torch
    torch.cuda.set_device(0)
    x = torch.ones(32, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(sys.argv[1])
    y = x + 1
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    assert y.sum().item() == 64
    print(sys.argv[1], 'completed; zero model updates', flush=True)

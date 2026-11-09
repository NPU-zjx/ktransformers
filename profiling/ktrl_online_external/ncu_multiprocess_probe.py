"""Eight native SGLang imports/contexts under NCU, no model or optimizer."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('root', type=Path)
p.add_argument('--rank', type=int)
a = p.parse_args()
if a.rank is None:
    sys.path.insert(0, str(a.source))
    from ktrl.trainer.online_native import environment
    children = []
    for rank in range(8):
        folder = a.root / f'rank{rank}'
        env = environment(folder, False)
        log = (folder/'import.log').open('x')
        proc = subprocess.Popen([sys.executable, '-B', __file__, str(a.source),
            str(a.root), '--rank', str(rank)], env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((rank, proc, log))
    results = []
    for rank, proc, log in children:
        results.append(dict(rank=rank, exit_code=proc.wait(timeout=300)))
        log.close()
    (a.root/'validated.json').write_text(json.dumps(results, indent=2))
    assert all(r['exit_code'] == 0 for r in results), results
else:
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.managers.scheduler import Scheduler
    torch.cuda.set_device(a.rank)
    x = torch.ones(32, 128, device='cuda', dtype=torch.bfloat16)
    if a.rank == 0:
        from flashinfer.norm import rmsnorm
        w = torch.ones(128, device='cuda', dtype=torch.bfloat16)
        rmsnorm(x, w)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push('rollout-extend-tp0')
        y = rmsnorm(x, w)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        assert torch.isfinite(y).all()
    print(json.dumps(dict(rank=a.rank, import_passed=True, model_loaded=False,
                         optimizer_updates=0)), flush=True)

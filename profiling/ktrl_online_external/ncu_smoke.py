"""Real selected kernels and graph replay; no model and zero optimizer updates."""
import json
from pathlib import Path
import subprocess
import sys

source, out = Path(sys.argv[1]), Path(sys.argv[2])
if len(sys.argv) == 3:
    sys.path.insert(0, str(source))
    from ktrl.trainer.online_native import environment
    for role in ('rollout', 'actor'):
        env = environment(out / role, role == 'actor')
        subprocess.run([sys.executable, '-B', __file__, str(source), str(out), role],
                       env=env, check=True)
else:
    import torch
    role = sys.argv[3]
    torch.cuda.set_device(0)
    stages = []

    def capture(label, function):
        from cpu_observer import CPUWindow
        torch.cuda.synchronize()
        cpu = CPUWindow(out, label, 0)
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(label)
        try:
            result = function()
            torch.cuda.synchronize()
            stages.append(label)
            return result
        finally:
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            summary = cpu.finish()
            assert summary['operators'] and summary['record_shapes']

    if role == 'rollout':
        from flashinfer.norm import rmsnorm
        x = torch.randn(32, 2048, device='cuda', dtype=torch.bfloat16)
        weight = torch.ones(2048, device='cuda', dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                rmsnorm(x, weight)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = rmsnorm(x, weight)
        capture('rollout-extend-tp0', lambda: rmsnorm(x, weight))
        capture('rollout-decode-tp0', graph.replay)
        assert torch.isfinite(y).all()
    else:
        # Match the fused implementation observed in the actual Actor NSys trace.
        from fla.modules.fused_norm_gate import FusedRMSNormGated
        norm = FusedRMSNormGated(128, device='cuda', dtype=torch.bfloat16)
        x = torch.randn(32, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        z = torch.randn_like(x, requires_grad=True)
        for _ in range(2):
            norm(x, z).sum().backward()
            x.grad = z.grad = None
            norm.zero_grad()
        y = capture('actor_actor_forward_start_rank0', lambda: norm(x, z))
        capture('actor_backward_start_rank0', lambda: y.sum().backward())
        assert torch.isfinite(x.grad).all()
    (out / (role + '-proof.json')).write_text(json.dumps(
        dict(stages=stages, optimizer_updates=0, graph_replay=role == 'rollout')))

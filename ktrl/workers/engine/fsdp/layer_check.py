"""Two-rank numerical gate for KT's native BF16 Full expert backward."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from kt_kernel.sft.amx import AMXSFTMoEWrapper
from kt_kernel.sft.arch import MOEArchConfig
from kt_kernel.sft.layer import KTMoELayerWrapper


def compare(name, actual, expected, metrics):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    delta = actual - expected
    relative = (delta.norm() / expected.norm().clamp_min(1e-10)).item()
    metrics[name] = {"relative_l2": relative, "max_abs": delta.abs().max().item(),
                     "actual_norm": actual.norm().item(), "reference_norm": expected.norm().item()}
    if not torch.isfinite(actual).all() or not relative <= 0.06:
        raise ValueError(f"{name} differs from BF16 reference: {metrics[name]}")


def main(folder):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16
    experts, width, topk = 4, 256, 2
    rng = torch.Generator().manual_seed(730)
    matrices = [(torch.randn(experts, width, width, generator=rng) / width**0.5).to(dtype) for _ in range(3)]
    ref = [torch.nn.Parameter(w.to(device)) for w in matrices]
    wrapper = None
    if rank == 0:
        wrapper = AMXSFTMoEWrapper(layer_idx=0, num_experts=experts, num_experts_per_tok=topk,
                                   hidden_size=width, moe_intermediate_size=width, num_gpu_experts=0,
                                   cpuinfer_threads=4, threadpool_count=1, weight_path="", chunked_prefill_size=64,
                                   lora_rank=0, full_weight_grad=True)
        wrapper.load_weights_from_tensors(*matrices, torch.arange(experts, dtype=torch.long))
        wrapper.init_full_weight_grad_buffers(*matrices)
    params = (wrapper.gate_proj_buf, wrapper.up_proj_buf, wrapper.down_proj_buf) if wrapper else (None,) * 3
    fixture = torch.nn.Module()
    fixture.gate = torch.nn.Linear(width, experts, bias=False)
    fixture.experts = torch.nn.ModuleList()
    arch = MOEArchConfig("mlp", "gate", "experts", ("gate_proj", "up_proj", "down_proj"), experts, width, topk)
    layer = KTMoELayerWrapper(fixture, wrapper, None, arch, width, 0, full_weight_grad=True,
                              uses_authoritative_optimizer_grads=True)
    metrics = {}
    try:
        for micro in range(2):
            torch.manual_seed(101 + 7 * rank + micro)
            n = 8 + rank
            x = torch.randn(1, n, width, device=device, dtype=dtype).requires_grad_()
            router = torch.rand(n, topk, device=device, dtype=dtype).requires_grad_()
            ids = torch.tensor([[0, 1], [1, 2]], device=device).repeat((n + 1) // 2, 1)[:n].contiguous()
            x_ref = x.detach().clone().requires_grad_()
            r_ref = router.detach().clone().requires_grad_()
            actual = layer._forward_routed(x, ids, router)
            result = torch.zeros_like(x_ref[0])
            for e in range(experts):
                rows, choices = torch.where(ids == e)
                selected = x_ref[0, rows]
                y = F.linear(F.silu(F.linear(selected, ref[0][e])) * F.linear(selected, ref[1][e]), ref[2][e])
                result = result.index_add(0, rows, y * r_ref[rows, choices, None])
            dy = torch.randn_like(actual)
            actual.backward(dy)
            result.backward(dy[0])
            compare(f"forward_{micro}", actual[0], result, metrics)
            compare(f"input_grad_{micro}", x.grad, x_ref.grad, metrics)
            compare(f"routing_grad_{micro}", router.grad, r_ref.grad, metrics)
        for i, parameter in enumerate(ref):
            grad = parameter.grad.float()
            dist.all_reduce(grad)
            grad /= world
            if rank == 0:
                compare(f"expert_grad_{i}", params[i].grad, grad, metrics)
                assert params[i].grad[3].count_nonzero().item() == 0
        if rank == 0:
            opt = torch.optim.AdamW(params, lr=1e-6, weight_decay=0.01, foreach=False)
            before = params[0].detach().clone()
            opt.step()
            wrapper.update_base_weights()
            changed = (before != params[0]).sum().item()
            assert changed and int(opt.state[params[0]]["step"]) == 1
            metrics["optimizer_changed_elements"] = changed
        (folder / f"rank-{rank}.json").write_text(json.dumps(metrics, indent=2))
        dist.barrier()
        if rank == 0:
            (folder / "accepted.json").write_text(json.dumps({"world": world, "microbatches": 2, "max_relative_l2": 0.06}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args().out)

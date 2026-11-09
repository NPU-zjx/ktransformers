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
from ktrl.trainer.master_adamw import MasterAdamW,assert_compute_copy


def compare(name, actual, expected, metrics):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    delta = actual - expected
    relative = (delta.norm() / expected.norm().clamp_min(1e-10)).item()
    metrics[name] = {"relative_l2": relative, "max_abs": delta.abs().max().item(),
                     "actual_norm": actual.norm().item(), "reference_norm": expected.norm().item()}
    if not torch.isfinite(actual).all() or not relative <= 0.06:
        raise ValueError(f"{name} differs from BF16 reference: {metrics[name]}")


def main(folder,realistic=False):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16
    experts, width, topk = (16,2048,8) if realistic else (4,256,2)
    intermediate=512 if realistic else width
    rng = torch.Generator().manual_seed(730)
    matrices = [(torch.randn(shape,generator=rng)/shape[-1]**.5).to(dtype) for shape in
        ((experts,intermediate,width),(experts,intermediate,width),(experts,width,intermediate))]
    ref = [torch.nn.Parameter(w.to(device)) for w in matrices]
    wrapper = None
    if rank == 0:
        wrapper = AMXSFTMoEWrapper(layer_idx=0, num_experts=experts, num_experts_per_tok=topk,
                                   hidden_size=width, moe_intermediate_size=intermediate, num_gpu_experts=0,
                                   cpuinfer_threads=4, threadpool_count=1, weight_path="", chunked_prefill_size=1024 if realistic else 64,
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
            n = (256 if realistic else 8) + rank
            x = torch.randn(1, n, width, device=device, dtype=dtype).requires_grad_()
            router = torch.rand(n, topk, device=device, dtype=dtype).requires_grad_()
            ids = ((torch.arange(n,device=device)[:,None]+torch.arange(topk,device=device)[None,:])%(experts-1)).contiguous()
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
                actual = params[i].grad.float().to(device)
                coefficient = (actual.flatten() @ grad.flatten() / grad.square().sum()).item()
                assert abs(coefficient-1) < .06, ('owner normalization',coefficient)
                metrics[f'expert_scale_{i}'] = dict(actual_over_expected_mean=coefficient,
                    expected='sum of rank gradients / world, after two accumulated backward calls',
                    double_division_would_be=1/world)
                assert params[i].grad[experts-1].count_nonzero().item() == 0
        if rank == 0:
            opt = MasterAdamW(params,cpu_parameters=params,cpu_threads=4,lr=1e-6,weight_decay=0.01)
            before = params[0].detach().clone()
            opt.step()
            wrapper.update_base_weights()
            changed = (before != params[0]).sum().item()
            assert changed and int(opt.state[params[0]]["step"]) == 1
            metrics["optimizer_changed_elements"] = changed
            for parameter in params:
                assert_compute_copy(parameter,opt.state[parameter]['master'])
                assert opt.state[parameter]['exp_avg'].dtype==torch.float32
            torch.save(opt.state_dict(),folder/'master-optimizer.pt')
            resumed=MasterAdamW(params,cpu_parameters=params,cpu_threads=4,lr=1e-6,weight_decay=.01)
            resumed.load_state_dict(torch.load(folder/'master-optimizer.pt',weights_only=True))
            for parameter in params:
                for key in ('master','exp_avg','exp_avg_sq','step'):
                    assert torch.equal(resumed.state[parameter][key],opt.state[parameter][key])
            metrics['fp32_master_resume_exact']=True
        # Small four-expert numerical fixture only: copy reference tensors to the
        # reference GPU implementation; real KTRL expert publication stays on CPU.
        with torch.no_grad():
            for i,parameter in enumerate(ref):
                updated=params[i].detach().to(device) if rank==0 else torch.empty_like(parameter)
                dist.broadcast(updated,src=0)
                parameter.copy_(updated)
            actual=layer._forward_routed(x.detach(),ids,router.detach())
            expected=torch.zeros_like(x_ref[0])
            for expert in range(experts):
                rows,choices=torch.where(ids==expert)
                selected=x.detach()[0,rows]
                value=F.linear(F.silu(F.linear(selected,ref[0][expert]))*F.linear(selected,ref[1][expert]),ref[2][expert])
                expected=expected.index_add(0,rows,value*router.detach()[rows,choices,None])
            compare('updated_amx_forward',actual[0],expected,metrics)
        (folder / f"rank-{rank}.json").write_text(json.dumps(metrics, indent=2))
        dist.barrier()
        if rank == 0:
            (folder / "accepted.json").write_text(json.dumps({"world": world, "microbatches": 2, "max_relative_l2": 0.06,
                'fp32_master':True,'master_checkpoint_resume':True,'updated_amx_forward':True}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument('--realistic',action='store_true')
    args=p.parse_args();main(args.out,args.realistic)

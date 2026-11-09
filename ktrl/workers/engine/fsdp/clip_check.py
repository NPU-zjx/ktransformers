"""Two GPU ranks: exact known gradient norm through real Accelerate-KT/FSDP2."""
import argparse
import json
from pathlib import Path
import torch
import torch.distributed as dist
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import MixedPrecisionPolicy
from ktrl.trainer.gradient_audit import gradient_snapshot, verify_clip


def main(out):
    plugin=FullyShardedDataParallelPlugin(fsdp_version=2,reshard_after_forward=True,
        mixed_precision_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32))
    accelerator=Accelerator(fsdp_plugin=plugin,mixed_precision='bf16')
    assert accelerator.is_fsdp2 and accelerator.num_processes==2
    rank=accelerator.process_index
    model=torch.nn.Linear(8,8,bias=False)
    model.weight.data.fill_(.125)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    model,optimizer=accelerator.prepare(model,optimizer)
    x=torch.ones(1,8,device=accelerator.device)*(rank+1)
    accelerator.backward(model(x).sum())
    for p in model.parameters():
        assert p.dtype==p.grad.dtype==torch.float32
        torch.testing.assert_close(p.grad.to_local(),torch.full_like(p.grad.to_local(),1.5),rtol=0,atol=0)
    optimizer.zero_grad()
    for p in model.parameters():
        assert hasattr(p,'to_local')
        p.grad=torch.ones_like(p)*12
    expert=torch.nn.Parameter(torch.tensor([3.,4.])) if rank==0 else None
    cpu=[] if expert is None else [expert]
    if expert is not None: expert.grad=torch.tensor([3.,4.])
    parameters=list(model.named_parameters())+([] if expert is None else [('cpu_expert',expert)])
    value,rows=gradient_snapshot(parameters,{id(p) for p in cpu},rank)
    scalar=torch.tensor(value,dtype=torch.float64,device=accelerator.device)
    dist.all_reduce(scalar)
    expected=(64*144+25)**.5
    assert abs(scalar.sqrt().item()-expected)<1e-8
    norm=accelerator.clip_grad_norm_(model.parameters(),1.,rank_local_parameters=cpu)
    proof=verify_clip(rows,expected,float(norm),1.)
    proof.update(rank=rank,fsdp2=True,world=2,expected_global_elements=66,
                 fp32_master_shards=True,bf16_compute=True,fp32_reduce_gradient=1.5,
                 zero_optimizer_updates=True,device=str(accelerator.device))
    (out/f'clip-rank{rank}.json').write_text(json.dumps(proof,indent=2))
    dist.barrier()
    if rank==0:(out/'accepted.json').write_text(json.dumps(dict(native_fsdp2_joint_clip=True,updates=0)))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)

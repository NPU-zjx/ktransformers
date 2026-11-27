"""Real FSDP2/AdamW CUDA shard parking equivalence, with untouched BF16 CPU state."""
import argparse,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
from ktrl.trainer.master_adamw import MasterAdamW
from ktrl.workers.engine.fsdp.residency import GPUStateParking,release_cuda_gradients

def local(t):return t.to_local() if hasattr(t,'to_local') else t
def snapshot(model,opt,cpu):
    d={f'p:{k}':local(p).detach().cpu().clone() for k,p in model.named_parameters()}
    d['cpu']=cpu.detach().clone()
    for i,p in enumerate([*model.parameters(),cpu]):
        for k,v in opt.state[p].items():
            if isinstance(v,torch.Tensor):d[f's:{i}:{k}']=local(v).detach().cpu().clone()
    return d
def equal(a,b):
    assert a.keys()==b.keys()
    for k in a:assert torch.equal(a[k],b[k]),k
def trial(park,device):
    torch.manual_seed(41)
    model=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,64)).to(device)
    policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.bfloat16)
    fully_shard(model[0],mp_policy=policy);fully_shard(model[2],mp_policy=policy);fully_shard(model,mp_policy=policy,reshard_after_forward=True)
    cpu=nn.Parameter(torch.ones(16,16,dtype=torch.bfloat16))
    opt=MasterAdamW([*model.parameters(),cpu],cpu_parameters=[cpu],cpu_threads=1,lr=1e-4)
    parking=GPUStateParking();records=[];copies=[];ids=[id(p) for p in model.parameters()]
    for step in range(1,9):
        gen=torch.Generator(device=device).manual_seed(100+step)
        x=torch.randn(8,128,device=device,generator=gen);y=torch.randn(8,64,device=device,generator=gen)
        loss=(model(x).float()-y).square().mean();loss.backward()
        cpu.grad=torch.full_like(cpu,step/100)
        opt.step();cpu_grad_id=id(cpu.grad);n=release_cuda_gradients(opt)
        assert id(cpu.grad)==cpu_grad_id and cpu.device.type=='cpu'
        assert all(p.grad is None for p in model.parameters()) and n>0
        records.append(snapshot(model,opt,cpu));opt.zero_grad(set_to_none=False)
        if park and step%2==0:
            before=snapshot(model,opt,cpu);before_cpu=cpu.detach().clone()
            cpu_state_ids={k:id(v) for k,v in opt.state[cpu].items()}
            off=parking.park(model,opt)
            assert off['bytes']>0 and off['cpu_expert_transfer_bytes']==0
            assert [id(p) for p in model.parameters()]==ids
            assert all(s.nbytes()==0 for s,_,_,_ in parking.parked)
            on=parking.restore();equal(before,snapshot(model,opt,cpu))
            assert torch.equal(before_cpu,cpu) and cpu_state_ids=={k:id(v) for k,v in opt.state[cpu].items()}
            copies.append(dict(step=step,offload=off,reload=on))
    return records,copies

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank);dist.init_process_group('nccl',device_id=device)
start=time.monotonic();control,_=trial(False,device);result,copies=trial(True,device)
for a,b in zip(control,result):equal(a,b)
args.out.mkdir(exist_ok=True,parents=True)
(args.out/f'residency-rank{rank}.json').write_text(json.dumps(dict(passed=True,rank=rank,optimizer_updates=8,simulated_rounds=4,all_parameters_and_adam_slots_bit_exact=True,cpu_state_identity_preserved=True,copies=copies,seconds=time.monotonic()-start),indent=2))
print(f'rank {rank}: eight real FSDP2/AdamW updates bit-exact with/without parking',flush=True)
dist.destroy_process_group()

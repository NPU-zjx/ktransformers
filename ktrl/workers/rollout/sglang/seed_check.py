"""Actual native GPU sampler: request seeds survive batching/order changes."""
import argparse
import json
from pathlib import Path
import torch
from sglang.srt.layers.sampler import sampling_from_probs_torch


def main(out):
    out.mkdir()
    device='cuda:0'
    probabilities=torch.full((64,128),1/128,device=device)
    seeds=torch.arange(64,device=device,dtype=torch.int64)+42
    positions=torch.arange(64,device=device,dtype=torch.int64)
    a=sampling_from_probs_torch(probabilities,sampling_seed=seeds,positions=positions)
    b=sampling_from_probs_torch(probabilities,sampling_seed=seeds,positions=positions)
    assert torch.equal(a,b)
    order=torch.arange(63,-1,-1,device=device)
    c=sampling_from_probs_torch(probabilities[order],sampling_seed=seeds[order],positions=positions[order])
    assert torch.equal(a[order],c)
    d=sampling_from_probs_torch(probabilities,sampling_seed=seeds+1000,positions=positions)
    assert (a!=d).any()
    (out/'accepted.json').write_text(json.dumps(dict(native_gpu_sampler=True,
        repeat_exact=True,batch_reordering_exact=True,changed_seed_changes=int((a!=d).sum()))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)

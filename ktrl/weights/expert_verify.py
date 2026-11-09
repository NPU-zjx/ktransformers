"""Check packed native CPU experts against checkpoint tensors without GPU staging."""
import json
from pathlib import Path


def verify_native_expert(wrapper,path):
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    path=Path(path)
    index=json.loads((path/'kt_full_experts.index.json').read_text())
    entry=index['layers'][str(wrapper.layer_idx)]
    count,intermediate,hidden=entry['shapes']['gate_proj']
    topk=wrapper.num_experts_per_tok
    ids=torch.linspace(0,count-1,topk,dtype=torch.float64).long().reshape(1,-1)
    weights=torch.full((1,topk),1/topk,dtype=torch.float32)
    x=(torch.sin(torch.arange(hidden,dtype=torch.float32))*.01).bfloat16().reshape(1,-1)
    output=torch.zeros_like(x)
    length=torch.tensor([1],dtype=torch.int32)
    wrapper.cpu_infer.sync()
    wrapper.cpu_infer.submit(wrapper.moe.forward_task(length.data_ptr(),topk,ids.data_ptr(),
        weights.data_ptr(),x.data_ptr(),output.data_ptr(),False))
    wrapper.cpu_infer.sync()
    reference=torch.zeros(1,hidden,dtype=torch.float32)
    samples=[]
    with safe_open(path/entry['file'],framework='pt',device='cpu') as source:
        keys={short:next(k for k in source.keys() if k.endswith(short+'_proj')) for short in ('gate','up','down')}
        matrices={short:source.get_tensor(key) for short,key in keys.items()}
        for e in ids[0].tolist():
            gate,up,down=(matrices[s][e] for s in ('gate','up','down'))
            value=F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)
            reference.add_(value.float(),alpha=1/topk)
            for short in keys:
                tensor=matrices[short][e].reshape(-1)
                for pos in (0,tensor.numel()//2,tensor.numel()-1):
                    samples.append(dict(expert=e,projection=short,index=pos,value=tensor[pos].item()))
    delta=output.float()-reference
    relative=(delta.norm()/reference.norm().clamp_min(1e-12)).item()
    if not torch.isfinite(output).all() or relative>.06:
        raise RuntimeError(f'Packed CPU expert layer {wrapper.layer_idx} differs from checkpoint: {relative}')
    return dict(layer=wrapper.layer_idx,selected_experts=ids[0].tolist(),source_samples=samples,
        packed_native_forward_relative_l2=relative,max_absolute_error=delta.abs().max().item(),
        inputs_and_weights_device='cpu',gpu_staging_bytes=0)

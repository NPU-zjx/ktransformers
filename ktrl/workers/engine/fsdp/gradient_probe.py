"""Full 35B KT/GPU GRPO backward comparison; always stops before optimizer.step."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText,Trainer,TrainerCallback,TrainingArguments
from torch.distributed.fsdp import MixedPrecisionPolicy
from kt_kernel.sft import get_kt_named_trainable_params
from ktrl.configs.recipe import load_plan
from ktrl.trainer.grpo import policy_loss_sum
from .actor_update import TokenLogProbs
from . import checkpoint_context

LAYERS={0,10,20,30,39}
EXPERTS=[0,7,63,127,255]

class Captured(Exception):pass


def capture(model,out,backend,kind='gradients'):
    result={}
    for name,p in model.named_parameters():
        match=re.search(r'\.layers\.(\d+)\.(.*)',name)
        data=p if kind=='weights' else p.grad
        if not match or data is None:continue
        layer,tail=int(match[1]),match[2]
        select=tail=='mlp.gate.weight' or ('.self_attn.' in '.'+tail and tail.endswith(('q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight')))
        expert=backend=='gpu' and layer in LAYERS and tail.startswith('mlp.experts.')
        if not (select or expert):continue
        grad=data.full_tensor() if hasattr(data,'full_tensor') else data
        if dist.get_rank()==0:
            if expert:
                chosen=grad[EXPERTS].detach().float().cpu()
                if tail.endswith('gate_up_proj'):
                    gate,up=chosen.chunk(2,dim=1)
                    result[f'{layer}/expert/gate']=gate.contiguous()
                    result[f'{layer}/expert/up']=up.contiguous()
                elif tail.endswith('down_proj'):
                    result[f'{layer}/expert/down']=chosen
                else:raise RuntimeError(f'Unexpected GPU expert layout: {name}')
            else:result[f'{layer}/{tail}']=grad.detach().float().cpu()
        del grad
    if backend=='kt' and dist.get_rank()==0:
        for name,p in get_kt_named_trainable_params(model):
            match=re.match(r'kt.layers.(\d+).experts.base.(gate|up|down)_proj',name)
            if match and int(match[1]) in LAYERS:
                data=p if kind=='weights' else p.grad
                assert p.device.type=='cpu' and data is not None
                result[f'{match[1]}/expert/{match[2]}']=data[EXPERTS].detach().float().clone()
    if dist.get_rank()==0:
        assert len(result)>=55,len(result)
        torch.save(result,out/f'{kind}.pt')
    dist.barrier()


def main(config,rows_file,out,backend,trace=False,fp32_expert_accum=False,no_checkpoint=False,replay_mlp_from=None):
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=timedelta(minutes=75))
    plan=load_plan(config)
    row=json.loads(rows_file.read_text())[0]
    response=row['response'][:512]
    tokens=row['prompt']+response
    n=len(response)
    kt=dict(plan['actor']['kt_config'],kt_skip_expert_loading=True,kt_lora_rank=0,
        kt_model_max_length=len(tokens),kt_activation_policy={'cpu':'recompute','gpu':'recompute'},kt_max_cache_depth=1)
    if no_checkpoint:kt['kt_activation_policy']={'cpu':'retain','gpu':'retain'}
    args=TrainingArguments(output_dir=str(out/'unused'),max_steps=1,per_device_train_batch_size=1,
        gradient_accumulation_steps=1,learning_rate=1e-6,optim='adamw_torch',max_grad_norm=0,
        bf16=True,report_to='none',save_strategy='no',remove_unused_columns=False,dataloader_pin_memory=False,
        gradient_checkpointing=not no_checkpoint,gradient_checkpointing_kwargs=dict(use_reentrant=False,**({'context_fn':checkpoint_context} if backend=='kt' else {})),
        fsdp='full_shard auto_wrap',fsdp_config=dict(version=2,cpu_ram_efficient_loading=True,
            reshard_after_forward=True,transformer_layer_cls_to_wrap=['Qwen3_5MoeDecoderLayer','Qwen3_5MoeVisionBlock']),
        kt_config=kt if backend=='kt' else None)
    model=AutoModelForImageTextToText.from_pretrained(plan['model']['path'],dtype=torch.bfloat16,
        device_map={'':f'cuda:{rank}' if backend=='kt' else 'cpu'},attn_implementation='sdpa',local_files_only=True)
    if backend=='gpu':
        from kt_kernel.sft.conv3d_compat import patch_vlm_conv3d
        patch_vlm_conv3d(model)
    if fp32_expert_accum:
        assert backend=='gpu'
        import types
        import torch.nn.functional as F
        def expert_forward(self,hidden_states,top_k_index,top_k_weights):
            result=torch.zeros_like(hidden_states,dtype=torch.float32)
            for expert in torch.unique(top_k_index).tolist():
                choices,tokens=None,None
                tokens,choices=torch.where(top_k_index==expert)
                gate,up=F.linear(hidden_states[tokens],self.gate_up_proj[expert]).chunk(2,dim=-1)
                value=F.linear(self.act_fn(gate)*up,self.down_proj[expert])
                result=result.index_add(0,tokens,value.float()*top_k_weights[tokens,choices,None].float())
            return result.to(hidden_states.dtype)
        patched=0
        for module in model.modules():
            if type(module).__name__=='Qwen3_5MoeExperts':
                module.forward=types.MethodType(expert_forward,module);patched+=1
        assert patched > 0, "No compatible GPU expert modules found"
    for p in model.parameters():
        if p.requires_grad and p.numel():p.data=p.data.float()
    model.config.use_cache=False

    # Diagnostic only: hold the forward trajectory fixed while retaining the
    # independent GPU expert Jacobian. Never used by the Actor training path.
    if replay_mlp_from:
        assert backend=='gpu'
        class Replay(torch.autograd.Function):
            @staticmethod
            def forward(ctx,value,target):return target.clone()
            @staticmethod
            def backward(ctx,gradient):return gradient,None
        count=0
        for name,module in model.named_modules():
            if name.endswith('.mlp') and '.layers.' in name:
                target=torch.load(replay_mlp_from/f'{name}-forward.pt',weights_only=True).to(rank)
                def replace(module,inputs,output,target=target):
                    assert output.shape==target.shape and output.dtype==target.dtype
                    return Replay.apply(output,target)
                module.register_forward_hook(replace)
                count+=1
        assert count == plan["model"]["text_layers"], count

    class Check(TrainerCallback):
        def on_train_begin(self,args,state,control,model,**kwargs):
            if trace:
                capture(model,out,backend,'weights')
                (out/'trace').mkdir(exist_ok=True)
                def hook(name):
                    def record(module,inputs,output):
                        if rank!=0:return
                        value=output[0] if isinstance(output,tuple) else output
                        if not isinstance(value,torch.Tensor):return
                        path=out/'trace'/f'{name}-forward.pt'
                        if not path.exists():torch.save(value.detach().cpu(),path)
                        if value.requires_grad:
                            def backward(gradient):
                                target=out/'trace'/f'{name}-backward.pt'
                                if not target.exists():torch.save(gradient.detach().cpu(),target)
                            value.register_hook(backward)
                    return record
                for name,module in model.named_modules():
                    if name.endswith(('embed_tokens','input_layernorm','post_attention_layernorm','mlp','linear_attn','self_attn')):
                        module.register_forward_hook(hook(name))
        def on_pre_optimizer_step(self,args,state,control,model,optimizer,**kwargs):
            assert state.global_step==0 and not optimizer.state
            capture(model,out,backend)
            if rank==0:
                (out/'accepted.json').write_text(json.dumps(dict(backend=backend,optimizer_updates=0,
                    response_tokens=n,prompt_tokens=len(row['prompt']),advantage=1,world=dist.get_world_size(),
                    gradient_checkpointing=not no_checkpoint,
                    diagnostic_forward_replay=str(replay_mlp_from) if replay_mlp_from else None,
                    gpu_expert_accumulation='float32' if fp32_expert_accum else 'native')))
            raise Captured()

    class Probe(Trainer):
        def compute_loss(self,model,inputs,**kwargs):
            ids=inputs['input_ids']
            logits=model(input_ids=ids,use_cache=False,logits_to_keep=n+1).logits[0,:-1]
            logp=TokenLogProbs.apply(logits,ids[0,-n:])
            if trace and rank==0:torch.save(logp.detach().cpu(),out/'logp.pt')
            return policy_loss_sum(logp,logp.detach(),torch.tensor(1.,device=logp.device),torch.ones_like(logp,dtype=torch.bool))/n

    trainer=Probe(model=model,args=args,train_dataset=[{'input_ids':tokens} for _ in range(dist.get_world_size())],
        data_collator=lambda batch:{'input_ids':torch.tensor([batch[0]['input_ids']])},callbacks=[Check()])
    trainer.model_accepts_loss_kwargs=False
    trainer.accelerator.state.fsdp_plugin.mixed_precision_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
    try:trainer.train()
    except Captured:pass
    else:raise RuntimeError('Gradient probe failed to stop before optimizer')
    dist.destroy_process_group()


def compare(kt_folder,gpu_folder,out):
    a=torch.load(kt_folder/'gradients.pt',weights_only=True)
    b=torch.load(gpu_folder/'gradients.pt',weights_only=True)
    assert set(a)==set(b),(set(a)-set(b),set(b)-set(a))
    rows=[]
    for name in sorted(a):
        x,y=a[name].reshape(-1).double(),b[name].reshape(-1).double()
        assert x.shape==y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
        rel=((x-y).norm()/y.norm().clamp_min(1e-20)).item()
        cosine=(torch.nn.functional.cosine_similarity(x,y,dim=0).item() if x.norm() and y.norm() else 1. if torch.equal(x,y) else 0.)
        rows.append(dict(name=name,relative_l2=rel,cosine=cosine,kt_norm=x.norm().item(),gpu_norm=y.norm().item(),passed=rel<=.06 and cosine>=.99))
    report=dict(accepted=all(r['passed'] for r in rows),optimizer_updates=0,comparisons=rows)
    out.write_text(json.dumps(report,indent=2))
    assert report['accepted'],'Full model gradients differ; inspect report before opening RL gate'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--rows',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=['kt','gpu'],required=True)
    p.add_argument('--trace',action='store_true')
    p.add_argument('--fp32-expert-accum',action='store_true')
    p.add_argument('--no-checkpoint',action='store_true')
    p.add_argument('--replay-mlp-from',type=Path)
    a=p.parse_args();main(a.config,a.rows,a.out,a.backend,a.trace,a.fp32_expert_accum,a.no_checkpoint,a.replay_mlp_from)

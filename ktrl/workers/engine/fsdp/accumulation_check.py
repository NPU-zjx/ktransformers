"""Actual Trainer GAS4 + FP32 FSDP2 reduction with an unused wrapped branch."""
import argparse
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import Trainer,TrainerCallback,TrainingArguments


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.full((8,8),.125))
    def forward(self,x):return torch.nn.functional.linear(x,self.weight)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.used=Block();self.unused=Block()
    def forward(self,x):return self.used(x)


class Done(Exception):pass


def main(out):
    seen=[]
    class Check(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model,optimizer,**kwargs):
            assert len(seen)==4 and state.global_step==0 and not optimizer.state
            expected=torch.tensor(sum(seen)/4,device='cuda',dtype=torch.float32)
            dist.all_reduce(expected);expected/=dist.get_world_size()
            assert model.unused.weight.grad is None
            gradient=model.used.weight.grad
            assert gradient.dtype==torch.float32
            torch.testing.assert_close(gradient.to_local(),torch.full_like(gradient.to_local(),expected.item()),rtol=0,atol=0)
            if dist.get_rank()==0:
                (out/'accepted.json').write_text(json.dumps(dict(accepted=True,gas=4,sync_each_batch=True,
                    gradient=expected.item(),unused_branch_no_grad=True,optimizer_updates=0)))
            raise Done()
    class Tiny(Trainer):
        def compute_loss(self,model,inputs,**kwargs):
            value=float(inputs['value'])*(dist.get_rank()+1);seen.append(value)
            x=torch.full((1,8),value,device=self.args.device)
            with torch.no_grad():model(x)
            return model(x).sum()
    args=TrainingArguments(output_dir=str(out/'trainer'),max_steps=1,per_device_train_batch_size=1,
        gradient_accumulation_steps=4,bf16=True,max_grad_norm=0,save_strategy='no',report_to='none',
        remove_unused_columns=False,dataloader_pin_memory=False,
        accelerator_config={'gradient_accumulation_kwargs':{'sync_each_batch':True}},
        fsdp='full_shard auto_wrap',fsdp_config=dict(version=2,reshard_after_forward=True,
            transformer_layer_cls_to_wrap=['Block']))
    trainer=Tiny(model=Model(),args=args,train_dataset=[{'value':i+1} for i in range(8)],
        data_collator=lambda rows:rows[0],callbacks=[Check()])
    trainer.model_accepts_loss_kwargs=False
    trainer.accelerator.state.fsdp_plugin.mixed_precision_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
    try:trainer.train()
    except Done:pass
    else:raise RuntimeError('Probe did not stop before optimizer')
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)

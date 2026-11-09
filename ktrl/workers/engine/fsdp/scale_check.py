"""Exercise installed Trainer/Accelerate loss division, with unequal token counts."""
import argparse
import json
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from ktrl.trainer.gradient_audit import watch_loss_scale


def main(out):
    out.mkdir()
    lengths=[1,3,5,7]
    model=torch.nn.Linear(1,1,bias=False)
    model.weight.data.fill_(.25)
    observations=[]
    reference=sum(n*(.25-n)*2 for n in lengths)/sum(lengths)

    class Check(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model,**kwargs):
            actual=model.weight.grad.item()
            assert abs(actual-reference)<1e-5,(actual,reference)
            assert len(observations)==4 and all(x['backward_scale']==.25 for x in observations)
            (out/'accepted.json').write_text(json.dumps(dict(gradient=actual,reference=reference,
                scales=observations,installed_trainer=True,gas=4,device='cpu')))

    class Tiny(Trainer):
        def compute_loss(self,model,inputs,**kwargs):
            n=inputs['n']
            loss=(model(torch.ones(1,1))-n).square().sum()*n*4/sum(lengths)
            watch_loss_scale(loss,4,observations.append)
            return loss

    args=TrainingArguments(output_dir=str(out/'trainer'),use_cpu=True,report_to='none',
        max_steps=1,per_device_train_batch_size=1,gradient_accumulation_steps=4,
        max_grad_norm=0,save_strategy='no',remove_unused_columns=False,
        dataloader_pin_memory=False)
    trainer=Tiny(model=model,args=args,train_dataset=[{'n':n} for n in lengths],
        data_collator=lambda rows:rows[0],callbacks=[Check()])
    trainer.model_accepts_loss_kwargs=False
    trainer.train()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)

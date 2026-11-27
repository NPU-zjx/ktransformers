"""Two-rank regression: actual Trainer GAS=2, BF16 rounding, clip/update/restore.

Run with torchrun --standalone --nproc_per_node=2, not a full-model RL run.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import Trainer, TrainerCallback, TrainingArguments
from torch.utils.data import SequentialSampler

from ktrl.workers.engine.fsdp.gradient_accumulation import (
    BF16AccumulationTrainerMixin, BF16ShardAccumulator,
)
from ktrl.trainer.master_adamw import MasterAdamW


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4, 4), 0.25))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = TinyBlock()

    def forward(self, x):
        return self.block(x)


def local(t):
    return t.to_local() if hasattr(t, 'to_local') else t


class ProbeTrainer(BF16AccumulationTrainerMixin, Trainer):
    def __init__(self, *args, accumulation_dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation_dtype = accumulation_dtype
        self.model_accepts_loss_kwargs = False
        self.observations = []
        self.backward_calls = 0
        self.before_clip = []

    def _clip_grad_norm(self, model):
        self.before_clip.append(float(local(model.block.weight.grad).flatten()[0]))
        return super()._clip_grad_norm(model)

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    def compute_loss(self, model, inputs, **kwargs):
        x = torch.full((1, 4), float(inputs['value']), device=self.args.device, dtype=torch.bfloat16)
        return model(x).sum()

    def training_step(self, *args, **kwargs):
        result = super().training_step(*args, **kwargs)
        self.backward_calls += 1
        return result

    def on_bf16_accumulation(self, snapshot, accumulator):
        buffer = next(iter(accumulator.buffers.values()))[2]
        assert buffer.dtype == torch.bfloat16
        assert snapshot['fp32_gradients_cleared']
        self.observations.append(dict(snapshot, value=float(local(buffer).flatten()[0])))


def trainer_case(out, mode, tail=False):
    records = []
    class ClipProof(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            p = model.block.weight
            records.append({'step': state.global_step, 'grad_dtype': str(p.grad.dtype),
                            'grad_after_clip': float(local(p.grad).flatten()[0]),
                            'param_before': float(local(p.detach()).flatten()[0])})
        def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
            records[-1]['param_after'] = float(local(model.block.weight.detach()).flatten()[0])
            records[-1]['adam_step'] = int(optimizer.state[model.block.weight]['step'])
            assert records[-1]['param_after'] != records[-1]['param_before']

    args = TrainingArguments(output_dir=str(out / (mode + ('-tail' if tail else ''))), max_steps=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        learning_rate=0.01, lr_scheduler_type='constant', max_grad_norm=1.0,
        weight_decay=0., bf16=True, save_strategy='no', report_to='none',
        remove_unused_columns=False, dataloader_pin_memory=False, disable_tqdm=True,
        accelerator_config={'gradient_accumulation_kwargs': {'sync_each_batch': True}},
        fsdp='full_shard auto_wrap', fsdp_config={'version':2, 'reshard_after_forward':True,
                                                'transformer_layer_cls_to_wrap':['TinyBlock']})
    trainer = ProbeTrainer(model=TinyModel(), args=args, accumulation_dtype=mode,
        # Four global rows -> two local microbatches per rank, per update.
        train_dataset=[{'value':v} for v in ((512.,512.,2.,2.,4.,4.) if tail else (512.,512.,2.,2.))],
        data_collator=lambda rows:rows[0], callbacks=[ClipProof()])
    trainer.accelerator.state.fsdp_plugin.mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    trainer.train()
    assert trainer.backward_calls == (3 if tail else 4), trainer.backward_calls
    assert [r['adam_step'] for r in records] == [1,2]
    assert all(abs(r['grad_after_clip'] - 0.25) < 1e-6 for r in records)
    if mode == 'bfloat16':
        # BF16 256+1 rounds back to 256; FP32 would retain 257.
        assert [r['value'] for r in trainer.observations] == ([256.,256.,4.] if tail else [256.,256.,256.,256.])
        assert [r['microsteps'] for r in trainer.observations] == ([1,2,1] if tail else [1,2,1,2])
        assert trainer.before_clip == ([256.,4.] if tail else [256.,256.]), trainer.before_clip
        assert not trainer.bf16_accumulator.buffers
    else:
        assert trainer.before_clip == [257.,257.], trainer.before_clip
    result = {'backward_calls_per_rank':trainer.backward_calls,
              'accumulation':trainer.observations,'optimizer':records,'grad_before_clip':trainer.before_clip}
    del trainer
    return result


def joint_restore_case(out):
    rank = dist.get_rank()
    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    accelerator = Accelerator(fsdp_plugin=FullyShardedDataParallelPlugin(
        fsdp_version=2, reshard_after_forward=True,
        mixed_precision_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)),
        mixed_precision='bf16')
    model = TinyBlock().cuda()
    model = accelerator.prepare(model)
    p = model.weight
    cpu = torch.nn.Parameter(torch.full((4,8),.5,dtype=torch.bfloat16)) if rank == 0 else None
    owned = [cpu] if cpu is not None else []
    optimizer = MasterAdamW([p]+owned,cpu_parameters=owned,cpu_threads=2,lr=.01,weight_decay=0.)

    def one_update():
        acc = BF16ShardAccumulator()
        values = []
        for value in (512.,2.):
            (model(torch.full((1,4),value,device='cuda',dtype=torch.bfloat16)).sum()/2).backward()
            acc.collect(model.named_parameters())
            values.append(float(local(next(iter(acc.buffers.values()))[2]).flatten()[0]))
            if cpu is not None:
                addition = torch.full_like(cpu,1.)
                cpu.grad = addition if cpu.grad is None else cpu.grad.add_(addition)
        assert values == [256.,256.]
        acc.materialize()
        # Same native Accelerate-KT unified clipping used by the Actor.
        norm = accelerator.clip_grad_norm_([p],1.,rank_local_parameters=owned)
        expected = (16*256.**2 + 32*2.**2)**.5
        assert abs(float(norm)-expected) < 1e-3
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(norm)

    norm = one_update()
    path = out/f'restore-rank{rank}.pt'
    torch.save({'parameter':p.detach().clone(),'cpu':cpu.detach().clone() if cpu is not None else None,
                'optimizer':optimizer.state_dict()},path)
    one_update()
    expected_p = p.detach().clone()
    expected_opt = copy.deepcopy(optimizer.state_dict())
    expected_cpu = cpu.detach().clone() if cpu is not None else None
    saved = torch.load(path,weights_only=False)
    with torch.no_grad():
        p.copy_(saved['parameter'])
        if cpu is not None:cpu.copy_(saved['cpu'])
    optimizer.load_state_dict(saved['optimizer'])
    assert int(optimizer.state[p]['step']) == 1
    one_update()
    assert torch.equal(local(p),local(expected_p))
    if cpu is not None:assert torch.equal(cpu,expected_cpu)
    actual_opt=optimizer.state_dict()
    for key, state in expected_opt['state'].items():
        for name, tensor in state.items():
            if torch.is_tensor(tensor):assert torch.equal(local(tensor),local(actual_opt['state'][key][name]))
    return {'joint_norm':norm,'resume_next_update_exact':True,
            'gpu_adam_step':int(optimizer.state[p]['step']),
            'cpu_state_dtypes':{k:str(v.dtype) for k,v in optimizer.state[cpu].items()} if cpu is not None else {}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    import os
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    args.out.mkdir(parents=True,exist_ok=True)
    result={'rank':dist.get_rank(),'trainer':trainer_case(args.out,'bfloat16')}
    result['native_fp32_control']=trainer_case(args.out,'float32')
    result['partial_final_batch']=trainer_case(args.out,'bfloat16',tail=True)
    result['joint_restore']=joint_restore_case(args.out)
    result['passed']=True
    (args.out/f'verified-rank{dist.get_rank()}.json').write_text(json.dumps(result,indent=2))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()

"""Replay one full-model GRPO update through the native KT Transformers Trainer."""

import argparse
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, Trainer, TrainerCallback, TrainingArguments
from kt_kernel.sft import get_kt_named_trainable_params, get_kt_rank_local_parameter_names
from kt_kernel.sft.profiler import collect_kt_sft_profile

from ktrl.configs.recipe import load_plan
from ktrl.trainer.grpo import policy_loss_sum
from . import checkpoint_context


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


class FixedRows(Dataset):
    def __init__(self, folder):
        audit = json.loads((folder / "batch-audit.json").read_text())
        raw = (folder / "fixed-rows.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != audit["fixed_sha256"]:
            raise ValueError("fixed batch changed since the GRPO audit")
        self.rows = json.loads(raw)
        self.valid_tokens = sum(len(r["response"]) for r in self.rows)
        assert self.valid_tokens == audit["valid_response_tokens"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows):
    row, = rows
    return {"input_ids": torch.tensor([row["prompt"] + row["response"]]),
            "advantage": torch.tensor(row["advantage"], dtype=torch.float32),
            "response_length": len(row["response"]), "uid": row["uid"]}


class TokenLogProbs(torch.autograd.Function):
    """Categorical log probabilities with bounded FP32 softmax workspace."""

    @staticmethod
    def forward(ctx, logits, labels):
        ctx.save_for_backward(logits, labels)
        result = torch.empty(labels.shape, device=logits.device, dtype=torch.float32)
        for start in range(0, labels.numel(), 128):
            part = slice(start, start + 128)
            block = logits[part].float()
            result[part] = block.gather(1, labels[part, None]).squeeze(1) - block.logsumexp(-1)
        return result

    @staticmethod
    def backward(ctx, derivative):
        logits, labels = ctx.saved_tensors
        grad = torch.empty_like(logits)
        for start in range(0, labels.numel(), 128):
            part = slice(start, start + 128)
            block = -logits[part].float().softmax(-1)
            block.scatter_add_(1, labels[part, None], torch.ones_like(derivative[part, None]))
            grad[part].copy_(block * derivative[part, None])
        return grad, None


def local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def validate_loss_math(device):
    rng = torch.Generator(device=device).manual_seed(41)
    x = torch.randn(137, 53, device=device, generator=rng, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    labels = torch.arange(137, device=device) % 53
    got = TokenLogProbs.apply(x, labels)
    wanted = reference.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
    torch.testing.assert_close(got, wanted, atol=2e-6, rtol=1e-5)
    mask = torch.arange(137, device=device) % 7 != 0
    old = wanted.detach() + torch.linspace(-2, 2, 137, device=device)
    adv = torch.where(labels % 2 == 0, 1.0, -0.5)
    loss = policy_loss_sum(got, old, adv, mask)
    ratio = (wanted - old).exp()
    ref = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.28) * adv)
    ref = torch.where(adv < 0, torch.minimum(ref, -3 * adv), ref)[mask].sum()
    loss.backward()
    ref.backward()
    torch.testing.assert_close(x.grad, reference.grad, atol=2e-6, rtol=1e-4)
    assert x.grad[~mask].count_nonzero().item() == 0


class Evidence(TrainerCallback):
    def __init__(self, folder, dataset):
        self.out, self.data = folder, dataset
        self.rank = dist.get_rank()
        self.trainer = None
        self.samples = []
        self.probe_before = None

    def event(self, phase, **fields):
        item = dict(phase=phase, rank=self.rank, time_unix_ns=time.time_ns(),
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(), **fields)
        with (self.out / f"events-rank{self.rank}.jsonl").open("a") as f:
            f.write(json.dumps(item, allow_nan=False) + "\n")

    def probe(self, model):
        ids = torch.tensor([self.data[0]["prompt"]], device=torch.cuda.current_device())
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].detach().float().cpu()
        model.train()
        if not torch.isfinite(logits).all():
            raise RuntimeError("fixed probe logits are not finite")
        return logits

    def on_train_begin(self, args, state, control, model, optimizer, **kwargs):
        trainer = self.trainer
        assert trainer.is_kt_enabled and trainer.accelerator.is_fsdp2 and trainer.accelerator.gradient_accumulation_steps == 1
        owned = list(get_kt_named_trainable_params(model))
        cpu = [(n, p) for n, p in owned if p.device.type == "cpu" and p.ndim == 3]
        assert len(cpu) == (120 if self.rank == 0 else 0), (self.rank, len(cpu))
        shards = [(n, p) for n, p in model.named_parameters() if hasattr(p, "to_local")]
        assert shards and len(model._kt_wrappers) == 40
        placeholders = list(get_kt_rank_local_parameter_names(model))
        params = dict(model.named_parameters())
        for n in placeholders:
            p = params[n]
            if hasattr(p, "to_local") or (p.device.type == "cuda" and p.numel()):
                raise RuntimeError(f"FSDP claimed a real expert parameter: {n}")
        membership = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert all(id(p) in membership for _, p in owned)
        gpu_names = [next((n,p) for n,p in shards if ".layers.0.mlp.gate.weight" in n),
                     next((n,p) for n,p in shards if ".self_attn.q_proj.weight" in n)]
        self.samples = [("cpu_expert", *cpu[0])] if cpu else []
        self.samples.extend((role,n,p) for role,(n,p) in zip(("gpu_router","gpu_attention"),gpu_names))
        save(self.out / f"inventory-rank{self.rank}.json", dict(
            fsdp2=True, kt=True, wrappers=len(model._kt_wrappers), cpu_parameters=[
                {"name":n, "shape":list(p.shape), "dtype":str(p.dtype),"device":str(p.device)} for n,p in cpu],
            fsdp_shards=len(shards), rank_local_placeholders=placeholders,
            optimizer_class=type(getattr(optimizer,"optimizer",optimizer)).__name__,
            normalization="native Trainer divides by GAS, FSDP2 averages over ranks, KT owner applies 1/world on native backward"))
        self.probe_before = self.probe(model)
        self.event("begin", wrappers=len(model._kt_wrappers), cpu_owned=len(cpu))

    def on_pre_optimizer_step(self, args, state, control, optimizer, **kwargs):
        self.before = []
        for role, name, p in self.samples:
            flat = local_tensor(p.detach()).reshape(-1)
            grad = local_tensor(p.grad).reshape(-1)
            idx = torch.linspace(0,flat.numel()-1,min(8192,flat.numel()), device=flat.device,
                                 dtype=torch.float64).long()
            val = flat[idx].float().cpu()
            g = grad[idx].float().cpu()
            assert torch.isfinite(g).all() and g.count_nonzero(), (role, name)
            self.before.append((idx,val,g))
        save(self.out / f"preoptimizer-rank{self.rank}.json",
             {"grad_norm_preclip":self.trainer.grad_norm,"lr":[float(g["lr"]) for g in optimizer.param_groups]})
        dist.barrier()
        if self.rank == 0:
            with (self.out.parent / "optimizer-started.json").open("x") as f:
                json.dump({"attempt":self.out.name, "step":int(state.global_step)+1,"time":time.time()},f)
        dist.barrier()
        self.event("optimizer_start")

    def on_optimizer_step(self, args, state, control, optimizer, model, **kwargs):
        records = []
        for (role,name,p),(idx,old,grad) in zip(self.samples,self.before):
            flat = local_tensor(p.detach()).reshape(-1)
            new = flat[idx].float().cpu()
            changed = (old != new) & (grad != 0)
            st = optimizer.state[p]
            examples = []
            for k in changed.nonzero().flatten()[:12].tolist():
                i = int(idx[k].item())
                examples.append(dict(local_flat_index=i,expert_id=i//(p.shape[1]*p.shape[2]) if role=="cpu_expert" else None,
                    before=old[k].item(), after=new[k].item(), clipped_grad=grad[k].item(),
                    exp_avg=local_tensor(st["exp_avg"]).reshape(-1)[i].item(),
                    exp_avg_sq=local_tensor(st["exp_avg_sq"]).reshape(-1)[i].item()))
            records.append(dict(role=role,name=name,device=str(flat.device),dtype=str(flat.dtype),
                local_shape=list(local_tensor(p).shape),sampled=len(idx),nonzero_gradient=int(grad.count_nonzero()),
                changed_with_nonzero_grad=int(changed.sum()),step=int(st["step"].item()),examples=examples))
        save(self.out / f"update-rank{self.rank}.json",records)
        self.event("optimizer_return")

    def on_step_end(self, args, state, control, model, **kwargs):
        assert state.global_step == 1
        control.should_training_stop = True
        control.should_save = True
        return control

    def check_postforward(self, model):
        now = self.probe(model)
        torch.save(now, self.out / f"postforward-logits-rank{self.rank}.pt")
        delta = now - self.probe_before
        save(self.out / f"postforward-rank{self.rank}.json",dict(
            logits_max_abs=delta.abs().max().item(),logits_l2=delta.norm().item(),
            dirty_experts=sum(bool(getattr(w.wrapper,"_base_weights_dirty",False)) for w in model._kt_wrappers),
            gpu_allocated_peak=torch.cuda.max_memory_allocated(),gpu_reserved_peak=torch.cuda.max_memory_reserved()))
        self.event("step_verified", step=1)


class GRPOTrainer(Trainer):
    def __init__(self, dataset, evidence, **kwargs):
        self.evidence, self.seen = evidence, []
        super().__init__(train_dataset=dataset, callbacks=[evidence], **kwargs)
        self.model_accepts_loss_kwargs = False
        self.grad_norm = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        uid, ids = inputs["uid"], inputs["input_ids"]
        n = inputs["response_length"]
        targets = ids[0,-n:]
        def token_logp():
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=n+1).logits[0,:-1]
            return TokenLogProbs.apply(logits, targets)
        self.evidence.event("old_logp_start", uid=uid, response_tokens=n)
        model.eval()
        with torch.no_grad():
            old = token_logp()
        model.train()
        assert torch.isfinite(old).all()
        torch.save(old.cpu(), self.evidence.out / "old_logp" / (uid + ".pt"))
        self.evidence.event("actor_forward_start", uid=uid)
        current = token_logp()
        assert torch.isfinite(current).all()
        # Trainer divides by GAS; FSDP and KT divide by world size once.
        numerator = policy_loss_sum(current, old, inputs["advantage"], torch.ones_like(targets,dtype=torch.bool))
        loss = numerator * dist.get_world_size() * self.current_gradient_accumulation_steps / self.train_dataset.valid_tokens
        self.seen.append(uid)
        self.evidence.event("backward_start", uid=uid, loss=loss.detach().item(), old_new_max=(current.detach()-old).abs().max().item())
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        torch.cuda.synchronize()
        self.evidence.event("backward_complete", uid=inputs["uid"], microbatch=len(self.seen))
        return loss

    def _clip_grad_norm(self, model):
        value = super()._clip_grad_norm(model)
        self.grad_norm = float(value.item())
        if not 0 < self.grad_norm < float("inf"):
            raise RuntimeError(f"invalid full gradient norm: {self.grad_norm}")
        return value


def main(config_path, batch_dir, folder, restore_from=None):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=75), device_id=torch.device("cuda",rank))
    plan = load_plan(config_path)
    world = dist.get_world_size()
    assert world == plan["actor"]["world_size"] == 8
    dataset = FixedRows(batch_dir)
    assert len(dataset) == plan["inputs"]["responses_per_update"] == 128
    evidence = Evidence(folder, dataset)
    (folder/"old_logp").mkdir(exist_ok=True)
    evidence.event("load_start")
    validate_loss_math(torch.device("cuda", rank))
    kt = dict(plan["actor"]["kt_config"],kt_skip_expert_loading=True,kt_lora_rank=0,
              kt_model_max_length=max(len(r["prompt"])+len(r["response"]) for r in dataset.rows),
              kt_activation_policy={"cpu":"recompute","gpu":"recompute"},kt_max_cache_depth=1)
    if restore_from:
        kt["kt_expert_checkpoint_path"] = str(restore_from)
    args = TrainingArguments(output_dir=str(folder/"trainer"), per_device_train_batch_size=1,
        gradient_accumulation_steps=len(dataset)//world, learning_rate=1e-6, weight_decay=0.01,
        adam_beta1=0.9,adam_beta2=0.999,adam_epsilon=1e-8,optim="adamw_torch",max_grad_norm=1.0,
        lr_scheduler_type="constant",max_steps=418,logging_steps=1,save_strategy="steps",save_steps=1,report_to="none",
        dataloader_num_workers=0, dataloader_pin_memory=False,remove_unused_columns=False,
        bf16=True,seed=42,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False,"context_fn":checkpoint_context},
        fsdp="full_shard auto_wrap",fsdp_config={"version":2,"cpu_ram_efficient_loading":True,
           "reshard_after_forward":True,"transformer_layer_cls_to_wrap":["Qwen3_5MoeDecoderLayer","Qwen3_5MoeVisionBlock"]},
        kt_config=kt)
    model = AutoModelForImageTextToText.from_pretrained(str(restore_from or plan["model"]["path"]),dtype=torch.bfloat16,
                device_map={"":f"cuda:{rank}"},attn_implementation="sdpa",local_files_only=True)
    assert len(model._kt_wrappers)==40
    model.config.use_cache=False
    evidence.event("loaded",wrappers=len(model._kt_wrappers))
    trainer = GRPOTrainer(dataset,evidence,model=model,args=args,data_collator=collate)
    evidence.trainer=trainer
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    if restore_from:
        from .restore import restore_only
        restore_only(trainer, evidence, restore_from)
        dist.destroy_process_group()
        return
    trainer.train()
    evidence.check_postforward(trainer.model)
    if rank == 0:
        save(folder/"native-expert-profile.json", collect_kt_sft_profile(trainer.model))
    assert trainer.state.global_step==1 and len(trainer.seen)==len(dataset)//world
    all_uids=[None]*world
    dist.all_gather_object(all_uids,trainer.seen)
    assert sorted(uid for part in all_uids for uid in part)==sorted(r["uid"] for r in dataset.rows)
    dist.barrier()
    if rank==0:
        all_records=[x for i in range(world) for x in json.loads((folder/f"update-rank{i}.json").read_text())]
        for role in ("cpu_expert","gpu_router","gpu_attention"):
            assert any(x["role"]==role and x["step"]==1 and x["examples"] for x in all_records),role
        for i in range(world):
            probe=json.loads((folder/f"postforward-rank{i}.json").read_text())
            assert probe["logits_max_abs"]>0 and probe["dirty_experts"]==0
        checkpoint = folder/"trainer/checkpoint-1"
        for key in ("kt_optimizer.index.json","kt_full_experts.index.json","trainer_state.json","scheduler.pt"):
            assert (checkpoint/key).is_file(), key
        save(folder/"accepted.json",dict(world=world,updates=1,roles=[x["role"] for x in all_records],
            unique_responses=128,valid_response_tokens=dataset.valid_tokens,optimizer_schedule_steps=418))
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("config","batch-dir","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--restore-from",type=Path)
    opts=parser.parse_args()
    main(opts.config,opts.batch_dir,opts.out,opts.restore_from)

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
from ktrl.trainer.gradient_audit import watch_loss_scale, gradient_snapshot, verify_clip
from ktrl.trainer.master_adamw import MasterAdamW
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
            "response_length": len(row["response"]), "uid": row["uid"],
            "rollout_logprobs": torch.tensor(row.get('rollout_logprobs',[]),dtype=torch.float32),
            "policy_version": row.get('policy_version')}


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
            # d log p(label) / d logits = one_hot(label) - softmax(logits).
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
    def __init__(self, folder, dataset, start_step=0):
        self.out, self.data = folder, dataset
        self.start_step, self.target_step = start_step, start_step+1
        self.rank = dist.get_rank()
        self.trainer = None
        self.samples = []
        self.probe_before = None
        self.resident_rounds = None

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
        self.cpu_parameters = cpu
        assert state.global_step == self.start_step
        if self.start_step:
            assert trainer.lr_scheduler.last_epoch == self.start_step
            assert optimizer.state, 'Resume must restore nonempty Adam state'
            for record in optimizer.state.values():
                if 'step' in record:
                    assert int(record['step']) == self.start_step
            previous = self.previous_round if self.resident_rounds else self.resume_checkpoint.parent.parent
            records = json.loads((previous/f'update-rank{self.rank}.json').read_text())
            named = dict(model.named_parameters())
            named.update(owned)
            checked = 0
            for record in records:
                parameter = named[record['name']]
                slots = optimizer.state[parameter]
                assert int(slots['step']) == self.start_step
                for point in record['examples']:
                    i = point['local_flat_index']
                    assert local_tensor(parameter).detach().reshape(-1)[i].item() == point['after']
                    for key in ('exp_avg','exp_avg_sq')+ (('master',) if record['role']=='cpu_expert' else ()):
                        assert local_tensor(slots[key]).reshape(-1)[i].item() == point[key]
                    checked += 1
            assert checked > 0
            save(self.out/f'resume-rank{self.rank}.json',dict(checkpoint=str(self.resume_checkpoint),
                restored_step=self.start_step,scheduler_epoch=trainer.lr_scheduler.last_epoch,
                exact_parameter_and_moment_positions=checked,verified_before_backward=True,
                resident=bool(self.resident_rounds),pid=os.getpid()))
        assert len(cpu) == (120 if self.rank == 0 else 0), (self.rank, len(cpu))
        shards = [(n, p) for n, p in model.named_parameters() if hasattr(p, "to_local")]
        assert all(p.dtype==torch.float32 for _,p in shards if p.requires_grad)
        policy=trainer.accelerator.state.fsdp_plugin.mixed_precision_policy
        assert policy.param_dtype==torch.bfloat16 and policy.reduce_dtype==torch.float32
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
        assert self.trainer.scale_checks == len(self.data)//dist.get_world_size()
        assert self.trainer.clip_audit_passed, 'Native unified clipping was not verified'
        self.before = []
        self.master_before = []
        for role, name, p in self.samples:
            flat = local_tensor(p.detach()).reshape(-1)
            grad = local_tensor(p.grad).reshape(-1)
            idx = torch.linspace(0,flat.numel()-1,min(8192,flat.numel()), device=flat.device,
                                 dtype=torch.float64).long()
            val = flat[idx].float().cpu()
            g = grad[idx].float().cpu()
            assert torch.isfinite(g).all() and g.count_nonzero(), (role, name)
            self.before.append((idx,val,g))
            st=optimizer.state.get(p,{})
            self.master_before.append(local_tensor(st['master']).reshape(-1)[idx].float().cpu()
                                      if 'master' in st else val.clone())
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
        for (role,name,p),(idx,old,grad),master_old in zip(self.samples,self.before,self.master_before):
            flat = local_tensor(p.detach()).reshape(-1)
            new = flat[idx].float().cpu()
            changed = (old != new) & (grad != 0)
            st = optimizer.state[p]
            master_new=local_tensor(st['master']).reshape(-1)[idx].float().cpu() if role=='cpu_expert' else new
            master_changed=(master_new!=master_old)&(grad!=0)
            examples = []
            for k in changed.nonzero().flatten()[:12].tolist():
                i = int(idx[k].item())
                examples.append(dict(local_flat_index=i,expert_id=i//(p.shape[1]*p.shape[2]) if role=="cpu_expert" else None,
                    before=old[k].item(), after=new[k].item(), clipped_grad=grad[k].item(),
                    exp_avg=local_tensor(st["exp_avg"]).reshape(-1)[i].item(),
                    exp_avg_sq=local_tensor(st["exp_avg_sq"]).reshape(-1)[i].item()))
                if role=='cpu_expert':examples[-1]['master']=st['master'].reshape(-1)[i].item()
            records.append(dict(role=role,name=name,device=str(flat.device),dtype=str(flat.dtype),
                local_shape=list(local_tensor(p).shape),sampled=len(idx),nonzero_gradient=int(grad.count_nonzero()),
                changed_with_nonzero_grad=int(changed.sum()),master_changed_with_nonzero_grad=int(master_changed.sum()),
                master_change_fraction=float(master_changed.sum()/grad.count_nonzero()),
                master_dtype=str(st['master'].dtype) if role=='cpu_expert' else str(flat.dtype),
                moment_dtypes={k:str(st[k].dtype) for k in ('exp_avg','exp_avg_sq')},
                step=int(st["step"].item()),examples=examples))
        save(self.out / f"update-rank{self.rank}.json",records)
        self.event("optimizer_return")

    def on_step_end(self, args, state, control, model, **kwargs):
        assert state.global_step == self.target_step
        if self.resident_rounds:
            trainer = self.trainer
            control.should_save = False
            trainer.args.output_dir = str(self.out / "trainer")
            trainer._save_checkpoint(model, None)
            finish_round(trainer, self, model, model_only=True, synchronize=True)
            if state.global_step == self.resident_rounds:
                control.should_training_stop = True
                return control
            self.previous_round = self.out
            next_round = self.out.parent.parent / f"round-{state.global_step + 1:02d}"
            deadline = time.monotonic() + 7200
            while not (next_round / "actor-ready.json").is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"No training batch released for {next_round}")
                time.sleep(0.2)
            dataset = FixedRows(next_round / "batch")
            validate_rows(dataset, self.plan)
            self.data.rows, self.data.valid_tokens = dataset.rows, dataset.valid_tokens
            self.out = next_round / "actor"
            (self.out / "old_logp").mkdir(exist_ok=True)
            self.start_step, self.target_step = state.global_step, state.global_step + 1
            trainer.seen.clear()
            trainer.scale_checks = 0
            trainer.clip_audit_passed = False
            self.on_train_begin(args, state, control, model, trainer.optimizer)
            return control
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
        self.event("step_verified", step=self.target_step)

    def check_update_direction(self,model):
        model.eval()
        lookup={r['uid']:r for r in self.data.rows}
        records=[]
        with torch.no_grad():
            for uid in self.trainer.seen:
                row=lookup[uid];n=len(row['response'])
                ids=torch.tensor([row['prompt']+row['response']],device=torch.cuda.current_device())
                logits=model(input_ids=ids,use_cache=False,logits_to_keep=n+1).logits[0,:-1]
                new=TokenLogProbs.apply(logits,ids[0,-n:]).float().cpu()
                old=torch.load(self.out/'old_logp'/f'{uid}.pt',weights_only=True)
                delta=(new-old).double()
                assert torch.isfinite(delta).all()
                records.append(dict(uid=uid,advantage=row['advantage'],tokens=n,
                    mean_delta=delta.mean().item(),mean_absolute_delta=delta.abs().mean().item(),
                    weighted_delta_sum=(delta*row['advantage']).sum().item()))
                del logits,ids,new,old,delta
        gathered=[None]*dist.get_world_size()
        dist.all_gather_object(gathered,records)
        all_rows=[r for part in gathered for r in part]
        total=sum(r['weighted_delta_sum'] for r in all_rows)
        if self.rank==0:
            groups={}
            for name,sign in [('positive',1),('negative',-1),('zero',0)]:
                selected=[r for r in all_rows if (1 if r['advantage']>0 else -1 if r['advantage']<0 else 0)==sign]
                count=sum(r['tokens'] for r in selected)
                groups[name]=dict(responses=len(selected),tokens=count,
                    mean_delta=sum(r['mean_delta']*r['tokens'] for r in selected)/count if count else None,
                    mean_absolute_delta=sum(r['mean_absolute_delta']*r['tokens'] for r in selected)/count if count else None)
            save(self.out/'update-direction.json',dict(weighted_delta_sum=total,groups=groups,rows=all_rows,
                note='Zero-advantage samples can change through shared parameters; per-answer signs are diagnostics, not universal guarantees'))
        assert total>0, f'Observed GRPO proxy did not increase: {total}'
        self.event('update_direction_verified',weighted_delta_sum=total)


class GRPOTrainer(Trainer):
    def _save_checkpoint(self, model, trial):
        started = time.monotonic()
        super()._save_checkpoint(model, trial)
        self.evidence.event("checkpoint_saved", seconds=time.monotonic() - started,
                            model_only=self.args.save_only_model)

    def create_optimizer(self,model=None):
        if self.optimizer is None:
            self.optimizer_cls_and_kwargs=(MasterAdamW,dict(lr=self.args.learning_rate,
                betas=(self.args.adam_beta1,self.args.adam_beta2),eps=self.args.adam_epsilon,
                cpu_parameters=getattr(self,'_kt_rank_local_optimizer_parameters',()),cpu_threads=48))
        return super().create_optimizer(model)

    def __init__(self, dataset, evidence, **kwargs):
        self.evidence, self.seen = evidence, []
        super().__init__(train_dataset=dataset, callbacks=[evidence], **kwargs)
        self.model_accepts_loss_kwargs = False
        self.grad_norm = None
        self.scale_checks = 0
        self.clip_audit_passed = False

    def record_scale(self, record):
        self.scale_checks += 1
        self.evidence.event('backward_loss_scale', **record)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        uid, ids = inputs["uid"], inputs["input_ids"]
        if uid in self.seen:
            raise RuntimeError('A round permits one update only; old probabilities must stay frozen')
        assert self.state.global_step == self.evidence.start_step
        if inputs['policy_version'] is not None:
            assert inputs['policy_version']==f'v{self.evidence.start_step}'
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
        sampling = inputs['rollout_logprobs']
        if sampling.numel():
            assert sampling.shape == old.shape and torch.isfinite(sampling).all()
            delta=(old-sampling.to(old.device)).detach()
            sequence_log_ratio=delta.double().sum().item()
            self.evidence.event('actor_rollout_logprob_difference',uid=uid,
                mean_abs=delta.abs().mean().item(),max_abs=delta.abs().max().item(),
                absolute_quantiles=torch.quantile(delta.abs(),torch.tensor([.5,.95,.99],device=delta.device)).tolist(),
                sequence_log_ratio=sequence_log_ratio,
                sequence_ratio=__import__('math').exp(sequence_log_ratio) if sequence_log_ratio<700 else None,
                sequence_ratio_overflow=sequence_log_ratio>=700,
                length_bucket='0-2k' if n<=2048 else '2-4k' if n<=4096 else '4-8k',
                log_ratio_quantiles=torch.quantile(delta,torch.tensor([0.,.5,.95,1.],device=delta.device)).tolist())
        self.evidence.event("actor_forward_start", uid=uid)
        current = token_logp()
        assert torch.isfinite(current).all()
        # Trainer divides by GAS; FSDP and KT divide by world size once.
        numerator = policy_loss_sum(current, old, inputs["advantage"], torch.ones_like(targets,dtype=torch.bool))
        loss = numerator * dist.get_world_size() * self.current_gradient_accumulation_steps / self.train_dataset.valid_tokens
        watch_loss_scale(loss,self.current_gradient_accumulation_steps,self.record_scale)
        self.seen.append(uid)
        self.evidence.event("backward_start", uid=uid, loss=loss.detach().item(), old_new_max=(current.detach()-old).abs().max().item())
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        torch.cuda.synchronize()
        self.evidence.event("backward_complete", uid=inputs["uid"], microbatch=len(self.seen))
        return loss

    def _clip_grad_norm(self, model):
        # Synchronous audit before native clipping, not the post-clip callback.
        torch.cuda.synchronize()
        owned = self.evidence.cpu_parameters
        names = {id(p):n for n,p in model.named_parameters()}
        names.update({id(p):n for n,p in owned})
        parameters = [(names.get(id(p),f'optimizer-{i}-{j}'),p)
            for i,g in enumerate(self.optimizer.param_groups) for j,p in enumerate(g['params'])]
        local_sum, rows = gradient_snapshot(parameters,{id(p) for _,p in owned},dist.get_rank())
        total = torch.tensor(local_sum,dtype=torch.float64,device=torch.cuda.current_device())
        dist.all_reduce(total)
        expected_norm = total.sqrt().item()
        value = super()._clip_grad_norm(model)
        self.grad_norm = float(value.item())
        if not 0 < self.grad_norm < float("inf"):
            raise RuntimeError(f"invalid full gradient norm: {self.grad_norm}")
        proof = verify_clip(rows,expected_norm,self.grad_norm,self.args.max_grad_norm)
        save(self.evidence.out/f'joint-clip-rank{dist.get_rank()}.json',proof)
        self.clip_audit_passed = True
        return value


def validate_rows(dataset, plan):
    assert len(dataset) == plan["inputs"]["responses_per_update"]
    assert len(dataset) // plan["actor"]["world_size"] == plan["gradient_accumulation_steps"]
    for row in dataset.rows:
        assert 0 < len(row["response"]) <= plan["inputs"]["max_response_tokens"]
        assert len(row["prompt"]) + len(row["response"]) <= plan["inputs"]["context_length"]


def main(config_path, batch_dir, folder, restore_from=None, resume_training=False,
         validation=False, save_model_only=False, resident_rounds=None):
    if resume_training and not (restore_from / "kt_optimizer.index.json").is_file():
        raise ValueError(f"Training resume requires a full KT checkpoint: {restore_from}")
    # Online updates remain blocked until the revised teacher-review gate is
    # explicitly produced by the completed tests, not by submitting a run.
    if validation:
        validation_root = folder.parent.parent if resident_rounds else folder.parent
        assert validation_root.name.startswith('validation-fixed-') and not resume_training
    elif json.loads((batch_dir/'batch-audit.json').read_text()).get('policy_version') is not None:
        from ktrl.trainer.precision_gate import verify
        verify(config_path)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=75), device_id=torch.device("cuda",rank))
    plan = load_plan(config_path)
    world = dist.get_world_size()
    assert world == plan["actor"]["world_size"] == 8
    dataset = FixedRows(batch_dir)
    validate_rows(dataset, plan)
    start_step = json.loads((restore_from/'trainer_state.json').read_text())['global_step'] if resume_training else 0
    evidence = Evidence(folder, dataset,start_step)
    evidence.resume_checkpoint = restore_from
    evidence.resident_rounds = resident_rounds
    evidence.resident_smoke = validation and bool(resident_rounds)
    evidence.plan = plan
    (folder/"old_logp").mkdir(exist_ok=True)
    evidence.event("load_start")
    validate_loss_math(torch.device("cuda", rank))
    kt = dict(plan["actor"]["kt_config"],kt_skip_expert_loading=True,kt_lora_rank=0,
              kt_model_max_length=(plan["inputs"]["context_length"] if resident_rounds
                                   else max(len(r["prompt"])+len(r["response"]) for r in dataset.rows)),
              kt_activation_policy={"cpu":"recompute","gpu":"recompute"},kt_max_cache_depth=1)
    if restore_from:
        kt["kt_expert_checkpoint_path"] = str(restore_from)
    args = TrainingArguments(output_dir=str(folder/"trainer"), per_device_train_batch_size=1,
        gradient_accumulation_steps=len(dataset)//world, learning_rate=1e-6, weight_decay=0.01,
        adam_beta1=0.9,adam_beta2=0.999,adam_epsilon=1e-8,optim="adamw_torch",max_grad_norm=1.0,
        lr_scheduler_type="constant",max_steps=plan['optimizer_schedule_steps'],logging_steps=1,save_strategy="steps",save_steps=1,report_to="none",
        dataloader_num_workers=0, dataloader_pin_memory=False,remove_unused_columns=False,
        accelerator_config={'gradient_accumulation_kwargs':{'sync_each_batch':True}},
        bf16=True,seed=plan['inputs']['seed'],ignore_data_skip=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False,"context_fn":checkpoint_context},
        fsdp="full_shard auto_wrap",fsdp_config={"version":2,"cpu_ram_efficient_loading":True,
           "reshard_after_forward":True,"transformer_layer_cls_to_wrap":["Qwen3_5MoeDecoderLayer","Qwen3_5MoeVisionBlock"]},
        kt_config=kt)
    master_dtype = getattr(torch,plan['actor']['nonexpert_parameter_dtype'])
    model = AutoModelForImageTextToText.from_pretrained(str(restore_from or plan["model"]["path"]),
                dtype=master_dtype if restore_from else torch.bfloat16,
                device_map={"":f"cuda:{rank}"},attn_implementation="sdpa",local_files_only=True)
    # LlamaFactory full tuning upcasts registered trainable parameters; native KT
    # owns its BF16 CPU expert buffers separately. Never upcast/migrate those buffers.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.numel():
            assert parameter.device.type=='cuda'
            parameter.data=parameter.data.to(master_dtype)
    assert len(model._kt_wrappers)==40
    model.config.use_cache=False
    evidence.event("loaded",wrappers=len(model._kt_wrappers))
    trainer = GRPOTrainer(dataset,evidence,model=model,args=args,data_collator=collate)
    from torch.distributed.fsdp import MixedPrecisionPolicy
    trainer.accelerator.state.fsdp_plugin.mixed_precision_policy=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
    evidence.trainer=trainer
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Trainer initializes the FSDP plugin in sharded mode and rejects model-only
    # saves in that mode. Enable them after selecting the native full export.
    trainer.args.save_only_model = save_model_only or bool(resident_rounds)
    if restore_from and not resume_training:
        from .restore import restore_only
        restore_only(trainer, evidence, restore_from)
        dist.destroy_process_group()
        return
    trainer.train(resume_from_checkpoint=str(restore_from) if resume_training else None)
    if not resident_rounds:
        finish_round(trainer, evidence, trainer.model, model_only=save_model_only)
    dist.destroy_process_group()


def finish_round(trainer, evidence, model, *, model_only, synchronize=False):
    folder, dataset = evidence.out, evidence.data
    rank, world = dist.get_rank(), dist.get_world_size()
    evidence.check_postforward(model)
    evidence.check_update_direction(model)
    if rank == 0:
        save(folder/"native-expert-profile.json", collect_kt_sft_profile(model))
    assert trainer.state.global_step==evidence.target_step and len(trainer.seen)==len(dataset)//world
    all_uids=[None]*world
    dist.all_gather_object(all_uids,trainer.seen)
    assert sorted(uid for part in all_uids for uid in part)==sorted(r["uid"] for r in dataset.rows)
    dist.barrier()
    if synchronize:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        save(folder/f"resident-rank{rank}.json", dict(pid=os.getpid(), step=trainer.state.global_step,
             scheduler_epoch=trainer.lr_scheduler.last_epoch, optimizer_id=id(trainer.optimizer),
             allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved()))
        dist.barrier()
    if rank==0:
        all_records=[x for i in range(world) for x in json.loads((folder/f"update-rank{i}.json").read_text())]
        for role in ("cpu_expert","gpu_router","gpu_attention"):
            assert any(x["role"]==role and x["step"]==evidence.target_step and x["examples"] for x in all_records),role
        for record in all_records:
            assert record['master_dtype']=='torch.float32'
            assert set(record['moment_dtypes'].values())=={'torch.float32'}
            fraction = record["master_change_fraction"]
            if evidence.resident_smoke:
                assert fraction > 0, record["role"]
            else:
                assert fraction >= .99, (record["role"], fraction)
        for i in range(world):
            probe=json.loads((folder/f"postforward-rank{i}.json").read_text())
            assert probe["logits_max_abs"]>0 and probe["dirty_experts"]==0
        checkpoint = folder/f"trainer/checkpoint-{evidence.target_step}"
        for key in ("kt_full_experts.index.json", "trainer_state.json"):
            assert (checkpoint/key).is_file(), key
        assert any(checkpoint.glob("model*.safetensors")), "Missing nonexpert weights"
        for key in ("kt_optimizer.index.json", "scheduler.pt"):
            assert (checkpoint/key).is_file() != model_only, key
        save(folder/"accepted.json",dict(world=world,updates=1,roles=[x["role"] for x in all_records],
            start_step=evidence.start_step,end_step=evidence.target_step,unique_responses=len(dataset),
            checkpoint_kind="model_only" if model_only else "full",
            valid_response_tokens=dataset.valid_tokens,optimizer_schedule_steps=evidence.plan['optimizer_schedule_steps']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("config","batch-dir","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--restore-from",type=Path)
    parser.add_argument("--resume-training",action='store_true')
    parser.add_argument("--save-model-only", action="store_true")
    parser.add_argument("--resident-rounds", type=int, choices=(1, 2))
    parser.add_argument('--validation-run',action='store_true',help='Only an isolated validation-fixed-* directory; never an online round')
    opts=parser.parse_args()
    if opts.resume_training and opts.restore_from is None:
        parser.error('--resume-training requires --restore-from')
    if opts.resident_rounds and opts.restore_from:
        parser.error("--resident-rounds requires a fresh online run")
    main(opts.config,opts.batch_dir,opts.out,opts.restore_from,opts.resume_training,
         opts.validation_run,opts.save_model_only,opts.resident_rounds)

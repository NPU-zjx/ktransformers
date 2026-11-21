#!/usr/bin/env python3
"""2-GPU synthetic probe for Torch 2.9.1 + Accelerate 1.14 FSDP2 with reduce_dtype=bfloat16."""

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import Trainer, TrainerCallback, TrainingArguments

from ktrl.trainer.gradient_audit import gradient_snapshot, verify_clip
from ktrl.trainer.master_adamw import MasterAdamW


class TinyBlock(torch.nn.Module):
    def __init__(self, dim=16, init_val=0.25):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((dim, dim), init_val, dtype=torch.float32))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class TinyModel(torch.nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.block1 = TinyBlock(dim, 0.25)
        self.block2 = TinyBlock(dim, 0.125)

    def forward(self, x):
        return self.block2(self.block1(x))


def run_trainer_accum_case(sync_each_batch: bool, out_dir: Path) -> dict:
    """Test actual Transformers Trainer + Accelerate FSDP2 cross-microbatch behavior."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    case_dir = out_dir / f"sync_{sync_each_batch}"
    case_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    model = TinyModel(16)

    substep_grad_dtypes = []
    substep_grad_means = []
    preopt_record = {}
    postopt_record = {}

    class ProbeCallback(TrainerCallback):
        def on_substep_end(self, args, state, control, model=None, **kwargs):
            g = model.block1.weight.grad
            substep_grad_dtypes.append(str(g.dtype) if g is not None else None)
            substep_grad_means.append(float(g.to_local().float().mean().item()) if g is not None else None)

        def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
            g1 = model.block1.weight.grad
            p1 = model.block1.weight
            preopt_record["param_dtype"] = str(p1.dtype)
            preopt_record["grad_dtype_pre_opt"] = str(g1.dtype)
            preopt_record["grad_mean_pre_opt"] = float(g1.to_local().float().mean().item())
            preopt_record["param_before_mean"] = float(p1.to_local().float().mean().item())

        def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
            p1 = model.block1.weight
            after_mean = float(p1.to_local().float().mean().item())
            postopt_record["param_after_mean"] = after_mean
            postopt_record["gpu_param_changed"] = (after_mean != preopt_record["param_before_mean"])

    class ProbeTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            out = super().training_step(model, inputs, num_items_in_batch)
            g = model.block1.weight.grad
            substep_grad_dtypes.append(str(g.dtype) if g is not None else None)
            substep_grad_means.append(float(g.to_local().float().mean().item()) if g is not None else None)
            return out

        def compute_loss(self, model, inputs, **kwargs):
            # Simple linear loss on block1.weight via block2:
            # block1 output: x @ w1^T, sum = sum_j (sum_i x_i * w1_ji), but we pass x of ones*(val*(rank+1))
            scale = float(inputs["val"]) * (rank + 1)
            x = torch.full((1, 16), scale, device=self.args.device, dtype=torch.bfloat16)
            return model(x).sum()

    args = TrainingArguments(
        output_dir=str(case_dir / "trainer"),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-2,
        max_grad_norm=1.0,
        bf16=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        accelerator_config={"gradient_accumulation_kwargs": {"sync_each_batch": sync_each_batch}},
        fsdp="full_shard auto_wrap",
        fsdp_config={"version": 2, "reshard_after_forward": True, "transformer_layer_cls_to_wrap": ["TinyBlock"]},
    )
    trainer = ProbeTrainer(
        model=model,
        args=args,
        train_dataset=[{"val": 1.0}, {"val": 3.0}],
        data_collator=lambda rows: rows[0],
        callbacks=[ProbeCallback()],
    )
    trainer.model_accepts_loss_kwargs = False
    trainer.accelerator.state.fsdp_plugin.mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    )
    trainer.train()
    dist.barrier()

    return {
        "rank": rank,
        "world_size": world,
        "sync_each_batch": sync_each_batch,
        "substep_grad_dtypes": substep_grad_dtypes,
        "substep_grad_means": substep_grad_means,
        "preopt": preopt_record,
        "postopt": postopt_record,
    }


def run_joint_clip_and_restore_case() -> dict:
    """Test FSDP2(bf16 reduce) + rank-0 CPU MasterAdamW joint clip, update, and exact restore."""
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=True,
        mixed_precision_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
    )
    accelerator = Accelerator(fsdp_plugin=plugin, mixed_precision="bf16")
    rank = accelerator.process_index

    torch.manual_seed(7)
    model = TinyModel(16)
    model = accelerator.prepare(model)

    cpu_expert = torch.nn.Parameter(torch.full((4, 8), 0.5, dtype=torch.bfloat16)) if rank == 0 else None
    cpu_list = [cpu_expert] if cpu_expert is not None else []
    optimizer = MasterAdamW(list(model.parameters()) + cpu_list, cpu_parameters=cpu_list, cpu_threads=4, lr=1e-2, weight_decay=0.0)
    optimizer = accelerator.prepare(optimizer)

    # Microbatch 1 & 2 manual accumulation with GAS=2 scaling
    x1 = torch.ones((1, 16), device=accelerator.device, dtype=torch.bfloat16) * (rank + 1)
    accelerator.backward(model(x1).sum() / 2.0)
    g1 = next(model.parameters()).grad
    mb1_dtype = str(g1.dtype)
    mb1_mean = float(g1.to_local().float().mean().item())

    x2 = torch.ones((1, 16), device=accelerator.device, dtype=torch.bfloat16) * 3.0 * (rank + 1)
    accelerator.backward(model(x2).sum() / 2.0)
    g2 = next(model.parameters()).grad
    mb2_dtype = str(g2.dtype)
    mb2_mean = float(g2.to_local().float().mean().item())

    if cpu_expert is not None:
        cpu_expert.grad = torch.full_like(cpu_expert, 2.0, dtype=torch.bfloat16)

    named = list(model.named_parameters()) + ([("cpu_expert", cpu_expert)] if cpu_expert is not None else [])
    val, rows = gradient_snapshot(named, {id(cpu_expert)} if cpu_expert is not None else set(), rank)
    sq = torch.tensor(val, dtype=torch.float64, device=accelerator.device)
    dist.all_reduce(sq)
    expected_norm = math.sqrt(float(sq.item()))

    norm = accelerator.clip_grad_norm_(model.parameters(), 1.0, rank_local_parameters=cpu_list)
    proof = verify_clip(rows, expected_norm, float(norm), 1.0)

    gpu_p = next(model.parameters())
    gpu_before = gpu_p.to_local().detach().clone()
    cpu_before = cpu_expert.detach().clone() if cpu_expert is not None else None

    optimizer.step()
    optimizer.zero_grad()

    gpu_after = gpu_p.to_local().detach().clone()
    gpu_changed = not torch.equal(gpu_before, gpu_after)

    cpu_info = {}
    if cpu_expert is not None:
        st = optimizer.state[cpu_expert]
        cpu_changed = not torch.equal(cpu_before, cpu_expert.detach())
        saved_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        saved_opt = optimizer.state_dict()

        # Verify exact restore into a fresh MasterAdamW + model
        cpu_clone = torch.nn.Parameter(cpu_expert.detach().clone())
        opt2 = MasterAdamW(
            list(model.parameters()) + [cpu_clone],
            cpu_parameters=[cpu_clone],
            cpu_threads=4,
            lr=1e-2,
            weight_decay=0.0,
        )
        opt2.load_state_dict(saved_opt)
        st2 = opt2.state[cpu_clone]
        model.load_state_dict(saved_model)
        cpu_info = {
            "cpu_expert_changed": cpu_changed,
            "cpu_param_dtype": str(cpu_expert.dtype),
            "cpu_master_dtype": str(st["master"].dtype),
            "cpu_exp_avg_dtype": str(st["exp_avg"].dtype),
            "cpu_exp_avg_sq_dtype": str(st["exp_avg_sq"].dtype),
            "master_adamw_restore_exact": bool(
                torch.equal(st["master"], st2["master"])
                and torch.equal(st["exp_avg"], st2["exp_avg"])
                and torch.equal(st["exp_avg_sq"], st2["exp_avg_sq"])
            ),
        }

    return {
        "gpu_shard_param_dtype": str(gpu_p.dtype),
        "mb1_grad_dtype": mb1_dtype,
        "mb1_grad_mean": mb1_mean,
        "mb2_accum_grad_dtype": mb2_dtype,
        "mb2_accum_grad_mean": mb2_mean,
        "expected_unclipped_norm": expected_norm,
        "actual_clip_norm": float(norm),
        "clip_coefficient": proof["coefficient"],
        "gpu_shard_changed": gpu_changed,
        "cpu_rank0": cpu_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    with tempfile.TemporaryDirectory() as tmp:
        res_true = run_trainer_accum_case(True, Path(tmp))
        res_false = run_trainer_accum_case(False, Path(tmp))
    res_joint = run_joint_clip_and_restore_case()

    from torch.distributed.fsdp import fully_shard
    def _fingerprint(rdtype):
        w = torch.nn.Linear(4, 4, bias=False).to(device="cuda", dtype=torch.float32)
        with torch.no_grad():
            w.weight.fill_(1.0)
        fully_shard(w, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=rdtype))
        val = 1024.0 if local_rank == 0 else 4.0
        x = torch.full((1, 4), val / 4.0, device="cuda", dtype=torch.bfloat16)
        w(x).sum().backward()
        g1 = w.weight.grad
        mb1 = {"grad_dtype": str(g1.dtype), "val": float(g1.to_local().view(-1)[0].item())}
        x2 = torch.full((1, 4), 1.0, device="cuda", dtype=torch.bfloat16)
        w(x2).sum().backward()
        g2 = w.weight.grad
        mb2 = {"grad_dtype": str(g2.dtype), "val": float(g2.to_local().view(-1)[0].item())}
        return {"mb1": mb1, "mb2": mb2}

    fp_bf16 = _fingerprint(torch.bfloat16)
    fp_fp32 = _fingerprint(torch.float32)

    if local_rank == 0:
        summary = {
            "torch_version": torch.__version__,
            "accelerate_version": __import__("accelerate").__version__,
            "policy": {"param_dtype": "bfloat16", "reduce_dtype": "bfloat16"},
            "trainer_sync_each_batch_true": res_true,
            "trainer_sync_each_batch_false": res_false,
            "joint_clip_update_restore": res_joint,
            "reduce_dtype_fingerprint": {"bfloat16_reduce": fp_bf16, "float32_reduce": fp_fp32},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

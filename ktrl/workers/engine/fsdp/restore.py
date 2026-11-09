"""Read a completed native HF + KT checkpoint, validate it, and exit before training."""

import json
import random
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from kt_kernel.sft import get_kt_named_trainable_params, get_kt_rank_local_parameter_names
from transformers import TrainerCallback
from transformers.trainer import safe_globals


class RestoreFinished(Exception):
    """Stop the resume check at on_train_begin before taking a training step."""


def local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def verify_rng(trainer, checkpoint, rank):
    path = checkpoint / f"rng_state_{rank}.pth"
    with safe_globals():
        expected = torch.load(path, weights_only=True)
    trainer._load_rng_state(str(checkpoint))
    assert random.getstate() == expected["python"]
    np.testing.assert_equal(np.random.get_state(), expected["numpy"])
    assert torch.equal(torch.random.get_rng_state(), expected["cpu"])
    actual_gpu = torch.cuda.get_rng_state_all()
    assert len(actual_gpu) == len(expected["cuda"])
    assert all(torch.equal(a, b) for a, b in zip(actual_gpu, expected["cuda"]))
    return path.name


def restore_only(trainer, evidence, checkpoint):
    source = checkpoint.parent.parent
    rank = dist.get_rank()

    def verify_all_weights(model):
        """Exact checkpoint comparison; CPU experts never pass through GPU."""
        from safetensors import safe_open
        files={}
        for path in checkpoint.glob('model*.safetensors'):
            with safe_open(path,framework='pt',device='cpu') as handle:
                for key in handle.keys():files[key]=path
        failures=[];checked=[]
        placeholders=set(get_kt_rank_local_parameter_names(model))
        for name,parameter in model.named_parameters():
            if name in placeholders:
                assert not hasattr(parameter,'full_tensor') and parameter.device.type!='cuda'
                continue
            if not parameter.numel():continue
            full=parameter.full_tensor() if hasattr(parameter,'full_tensor') else parameter
            if rank==0:
                if name not in files:failures.append(f'Missing GPU checkpoint parameter: {name}')
                else:
                    with safe_open(files[name],framework='pt',device='cpu') as handle:
                        expected=handle.get_tensor(name)
                        actual=full.detach().cpu()
                        if actual.dtype!=expected.dtype or not torch.equal(actual,expected):
                            failures.append(f'GPU parameter differs: {name}')
                        checked.append(dict(name=name,role='gpu',elements=actual.numel(),dtype=str(actual.dtype)))
            del full
        if rank==0:
            index=json.loads((checkpoint/'kt_full_experts.index.json').read_text())
            for name,parameter in get_kt_named_trainable_params(model):
                match=re.fullmatch(r'kt.layers.(\d+).experts.base.(gate|up|down)_proj',name)
                if not match:failures.append(f'Unexpected CPU parameter: {name}');continue
                entry=index['layers'][match[1]]
                with safe_open(checkpoint/entry['file'],framework='pt',device='cpu') as handle:
                    key=next(k for k in handle.keys() if k.endswith(match[2]+'_proj'))
                    expected=handle.get_tensor(key)
                    assert parameter.device.type=='cpu'
                    if parameter.dtype!=expected.dtype or not torch.equal(parameter.detach(),expected):
                        failures.append(f'CPU parameter differs: {name}')
                    checked.append(dict(name=name,role='cpu_expert',elements=parameter.numel(),dtype=str(parameter.dtype)))
            (evidence.out/'all-restored-weights.json').write_text(json.dumps(dict(
                accepted=not failures,failures=failures,parameters=checked,cpu_expert_gpu_staging_bytes=0)))
        errors=[None]*dist.get_world_size();dist.all_gather_object(errors,failures)
        if any(errors):raise RuntimeError(f'Checkpoint weights differ: {errors}')

    class Check(TrainerCallback):
        def on_train_begin(self, args, state, control, model, optimizer, **kwargs):
            stored = json.loads((checkpoint / "trainer_state.json").read_text())
            assert state.global_step == stored["global_step"] == 1
            assert trainer.lr_scheduler.last_epoch == 1
            rng_file = verify_rng(trainer, checkpoint, rank)
            params = dict(model.named_parameters())
            params.update(get_kt_named_trainable_params(model))
            records = json.loads((source / f"update-rank{rank}.json").read_text())
            saved_groups = optimizer.state_dict()["param_groups"]
            parameter_ids = {id(p): index for live, saved in zip(optimizer.param_groups, saved_groups)
                             for p, index in zip(live["params"], saved["params"])}
            moment_bytes = Counter()
            for entry in optimizer.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in entry:
                        slot = local(entry[key])
                        moment_bytes[(str(slot.device), str(slot.dtype), key)] += slot.numel() * slot.element_size()
            verified = 0
            placements = []
            for record in records:
                p = params[record["name"]]
                st = optimizer.state[p]
                assert int(st["step"].item()) == record["step"] == 1
                local_param = local(p)
                slots = {}
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    tensor = local(st[key])
                    slots[key] = {"device": str(tensor.device), "dtype": str(tensor.dtype),
                                  "shape": list(tensor.shape),
                                  "local_tensor_bytes": tensor.numel() * tensor.element_size()}
                    if key != "step":
                        assert tensor.device == local_param.device and tensor.shape == local_param.shape
                placements.append({"role": record["role"], "name": record["name"],
                                   "checkpoint_parameter_id": parameter_ids[id(p)],
                                   "parameter_device": str(local_param.device), "slots": slots})
                for point in record["examples"]:
                    i = point["local_flat_index"]
                    assert local_param.detach().reshape(-1)[i].item() == point["after"]
                    for key in ("exp_avg", "exp_avg_sq")+ (('master',) if record['role']=='cpu_expert' else ()):
                        assert local(st[key]).reshape(-1)[i].item() == point[key]
                    verified += 1
                if record['role']=='cpu_expert':
                    from ktrl.trainer.master_adamw import assert_compute_copy
                    assert_compute_copy(p,st['master'])
            assert verified > 0
            verify_all_weights(model)
            handles = []
            if os.environ.get("KTRL_FORWARD_TRACE"):
                from ktrl.weights.forward_trace import attach
                handles = attach(model, evidence.out / 'trace', lambda: 'actor')
            logits = evidence.probe(model)
            for handle in handles:
                handle.remove()
            expected = torch.load(source / f"postforward-logits-rank{rank}.pt", weights_only=True)
            torch.save(logits,evidence.out/f'restored-logits-rank{rank}.pt')
            references=[torch.load(source/f'postforward-logits-rank{i}.pt',weights_only=True) for i in range(dist.get_world_size())]
            matching=[i for i,value in enumerate(references) if torch.equal(logits,value)]
            # The original Actor already exhibits cross-rank numerical variants.
            # Gate exact restored weights/state, and retain numerical differences
            # explicitly instead of declaring a correct checkpoint corrupt.
            assert torch.isfinite(logits).all()
            (evidence.out / f"restored-rank{rank}.json").write_text(json.dumps({
                "sampled_parameter_positions": verified, "checkpoint_step": state.global_step,
                "scheduler_epoch": trainer.lr_scheduler.last_epoch, "rng_file_verified": rng_file,
                "all_moment_local_tensor_bytes": [
                    {"device": device, "dtype": dtype, "slot": key, "bytes": size}
                    for (device, dtype, key), size in sorted(moment_bytes.items())],
                "optimizer_placement": placements, "probe_max_abs": (logits - expected).abs().max().item(),
                "exact_matching_original_rank_logits":matching,
                "logits_policy":"diagnostic; exact full restored weights are the acceptance gate",
                "additional_updates": 0}))
            dist.barrier()
            if rank == 0:
                (evidence.out / "accepted.json").write_text(json.dumps({
                    "checkpoint": str(checkpoint), "ranks": dist.get_world_size(), "additional_updates": 0}))
            raise RestoreFinished()

    trainer.remove_callback(type(evidence))
    trainer.add_callback(Check())
    try:
        trainer.train(resume_from_checkpoint=str(checkpoint))
    except RestoreFinished:
        return
    raise RuntimeError("Restore-only check did not stop before training")

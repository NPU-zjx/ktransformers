"""Read a completed native HF + KT checkpoint, validate it, and exit before training."""

import json
import random
import os
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from kt_kernel.sft import get_kt_named_trainable_params
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
                    for key in ("exp_avg", "exp_avg_sq"):
                        assert local(st[key]).reshape(-1)[i].item() == point[key]
                    verified += 1
            assert verified > 0
            handles = []
            if os.environ.get("KTRL_FORWARD_TRACE"):
                from ktrl.weights.forward_trace import attach
                handles = attach(model, evidence.out / 'trace', lambda: 'actor')
            logits = evidence.probe(model)
            for handle in handles:
                handle.remove()
            expected = torch.load(source / f"postforward-logits-rank{rank}.pt", weights_only=True)
            torch.testing.assert_close(logits, expected, rtol=0.005, atol=0.0625)
            (evidence.out / f"restored-rank{rank}.json").write_text(json.dumps({
                "sampled_parameter_positions": verified, "checkpoint_step": state.global_step,
                "scheduler_epoch": trainer.lr_scheduler.last_epoch, "rng_file_verified": rng_file,
                "all_moment_local_tensor_bytes": [
                    {"device": device, "dtype": dtype, "slot": key, "bytes": size}
                    for (device, dtype, key), size in sorted(moment_bytes.items())],
                "optimizer_placement": placements, "probe_max_abs": (logits - expected).abs().max().item(),
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

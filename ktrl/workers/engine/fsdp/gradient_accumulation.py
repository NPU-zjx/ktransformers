"""BF16 cross-microbatch accumulation over already reduced FSDP2 shards.

FSDP2 casts each reduced shard to its FP32 parameter dtype. We clear that
transient gradient after every backward so FSDP2 cannot accumulate it in FP32.
Only at the optimizer boundary is the BF16 sum materialized as an FP32 grad
for the existing joint clipping and optimizer. CPU KT gradients are untouched.
"""
import torch


class BF16ShardAccumulator:
    def __init__(self):
        self.buffers = {}
        self.microsteps = 0

    @torch.no_grad()
    def collect(self, named_parameters):
        collected = 0
        for name, parameter in named_parameters:
            if not parameter.requires_grad or parameter.device.type != "cuda":
                continue
            if not hasattr(parameter, "to_local"):
                raise RuntimeError(f"BF16 accumulation requires FSDP2 shards: {name}")
            gradient = parameter.grad
            if gradient is None:
                continue
            if parameter.dtype != torch.float32 or gradient.dtype != torch.float32:
                raise RuntimeError(f"Expected FP32 FSDP2 parameter/temporary gradient: {name}")
            contribution = gradient.detach().to(torch.bfloat16)
            key = id(parameter)
            if key in self.buffers:
                previous_name, previous_parameter, buffer = self.buffers[key]
                if previous_parameter is not parameter or previous_name != name:
                    raise RuntimeError("FSDP2 parameter identity changed during accumulation")
                buffer.add_(contribution)
            else:
                self.buffers[key] = (name, parameter, contribution)
            parameter.grad = None
            collected += 1
        self.microsteps += 1
        if not collected and not self.buffers:
            raise RuntimeError("No reduced FSDP2 gradients; asynchronous/no_sync accumulation is unsupported")

    def snapshot(self):
        return {
            "microsteps": self.microsteps,
            "buffer_dtype": "bfloat16",
            "buffer_count": len(self.buffers),
            "local_buffer_bytes": sum(b.to_local().numel() * b.element_size() for _, _, b in self.buffers.values()),
            "fp32_gradients_cleared": all(p.grad is None for _, p, _ in self.buffers.values()),
        }

    @torch.no_grad()
    def materialize(self):
        for _, parameter, buffer in self.buffers.values():
            if parameter.grad is not None:
                raise RuntimeError("Unexpected gradient bypassed BF16 accumulation")
            parameter.grad = buffer.to(parameter.dtype)
        self.buffers.clear()
        self.microsteps = 0


class BF16AccumulationTrainerMixin:
    """Opt-in Trainer adapter. Requires reduction after every microbatch.

Set ``gradient_accumulation_dtype`` on the Trainer to ``bfloat16`` to enable.
The original Trainer remains responsible for GAS/world-size normalization.
"""
    def training_step(self, model, inputs, num_items_in_batch=None):
        if getattr(self, "gradient_accumulation_dtype", "float32") != "bfloat16":
            return super().training_step(model, inputs, num_items_in_batch)
        config = self.args.accelerator_config.gradient_accumulation_kwargs or {}
        if not self.accelerator.is_fsdp2 or not config.get("sync_each_batch", False):
            raise RuntimeError("BF16 shard accumulation requires FSDP2 and sync_each_batch=True")
        if not hasattr(self, "bf16_accumulator"):
            self.bf16_accumulator = BF16ShardAccumulator()
        accumulator = self.bf16_accumulator
        loss = super().training_step(model, inputs, num_items_in_batch)
        # FSDP2 completes its reduce-scatter stream dependency before backward
        # returns; these operations run on the same current stream as the next
        # forward. No extra device synchronization or second reduction is needed.
        accumulator.collect(model.named_parameters())
        snapshot = accumulator.snapshot()
        boundary = accumulator.microsteps == self.current_gradient_accumulation_steps
        if accumulator.microsteps > self.current_gradient_accumulation_steps:
            raise RuntimeError("BF16 accumulation exceeded the Trainer's optimizer boundary")
        snapshot["optimizer_boundary"] = boundary
        self.on_bf16_accumulation(snapshot, accumulator)
        if boundary:
            accumulator.materialize()
        return loss

    def on_bf16_accumulation(self, snapshot, accumulator):
        """Optional evidence hook before the BF16 buffer is materialized."""

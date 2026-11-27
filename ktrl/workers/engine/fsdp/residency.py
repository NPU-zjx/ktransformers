"""Actor lifecycle: discard disposable CUDA grads and optionally park GPU shards.

CPU KT experts are never copied. Storage aliases and Parameter/optimizer identity
are retained across parking, including FSDP2's cached sharded storage views.
"""
import time
import torch


def local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


def release_cuda_gradients(optimizer):
    released = 0
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.device.type != 'cuda' or parameter.grad is None:
                continue
            grad = local_tensor(parameter.grad)
            storage = grad.untyped_storage()
            key = storage._cdata
            if key not in seen:
                released += storage.nbytes(); seen.add(key)
            parameter.grad = None
    return released


def release_unused_router_weights(model):
    """KT routes with gate(); _original_router supplies only routing metadata.

    Remove only the proven duplicate alias, preserving correction-bias buffers
    and all routing attributes. Do this before FSDP replaces gate.weight.
    """
    count = capacity = 0
    for module in model.modules():
        if not type(module).__module__.startswith('kt_kernel.sft.layer'):
            continue
        original = getattr(module, '_original_router', None)
        gate = getattr(module, getattr(module, '_router_attr', 'gate'), None)
        if original is None or gate is None:
            continue
        weight = getattr(original, 'weight', None)
        if weight is not None and weight is getattr(gate, 'weight', None):
            capacity += weight.numel() * weight.element_size()
            original.register_parameter('weight', None)
            count += 1
    return dict(duplicate_router_parameters_removed=count, bytes=capacity)


class GPUStateParking:
    """Copy only existing CUDA parameter/state storages, never CPU experts.

    Parking is allowed only after export/checks at a synchronized idle boundary.
    No model/optimizer operation may run until restore() has completed. The
    device tensors, DTensors, shapes, strides, and optimizer keys are unchanged.
    """
    def __init__(self):
        self.buffers = {}
        self.parked = []

    @torch.no_grad()
    def park(self, model, optimizer):
        if self.parked:
            raise RuntimeError('Actor state is already parked')
        begin = time.perf_counter()
        tensors = list(model.parameters())
        tensors.extend(v for state in optimizer.state.values() for v in state.values()
                       if isinstance(v, torch.Tensor))
        seen = set(); entries = []
        for tensor in tensors:
            if tensor.device.type != 'cuda':
                continue
            local = local_tensor(tensor); storage = local.untyped_storage()
            key = storage._cdata; size = storage.nbytes()
            if not size or key in seen:
                continue
            seen.add(key)
            byte_view = torch.empty(0, dtype=torch.uint8, device=local.device).set_(storage, 0, (size,), (1,))
            buffer = self.buffers.get(key)
            if buffer is None or buffer.numel() != size:
                buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
                self.buffers[key] = buffer
            entries.append((storage, byte_view, buffer, size))
        torch.cuda.synchronize()
        copy_begin = time.perf_counter()
        for _, device, cpu, _ in entries:
            cpu.copy_(device, non_blocking=True)
        torch.cuda.synchronize()
        copy_seconds = time.perf_counter() - copy_begin
        for storage, _, _, _ in entries:
            storage.resize_(0)
        self.parked = entries
        torch.cuda.empty_cache()
        return dict(bytes=sum(e[3] for e in entries), storages=len(entries),
                    cpu_expert_transfer_bytes=0, copy_seconds=copy_seconds,
                    seconds=time.perf_counter()-begin)

    @torch.no_grad()
    def restore(self):
        if not self.parked:
            raise RuntimeError('No parked Actor state to restore')
        begin = time.perf_counter()
        for storage, _, _, size in self.parked:
            storage.resize_(size)
        copy_begin = time.perf_counter()
        for _, device, cpu, _ in self.parked:
            device.copy_(cpu, non_blocking=True)
        torch.cuda.synchronize()
        result = dict(bytes=sum(e[3] for e in self.parked), storages=len(self.parked),
                      cpu_expert_transfer_bytes=0, copy_seconds=time.perf_counter()-copy_begin,
                      seconds=time.perf_counter()-begin)
        self.parked = []
        return result

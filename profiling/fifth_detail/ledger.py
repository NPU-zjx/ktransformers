"""Read tensor storage metadata without cloning, moving, or reading tensor values."""
import os
import time
from collections import Counter


def inventory(model=None, optimizer=None, extra=None):
    import torch
    storages, errors = {}, []

    def add(value, role, name):
        if not isinstance(value, torch.Tensor):
            return
        try:
            value = value.to_local() if hasattr(value, 'to_local') else value
            if value.device.type == 'meta':
                return
            storage = value.untyped_storage()
            size = storage.nbytes()
            if not size:
                return
            key = (str(value.device), storage.data_ptr())
            row = storages.setdefault(key, dict(device=str(value.device), address=storage.data_ptr(),
                storage_bytes=size, category=role, aliases=[]))
            row['aliases'].append(dict(name=name, role=role, dtype=str(value.dtype),
                shape=list(value.shape), logical_bytes=value.numel()*value.element_size(),
                pinned=bool(value.is_pinned()) if value.device.type == 'cpu' else False,
                shared=bool(value.is_shared()) if value.device.type == 'cpu' else False))
        except Exception as error:
            errors.append(dict(name=name, role=role, error=repr(error)))

    def walk(value, role, name, depth=0):
        if isinstance(value, torch.Tensor):
            add(value, role, name)
        elif depth < 8 and isinstance(value, dict):
            for k, v in value.items():
                walk(v, role, f'{name}.{k}', depth+1)
        elif depth < 8 and isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(v, role, f'{name}[{i}]', depth+1)

    if optimizer is not None:
        for gi, group in enumerate(optimizer.param_groups):
            for pi, p in enumerate(group['params']):
                name = f'optimizer.group{gi}.p{pi}'
                add(p, 'parameter', name)
                add(p.grad, 'gradient', name+'.grad')
                for key, value in optimizer.state.get(p, {}).items():
                    walk(value, 'optimizer.'+str(key), name+'.'+str(key))
    if model is not None:
        for name, p in model.named_parameters():
            add(p, 'parameter', name)
            add(p.grad, 'gradient', name+'.grad')
        for name, p in model.named_buffers():
            add(p, 'model_buffer', name)
        # FSDP2 keeps compute copies/all-gather storage off named_parameters.
        # Read existing fields only; never call unshard/redistribute/materialize.
        seen_groups = set()
        fields = {'_unsharded_param':'fsdp.compute_parameter',
                  'all_gather_outputs':'fsdp.all_gather_buffer',
                  '_unsharded_inner_tensors':'fsdp.unsharded_inner',
                  '_sharded_param_data':'fsdp.sharded_parameter_storage',
                  '_sharded_post_forward_param_data':'fsdp.post_forward_shard',
                  'unsharded_accumulated_grad':'fsdp.unsharded_accumulated_gradient'}
        for module_name, module in model.named_modules():
            getter = getattr(module, '_get_fsdp_state', None)
            if getter is None:
                continue
            try:
                group = getattr(getter(), '_fsdp_param_group', None)
                if group is None or id(group) in seen_groups:
                    continue
                seen_groups.add(id(group))
                for index, param in enumerate(group.fsdp_params):
                    attributes = vars(param)
                    for field, role in fields.items():
                        walk(attributes.get(field), role, f'{module_name}.fsdp[{index}].{field}')
            except Exception as error:
                errors.append(dict(name=module_name, role='fsdp_internal', error=repr(error)))
    if model is not None:
        for index, layer in enumerate(getattr(model, '_kt_wrappers', [])):
            wrapper = getattr(layer, 'wrapper', layer)
            for field in ('grad_gate_proj_buf', 'grad_up_proj_buf', 'grad_down_proj_buf'):
                add(getattr(wrapper, field, None), 'kt_cpu_gradient_buffer', f'kt[{index}].{field}')
    for role, value in (extra or {}).items():
        walk(value, role, role)
    totals = Counter()
    for row in storages.values():
        totals[row['device']+'/'+row['category']] += row['storage_bytes']
    return dict(time_unix_ns=time.time_ns(), pid=os.getpid(),
                storages=list(storages.values()), bytes_by_device_category=dict(totals), errors=errors,
                scope='Process-local storage capacity; not RSS/PSS; aliases counted once. Unexposed native allocations excluded.')

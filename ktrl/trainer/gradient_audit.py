"""Observe backward scaling and audit native joint clipping; never replace it."""
import math
import torch


def watch_loss_scale(loss, gas, record):
    """Observe d(backward_loss)/d(compute_loss) at the actual autograd boundary."""
    if gas < 1:
        raise ValueError('GAS must be positive')
    def check(incoming):
        actual = incoming.detach().float().item()
        expected = 1.0 / gas
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8):
            raise RuntimeError(f'Backward loss scale {actual} != 1/GAS {expected}')
        record(dict(compute_loss=float(loss.detach()), gas=gas, backward_scale=actual,
                    effective_backward_loss=float(loss.detach())*actual))
        return incoming
    return loss.register_hook(check)


def local(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


def gradient_snapshot(named_parameters, required_ids, rank=0, chunk=1048576):
    """Bounded workspace scan of ALL gradients, plus samples for clipping checks.

    Counts one-dimensional FSDP shards once, replicated parameters only on rank 0,
    and rank-local KT CPU experts once. Unknown DTensor layouts fail closed.
    """
    seen, rows, square_sum = set(), [], 0.
    required_seen = set()
    for name, parameter in named_parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        required = id(parameter) in required_ids
        if required:
            required_seen.add(id(parameter))
        gradient = parameter.grad
        if gradient is None:
            if required:
                raise RuntimeError(f'CPU expert gradient not ready before clipping: {name}')
            continue
        part = local(gradient).detach()
        if not part.is_contiguous():
            raise RuntimeError(f'Unsupported noncontiguous gradient: {name}')
        flat = part.view(-1)
        if required and (part.device.type != 'cpu' or rank != 0):
            raise RuntimeError(f'Unexpected expert owner/device: {name}')
        counts = required or rank == 0
        if hasattr(gradient, 'placements'):
            placements = gradient.placements
            if len(placements) != 1 or type(placements[0]).__name__ not in ('Shard','Replicate'):
                raise RuntimeError(f'Unsupported distributed gradient layout: {name}: {placements}')
            counts = type(placements[0]).__name__ == 'Shard' or rank == 0
        norm2 = 0.
        for start in range(0, flat.numel(), chunk):
            block = flat[start:start+chunk]
            if not torch.isfinite(block).all().item():
                raise RuntimeError(f'Nonfinite gradient before clipping: {name}')
            norm2 += block.double().square().sum().item()
        if counts:
            square_sum += norm2
        indices = torch.linspace(0,max(flat.numel()-1,0),min(flat.numel(),64),
                                 dtype=torch.float64,device=flat.device).long()
        rows.append(dict(name=name, parameter=parameter, indices=indices,
                         before=flat[indices].clone(), norm2=norm2, counts=counts,
                         elements=flat.numel(), cpu_expert=required))
    if required_seen != required_ids:
        raise RuntimeError('CPU experts missing from optimizer parameters')
    return square_sum, rows


def verify_clip(rows, total_norm, returned_norm, max_norm):
    if not math.isfinite(total_norm) or not math.isclose(total_norm, returned_norm, rel_tol=.02, abs_tol=1e-7):
        raise RuntimeError(f'Native norm {returned_norm} != independently deduplicated norm {total_norm}')
    coefficient = min(1., max_norm/(returned_norm+1e-6))
    for row in rows:
        gradient = local(row['parameter'].grad).detach().view(-1)
        actual = gradient[row['indices']]
        expected = (row['before'].float()*coefficient).to(actual.dtype)
        # BF16 native scalar multiplication can differ by a rounding unit.
        tolerance = 2*torch.finfo(actual.dtype).eps
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=1e-10,
                                   msg=f'Joint clipping mismatch: {row["name"]}')
    return dict(preclip_norm=returned_norm, independently_computed_norm=total_norm,
                coefficient=coefficient, gradients_scanned=len(rows),
                cpu_expert_gradients_scanned=sum(r['cpu_expert'] for r in rows),
                finite_scan='all elements before native clipping',
                postclip_check='up to 64 positions per gradient, including zeros')

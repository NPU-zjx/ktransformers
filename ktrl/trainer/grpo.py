"""GRPO outcome advantages and the clipped per-response-token objective."""

from collections import defaultdict
import math


def outcome_advantages(groups, rewards, group_size, epsilon=1e-6):
    """Normalize by each prompt's sample standard deviation, preserving row order."""
    if len(groups) != len(rewards) or not groups or group_size < 2:
        raise ValueError("expected aligned responses and at least two answers per prompt")
    members = defaultdict(list)
    for i, (group, reward) in enumerate(zip(groups, rewards)):
        if not math.isfinite(reward):
            raise ValueError(f"non-finite reward in row {i}")
        members[group].append(i)
    advantages = [0.0] * len(rewards)
    for indices in members.values():
        if len(indices) != group_size:
            raise ValueError(f"expected {group_size} answers, got {len(indices)}")
        mean = math.fsum(rewards[i] for i in indices) / group_size
        variance = math.fsum((rewards[i] - mean) ** 2 for i in indices) / (group_size - 1)
        for i in indices:
            advantages[i] = (rewards[i] - mean) / (math.sqrt(variance) + epsilon)
    return advantages


def token_tis_weights(old_logp, rollout_logp, threshold):
    """veRL decoupled token TIS: a detached weight for rollout-to-Actor drift."""
    import torch

    if rollout_logp.shape != old_logp.shape or not rollout_logp.numel():
        raise ValueError("token TIS requires one rollout log probability per response token")
    log_ratio = old_logp.detach().float() - rollout_logp.detach().to(old_logp.device, torch.float32)
    if not torch.isfinite(log_ratio).all():
        raise ValueError("token TIS requires finite log probabilities")
    return log_ratio.clamp(-20, 20).exp().clamp(max=threshold)


def policy_loss_sum(logp, old_logp, advantage, mask, clip_low=0.2, clip_high=0.28, dual_clip=3.0,
                    *, rollout_is_weights=None):
    """Return a sum over valid response tokens for global token-mean normalization."""
    import torch

    ratio = (logp[mask].float() - old_logp[mask].detach().float()).clamp(-20, 20).exp()
    adv = advantage.expand_as(logp)[mask]
    unclipped = -adv * ratio
    clipped = -adv * ratio.clamp(1 - clip_low, 1 + clip_high)
    loss = torch.maximum(unclipped, clipped)
    loss = torch.where(adv < 0, torch.minimum(loss, -adv * dual_clip), loss)
    if rollout_is_weights is not None:
        loss = loss * rollout_is_weights.detach()[mask]
    return loss.sum()

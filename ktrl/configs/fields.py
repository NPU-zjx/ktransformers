"""Validation shared by the offline recipe and its backend planners."""

from pathlib import Path


def fields(value, label, allowed):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{label}: unknown field(s): {', '.join(sorted(unknown))}")
    return value


def integer(value, label, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{label} must be an integer in {bounds}, got {value!r}")
    return value


def require(value, label, choices):
    if value not in choices:
        raise ValueError(f"{label} must be one of {choices}, got {value!r}")
    return value


def path_from(value, label, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path")
    return (Path(base) / Path(value).expanduser()).resolve()


def uniform_placement(total, gpu):
    """Report inclusive expert-ID ranges per MoE layer for a uniform budget."""
    return {
        "policy": "uniform",
        "routed_experts_per_moe_layer": total,
        "gpu": {"count": gpu, "ids_inclusive": [0, gpu - 1] if gpu else None},
        "cpu": {"count": total - gpu, "ids_inclusive": [gpu, total - 1] if gpu < total else None},
    }

"""Integration point for the HF/FSDP2 actor and KT-managed CPU experts."""


def checkpoint_context():
    """Picklable native KT checkpoint context for saved HF training arguments."""
    from kt_kernel.sft import get_activation_checkpoint_context_fn
    return get_activation_checkpoint_context_fn()()

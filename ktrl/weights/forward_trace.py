"""Optional forward-only evidence for the fixed M3 prompt."""

from pathlib import Path
import torch
import torch.distributed as dist


def attach(model, folder, label):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    handles = []

    def capture(name):
        def hook(module, args, output):
            if dist.get_rank() != 0 or torch.cuda.is_current_stream_capturing():
                return
            values = output if isinstance(output, tuple) else (output,)
            tensors = [v for v in values if isinstance(v, torch.Tensor) and v.numel() > 0]
            if not tensors or tensors[0].numel() // tensors[0].shape[-1] < 64:
                return
            torch.save([v.detach().cpu() for v in tensors], folder / (label() + '-' + name + '.pt'))
        return hook

    for name, module in model.named_modules():
        if name.endswith(('embed_tokens', 'norm', 'input_layernorm', 'post_attention_layernorm', 'mlp', 'linear_attn', 'self_attn')) or name.rsplit('.', 1)[-1].isdigit() and '.layers.' in name:
            handles.append(module.register_forward_hook(capture(name)))
    return handles

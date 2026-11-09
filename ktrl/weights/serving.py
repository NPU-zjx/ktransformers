"""Scoped M3 publication hooks for a dedicated, drained KT-SGLang service."""

import json
import os
import hashlib
from pathlib import Path


def install():
    import torch
    import torch.distributed as dist
    from kt_kernel.utils.amx import NativeMoEWrapper
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from safetensors import safe_open
    from .checkpoint import ExpertCheckpoint

    root = Path(os.environ["KTRL_PUBLICATION_OUTPUT"])
    version = {"value": "v0", "checkpoint": None, "failed": False, "updating": False}
    def parameter_hashes(model, stage):
        if not os.environ.get('KTRL_FORWARD_TRACE'):
            return
        records = {}
        for name, parameter in model.named_parameters():
            data = parameter.detach().cpu().contiguous()
            records[name] = {'shape': list(data.shape), 'dtype': str(data.dtype),
                             'sha256': hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()}
        (root / f'parameters-{stage}-rank{dist.get_rank()}.json').write_text(json.dumps(records))

    if os.environ.get("KTRL_FORWARD_TRACE"):
        from .forward_trace import attach
        original_load = ModelRunner.load_model

        def load_model(self, *args, **kwargs):
            result = original_load(self, *args, **kwargs)
            parameter_hashes(self.model, 'cold')
            self._ktrl_trace_handles = attach(self.model, root / 'trace', lambda: version['value'])
            return result

        ModelRunner.load_model = load_model
    original_loader = NativeMoEWrapper._create_loader

    def create_loader(method, path):
        if (Path(path) / "kt_full_experts.index.json").exists():
            if method != "BF16":
                raise ValueError("M3 requires BF16")
            return ExpertCheckpoint(path)
        return original_loader(method, path)

    NativeMoEWrapper._create_loader = staticmethod(create_loader)
    original_replay = CudaGraphRunner.replay
    counts = {}

    def replay(self, *args, **kwargs):
        result = original_replay(self, *args, **kwargs)
        label = version["value"]
        counts[label] = counts.get(label, 0) + 1
        (root / f"replays-rank{dist.get_rank()}.json").write_text(json.dumps(counts))
        return result

    CudaGraphRunner.replay = replay
    original_logits = LogitsProcessor.forward

    def logits(self, *args, **kwargs):
        if version["failed"] and not version["updating"]:
            raise RuntimeError("Publication failed, service requires recovery")
        result = original_logits(self, *args, **kwargs)
        metadata = args[3] if len(args) > 3 else kwargs["logits_metadata"]
        if metadata.forward_mode.is_extend() and result.next_token_logits is not None and not os.environ.get("KTRL_SKIP_LOGITS_DUMP"):
            if dist.get_rank() == 0:
                torch.save(result.next_token_logits.detach().float().cpu(),
                           root / (version["value"] + "-logits.pt"))
        return result

    LogitsProcessor.forward = logits

    def update(self, model_path, load_format, weight_name_filter=None, recapture_cuda_graph=False):
        path = Path(model_path)
        # Validate all files before mutation. A failed publication poisons generation.
        index = ExpertCheckpoint(path)
        for item in index.index["layers"].values():
            file = path / item["file"]
            if file.parent != path or file.stat().st_size != item["size_bytes"]:
                return False, "Incomplete expert checkpoint"
        if version["checkpoint"] == str(path):
            return True, "Already published"
        if version["failed"]:
            return False, "Service requires recovery"
        version["failed"] = True
        version["updating"] = True
        try:
            torch.cuda.synchronize()
            methods = [(module, module.quant_method) for module in self.model.modules()
                       if type(getattr(module, "quant_method", None)).__name__ == "KTEPWrapperMethod"]
            assert len(methods) == 40
            NativeMoEWrapper.force_release_loader()
            for module, method in methods:
                assert int(module.w13_weight.shape[0]) == 0
                if method.wrapper is not None:
                    method.wrapper.cpu_infer.sync()
                    method.wrapper.weight_path = str(path)
                method.kt_config.weight_path = str(path)
            # The HF model file contains GPU parameters and zero-sized expert placeholders.
            loaded = []
            def tensors():
                for file in sorted(path.glob("model*.safetensors")):
                    with safe_open(file, framework="pt", device="cpu") as source:
                        for key in source.keys():
                            if ".experts." in key:
                                continue
                            tensor = source.get_tensor(key)
                            loaded.append({"name": key, "bytes": tensor.numel() * tensor.element_size()})
                            yield key, tensor
            self.model.load_weights(tensors())
            router_name, router = next((name, param) for name, param in self.model.named_parameters()
                                      if name.endswith("layers.0.mlp.gate.weight"))
            router_value = router.detach().reshape(-1)[64].item()
            with safe_open(next(path.glob("model*.safetensors")), framework="pt", device="cpu") as source:
                key = next(name for name in source.keys() if name.endswith("layers.0.mlp.gate.weight"))
                expected_router = source.get_tensor(key).reshape(-1)[64].item()
            assert router_value == expected_router
            for module, method in methods:
                method.process_weights_after_loading(module)
            parameter_hashes(self.model, 'published')
            dist.barrier()
            self.init_device_graphs()
            torch.cuda.synchronize()
            dist.barrier()
            rank = dist.get_rank()
            (root / f"published-rank{rank}.json").write_text(json.dumps({
                "rank": rank, "version": str(path), "cpu_expert_layers": 40 if rank == 0 else 0,
                "router_name": router_name, "router_flat_index": 64,
                "router_value": router_value, "checkpoint_router_value": expected_router,
                "expert_gpu_staging_bytes": 0, "nonexpert_tensors": loaded,
                "graph_recaptured": True}))
            version["checkpoint"] = str(path)
            version["value"] = "v1"
            version["failed"] = False
            version["updating"] = False
            return True, "Published M2 step 1"
        except Exception:
            import traceback
            traceback.print_exc()
            # Abort all ranks to prevent serving a mixed policy after partial mutation.
            os._exit(81)

    ModelRunner.update_weights_from_disk = update

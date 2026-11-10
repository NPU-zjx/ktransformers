"""Scoped M3 publication hooks for a dedicated, drained KT-SGLang service."""

import json
import os
import hashlib
import time
from datetime import timedelta
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
    from .identity import validate_manifest

    root = Path(os.environ["KTRL_PUBLICATION_OUTPUT"])
    version = {"value": "v0", "checkpoint": None, "failed": False, "updating": False}
    def parameter_hashes(model, stage):
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
    original_expert_load=NativeMoEWrapper.load_weights
    def load_expert(self,*args,**kwargs):
        result=original_expert_load(self,*args,**kwargs)
        if (Path(self.weight_path)/'kt_full_experts.index.json').exists():
            mapping=args[0] if args else kwargs['physical_to_logical_map_cpu']
            assert mapping.device.type=='cpu'
            assert torch.equal(mapping.reshape(-1),torch.arange(mapping.numel(),dtype=mapping.dtype)), 'Phase 1 requires identity expert placement'
            from .expert_verify import verify_native_expert
            proof=verify_native_expert(self,self.weight_path)
            step=json.loads((Path(self.weight_path)/'trainer_state.json').read_text())['global_step']
            (root/f'cpu-verified-v{step}-layer{self.layer_idx}.json').write_text(json.dumps(proof))
        return result
    NativeMoEWrapper.load_weights=load_expert
    original_replay = CudaGraphRunner.replay
    counts = {}

    def flush_replay_counts():
        # Avoid synchronous filesystem I/O on every decode. Atomic replacement
        # also prevents readers observing a partially truncated JSON document.
        target = root / f"replays-rank{dist.get_rank()}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(counts))
        temporary.replace(target)

    if os.environ.get("KTRL_ONLINE_PHASES") == "1":
        import phase_hooks
        original_phase_end = phase_hooks.end
        def phase_end():
            result = original_phase_end()
            flush_replay_counts()
            return result
        phase_hooks.end = phase_end

    def replay(self, *args, **kwargs):
        result = original_replay(self, *args, **kwargs)
        label = version["value"]
        counts[label] = counts.get(label, 0) + 1
        if counts[label] == 1:
            flush_replay_counts()
        return result

    CudaGraphRunner.replay = replay
    original_logits = LogitsProcessor.forward

    def logits(self, *args, **kwargs):
        if version["failed"] and not version["updating"]:
            raise RuntimeError("Publication failed, service requires recovery")
        result = original_logits(self, *args, **kwargs)
        metadata = args[3] if len(args) > 3 else kwargs["logits_metadata"]
        requested=root/'capture-next-prefill.json'
        if dist.get_rank()==0 and requested.exists() and metadata.forward_mode.is_extend() and result.next_token_logits is not None:
            name=json.loads(requested.read_text())['filename']
            assert Path(name).name==name and name.endswith('.pt')
            torch.save(result.next_token_logits.detach().float().cpu(),root/name)
            requested.unlink()
        if metadata.forward_mode.is_extend() and result.next_token_logits is not None and not os.environ.get("KTRL_SKIP_LOGITS_DUMP"):
            if dist.get_rank() == 0:
                torch.save(result.next_token_logits.detach().float().cpu(),
                           root / (version["value"] + "-logits.pt"))
        return result

    LogitsProcessor.forward = logits

    publication_group = None
    publication_timeout = timedelta(minutes=15)

    def update(self, model_path, load_format, weight_name_filter=None, recapture_cuda_graph=False):
        nonlocal publication_group
        if publication_group is None:
            # Rank 0 loads and verifies CPU experts while its peers wait. A
            # GPU barrier would start the NCCL watchdog during that CPU work.
            publication_group = dist.new_group(backend="gloo", timeout=publication_timeout)
        flush_replay_counts()
        path = Path(model_path)
        error=None
        try:
            step,manifest_sha=validate_manifest(path,verify_content=dist.get_rank()==0)
        except Exception as exc:
            error=str(exc)
        errors=[None]*dist.get_world_size()
        dist.all_gather_object(errors, error, group=publication_group)
        if any(errors):return False,f'Publication preflight failed: {errors}'
        label = f'v{step}'
        if step < 1:
            return False, 'Invalid policy step'
        # Validate all files before mutation. A failed publication poisons generation.
        index = ExpertCheckpoint(path)
        for item in index.index["layers"].values():
            file = path / item["file"]
            if file.parent != path or file.stat().st_size != item["size_bytes"]:
                return False, "Incomplete expert checkpoint"
        if version["checkpoint"] == str(path):
            if version.get('manifest_sha256')!=manifest_sha:
                return False,'Same checkpoint path has changed content'
            return True, "Already published"
        if version.get('manifest_sha256')==manifest_sha:
            return True,'Already published identical policy content'
        if step<=int(version['value'][1:]):return False,'Policy rollback is forbidden'
        if version["failed"]:
            return False, "Service requires recovery"
        version["failed"] = True
        version["updating"] = True
        try:
            torch.cuda.synchronize()
            methods = [(module, module.quant_method) for module in self.model.modules()
                       if type(getattr(module, "quant_method", None)).__name__ == "KTEPWrapperMethod"]
            native_layers = {str(method.kt_config.layer_idx) for _, method in methods}
            if native_layers != set(index.index["layers"]) or len(native_layers) != len(methods):
                raise ValueError("Published CPU expert layers do not match the rollout model")
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
            injection=root/'inject-publication-failure.json'
            if injection.exists():
                fault=json.loads(injection.read_text())
                if fault.get('rank')==dist.get_rank() and fault.get('stage')=='after_gpu_load':
                    raise RuntimeError('Explicit publication fault injection after GPU load')
            first_layer = min(map(int, native_layers))
            router_suffix = f"layers.{first_layer}.mlp.gate.weight"
            router_name, router = next((name, param) for name, param in self.model.named_parameters()
                                      if name.endswith(router_suffix))
            router_sample = min(64, router.numel() - 1)
            router_value = router.detach().reshape(-1)[router_sample].item()
            expected_router = None
            for file in sorted(path.glob('model*.safetensors')):
                with safe_open(file,framework='pt',device='cpu') as source:
                    keys=[name for name in source.keys() if name.endswith(router_suffix)]
                    if keys:
                        assert expected_router is None and len(keys)==1
                        expected_router=source.get_tensor(keys[0]).reshape(-1)[router_sample].to(router.dtype).item()
            assert expected_router is not None
            assert router_value == expected_router
            expert_load_start = time.monotonic()
            for module, method in methods:
                method.process_weights_after_loading(module)
            expert_load_seconds = time.monotonic() - expert_load_start
            parameter_hashes(self.model, f'published-v{step}')
            dist.monitored_barrier(group=publication_group, timeout=publication_timeout)
            graph_start = time.monotonic()
            self.init_device_graphs()
            torch.cuda.synchronize()
            graph_capture_seconds = time.monotonic() - graph_start
            dist.monitored_barrier(group=publication_group, timeout=publication_timeout)
            rank = dist.get_rank()
            proof = {
                "rank": rank, "version": str(path), "cpu_expert_layers": len(methods) if rank == 0 else 0,
                "router_name": router_name, "router_flat_index": router_sample,
                "router_value": router_value, "checkpoint_router_value": expected_router,
                "expert_gpu_staging_bytes": 0, "nonexpert_tensors": loaded,
                "graph_recaptured": True, "policy_version":label, "actor_step":step,
                "publication_rendezvous_backend": "gloo",
                "expert_load_seconds": expert_load_seconds,
                "graph_capture_seconds": graph_capture_seconds,
                "manifest_sha256":manifest_sha}
            for filename in (f'published-rank{rank}.json',f'published-{label}-rank{rank}.json'):
                (root/filename).write_text(json.dumps(proof))
            version["checkpoint"] = str(path)
            version['manifest_sha256']=manifest_sha
            version["value"] = label
            version["failed"] = False
            version["updating"] = False
            return True, f'Published Actor step {step}'
        except Exception as exc:
            import traceback
            traceback.print_exc()
            (root/f'publication-failed-rank{dist.get_rank()}.json').write_text(json.dumps(dict(
                checkpoint=str(path),policy_version=label,manifest_sha256=manifest_sha,error=str(exc),
                exit_code=81,failed_closed=True)))
            # Abort all ranks to prevent serving a mixed policy after partial mutation.
            os._exit(81)

    ModelRunner.update_weights_from_disk = update

"""Load a real veRL standalone rollout replica and drive its vLLM HTTP server."""
import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path


async def main(args):
    import ray
    from verl.workers.config import HFModelConfig, RolloutConfig
    from verl.workers.rollout.replica import get_rollout_replica_class
    import gpu_measure

    run = Path(os.environ["GPU_RUN"])
    if args.mode == "ncu" and os.environ.get("GPU_NCU_APP_REPLAY") == "1":
        # The tool relaunches the application for each pass. Preserve every ledger.
        parent = run
        run = Path(tempfile.mkdtemp(prefix="pass-", dir=parent))
        run.chmod(0o755)
        (run/"host-ready").symlink_to(parent/"host-ready")
    model = HFModelConfig(path=args.model, trust_remote_code=True)
    config = RolloutConfig(name="vllm", dtype="bfloat16", tensor_model_parallel_size=args.tp,
        data_parallel_size=1, expert_parallel_size=1, max_model_len=args.context,
        prompt_length=1024, response_length=args.output_cap, max_num_seqs=args.concurrency,
        max_num_batched_tokens=args.chunk, gpu_memory_utilization=args.memory_fraction,
        enable_prefix_caching=False, enable_chunked_prefill=True, enforce_eager=False,
        cudagraph_capture_sizes=[1,2,4,8,12,16,24,32], seed=args.seed, load_format="auto",
        enable_sleep_mode=False, free_cache_engine=False, disable_log_stats=False,
        engine_kwargs={"vllm":{"language_model_only":True, "worker_extension_cls":"gpu_worker.ProfileWorker",
            "compilation_config":{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}}})
    ray.init(address="local", include_dashboard=False, num_cpus=32, num_gpus=8,
             object_store_memory=2*1024**3, _temp_dir="/tmp/ray")
    replica = get_rollout_replica_class("vllm")(replica_rank=0,config=config,model_config=model,gpus_per_node=args.tp)
    await replica.init_standalone()
    server = replica._server_handle
    async def rpc(function, *params):
        return await server.collective_rpc.remote("gpu_reference_"+function.__name__, timeout=900, args=params)
    identities=await rpc(gpu_measure.install)
    if {v["rank"] for v in identities} != set(range(8)):
        raise RuntimeError(f"Expected eight vLLM workers: {identities}")
    (run/"worker-identities.json").write_text(json.dumps(identities,indent=2))
    (run/"service.json").write_text(json.dumps({"url":"http://"+replica._server_address,
        "model":args.model, "runtime":str(type(replica)), "settings":vars(args)},indent=2))
    from gpu_client import measure
    await measure(run,"http://"+replica._server_address,args,rpc)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--tp",type=int,required=True)
    parser.add_argument("--context",type=int,required=True)
    parser.add_argument("--output-cap",type=int,required=True)
    parser.add_argument("--concurrency",type=int,required=True)
    parser.add_argument("--chunk",type=int,required=True)
    parser.add_argument("--memory-fraction",type=float,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--mode",choices=("formal","torch","nsys","ncu"),required=True)
    asyncio.run(main(parser.parse_args()))

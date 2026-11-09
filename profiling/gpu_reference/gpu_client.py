import asyncio
import json
import os
from pathlib import Path
import time

import aiohttp
import gpu_measure


async def measure(run,url,args,rpc):
    import client as baseline
    from analysis import analyze_requests
    rows=baseline.inputs()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=14400),trust_env=False) as session:
        async def one(i,row,limit,warm=False):
            payload=dict(model=args.model,prompt=row["prompt"],max_tokens=limit,stream=True,
                return_token_ids=True,add_special_tokens=False,temperature=0 if warm else 1.0,
                top_p=1.0,top_k=-1,ignore_eos=False,stream_options={"include_usage":True})
            start=time.perf_counter_ns();events=[];tokens=[];usage=None;reason=None
            async with session.post(url+"/v1/completions",json=payload) as result:
                if result.status!=200:raise RuntimeError((await result.text())[:2000])
                async for line in result.content:
                    if not line.startswith(b"data: "):continue
                    raw=line[6:].strip()
                    if raw==b"[DONE]":break
                    event=json.loads(raw);now=time.perf_counter_ns()
                    if event.get("error"):raise RuntimeError(str(event["error"]))
                    if event.get("usage"):usage=event["usage"]
                    for choice in event.get("choices",[]):
                        if choice.get("prompt_token_ids") and choice["prompt_token_ids"]!=row["prompt"]:raise ValueError("Prompt ID mismatch")
                        ids=choice.get("token_ids") or []
                        if ids:
                            tokens.extend(ids);events.append([now,len(tokens),len(ids)])
                        reason=choice.get("finish_reason") or reason
            end=time.perf_counter_ns()
            if not reason or not usage or usage["completion_tokens"]!=len(tokens) or usage["prompt_tokens"]!=len(row["prompt"]):
                raise RuntimeError(f"Incomplete token stream {i}: {len(tokens)}, {reason}, {usage}")
            record=dict(index=i,uid=row["uid"],prompt_tokens=len(row["prompt"]),output_tokens=len(tokens),
                input_ids=row["prompt"],output_ids=tokens,start_ns=start,end_ns=end,finish_reason=reason,events_ns_tokens_delta=events)
            (run/("warmup-response.json" if warm else f"baseline-request-{i:02d}.json")).write_text(json.dumps(record))
            return record

        await rpc(gpu_measure.begin,str(run/"warmup-phase"),"formal")
        try:
            warm=await one(0,rows[0],64,True)
        finally:
            await rpc(gpu_measure.finish)
        audit(run/"warmup-phase",warm["prompt_tokens"],warm["output_tokens"]-1,True)
        while not (run/"host-ready").exists():
            if time.time()-(run/"service.json").stat().st_mtime>600:raise TimeoutError("Host resource sampler not ready")
            await asyncio.sleep(1)
        await rpc(gpu_measure.begin,str(run/"phase"),args.mode)
        try:
            with (run/"ledger.json").open("x") as f:json.dump({"status":"submitted","requests":32,"mode":args.mode},f)
            (run/"current-phase.json").write_text(json.dumps({"phase":"formal_rollout_1" if args.mode=="formal" else args.mode,"time_unix":time.time()}))
            results=await asyncio.gather(*(one(i,row,args.output_cap if args.mode=="formal" else 32) for i,row in enumerate(rows)))
        finally:
            ends=await rpc(gpu_measure.finish)
            (run/"current-phase.json").write_text(json.dumps({"phase":"idle_after_rollout","time_unix":time.time()}))
        output=sum(v["output_tokens"] for v in results)
        audit(run/"phase",5712,output-32)
        analyze_requests.analyze(run)
        (run/"ledger.json").write_text(json.dumps({"status":"complete","output_tokens":output,"requests":32,"workers":ends}))


def audit(folder,expected_input,accepted_decode,warm=False):
    from analysis.analyze_phases import union_ns
    import gpu_phase_analysis
    traces=[json.loads(p.read_text()) for p in folder.glob("phase-TP*.json")]
    if len(traces)!=8 or {t["rank"] for t in traces}!=set(range(8)):raise ValueError("TP coverage")
    counts=lambda t:[(r["phase"],r["context_tokens"],r["generation_tokens"]) for r in t["rows"]]
    if any(counts(t)!=counts(traces[0]) for t in traces):raise ValueError("TP batch mismatch")
    if sum(r["context_tokens"] for r in traces[0]["rows"])!=expected_input:raise ValueError("Prefill token closure")
    scheduled=sum(r["generation_tokens"] for r in traces[0]["rows"])
    if scheduled<accepted_decode:raise ValueError(f"Decode closure: {scheduled}/{accepted_decode}")
    if any(not any(r["phase"]=="DECODE" and r["graph_replayed"] for r in t["rows"]) for t in traces):
        raise ValueError("Missing actual decode Graph replay")
    if warm and any(r["phase"]=="DECODE" and not r["graph_replayed"] for t in traces for r in t["rows"]):
        raise ValueError("Warmup contained a decode batch without actual graph replay")
    gpu_phase_analysis.summarize(folder,traces,expected_input,accepted_decode,scheduled,union_ns)

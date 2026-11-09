"""Submit one native KT-SGLang generation from KTRL and audit its decode replay."""

import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

from ktrl.trainer.online import save


class SGLangClient:
    def __init__(self, url):
        self.url = url
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, path, data=None, *, timeout=120):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(self.url + path, data=body, headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=timeout) as response:
            raw = response.read().decode()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def generate(self, payload):
        return self.call("/generate", payload, timeout=600)


def wait_ready(client, is_running, expected, out, *, timeout=2400):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_running():
            raise RuntimeError(f"KT-SGLang exited during startup; see {out}/server.log")
        try:
            models = client.call("/v1/models", timeout=8)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(5)
            continue
        if expected["served_model_name"] not in {model["id"] for model in models["data"]}:
            raise RuntimeError("HTTP endpoint belongs to a different service")
        # With HTTP warmup skipped, /server_info does not start TokenizerManager's
        # reply loop. Open and close an empty control session without generating.
        session = "ktrl-control-" + uuid.uuid4().hex
        opened = client.call("/open_session", {"session_id": session, "capacity_of_str_len": 1})
        if opened != session:
            raise RuntimeError(f"Unexpected control-session identity: {opened!r}")
        closed = client.call("/close_session", {"session_id": session})
        save(out / "control-handshake.json", {"session": session, "open": opened, "close": closed})
        info = client.call("/server_info")
        save(out / "server-info.json", info)
        differences = {k: (v, info.get(k)) for k, v in expected.items() if info.get(k) != v}
        if differences or info.get("enable_lora"):
            raise RuntimeError(f"Unexpected resolved service configuration: {differences}")
        return
    raise TimeoutError(f"KT-SGLang readiness timed out; see {out}/server.log")


def validate_response(payload, response, *, minimum_tokens=2):
    meta = response["meta_info"]
    ids = response["output_ids"]
    if meta["id"] != payload["rid"] or meta["prompt_tokens"] != len(payload["input_ids"]):
        raise ValueError("Response does not match the submitted request")
    if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("Invalid output token IDs")
    if not minimum_tokens <= len(ids) <= payload["sampling_params"]["max_new_tokens"] or meta["completion_tokens"] != len(ids):
        raise ValueError("Output token count does not cover decode")
    if meta["finish_reason"]["type"] not in {"stop", "length"}:
        raise ValueError(f"Generation did not finish normally: {meta['finish_reason']}")
    return ids


def verify_replays(folder, tp_size, input_count, output_count):
    ranks = {}
    batch_sequence = None
    for path in sorted(folder.glob("phase-*.json")):
        trace = json.loads(path.read_text())
        rank = trace["rank"]
        rows = sorted(trace["rows"], key=lambda r: r["index"])
        seq = [(r["phase"], r["tokens"]) for r in rows]
        prefill = [r for r in rows if r["phase"] == "EXTEND"]
        decode = [r for r in rows if r["phase"] == "DECODE"]
        if rank in ranks or len(rows) != len(prefill) + len(decode) or (batch_sequence is not None and seq != batch_sequence):
            raise ValueError(f"Duplicate window or inconsistent batch sequence in {path.name}")
        batch_sequence = seq
        if sum(r["tokens"] for r in prefill) != input_count or not output_count - 1 <= len(decode) <= output_count:
            raise ValueError(f"Missing prefill/decode coverage in {path.name}")
        for row in decode:
            if row["tokens"] != 1 or row["graph_replayed"] is not True or row["stream_ms"] <= 0:
                raise ValueError(f"Missing single-request Graph replay in {path.name}")
        if trace["replays"] < len(decode):
            raise ValueError(f"Replay count is inconsistent in {path.name}")
        ranks[rank] = {"rank": rank, "pid": trace["pid"], "gpu_uuid": trace["uuid"], "file": path.name,
                       "prefill_batches": len(prefill), "decode_batches": len(decode),
                       "decode_replay_batches": len(decode), "runner_replays": trace["replays"]}
    if set(ranks) != set(range(tp_size)) or len({r["gpu_uuid"] for r in ranks.values()}) != tp_size:
        raise ValueError("Missing TP rank or distinct GPU in worker traces")
    return [ranks[i] for i in range(tp_size)]


def run_one(config_path, output, max_new_tokens, prompt_file):
    from ktrl.configs.recipe import load_plan
    from ktrl.trainer.online_native import service
    from ktrl.trainer.phase_completion import wait_phase_traces

    config_path = Path(config_path).resolve()
    plan = load_plan(config_path)
    rollout = plan["rollout"]
    prompt = json.loads(Path(prompt_file).read_bytes())
    if (not isinstance(prompt, list) or not prompt
            or any(type(token) is not int or token < 0 for token in prompt)):
        raise ValueError("Prompt file must be a nonempty JSON array of token IDs")
    if len(prompt) + max_new_tokens > plan["inputs"]["context_length"]:
        raise ValueError("Prompt and requested output exceed data.context_length")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    phase = out / "phase"
    phase.mkdir()
    save(out / "resolved-rollout.json", rollout)
    (out / "recipe.yaml").write_bytes(config_path.read_bytes())
    rid = uuid.uuid4().hex
    payload = {"rid": rid, "input_ids": prompt, "stream": False,
               "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": False}}
    with service(config_path, out / "service", plan) as (client, _):
        client.call("/start_profile", {"activities": ["KT_PHASE"], "output_dir": str(phase),
                                       "record_shapes": False, "with_stack": False})
        try:
            # The exclusive output directory and this record protect the single submission.
            with (out / "request.json").open("x") as log:
                json.dump(payload, log)
            start = time.perf_counter()
            response = client.generate(payload)
            seconds = time.perf_counter() - start
            save(out / "response.json", response)
        finally:
            client.call("/stop_profile", {})
        ids = validate_response(payload, response)
        tp = rollout["server_args"]["tp_size"]
        wait_phase_traces(phase, world_size=tp)
        ranks = verify_replays(phase, tp, len(prompt), len(ids))
    result = {
        "scope": "one KTRL rollout request", "backend": rollout["backend"],
        "request_id": rid, "prompt_file": str(Path(prompt_file).resolve()),
        "submitted_requests": 1, "input_tokens": len(row["prompt"]), "output_tokens": len(ids), "output_ids": ids,
        "text": response["text"], "finish_reason": response["meta_info"]["finish_reason"],
        "client_request_seconds": seconds, "ranks": ranks,
        "decode_replay_batches_all_ranks": sum(r["decode_replay_batches"] for r in ranks),
        "exit": json.loads((out / "service" / "exit.json").read_text()),
    }
    save(out / "rollout-summary.json", result)
    return result

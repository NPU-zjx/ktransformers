"""Publish the existing M2 checkpoint into one persistent inference process."""

import json
import sys
import os
from pathlib import Path

from ktrl.configs.recipe import load_rollout_config
from . import qjh005
from .smoke import SGLangClient, wait_ready


def main():
    out = qjh005.RESULTS / (sys.argv[1] if len(sys.argv) > 1 else "ktrl-m3-r01")
    if out.parent != qjh005.RESULTS or not out.name.startswith("ktrl-m3-"):
        raise ValueError("Use a new ktrl-m3-* result directory")
    out.mkdir()
    checkpoint = qjh005.RESULTS / "ktrl-m2-r01/update-a03/trainer/checkpoint-1"
    batch = json.loads((checkpoint.parents[2] / "prepare-a01/fixed-rows.json").read_text())
    plan = load_rollout_config(Path("examples/ktrl/qwen36_35b_grpo.yaml"))
    cold = '--cold' in sys.argv
    if '--tp1' in sys.argv:
        if not cold:
            raise ValueError('TP1 diagnostic requires cold checkpoint loading')
        plan['server_args']['tp_size'] = 1
    if cold:
        view = out / 'model-view'
        view.mkdir()
        for file in checkpoint.iterdir():
            if file.name in ('config.json', 'generation_config.json') or file.name.startswith('model') and file.suffix in ('.safetensors', '.json'):
                (view / file.name).symlink_to(file)
        plan['server_args']['tokenizer_path'] = plan['server_args']['model_path']
        plan['server_args']['model_path'] = str(view)
        plan['server_args']['kt_weight_path'] = str(checkpoint)
    original = qjh005.command

    def command(*args):
        cmd = original(*args)
        image = cmd.index(qjh005.IMAGE)
        cmd[image:image] = ["-v", f"{checkpoint}:{checkpoint}:ro", "-e", "KTRL_PUBLICATION_OUTPUT=/output",
                            "-e", "KTRL_FORWARD_TRACE=" + os.environ.get('KTRL_FORWARD_TRACE', ''),
                            "-e", "PYTHONPATH=/repo/ktrl/workers/rollout/sglang/publication_worker:/repo:/repo/profiling/worker:/repo/third_party/sglang/python"]
        if cold:
            image = cmd.index(qjh005.IMAGE)
            tokenizer = plan['server_args']['tokenizer_path']
            cmd[image:image] = ['-v', f'{tokenizer}:{tokenizer}:ro']
        return cmd

    qjh005.command = command
    with qjh005.service(plan, out) as (cid, name, url):
        client = SGLangClient(url)
        expected = dict(plan["server_args"], served_model_name=name, skip_server_warmup=True)
        wait_ready(client, cid, expected, out)
        payload = {"input_ids": batch[0]["prompt"], "sampling_params": {
            "temperature": 0, "max_new_tokens": 2, "ignore_eos": True}}
        qjh005.save(out / "input.json", payload)
        qjh005.save(out / "v0-response.json", client.generate(payload))
        if cold:
            return
        request = {"model_path": str(checkpoint), "load_format": "safetensors", "flush_cache": True,
                   "recapture_cuda_graph": True}
        response = client.call("/update_weights_from_disk", request, timeout=1800)
        qjh005.save(out / "publication.json", response)
        if not response.get("success"):
            raise RuntimeError(response)
        qjh005.save(out / "v1-response.json", client.generate(payload))
        duplicate = client.call("/update_weights_from_disk", request, timeout=120)
        qjh005.save(out / "duplicate.json", duplicate)
        if not duplicate.get("success"):
            raise RuntimeError(duplicate)
    print("Publication requests completed. Logits and rank evidence require audit.", flush=True)


if __name__ == "__main__":
    main()

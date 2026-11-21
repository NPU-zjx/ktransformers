#!/usr/bin/env python3
"""Reproducible SFT token length & label-mask audit (HF chat template + LlamaFactory SupervisedDatasetProcessor)."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(values: list[int]) -> dict:
    arr = np.array(values, dtype=np.int64)
    return {
        "count": int(len(arr)),
        "mean": round(float(arr.mean()), 2) if len(arr) else 0.0,
        "min": int(arr.min()) if len(arr) else 0,
        "p25": int(np.percentile(arr, 25)) if len(arr) else 0,
        "p50": int(np.percentile(arr, 50)) if len(arr) else 0,
        "p75": int(np.percentile(arr, 75)) if len(arr) else 0,
        "p90": int(np.percentile(arr, 90)) if len(arr) else 0,
        "p95": int(np.percentile(arr, 95)) if len(arr) else 0,
        "p99": int(np.percentile(arr, 99)) if len(arr) else 0,
        "max": int(arr.max()) if len(arr) else 0,
        "gt_512": int((arr > 512).sum()),
        "gt_2560": int((arr > 2560).sum()),
        "gt_3072": int((arr > 3072).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=str, default="qwen3")
    parser.add_argument("--cutoff-len", type=int, default=3072)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    chat_tpl = getattr(tok, "chat_template", "") or ""

    rows = []
    with args.sft_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Path A: HuggingFace apply_chat_template approximation
    hf_prompts_txt = [
        tok.apply_chat_template([r["messages"][0]], tokenize=False, add_generation_prompt=True)
        for r in rows
    ]
    hf_fulls_txt = [
        tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in rows
    ]
    hf_prompt_lens, hf_asst_lens, hf_full_lens = [], [], []
    bs = 1024
    for i in range(0, len(rows), bs):
        p_ids = tok(hf_prompts_txt[i : i + bs], add_special_tokens=False)["input_ids"]
        f_ids = tok(hf_fulls_txt[i : i + bs], add_special_tokens=False)["input_ids"]
        for p, fl in zip(p_ids, f_ids):
            hf_prompt_lens.append(len(p))
            hf_full_lens.append(len(fl))
            hf_asst_lens.append(len(fl) - len(p))

    # Path B: Actual LlamaFactory SupervisedDatasetProcessor._encode_data_example (exact labels != -100)
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.hparams import DataArguments

    data_args = DataArguments(template=args.template, cutoff_len=args.cutoff_len, train_on_prompt=False)
    lf_tpl = get_template_and_fix_tokenizer(tok, data_args)
    processor = SupervisedDatasetProcessor(template=lf_tpl, tokenizer=tok, processor=None, data_args=data_args)

    lf_masked_prompt_lens, lf_supervised_lens, lf_full_lens = [], [], []
    answer_intact_count = 0
    answer_missing_examples = []
    for idx, r in enumerate(rows):
        prompt = [r["messages"][0]]
        response = [r["messages"][1]]
        input_ids, labels = processor._encode_data_example(
            prompt=prompt,
            response=response,
            system=None,
            tools=None,
            images=[],
            videos=[],
            audios=[],
        )
        sup_ids = [tid for tid, lbl in zip(input_ids, labels) if lbl != IGNORE_INDEX]
        masked_count = sum(1 for lbl in labels if lbl == IGNORE_INDEX)
        lf_masked_prompt_lens.append(masked_count)
        lf_supervised_lens.append(len(sup_ids))
        lf_full_lens.append(len(input_ids))
        decoded_sup = tok.decode(sup_ids, skip_special_tokens=False)
        if "<answer>" in decoded_sup and "</answer>" in decoded_sup:
            answer_intact_count += 1
        elif len(answer_missing_examples) < 5:
            answer_missing_examples.append({"idx": idx, "id": r.get("id"), "tail": decoded_sup[-80:]})

    report = {
        "command": " ".join(sys.argv),
        "inputs": {
            "sft_jsonl": str(args.sft_jsonl),
            "sft_jsonl_sha256": sha256_file(args.sft_jsonl),
            "model": str(args.model),
            "tokenizer_json_sha256": sha256_file(args.model / "tokenizer.json"),
            "tokenizer_config_json_sha256": sha256_file(args.model / "tokenizer_config.json"),
            "hf_chat_template_sha256": hashlib.sha256(chat_tpl.encode("utf-8")).hexdigest(),
            "llamafactory_template": args.template,
            "llamafactory_cutoff_len": args.cutoff_len,
            "llamafactory_efficient_eos": bool(lf_tpl.efficient_eos),
            "row_count": len(rows),
        },
        "hf_apply_chat_template": {
            "prompt": summarize(hf_prompt_lens),
            "assistant_diff": summarize(hf_asst_lens),
            "full_sequence": summarize(hf_full_lens),
        },
        "llamafactory_supervised_processor": {
            "masked_prompt_labels_eq_minus_100": summarize(lf_masked_prompt_lens),
            "supervised_labels_ne_minus_100": summarize(lf_supervised_lens),
            "full_input_ids": summarize(lf_full_lens),
            "answer_tags_intact_count": answer_intact_count,
            "answer_tags_truncated_or_missing_count": len(rows) - answer_intact_count,
            "answer_missing_samples": answer_missing_examples,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

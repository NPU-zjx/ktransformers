#!/usr/bin/env python3
"""Prepare HotpotQA (distractor) for KTRL, GPU FSDP2 RL (veRL), and KT native SFT."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def build_user_message(row: dict) -> str:
    titles = row["context"]["title"]
    sents_list = row["context"]["sentences"]
    passages = [
        f"[{i}] {t}: " + "".join(sents)
        for i, (t, sents) in enumerate(zip(titles, sents_list), 1)
    ]
    ctx = "\n\n".join(passages)
    q = row["question"].strip()
    return (
        "Answer the question based strictly on the provided passages. "
        "Reason briefly, then put your concise final answer inside <answer>...</answer> tags.\n\n"
        f"Passages:\n{ctx}\n\nQuestion: {q}"
    )


def build_sft_assistant_message(row: dict) -> str:
    titles = row["context"]["title"]
    sents_list = row["context"]["sentences"]
    title_to_sents = {t: sents for t, sents in zip(titles, sents_list)}
    sf = row.get("supporting_facts") or {}
    sf_titles = sf.get("title") or []
    sf_ids = sf.get("sent_id") or []
    evidences = []
    for t, sid in zip(sf_titles, sf_ids):
        sents = title_to_sents.get(t)
        if sents is not None and 0 <= sid < len(sents):
            evidences.append(sents[sid].strip())
    ans = (row.get("answer") or "").strip()
    if evidences:
        return "Relevant facts: " + " ".join(evidences) + f"\n<answer>{ans}</answer>"
    return f"<answer>{ans}</answer>"


def count_prompt_tokens(tok, prompts_messages: list[list[dict]], batch_size: int = 1024) -> list[int]:
    rendered = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompts_messages
    ]
    lengths = []
    for start in range(0, len(rendered), batch_size):
        enc = tok(rendered[start : start + batch_size], add_special_tokens=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


def percentile_summary(lengths: list[int]) -> dict:
    arr = np.array(lengths, dtype=np.int64)
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
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=2560)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    train_files = sorted(args.raw_dir.glob("train-*.parquet"))
    val_files = sorted(args.raw_dir.glob("validation-*.parquet"))
    if not train_files or not val_files:
        raise FileNotFoundError(f"Missing train/validation parquet in {args.raw_dir}")

    raw_train = []
    for fp in train_files:
        raw_train.extend(pq.read_table(fp).to_pylist())
    raw_val = []
    for fp in val_files:
        raw_val.extend(pq.read_table(fp).to_pylist())

    # Deduplicate validation first (preserve order), then remove any overlap + internal dups from train
    val_seen_q = set()
    val_seen_id = set()
    clean_val = []
    for r in raw_val:
        nq = norm_question(r["question"])
        rid = str(r["id"])
        if not nq or not (r.get("answer") or "").strip():
            continue
        if nq in val_seen_q or rid in val_seen_id:
            continue
        val_seen_q.add(nq)
        val_seen_id.add(rid)
        clean_val.append(r)

    train_seen_q = set()
    train_seen_id = set()
    clean_train = []
    removed_val_overlap = 0
    removed_train_dup = 0
    for r in raw_train:
        nq = norm_question(r["question"])
        rid = str(r["id"])
        if not nq or not (r.get("answer") or "").strip():
            continue
        if nq in val_seen_q or rid in val_seen_id:
            removed_val_overlap += 1
            continue
        if nq in train_seen_q or rid in train_seen_id:
            removed_train_dup += 1
            continue
        train_seen_q.add(nq)
        train_seen_id.add(rid)
        clean_train.append(r)

    def format_split(rows: list[dict], split_name: str):
        prompts = [[{"role": "user", "content": build_user_message(r)}] for r in rows]
        lengths = count_prompt_tokens(tok, prompts)
        raw_stats = percentile_summary(lengths)
        kept_rl = []
        kept_sft = []
        kept_lengths = []
        kept_q_set = set()
        for r, p, ln in zip(rows, prompts, lengths):
            if not (0 < ln <= args.max_prompt_tokens):
                continue
            ans = r["answer"].strip()
            kept_lengths.append(ln)
            kept_q_set.add(norm_question(r["question"]))
            kept_rl.append(
                {
                    "data_source": "hotpotqa",
                    "prompt": p,
                    "ability": "qa",
                    "reward_model": {"style": "rule", "ground_truth": ans},
                    "extra_info": {
                        "id": str(r["id"]),
                        "split": split_name,
                        "type": str(r.get("type", "")),
                        "level": str(r.get("level", "")),
                        "question": r["question"].strip(),
                        "prompt_tokens": int(ln),
                    },
                }
            )
            if split_name == "train":
                kept_sft.append(
                    {
                        "id": str(r["id"]),
                        "messages": [
                            p[0],
                            {"role": "assistant", "content": build_sft_assistant_message(r)},
                        ],
                    }
                )
        return kept_rl, kept_sft, raw_stats, percentile_summary(kept_lengths), kept_q_set

    train_rl, train_sft, train_raw_stats, train_kept_stats, train_q_set = format_split(clean_train, "train")
    val_rl, _, val_raw_stats, val_kept_stats, val_q_set = format_split(clean_val, "val")

    overlap_after = len(train_q_set & val_q_set)
    assert overlap_after == 0, f"Overlap must be 0, got {overlap_after}"

    train_pq = args.out_dir / "train.parquet"
    val_pq = args.out_dir / "val.parquet"
    train_sft_path = args.out_dir / "train_sft.jsonl"

    pq.write_table(pa.Table.from_pylist(train_rl), train_pq)
    pq.write_table(pa.Table.from_pylist(val_rl), val_pq)
    with train_sft_path.open("w", encoding="utf-8") as f:
        for item in train_sft:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "dataset": "hotpotqa/hotpot_qa (distractor)",
        "model_tokenizer": args.model,
        "max_prompt_tokens": args.max_prompt_tokens,
        "raw_counts": {"train": len(raw_train), "val": len(raw_val)},
        "dedup": {
            "removed_train_val_overlap": removed_val_overlap,
            "removed_train_internal_dup": removed_train_dup,
            "verified_train_val_overlap_after_filter": overlap_after,
        },
        "token_lengths_before_filter": {"train": train_raw_stats, "val": val_raw_stats},
        "token_lengths_after_filter": {"train": train_kept_stats, "val": val_kept_stats},
        "outputs": {
            "train_parquet": {"path": str(train_pq), "rows": len(train_rl), "sha256": sha256_file(train_pq)},
            "val_parquet": {"path": str(val_pq), "rows": len(val_rl), "sha256": sha256_file(val_pq)},
            "train_sft_jsonl": {"path": str(train_sft_path), "rows": len(train_sft), "sha256": sha256_file(train_sft_path)},
        },
    }
    stats_path = args.out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"PHASE2_DATA_PREP_PASS train={len(train_rl)} val={len(val_rl)} overlap={overlap_after}")


if __name__ == "__main__":
    main()

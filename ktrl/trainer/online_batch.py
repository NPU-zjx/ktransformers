"""Freeze prompt metadata, then score exact live response IDs without veRL runtime."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .grpo import outcome_advantages
from .math_reward import compute_score


def write(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, allow_nan=False)


def select_prompts(rows, tokenize, count, seed, max_prompt_tokens):
    """Select a reproducible prefix of one shuffled dataset pass."""
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = []
    for index in indices:
        row = rows[index]
        ground_truth = (row.get("reward_model") or {}).get("ground_truth")
        if row.get("data_source") != "math_dapo" or ground_truth is None:
            continue
        ids = tokenize(row["prompt"])
        if not 0 < len(ids) <= max_prompt_tokens:
            continue
        selected.append(dict(prompt=ids, ground_truth=ground_truth, data_source=row["data_source"],
                             dataset_row=index, group=f"dataset-row-{index}"))
        if len(selected) == count:
            break
    return selected


def prepare(out, model, parquet, count, seed, max_prompt_tokens):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = pq.read_table(parquet, columns=["prompt", "reward_model", "data_source"]).to_pylist()
    def tokenize(prompt):
        return tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
    selected = select_prompts(rows, tokenize, count, seed, max_prompt_tokens)
    if len(selected) != count:
        raise ValueError(f"Need {count} eligible dataset rows, found {len(selected)}")
    write(out/'prompts.json', selected)
    write(out/'prompt-provenance.json', dict(parquet=str(parquet), dataset_rows=len(rows), seed=seed,
          parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(), eligible_selected=len(selected),
          dataset_row_order=[row["dataset_row"] for row in selected], responses='must be generated live'))


def score(out, model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = json.loads((out/'prompts.json').read_bytes())
    recipe = json.loads((out/'round.json').read_text())
    maximum = recipe['max_response_tokens']
    buffer = recipe['overlong_buffer_tokens']
    rows, details = [], []
    started = time.time_ns()
    for wave in range(4):
        for index, question in enumerate(prompts):
            uid = f'm4-w{wave}-q{index:02d}'
            result = json.loads((out/'rollout'/f'{uid}.json').read_bytes())
            response = result['output_ids']
            meta = result['meta_info']
            assert meta['id'] == uid and len(response) == meta['completion_tokens']
            assert 0 < len(response) <= maximum and meta['prompt_tokens'] == len(question['prompt'])
            assert meta['finish_reason']['type'] in ('stop','length')
            text = tok.decode(response, skip_special_tokens=True)
            base = compute_score(text, question['ground_truth'])
            penalty = min(-(len(response)-(maximum-buffer))/buffer,0) if buffer else 0.
            logps = meta['output_token_logprobs']
            assert [x[1] for x in logps] == response
            assert all(math.isfinite(x[0]) for x in logps)
            reward = base['score'] + penalty
            rows.append(dict(uid=uid, group=question['group'], prompt=question['prompt'],
                             response=response, reward=reward,policy_version=recipe['policy_version'],
                             rollout_logprobs=[x[0] for x in logps]))
            details.append(dict(uid=uid, **base, overlong_reward=penalty, reward=reward,
                                response_tokens=len(response), ground_truth=question['ground_truth']))
    reward_end = time.time_ns()
    adv = outcome_advantages([r['group'] for r in rows], [r['reward'] for r in rows], 4)
    groups = defaultdict(list)
    for row, value in zip(rows, adv):
        row['advantage'] = value
        groups[row['group']].append(row)
    max_error = 0.
    zero_groups=0
    zero_tokens=0
    identical_groups=0
    for group in groups.values():
        assert len(group) == 4
        mean = math.fsum(r['reward'] for r in group)/4
        std = math.sqrt(math.fsum((r['reward']-mean)**2 for r in group)/3)
        for row in group:
            expected = (row['reward']-mean)/(std+1e-6)
            max_error = max(max_error, abs(expected-row['advantage']))
        assert abs(math.fsum(r['advantage'] for r in group)) < 1e-5
        if all(r['advantage']==0 for r in group):
            zero_groups+=1
            zero_tokens+=sum(len(r['response']) for r in group)
        identical_groups+=int(len({tuple(r['response']) for r in group})==1)
    assert max_error < 1e-6, 'GRPO reference mismatch'
    advantage_end = time.time_ns()
    batch = out/'batch'
    batch.mkdir()
    write(batch/'fixed-rows.json', rows)
    write(batch/'rewards.json', details)
    assert len(rows)==32 and len(prompts)==8
    write(batch/'batch-audit.json', dict(records=len(rows), questions=len(prompts),
          fixed_sha256=hashlib.sha256((batch/'fixed-rows.json').read_bytes()).hexdigest(),
          valid_response_tokens=sum(len(r['response']) for r in rows),
          max_sequence=max(len(r['prompt'])+len(r['response']) for r in rows),
          nonzero_advantage_responses=sum(a != 0 for a in adv),
          zero_advantage_groups=zero_groups,zero_advantage_tokens=zero_tokens,
          zero_advantage_token_fraction=zero_tokens/sum(len(r['response']) for r in rows),
          identical_answer_groups=identical_groups,filter_groups=False,
          rollout_importance_sampling=recipe.get("rollout_correction", {}).get("rollout_is"),
          advantage_reference_max_error=max_error,
          score_histogram=dict(Counter(str(r['score']) for r in details)),
          reward_interval_ns=[started,reward_end], advantage_interval_ns=[reward_end,advantage_end],
          provenance='Live KT-SGLang output IDs including actual EOS; no text re-encoding. Actor recomputes old probabilities before its sole update.',
          policy_version=recipe['policy_version'],
          reward_recipe=f'math_dapo score +/-1; overlong buffer={buffer}; maximum={maximum}'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare','score'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--parquet', type=Path)
    p.add_argument('--prompt-count', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--max-prompt-tokens', type=int)
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args.out,args.model,args.parquet,args.prompt_count,args.seed,args.max_prompt_tokens)
    else:
        score(args.out,args.model)

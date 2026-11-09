"""Freeze prompt metadata, then score exact live response IDs without veRL runtime."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

from .grpo import outcome_advantages
from .math_reward import compute_score


def write(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, allow_nan=False)


def prepare(out, model, parquet, reference):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    old = json.loads(reference.read_bytes())
    if isinstance(old,dict):
        old=old['rows']
    wanted = {tuple(r['prompt']) for r in old}
    assert len(wanted) == 8
    found = {}
    for row in pq.read_table(parquet).to_pylist():
        ids = tok.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True, return_dict=False)
        key = tuple(ids)
        if key not in wanted or key in found:
            continue
        assert row['data_source'] == 'math_dapo' and 0 < len(ids) <= 1024
        found[key] = dict(prompt=ids, ground_truth=row['reward_model']['ground_truth'],
                          data_source=row['data_source'], group=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    assert set(found) == wanted
    ordered = list(dict.fromkeys(tuple(r['prompt']) for r in old))
    write(out/'prompts.json', [found[key] for key in ordered])
    write(out/'prompt-provenance.json', dict(reference=str(reference),
          reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), parquet=str(parquet),
          parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
          historical_fields_used=['prompt'], responses='must be generated live'))


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
    assert max_error < 1e-6 and any(a != 0 for a in adv), 'No learning signal; do not update'
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
          identical_answer_groups=identical_groups,filter_groups=False,rollout_importance_sampling=None,
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
    p.add_argument('--reference', type=Path)
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args.out,args.model,args.parquet,args.reference)
    else:
        score(args.out,args.model)

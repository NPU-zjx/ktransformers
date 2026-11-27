# SynSQL data and reward

`qwen3_30b_synsql_grpo.yaml` selects SynSQL with the existing KT-SGLang rollout and FSDP2 Actor. Check the model and dataset paths before use. A configuration dry run does not start the model:

```bash
python -m ktrl run --config examples/ktrl/qwen3_30b_synsql_grpo.yaml --dry-run
```

Prepare train/validation Parquet files with disjoint database IDs. Required runtime columns:

```json
{
  "data_source": "synsql",
  "prompt": [{"role": "user", "content": "Complete schema, example values, external knowledge and question"}],
  "reward_model": {
    "style": "rule",
    "ground_truth": "{\"db_id\":\"example\",\"sql\":\"SELECT name FROM items\"}"
  }
}
```

The database must be at `reward.database_root/example/example.sqlite`. Reference reasoning may be stored separately as `reference_response` for SFT; the online prompt loader reads only prompt, reward_model and data_source. It never sends reference reasoning/SQL to the rollout model. Tokenize the complete chat input with the actual model tokenizer, without truncating the schema to make a sample fit.

The provided recipe uses 6144 input tokens, 2048 output tokens and an 8192-token context, 8 prompts x 4 responses, Actor minibatches of 4 prompts, one PPO epoch and GAS=2. The 4-step value is a configurable small-run default, not a hard limit. A four-rank deployment with GAS=4 completed six online batches, five nonzero-advantage rounds and ten optimizer updates. This example defaults to eight ranks and GAS=2; validate capacity for the selected hardware. Long-term learning quality remains unverified.

Reward is binary execution match, without a format bonus. It extracts the final SQL after thinking, runs reference and candidate against the same immutable, read-only SQLite database, and compares results. Read-only table_info/table_xinfo metadata table functions are allowed; RANDOM/RANDOMBLOB are rejected. Duplicate rows and column order are preserved. Row order is enforced when the reference has an outer ORDER BY; nested/window ORDER BY does not impose output order. Numeric comparison uses configurable tolerances (1e-6 by default) and bounded matching work. Query time, returned rows and result bytes are bounded. SQLite's VM progress timeout is not a separate OS process hard deadline.

Candidate syntax/limit errors get zero reward with a recorded reason. Missing, invalid or empty reference results raise a dataset error instead of silently assigning zero to the whole group. A one-row NULL or zero result is not an empty result. Formatting alone earns no correctness reward.

`validation_file` is parsed and recorded in the plan; it is not used as training data. Automatic validation/evaluation scheduling is not added by this change. Keep the validation split for a separate evaluation pipeline.

Execution match on a sparse database is a proxy: different SQL can coincidentally return identical rows. It does not prove query equivalence or correctness of every reasoning step. Filter invalid/empty references and check reward false positives separately. Preserve the source revision, tokenizer hash, preprocessing template and split manifest when comparing KTRL, GPU RL and KT SFT.

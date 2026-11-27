import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ktrl.configs.recipe import plan_inputs, plan_reward
from ktrl.trainer.online_batch import ParquetRows, score, select_prompts


class SynSQLIntegrationTest(unittest.TestCase):
    def test_config_accepts_validation_and_sql_options(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train.parquet').touch()
            (root / 'validation.parquet').touch()
            inputs = plan_inputs({'train_file': 'train.parquet', 'validation_file': 'validation.parquet'}, root)
            self.assertEqual(inputs['validation_file'], str(root / 'validation.parquet'))
            reward = plan_reward({'name': 'sql_execution', 'database_root': '.'}, root)
            self.assertEqual(reward['database_root'], str(root))
            for key, value in [('timeout_seconds', True), ('timeout_seconds', -1), ('float_abs_tol', float('nan')), ('max_rows', 0)]:
                with self.assertRaises(ValueError):
                    plan_reward({'name': 'sql_execution', 'database_root': '.', key: value}, root)
            with self.assertRaises(ValueError):
                plan_reward({'name': 'auto', 'database_root': '.'}, root)

    def test_lazy_parquet_selection_preserves_global_shuffle_and_bounded_cache(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = [{'prompt': [{'role': 'user', 'content': str(i)}], 'data_source': 'synsql',
                 'reward_model': {'ground_truth': json.dumps({'db_id': 'demo', 'sql': 'SELECT 1'})}} for i in range(30)]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.parquet'
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
            lazy = ParquetRows(path, ['prompt', 'data_source', 'reward_model'])
            tokenize = lambda prompt: [int(prompt[0]['content'])]
            self.assertEqual(select_prompts(lazy, tokenize, 16, 42, 8), select_prompts(rows, tokenize, 16, 42, 8))
            self.assertLessEqual(len(lazy.cache), 2)
            with self.assertRaises(IndexError):
                lazy[len(rows)]

    def test_live_scoring_dispatches_sql_reward_and_grpo(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'databases/demo').mkdir(parents=True)
            with sqlite3.connect(root / 'databases/demo/demo.sqlite') as c:
                c.execute('CREATE TABLE t(x INTEGER)')
                c.execute('INSERT INTO t VALUES(3)')
            c.close()
            (root / 'rollout').mkdir()
            gold = json.dumps({'db_id': 'demo', 'sql': 'SELECT x FROM t'})
            prompts = [{'prompt': [1], 'ground_truth': gold, 'data_source': 'synsql', 'group': 'q0'}]
            recipe = {'max_response_tokens': 2048, 'overlong_buffer_tokens': 0, 'responses_per_prompt': 2,
                      'prompts_per_step': 1, 'responses_per_update': 2, 'policy_version': 'v0',
                      'reward': {'name': 'sql_execution', 'database_root': str(root / 'databases')}}
            (root / 'prompts.json').write_text(json.dumps(prompts))
            (root / 'round.json').write_text(json.dumps(recipe))
            for wave, token in enumerate((10, 20)):
                uid = f'm4-w{wave}-q00'
                payload = {'output_ids': [token], 'meta_info': {'id': uid, 'completion_tokens': 1,
                           'prompt_tokens': 1, 'finish_reason': {'type': 'stop'}, 'output_token_logprobs': [[-.1, token]]}}
                (root / 'rollout' / (uid + '.json')).write_text(json.dumps(payload))
            tokenizer = SimpleNamespace(decode=lambda ids, **kw: '```sql\nSELECT x FROM t\n```' if ids[0] == 10 else '```sql\nSELECT 9\n```')
            fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
            with patch.dict('sys.modules', {'transformers': fake}):
                score(root, 'unused')
            rewards = json.loads((root / 'batch/rewards.json').read_text())
            self.assertEqual([r['score'] for r in rewards], [1., 0.])
            batch = json.loads((root / 'batch/fixed-rows.json').read_text())
            self.assertGreater(batch[0]['advantage'], 0)
            self.assertLess(batch[1]['advantage'], 0)
            self.assertAlmostEqual(sum(x['advantage'] for x in batch), 0)
            audit = json.loads((root / 'batch/batch-audit.json').read_text())
            self.assertIn('synsql', audit['reward_recipe'])
            self.assertEqual(audit['nonzero_advantage_responses'], 2)


if __name__ == '__main__':
    unittest.main()

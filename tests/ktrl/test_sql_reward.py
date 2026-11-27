import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from ktrl.trainer.sql_reward import SqlReward, extract_sql, has_top_level_order_by, parse_ground_truth, results_equal


class SqlRewardTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'demo').mkdir()
        self.db = self.root / 'demo/demo.sqlite'
        with sqlite3.connect(self.db) as c:
            c.execute('CREATE TABLE items(value REAL, kind TEXT)')
            c.executemany('INSERT INTO items VALUES (?, ?)', [(1, 'a'), (2, 'b'), (2, 'b'), (None, 'c')])
        c.close()
        self.scorer = SqlReward({'database_root': str(self.root)})

    def tearDown(self):
        self.temp.cleanup()

    def gold(self, sql):
        return json.dumps({'db_id': 'demo', 'sql': sql})

    def score(self, pred, gold='SELECT value, kind FROM items'):
        return self.scorer('Reasoning.\n```sql\n' + pred + '\n```', self.gold(gold))

    def test_nonempty_execution_match(self):
        result = self.score('SELECT value, kind FROM items')
        self.assertEqual(result['score'], 1)
        self.assertEqual(result['reference_rows'], 4)
        self.assertTrue(result['execution_match'])

    def test_bag_semantics_preserve_duplicates(self):
        self.assertEqual(self.score('SELECT DISTINCT value, kind FROM items')['score'], 0)
        self.assertEqual(self.score('SELECT value, kind FROM items ORDER BY value DESC')['score'], 1)

    def test_column_order_is_preserved(self):
        self.assertEqual(self.score('SELECT kind, value FROM items')['score'], 0)

    def test_ordered_reference_preserves_row_order(self):
        result = self.score('SELECT value FROM items WHERE value IS NOT NULL ORDER BY value DESC',
                            'SELECT value FROM items WHERE value IS NOT NULL ORDER BY value')
        self.assertTrue(result['ordered'])
        self.assertEqual(result['score'], 0)

    def test_top_level_order_detection_ignores_comments_quotes_and_windows(self):
        self.assertFalse(has_top_level_order_by("SELECT 'ORDER BY', ROW_NUMBER() OVER (ORDER BY x) FROM t /* ORDER BY y */"))
        self.assertFalse(has_top_level_order_by('WITH x AS (SELECT 1 ORDER BY 1) SELECT * FROM x'))
        self.assertTrue(has_top_level_order_by('WITH x AS (SELECT 1) SELECT * FROM x ORDER /*note*/ BY 1'))

    def test_numeric_tolerance_and_string_types(self):
        self.assertEqual(self.score('SELECT value + 0.0000001, kind FROM items')['score'], 1)
        self.assertEqual(self.score('SELECT CAST(value AS TEXT), kind FROM items')['score'], 0)

    def test_float_duplicates_need_non_greedy_matching(self):
        self.assertTrue(results_equal([(0.,), (.15,)], [(.1,), (0.,)], ordered=False, abs_tol=.1, rel_tol=0))
        self.assertFalse(results_equal([(0.,), (.15,)], [(0.,), (0.,)], ordered=False, abs_tol=.01, rel_tol=0))

    def test_null_and_zero_are_valid_nonempty_results(self):
        self.assertEqual(self.score('SELECT NULL', 'SELECT NULL')['score'], 1)
        self.assertEqual(self.score('SELECT 0', 'SELECT 0')['score'], 1)
        self.assertEqual(self.score('SELECT NULL', 'SELECT 0')['score'], 0)

    def test_empty_reference_is_dataset_error(self):
        with self.assertRaisesRegex(ValueError, 'Empty SynSQL reference'):
            self.score('SELECT 1', 'SELECT 1 WHERE 0')

    def test_invalid_reference_is_dataset_error(self):
        with self.assertRaisesRegex(ValueError, 'Invalid SynSQL reference'):
            self.score('SELECT 1', 'SELECT unknown FROM items')

    def test_invalid_candidate_is_zero_reward(self):
        result = self.score('SELECT unknown FROM items')
        self.assertEqual(result['candidate_status'], 'sql_error')
        self.assertEqual(result['score'], 0)

    def test_empty_candidate_is_not_reference_match(self):
        self.assertEqual(self.score('SELECT value, kind FROM items WHERE 0')['score'], 0)

    def test_writes_and_multiple_statements_cannot_change_database(self):
        for pred in ('WITH x AS (SELECT 1) DELETE FROM items RETURNING value',
                     'SELECT value, kind FROM items; DELETE FROM items;',
                     "SELECT load_extension('not-an-extension')"):
            self.assertEqual(self.score(pred)['score'], 0)
        with sqlite3.connect(self.db) as c:
            self.assertEqual(c.execute('SELECT COUNT(*) FROM items').fetchone()[0], 4)
        c.close()

    def test_pragma_and_attach_are_not_queries(self):
        self.assertEqual(extract_sql('PRAGMA user_version')[0], '')
        self.assertEqual(extract_sql("ATTACH DATABASE 'x' AS x")[0], '')

    def test_readonly_schema_table_function_is_supported(self):
        sql = "SELECT name, type FROM pragma_table_info('items')"
        self.assertEqual(self.score(sql, sql)['score'], 1)

    def test_random_functions_cannot_produce_rewards(self):
        self.assertEqual(self.score('SELECT RANDOM()')['score'], 0)
        with self.assertRaisesRegex(ValueError, 'Invalid SynSQL reference'):
            self.score('SELECT 1', 'SELECT RANDOM()')

    def test_only_final_sql_after_thinking_is_scored(self):
        self.assertEqual(extract_sql('<think>```sql\nSELECT 1\n```')[0], '')
        sql, _ = extract_sql('<think>```sql\nSELECT 2\n```</think>\n<answer>```sql\nSELECT 1\n```</answer>')
        self.assertEqual(sql, 'SELECT 1')

    def test_path_traversal_rejected(self):
        for db in ('../demo', '/demo', '..', 'C:\\demo'):
            with self.assertRaises(ValueError):
                parse_ground_truth({'db_id': db, 'sql': 'SELECT 1'})

    def test_timeout(self):
        scorer = SqlReward({'database_root': str(self.root), 'timeout_seconds': .02})
        pred = 'WITH RECURSIVE x(v) AS (SELECT 1 UNION ALL SELECT v+1 FROM x WHERE v<1000000000) SELECT SUM(v) FROM x'
        result = scorer('```sql\n' + pred + '\n```', self.gold('SELECT 1'))
        self.assertEqual(result['candidate_status'], 'timeout')
        self.assertEqual(result['score'], 0)

    def test_row_limit(self):
        scorer = SqlReward({'database_root': str(self.root), 'max_rows': 2})
        result = scorer('SELECT value FROM items', self.gold('SELECT 1'))
        self.assertEqual(result['candidate_status'], 'result_limit')

    def test_cached_reference_and_candidate_reexecution(self):
        self.assertEqual(self.score('SELECT 1', 'SELECT 1')['score'], 1)
        self.assertEqual(self.score('SELECT 2', 'SELECT 1')['score'], 0)
        self.assertEqual(len(self.scorer.cache), 1)


if __name__ == '__main__':
    unittest.main()

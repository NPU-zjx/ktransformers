import unittest

from ktrl.trainer.online_batch import select_prompts


class OnlineDatasetTest(unittest.TestCase):
    def test_one_shuffled_pass_preserves_all_rows_including_duplicate_text(self):
        rows = [
            {"prompt": [index % 11], "data_source": "math_dapo", "reward_model": {"ground_truth": index}}
            for index in range(32)
        ]
        selected = select_prompts(rows, list, 32, 42, 1024)
        self.assertEqual(selected, select_prompts(rows, list, 32, 42, 1024))
        self.assertEqual({row["dataset_row"] for row in selected}, set(range(32)))
        self.assertEqual(len({row["group"] for row in selected}), 32)
        self.assertEqual(len({tuple(row["prompt"]) for row in selected}), 11)
        self.assertTrue(all(row["ground_truth"] == row["dataset_row"] for row in selected))

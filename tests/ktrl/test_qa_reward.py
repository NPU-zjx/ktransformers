import unittest

from ktrl.trainer.online_batch import select_prompts
from ktrl.trainer.qa_reward import compute_qa_score, exact_match_score, f1_score


class QARewardTest(unittest.TestCase):
    def test_exact_match_and_format_tag(self):
        res = compute_qa_score("Brief reasoning.\n<answer>The United States!</answer>", "United States")
        self.assertTrue(res["acc"])
        self.assertEqual(res["em"], 1.0)
        self.assertEqual(res["f1"], 1.0)
        self.assertTrue(res["has_format"])
        self.assertEqual(res["score"], 1.0)

    def test_partial_f1_provides_non_binary_signal(self):
        res = compute_qa_score("Thinking...\n<answer>Scott Derrickson</answer>", "Scott Derrickson Jr.")
        self.assertFalse(res["acc"])
        self.assertGreater(res["f1"], 0.5)
        self.assertLess(res["f1"], 1.0)
        self.assertGreater(res["score"], 0.6)
        self.assertLess(res["score"], 1.0)

    def test_yes_no_strictness(self):
        self.assertEqual(f1_score("yes", "no"), 0.0)
        self.assertEqual(exact_match_score("Yes.", "yes"), 1.0)
        res = compute_qa_score("<answer>yes</answer>", "no")
        self.assertEqual(res["f1"], 0.0)
        self.assertEqual(res["score"], 0.1)

    def test_missing_tag_fallback_and_empty(self):
        res = compute_qa_score("Final answer:\nParis", "Paris")
        self.assertTrue(res["acc"])
        self.assertFalse(res["has_format"])
        self.assertEqual(res["score"], 0.9)
        empty = compute_qa_score("", "Paris")
        self.assertEqual(empty["score"], 0.0)

    def test_select_prompts_accepts_hotpotqa(self):
        rows = [
            {"prompt": [1, 2, 3], "data_source": "hotpotqa", "reward_model": {"ground_truth": "yes"}},
            {"prompt": [1, 2, 3, 4, 5], "data_source": "hotpotqa", "reward_model": {"ground_truth": "no"}},
            {"prompt": [1, 2], "data_source": "unknown", "reward_model": {"ground_truth": "x"}},
        ]
        selected = select_prompts(rows, lambda x: x, count=1, seed=7, max_prompt_tokens=4)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["data_source"], "hotpotqa")
        self.assertEqual(selected[0]["ground_truth"], "yes")


if __name__ == "__main__":
    unittest.main()

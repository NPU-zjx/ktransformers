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


class OverlongRewardBoundaryTest(unittest.TestCase):
    def _run_score_case(self, response_len, maximum, buffer, text="<answer>yes</answer>", gold="yes"):
        import json, tempfile
        from pathlib import Path
        from unittest import mock
        from ktrl.trainer.online_batch import score

        class FakeTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return text

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "rollout").mkdir()
            prompts = [{"prompt": [1, 2, 3], "ground_truth": gold, "data_source": "hotpotqa", "group": "g0"}]
            (out / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
            recipe = {
                "max_response_tokens": maximum,
                "overlong_buffer_tokens": buffer,
                "responses_per_prompt": 2,
                "prompts_per_step": 1,
                "responses_per_update": 2,
                "policy_version": 0,
            }
            (out / "round.json").write_text(json.dumps(recipe), encoding="utf-8")
            for w in range(2):
                uid = f"m4-w{w}-q00"
                ids = list(range(10, 10 + response_len))
                payload = {
                    "output_ids": ids,
                    "meta_info": {
                        "id": uid,
                        "completion_tokens": response_len,
                        "prompt_tokens": 3,
                        "finish_reason": {"type": "stop" if response_len < maximum else "length"},
                        "output_token_logprobs": [[-0.1, tid] for tid in ids],
                    },
                }
                (out / "rollout" / f"{uid}.json").write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                score(out, "/unused")
            details = json.loads((out / "batch" / "rewards.json").read_text(encoding="utf-8"))
            return details[0]

    def test_buffer_zero_has_no_overlong_penalty_at_max(self):
        d = self._run_score_case(response_len=512, maximum=512, buffer=0)
        self.assertEqual(d["f1"], 1.0)
        self.assertEqual(d["em"], 1.0)
        self.assertTrue(d["has_format"])
        self.assertEqual(d["overlong_reward"], 0.0)
        self.assertEqual(d["reward"], 1.0)

    def test_buffer_nonzero_thresholds(self):
        # threshold = 512 - 64 = 448
        below = self._run_score_case(response_len=447, maximum=512, buffer=64)
        at_thr = self._run_score_case(response_len=448, maximum=512, buffer=64)
        mid = self._run_score_case(response_len=480, maximum=512, buffer=64)
        at_max = self._run_score_case(response_len=512, maximum=512, buffer=64)
        self.assertEqual(below["overlong_reward"], 0.0)
        self.assertEqual(at_thr["overlong_reward"], 0.0)
        self.assertAlmostEqual(mid["overlong_reward"], -0.5, places=6)
        self.assertAlmostEqual(mid["reward"], 0.5, places=6)
        self.assertAlmostEqual(at_max["overlong_reward"], -1.0, places=6)
        self.assertAlmostEqual(at_max["reward"], 0.0, places=6)

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ktrl.trainer.online_batch import score, select_prompts
from ktrl.trainer.online_native import generate


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

    def test_non_baseline_rollout_and_scoring_use_recipe_shape(self):
        questions = [
            {"prompt": [index], "ground_truth": "10", "group": str(index), "dataset_row": index}
            for index in range(2)
        ]
        inputs = dict(seed=17, prompts_per_step=2, responses_per_prompt=3,
                      responses_per_update=6, max_response_tokens=17, overlong_buffer_tokens=0)

        class Client:
            def call(self, path, data, **kwargs):
                if path != "/generate":
                    return {}
                wave = int(data["rid"][4])
                ids = [10 + wave, 13]
                return {"output_ids": ids, "meta_info": {
                    "id": data["rid"], "completion_tokens": 2, "prompt_tokens": 1,
                    "finish_reason": {"type": "stop"}, "output_token_logprobs": [[-1.0, t] for t in ids],
                }}

        tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens[0]))
        fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
        with TemporaryDirectory() as temp, patch("ktrl.trainer.phase_completion.wait_phase_traces",
                                                 return_value=[{"rank": 0, "rows": [{"phase": "DECODE"}]}]) as traces:
            out = Path(temp)
            (out / "prompts.json").write_text(json.dumps(questions), encoding="utf-8")
            (out / "round.json").write_text(json.dumps(dict(inputs, policy_version="v0")), encoding="utf-8")
            self.assertEqual(generate(Client(), out / "rollout", questions, inputs, "v0", tp_size=1), 12)
            traces.assert_called_once_with(out / "rollout" / "phase", world_size=1)
            with patch.dict("sys.modules", {"transformers": fake_transformers}), patch(
                "ktrl.trainer.online_batch.compute_score",
                side_effect=lambda text, target: {"score": 1 if text == target else -1},
            ):
                score(out, "unused")
            audit = json.loads((out / "batch" / "batch-audit.json").read_bytes())
            self.assertEqual((audit["questions"], audit["records"], audit["nonzero_advantage_responses"]), (2, 6, 6))

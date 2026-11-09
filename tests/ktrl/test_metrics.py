import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ktrl.trainer.metrics import batch_metrics


class MetricsTest(unittest.TestCase):
    def test_reports_loss_and_token_weighted_logp_difference(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "round-03"
            (current / "actor").mkdir(parents=True)
            (current / "batch").mkdir()
            evidence = {
                "resolved-plan.json": {"actor": {"world_size": 2}, "gradient_accumulation_steps": 1},
                "round-03/round.json": {"dataset_batch": 4, "policy_version": "v2", "max_response_tokens": 3},
                "round-03/actor/accepted.json": {"end_step": 3},
                "round-03/actor/preoptimizer-rank0.json": {"grad_norm_preclip": 0.25},
                "round-03/batch/batch-audit.json": {"zero_advantage_groups": 0, "nonzero_advantage_responses": 2},
                "round-03/batch/rewards.json": [
                    {"uid": "a", "score": 1, "reward": 0.5, "response_tokens": 1},
                    {"uid": "b", "score": -1, "reward": -1, "response_tokens": 3},
                ],
            }
            for path, value in evidence.items():
                (root / path).write_text(json.dumps(value), encoding="utf-8")
            for rank, (uid, loss, drift) in enumerate((("a", -2, 0.1), ("b", 1, 0.3))):
                (current / "actor" / f"events-rank{rank}.jsonl").write_text(
                    json.dumps({"phase": "backward_start", "loss": loss}) + "\n"
                    + json.dumps({"phase": "actor_rollout_logprob_difference", "uid": uid, "mean_abs": drift}),
                    encoding="utf-8",
                )
            actual = batch_metrics(root, current)
            self.assertEqual(actual["dataset_batch"], 4)
            self.assertEqual(actual["optimizer_step"], 3)
            self.assertEqual(actual["loss"], -0.5)
            self.assertEqual(actual["answer_accuracy"], 0.5)
            self.assertEqual(actual["at_output_limit_fraction"], 0.5)
            self.assertAlmostEqual(actual["token_mean_abs_actor_rollout_logp_diff"], 0.25)

            (current / "skipped.json").touch()
            skipped = batch_metrics(root, current)
            self.assertTrue(skipped["skipped"])
            self.assertIsNone(skipped["optimizer_step"])
            self.assertNotIn("loss", skipped)

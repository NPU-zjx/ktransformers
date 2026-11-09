import unittest

from ktrl.trainer.grpo import outcome_advantages


class AdvantageTest(unittest.TestCase):
    def test_mixed_and_constant_rewards_with_interleaved_questions(self):
        scores = [-2, 1, -2, 1, -2, 1, 1, 1]
        actual = outcome_advantages(["a", "b"] * 4, scores, group_size=4)
        self.assertEqual(actual[1::2], [0.0] * 4)
        for got, centered in zip(actual[::2], [-0.75, -0.75, -0.75, 2.25]):
            self.assertAlmostEqual(got, centered / (1.5 + 1e-6))

    def test_incomplete_group_fails(self):
        with self.assertRaises(ValueError):
            outcome_advantages(["a"] * 3, [1, 0, 1], group_size=4)

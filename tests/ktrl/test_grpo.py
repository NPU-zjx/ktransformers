import unittest

from ktrl.trainer.grpo import outcome_advantages, policy_loss_sum, token_tis_weights


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

    def test_token_tis_reweights_gradients_even_when_policy_ratio_is_one(self):
        import torch

        old = torch.full((3,), -2.0, requires_grad=True)
        rollout = torch.tensor([0.25, 4.0, 1.0]).log().neg().sub(2.0).requires_grad_()
        current = old.detach().clone().requires_grad_()
        mask = torch.tensor([True, True, False])
        weights = token_tis_weights(old, rollout, 2.0)
        torch.testing.assert_close(weights, torch.tensor([0.25, 2.0, 1.0]))
        loss = policy_loss_sum(current, old, torch.tensor(1.0), mask, rollout_is_weights=weights)
        self.assertAlmostEqual(loss.item(), -2.25)
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([-0.25, -2.0, 0.0]))
        self.assertIsNone(old.grad)
        self.assertIsNone(rollout.grad)
        with self.assertRaises(ValueError):
            token_tis_weights(old, rollout[:0], 2.0)

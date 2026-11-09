"""Advance the constant LR schedule once per rollout update, as veRL does."""
from torch.optim.lr_scheduler import LambdaLR


class RolloutScheduler(LambdaLR):
    def __init__(self, optimizer, optimizer_steps_per_rollout):
        self.optimizer_steps_per_rollout = optimizer_steps_per_rollout
        self.optimizer_calls = 0
        self.initializing = True
        super().__init__(optimizer, lambda _: 1.)
        self.initializing = False

    def step(self, epoch=None):
        if self.initializing:
            return super().step(epoch)
        self.optimizer_calls += 1
        if self.optimizer_calls % self.optimizer_steps_per_rollout == 0:
            return super().step(epoch)

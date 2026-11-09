import unittest
import torch
from ktrl.trainer.ppo_scheduler import RolloutScheduler

class SchedulerTest(unittest.TestCase):
    def test_tick_once_per_rollout_and_resume(self):
        p=torch.nn.Parameter(torch.ones(1))
        o=torch.optim.AdamW([p],lr=1e-6)
        s=RolloutScheduler(o,4)
        for i in range(4):
            p.grad=torch.ones(1);o.step();s.step()
            self.assertEqual(s.last_epoch,(i+1)//4)
        state=s.state_dict()
        other=RolloutScheduler(o,4);other.load_state_dict(state)
        for i in range(4):
            p.grad=torch.ones(1);o.step();other.step()
        self.assertEqual(other.last_epoch,2)
        self.assertEqual(other.optimizer_calls,8)

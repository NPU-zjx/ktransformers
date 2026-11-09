import unittest
import torch
from ktrl.trainer.ppo_schedule import schedule
from ktrl.trainer.grpo import policy_loss_sum


class PPOScheduleTest(unittest.TestCase):
    def test_two_minibatches_two_epochs_cover_each_response_twice(self):
        rows = [{"uid":str(i), "response":[0]*(i+1)} for i in range(8)]
        entries = schedule(rows, 4, 2, shuffle=True, seed=7)
        self.assertEqual(len(entries), 16)
        for epoch in range(2):
            part = entries[epoch*8:(epoch+1)*8]
            self.assertEqual(sorted(x[0] for x in part), list(range(8)))
            for start in (0,4):
                block=part[start:start+4]
                self.assertEqual({x[3] for x in block}, {sum(len(rows[x[0]]["response"]) for x in block)})
        self.assertEqual(entries, schedule(rows,4,2,shuffle=True,seed=7))

    def test_frozen_old_survives_four_optimizer_steps_and_clip_activates(self):
        logits=torch.nn.Parameter(torch.tensor([0., 0.]))
        optimizer=torch.optim.SGD([logits],lr=.4)
        old=logits.detach().log_softmax(-1)[0].clone()
        reference=old.clone()
        ratios=[]
        for _ in range(4):
            optimizer.zero_grad()
            current=logits.log_softmax(-1)[0]
            ratios.append((current.detach()-old).exp().item())
            loss=policy_loss_sum(current[None],old[None],torch.tensor(1.),torch.ones(1,dtype=torch.bool))
            loss.backward()
            optimizer.step()
            torch.testing.assert_close(old,reference,rtol=0,atol=0)
        self.assertEqual(ratios[0],1.)
        self.assertGreater(ratios[-1],1.28)
        self.assertIsNone(old.grad)

    def test_minibatch_token_normalization_with_two_ranks(self):
        raw=torch.tensor([2.,3.,5.,7.],requires_grad=True)
        tokens=[2,4,3,5]
        # world=2, GAS=2; local losses divide by GAS then ranks average.
        losses=raw*4/sum(tokens)
        effective=(losses[0::2].sum()/2+losses[1::2].sum()/2)/2
        effective.backward()
        torch.testing.assert_close(raw.grad,torch.full((4,),1/sum(tokens)))

    def test_partial_minibatch_rejected(self):
        with self.assertRaises(ValueError):
            schedule([{"response":[1]}]*3,2,1)

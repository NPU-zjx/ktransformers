import copy
import unittest
import torch
from ktrl.trainer.master_adamw import MasterAdamW,assert_compute_copy


class MasterTest(unittest.TestCase):
    def test_small_updates_accumulate_and_resume_exactly(self):
        p=torch.nn.Parameter(torch.tensor([.01,-.02,.001],dtype=torch.bfloat16))
        original=p.detach().clone()
        reference=torch.nn.Parameter(p.detach().float())
        opt=MasterAdamW([p],cpu_parameters=[p],cpu_threads=2,lr=1e-6,weight_decay=.01)
        ref=torch.optim.AdamW([reference],lr=1e-6,weight_decay=.01,foreach=False)
        for step in range(80):
            grad=torch.tensor([.2,-.1,.3],dtype=torch.bfloat16)
            p.grad=grad.clone();reference.grad=grad.float()
            opt.step();ref.step()
            torch.testing.assert_close(opt.state[p]['master'],reference,rtol=0,atol=0)
            for key in ('exp_avg','exp_avg_sq'):
                torch.testing.assert_close(opt.state[p][key],ref.state[reference][key],rtol=0,atol=0)
            assert_compute_copy(p,opt.state[p]['master'])
            if step==0:
                self.assertTrue(torch.equal(p,original))
                self.assertTrue(torch.all(opt.state[p]['master']!=original.float()))
                state=copy.deepcopy(opt.state_dict())
                replacement=torch.nn.Parameter(p.detach().clone())
                resumed=MasterAdamW([replacement],cpu_parameters=[replacement],cpu_threads=2,lr=1e-6,weight_decay=.01)
                resumed.load_state_dict(state)
                self.assertEqual(resumed.state[replacement]['master'].dtype,torch.float32)
                p,opt=replacement,resumed
        self.assertTrue(torch.all(p!=original))
        self.assertEqual(int(opt.state[p]['step']),80)

    def test_legacy_state_rejected(self):
        p=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16))
        old=torch.optim.AdamW([p]);p.grad=torch.ones_like(p);old.step()
        new=MasterAdamW([p],cpu_parameters=[p],cpu_threads=2)
        with self.assertRaisesRegex(ValueError,'legacy'):
            new.load_state_dict(old.state_dict())

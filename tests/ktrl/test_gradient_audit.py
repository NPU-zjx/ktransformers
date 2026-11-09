import unittest
import torch
from ktrl.trainer.gradient_audit import watch_loss_scale, gradient_snapshot, verify_clip


class GradientAuditTest(unittest.TestCase):
    def test_actual_backward_boundary_and_double_division(self):
        records=[]
        p=torch.tensor(3.,requires_grad=True)
        loss=p.square()
        watch_loss_scale(loss,4,records.append)
        (loss/4).backward()
        self.assertEqual(records[0]['backward_scale'],.25)
        self.assertEqual(p.grad.item(),1.5)
        loss=p.square()
        watch_loss_scale(loss,4,records.append)
        with self.assertRaisesRegex(RuntimeError,'Backward loss scale'):
            (loss/16).backward()

    def test_unequal_token_counts_global_normalization(self):
        # Independent serial full-token gradient vs DP mean after four microbatches.
        lengths=[[1,3,5,2],[4,2,1,6]]
        total=sum(map(sum,lengths)); reference=torch.tensor(2.,requires_grad=True)
        data=[torch.arange(n,dtype=torch.float64)+rank+micro for rank,ns in enumerate(lengths) for micro,n in enumerate(ns)]
        sum(((reference*x-1)**2).sum() for x in data).div(total).backward()
        partial=[]
        for rank in range(2):
            p=torch.tensor(2.,requires_grad=True)
            for x in data[rank*4:(rank+1)*4]:
                (((p*x-1)**2).sum()*2*4/total/4).backward()
            partial.append(p.grad)
        torch.testing.assert_close(sum(partial)/2,reference.grad)
        self.assertFalse(torch.isclose(sum(partial)/4,reference.grad))

    def test_joint_clip_includes_owner_and_duplicate_parameter_once(self):
        cpu=torch.nn.Parameter(torch.ones(2)); gpu_role=torch.nn.Parameter(torch.ones(1))
        cpu.grad=torch.tensor([3.,4.]);gpu_role.grad=torch.tensor([12.])
        norm2,rows=gradient_snapshot([('expert',cpu),('alias',cpu),('nonexpert',gpu_role)],{id(cpu)})
        self.assertEqual(norm2,169.)
        value=torch.nn.utils.clip_grad_norm_([cpu,gpu_role],6.5)
        proof=verify_clip(rows,norm2**.5,float(value),6.5)
        self.assertEqual(proof['cpu_expert_gradients_scanned'],1)
        self.assertAlmostEqual(cpu.grad[0].item(),1.5,places=5)

    def test_omitted_expert_clip_rejected(self):
        p=torch.nn.Parameter(torch.ones(1));q=torch.nn.Parameter(torch.ones(1))
        p.grad=torch.tensor([3.]);q.grad=torch.tensor([4.])
        _,rows=gradient_snapshot([('expert',p),('other',q)],{id(p)})
        torch.nn.utils.clip_grad_norm_([q],1.)
        with self.assertRaises(AssertionError):
            verify_clip(rows,5.,5.,1.)

    def test_all_elements_scanned_for_nan_and_missing_grad(self):
        p=torch.nn.Parameter(torch.ones(4097))
        with self.assertRaisesRegex(RuntimeError,'not ready'):
            gradient_snapshot([('expert',p)],{id(p)})
        p.grad=torch.ones_like(p);p.grad[2305]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            gradient_snapshot([('expert',p)],{id(p)},chunk=128)

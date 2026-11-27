import unittest
import torch
from torch import nn
from ktrl.workers.engine.fsdp.residency import release_cuda_gradients, release_unused_router_weights
from ktrl.workers.engine.fsdp.config import plan_actor


class ResidencyTest(unittest.TestCase):
    def test_cpu_gradient_contract_is_untouched(self):
        p=nn.Parameter(torch.ones(4,4,dtype=torch.bfloat16));p.grad=torch.ones_like(p)
        optimizer=torch.optim.AdamW([p]);gradient=p.grad
        self.assertEqual(release_cuda_gradients(optimizer),0)
        self.assertIs(p.grad,gradient)

    def test_only_duplicate_kt_router_alias_removed(self):
        klass=type('KTFixture',(nn.Module,),{'__module__':'kt_kernel.sft.layer'})
        layer=klass();layer.gate=nn.Linear(4,8,bias=False);layer._router_attr='gate'
        layer._original_router=nn.Module();layer._original_router.weight=layer.gate.weight
        layer._original_router.top_k=2
        layer._original_router.register_buffer('e_score_correction_bias',torch.ones(8))
        x=torch.randn(3,4);before=layer.gate(x).detach().clone()
        result=release_unused_router_weights(layer)
        self.assertEqual(result['duplicate_router_parameters_removed'],1)
        self.assertIsNone(layer._original_router.weight)
        self.assertTrue(torch.equal(before,layer.gate(x)))
        self.assertEqual(layer._original_router.top_k,2)
        self.assertTrue(torch.equal(layer._original_router.e_score_correction_bias,torch.ones(8)))
        layer.gate(x).sum().backward();self.assertIsNotNone(layer.gate.weight.grad)
        self.assertEqual(release_unused_router_weights(layer)['duplicate_router_parameters_removed'],0)

    def test_offload_is_explicit_boolean(self):
        placement={'gpu':{'count':0},'cpu':{'count':128}}
        self.assertFalse(plan_actor({},placement)['offload_gpu_state_during_rollout'])
        self.assertTrue(plan_actor({'offload_gpu_state_during_rollout':True},placement)['offload_gpu_state_during_rollout'])
        for invalid in [1,'true',None]:
            with self.assertRaises(ValueError):plan_actor({'offload_gpu_state_during_rollout':invalid},placement)

if __name__=='__main__':unittest.main()

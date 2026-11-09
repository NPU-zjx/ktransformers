"""FP32 optimizer masters for native KT BF16 CPU buffers; GPU AdamW stays native."""
import torch
from torch.optim import AdamW
from torch.optim._functional import adamw


def assert_compute_copy(parameter,master,chunk=1048576):
    assert parameter.device.type==master.device.type=='cpu'
    assert parameter.dtype==torch.bfloat16 and master.dtype==torch.float32
    p,m=parameter.detach().view(-1),master.view(-1)
    for start in range(0,p.numel(),chunk):
        if not torch.equal(p[start:start+chunk],m[start:start+chunk].to(p.dtype)):
            raise RuntimeError('KT BF16 compute buffer differs from rounded FP32 master')


class MasterAdamW(AdamW):
    def __init__(self,params,*,cpu_parameters=(),cpu_threads=48,**kwargs):
        self.cpu_ids={id(p) for p in cpu_parameters}
        self.cpu_threads=cpu_threads
        kwargs.update(foreach=False,fused=False)
        super().__init__(params,**kwargs)
        found={id(p) for g in self.param_groups for p in g['params'] if id(p) in self.cpu_ids}
        assert found==self.cpu_ids
        for g in self.param_groups:
            assert not g['amsgrad'] and not g['maximize'] and not g['capturable'] and not g['differentiable']

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('MasterAdamW does not support closures')
        owned=[(g,p,p.grad) for g in self.param_groups for p in g['params'] if id(p) in self.cpu_ids]
        # Let PyTorch update only GPU FP32 shards. CPU BF16 buffers must never
        # receive a second update or get native zeros_like(BF16) moment state.
        try:
            for _,p,_ in owned:p.grad=None
            result=super().step()
        finally:
            for _,p,grad in owned:p.grad=grad
        threads=torch.get_num_threads()
        try:
            if owned:torch.set_num_threads(self.cpu_threads)
            for group,p,grad in owned:
                if grad is None:continue
                assert p.device.type==grad.device.type=='cpu' and p.dtype==grad.dtype==torch.bfloat16
                state=self.state[p]
                if not state:
                    state.update(step=torch.tensor(0.),master=p.detach().float(),
                        exp_avg=torch.zeros_like(p,dtype=torch.float32),
                        exp_avg_sq=torch.zeros_like(p,dtype=torch.float32))
                for key in ('master','exp_avg','exp_avg_sq'):
                    assert state[key].dtype==torch.float32 and state[key].device.type=='cpu'
                adamw([state['master']],[grad.float()],[state['exp_avg']],[state['exp_avg_sq']],[],[state['step']],
                    foreach=False,capturable=False,differentiable=False,fused=False,amsgrad=False,
                    beta1=group['betas'][0],beta2=group['betas'][1],lr=group['lr'],
                    weight_decay=group['weight_decay'],eps=group['eps'],maximize=False)
                p.copy_(state['master'])
                assert_compute_copy(p,state['master'])
        finally:
            if owned:torch.set_num_threads(threads)
        return result

    def load_state_dict(self,state_dict):
        # Optimizer.load_state_dict normally casts all floating state to p.dtype.
        # Strip KT state before that cast, then restore the ORIGINAL FP32 tensors.
        cpu=[]
        assert len(self.param_groups)==len(state_dict['param_groups'])
        sanitized=dict(state_dict,state=dict(state_dict['state']))
        for live,saved in zip(self.param_groups,state_dict['param_groups']):
            assert len(live['params'])==len(saved['params'])
            for p,index in zip(live['params'],saved['params']):
                if id(p) in self.cpu_ids and index in sanitized['state']:
                    entry=sanitized['state'][index]
                    if entry:
                        for key in ('master','exp_avg','exp_avg_sq'):
                            if key not in entry or entry[key].dtype!=torch.float32:
                                raise ValueError('Cannot resume a legacy BF16-only expert optimizer as FP32 master')
                        cpu.append((p,entry))
                        sanitized['state'][index]={}
        super().load_state_dict(sanitized)
        for p,entry in cpu:
            self.state[p]={k:v.to('cpu') if torch.is_tensor(v) else v for k,v in entry.items()}
            assert_compute_copy(p,self.state[p]['master'])

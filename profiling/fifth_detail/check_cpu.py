"""CPU-only acceptance for collectors, aliases and streaming latency semantics."""
import io
import json
from pathlib import Path
import tempfile
import torch
from ledger import inventory
from observe import container_sample, process_sample
from run import stream_request


def main():
    p = torch.nn.Parameter(torch.ones(32, dtype=torch.bfloat16))
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW([p], lr=1e-6)
    p.grad = torch.ones_like(p)
    opt.step()
    report = inventory(model, opt, {'alias':p[:8], 'another_alias':p})
    assert not report['errors'], report['errors']
    weight = next(r for r in report['storages'] if any(a['role']=='alias' for a in r['aliases']))
    assert weight['storage_bytes'] == 64
    assert len(weight['aliases']) == 3
    assert sum(report['bytes_by_device_category'].values()) == sum(r['storage_bytes'] for r in report['storages'])
    # A real extra storage hidden from named_parameters must be counted; an
    # all-gather alias of that storage must not be counted a second time.
    from types import SimpleNamespace
    class InternalCopies(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.ones(8,dtype=torch.float32))
            copy=torch.ones(8,dtype=torch.bfloat16)
            self.state=SimpleNamespace(_fsdp_param_group=SimpleNamespace(fsdp_params=[
                SimpleNamespace(_unsharded_param=copy,all_gather_outputs=[copy.view(-1)],
                                _sharded_param_data=self.weight.data)]))
        def _get_fsdp_state(self):return self.state
    hidden=inventory(InternalCopies())
    assert not hidden['errors']
    assert sum(hidden['bytes_by_device_category'].values())==48
    assert hidden['bytes_by_device_category']['cpu/fsdp.compute_parameter']==16
    assert int(container_sample()['memory.current']) > 0
    processes = process_sample()
    assert processes['records'] and not processes['errors'], processes['errors']
    class Headers:
        def get_content_type(self): return 'text/event-stream'
    class Response(io.BytesIO): headers = Headers()
    class Opener:
        def open(self, req, **kwargs):
            request = json.loads(req.data)
            assert request['stream'] is True
            rows = [dict(output_ids=[1],meta_info={'completion_tokens':1}),
                    dict(output_ids=[1,2,3],meta_info={'completion_tokens':3,'finish_reason':{'type':'stop'}})]
            return Response(b''.join(b'data: '+json.dumps(r).encode()+b'\n\n' for r in rows)+b'data: [DONE]\n\n')
    class Client: url='http://localhost'; opener=Opener()
    with tempfile.TemporaryDirectory() as folder:
        final = stream_request(Client(), {'rid':'test','input_ids':[8,9]}, Path(folder))
        data = json.loads((Path(folder)/'test.json').read_text())
        assert final['output_ids']==[1,2,3]
        assert data['coalesced_chunks']==1 and data['first_chunk_tokens']==1
        assert data['ttft_ms']>=0 and data['tpot_ms']>=0
    print('CPU_COLLECTOR_PASS storage_aliases=1 cgroup=1 pss=1 sse=1 coalesced_chunks=1')


if __name__=='__main__': main()

"""Wire explicit per-request seeds into the native sampler without changing attention."""
def install():
    import torch
    import os,json,hashlib
    from pathlib import Path
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    original=SamplingBatchInfo.from_schedule_batch.__func__
    @classmethod
    def build(cls,batch,vocab_size):
        result=original(cls,batch,vocab_size)
        seeds=[r.sampling_params.sampling_seed for r in batch.reqs]
        if any(s is not None for s in seeds):
            if not all(isinstance(s,int) for s in seeds):
                raise ValueError('KTRL batches require an explicit seed for every request')
            result.sampling_seed=torch.tensor(seeds,dtype=torch.int64,device=batch.device)
        return result
    SamplingBatchInfo.from_schedule_batch=build
    # Optional, bounded service-level evidence. Active only for short acceptance
    # requests; formal rollout does not copy probability tensors to the host.
    from sglang.srt.layers.sampler import Sampler,sampling_from_probs_torch
    original_sample=Sampler._sample_from_probs
    root=Path(os.environ['KTRL_PUBLICATION_OUTPUT'])
    def sample(self,probs,info,positions,simple):
        marker=root/'seed-audit-active.json'
        if not marker.exists():return original_sample(self,probs,info,positions,simple)
        request=json.loads(marker.read_text())
        seeds=info.sampling_seed
        record=dict(tag=request['tag'],seeds=seeds.tolist() if seeds is not None else None,
            positions=positions.tolist(),simple=bool(simple))
        target=root/f'seed-audit-rank{torch.distributed.get_rank()}.jsonl'
        if seeds is None or not torch.all(seeds==request['expected_seed']):
            with target.open('a') as handle:handle.write(json.dumps(record)+'\n')
            raise RuntimeError('Explicit request seed did not reach native sampler')
        assert simple
        record['probabilities_sha256']=hashlib.sha256(probs.detach().float().cpu().numpy().tobytes()).hexdigest()
        actual=original_sample(self,probs,info,positions,simple)
        repeat=sampling_from_probs_torch(probs,sampling_seed=seeds,positions=positions)
        record.update(output_ids=actual.tolist(),same_probabilities_repeat_exact=torch.equal(actual,repeat))
        with target.open('a') as handle:handle.write(json.dumps(record)+'\n')
        assert record['same_probabilities_repeat_exact']
        return actual
    Sampler._sample_from_probs=sample

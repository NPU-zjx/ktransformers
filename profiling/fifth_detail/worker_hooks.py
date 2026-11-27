import weakref
LIVE_TRAINER_REF = None
"""Version-checked observations of Trainer and SGLang; no optimizer/loss changes."""
import atexit
import functools
import json
import os
from pathlib import Path
import pickle
import threading
import time
import types

from ledger import inventory

OUT = Path(os.environ['KTRL_DETAIL_PROFILE'])
OUT.mkdir(parents=True, exist_ok=True)
ROLE = os.environ['KTRL_DETAIL_ROLE']
DIAGNOSTIC = os.environ.get('KTRL_DETAIL_MODE') == 'diagnostic'
LOCK = threading.Lock()
CAPTURE_LOCK = threading.Lock()
PENDING = []
CONTEXT = {}
SEEN = set()
HIGHWATER = {}
PEAK_METRICS = ('peak_allocated', 'peak_reserved', 'peak_requested_bytes')


def emit(name, row):
    with LOCK, (OUT/f'{name}-{ROLE}-pid{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps(dict(time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
            pid=os.getpid(), rank=os.environ.get('RANK'), **row), allow_nan=False)+'\n')


def cuda_stats():
    import torch
    if not torch.cuda.is_initialized():
        return dict(cuda_initialized=False)
    device = torch.cuda.current_device()
    s = torch.cuda.memory_stats(device)
    return dict(device=device, gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
        allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
        peak_allocated=torch.cuda.max_memory_allocated(device),
        peak_reserved=torch.cuda.max_memory_reserved(device),
        inactive_split_bytes=s.get('inactive_split_bytes.all.current'),
        active_bytes=s.get('active_bytes.all.current'),
        requested_bytes=s.get('requested_bytes.all.current'),
        peak_requested_bytes=s.get('requested_bytes.all.peak'),
        retries=s.get('num_alloc_retries'), oom=s.get('num_ooms'))


def drain(wait=False):
    remaining = []
    for label, begin, end, fields in PENDING:
        if wait: end.synchronize()
        if end.query():
            emit('cuda-intervals', dict(label=label, stream_ms=begin.elapsed_time(end), **fields))
        else: remaining.append((label, begin, end, fields))
    PENDING[:] = remaining


def measured(label, fn, *args, **kwargs):
    import torch
    fields = dict(start_ns=time.time_ns(), **CONTEXT)
    pair = None
    if torch.cuda.is_initialized():
        drain()
        pair = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        pair[0].record()
        torch.cuda.nvtx.range_push('KTRL_DETAIL/'+label)
    try:
        result = fn(*args, **kwargs)
    except BaseException as error:
        emit('errors', dict(label=label, error=repr(error)))
        raise
    finally:
        if pair:
            pair[1].record()
            torch.cuda.nvtx.range_pop()
            PENDING.append((label, *pair, fields))
        emit('timing', dict(label=label, end_ns=time.time_ns(), **fields, **cuda_stats()))
    return result


def mark(label, trainer=None, model=None, extra=None, shot=False):
    import torch
    drain()
    row = dict(label=label, **CONTEXT, **cuda_stats())
    if trainer is not None:
        model = trainer.model
        acc = getattr(trainer, 'bf16_accumulator', None)
        if acc is not None:
            extra = dict(extra or {}, bf16_accumulation=[v[2] for v in acc.buffers.values()])
    row['inventory'] = inventory(model, getattr(trainer, 'optimizer', None), extra)
    emit('boundaries', row)
    if shot and DIAGNOSTIC and torch.cuda.is_initialized():
        path = OUT/f'snapshot-{ROLE}-{os.getpid()}-{time.time_ns()}-{label}.pickle'
        with path.open('xb') as f: pickle.dump(torch.cuda.memory._snapshot(), f)
        emit('snapshots', dict(label=label, path=str(path)))


def _capture_highwater(label):
    """Durably retain the history when the allocator reports a new highwater."""
    if not DIAGNOSTIC:
        return
    import torch
    if not torch.cuda.is_initialized():
        return
    stats = cuda_stats()
    context = dict(CONTEXT)
    scope = str(context.get('window_id', 'lifetime')) if ROLE == 'rollout' else 'lifetime'
    improved = [name for name in PEAK_METRICS
                if (stats.get(name) or 0) > HIGHWATER.get((scope, name), 0)]
    if not improved:
        return
    snapshot = torch.cuda.memory._snapshot()
    snapshot['ktrl_highwater'] = dict(label=label, scope=scope, time_ns=time.time_ns(),
                                     metrics=improved, **context, **stats)
    prefix = f'highwater-{ROLE}-{os.getpid()}-{scope}'
    staging = OUT/f'{prefix}.tmp'
    with staging.open('wb') as handle:
        pickle.dump(snapshot, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        for name in improved:
            path = OUT/f'{prefix}-{name}.pickle'
            pending = path.with_suffix('.next')
            os.link(staging, pending)
            os.replace(pending, path)
            HIGHWATER[scope, name] = stats[name]
            emit('highwaters', dict(label=label, scope=scope, metric=name,
                path=str(path), **context, **stats))
    finally:
        staging.unlink()


def capture_highwater(label):
    if not DIAGNOSTIC:
        return
    with CAPTURE_LOCK:
        _capture_highwater(label)


def allocator_sampler():
    import torch
    device = torch.cuda.current_device()
    stop = threading.Event()
    def loop():
        # CUDA's current device is thread-local; a new thread otherwise reads
        # device 0 for ranks 1..7, producing a misleading zero-byte curve.
        torch.cuda.set_device(device)
        while not stop.wait(.5):
            try:
                emit('allocator', dict(**CONTEXT, **cuda_stats()))
                if ROLE == 'actor':
                    capture_highwater('periodic')
                    trainer = LIVE_TRAINER_REF() if LIVE_TRAINER_REF is not None else None
                    parking = getattr(trainer, 'rollout_state_parking', None)
                    key = ('actor_idle', CONTEXT.get('optimizer_step'))
                    if parking is not None and parking.parked and key not in SEEN:
                        SEEN.add(key)
                        mark('actor_idle', trainer=trainer,
                             extra={'parked_gpu_state_cpu_backup':[e[2] for e in parking.parked]}, shot=True)
            except Exception as error: emit('errors', dict(label='allocator', error=repr(error)))
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    def finish():
        stop.set()
        thread.join(2)
        try:
            torch.cuda.set_device(device)
            if ROLE == 'actor':
                capture_highwater('process_exit')
            emit('allocator-final', dict(**CONTEXT, **cuda_stats()))
            drain(True)
        except Exception as error: emit('errors', dict(label='drain', error=repr(error)))
    atexit.register(finish)


def wrap_instance(instance, name, label, before=None, after=None):
    original = getattr(instance, name)
    @functools.wraps(original)
    def call(*args, **kwargs):
        if before: before(*args, **kwargs)
        result = measured(label, original, *args, **kwargs)
        if after: after(*args, **kwargs)
        return result
    setattr(instance, name, call)


def patch_trainer(module):
    original_init = module.Trainer.__init__
    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        global LIVE_TRAINER_REF
        original_init(self, *args, **kwargs)
        LIVE_TRAINER_REF = weakref.ref(self)
        if type(self).__name__ != 'GRPOTrainer': return
        import torch
        from transformers import TrainerCallback
        emit('installed', dict(target='GRPOTrainer', mode='diagnostic' if DIAGNOSTIC else 'light'))
        if DIAGNOSTIC:
            torch.cuda.memory._record_memory_history(max_entries=262144, stacks='python')
            # Persist the pre-FSDP registered model state too. A lifetime peak
            # predating history cannot be explained by later trace replay.
            mark('model_pre_fsdp', trainer=self, shot=True)
        allocator_sampler()
        def context(*unused, **kw):
            CONTEXT.update(optimizer_step=self.state.global_step,
                           actor_out=str(self.evidence.out))
        def boundary(label): return lambda *a, **k: mark(label, trainer=self, shot=True)
        wrap_instance(self, 'freeze_old_logp', 'old_logp', context, boundary('old_logp_end'))
        wrap_instance(self, 'compute_loss', 'policy_forward_loss', context,
                      lambda *a, **k:capture_highwater('forward_end'))
        wrap_instance(self.accelerator, 'backward', 'backward',
                      after=lambda *a, **k:capture_highwater('backward_end'))
        wrap_instance(self.accelerator, 'clip_grad_norm_', 'clip_grad_norm',
                      after=boundary('after_clip'))
        wrap_instance(self, '_save_checkpoint', 'checkpoint_export',
                      before=boundary('export_start'), after=boundary('export_end'))
        original_acc = self.on_bf16_accumulation
        def accum(*a, **k):
            result = original_acc(*a, **k)
            tag = ('bf16', self.state.global_step)
            if tag not in SEEN:
                SEEN.add(tag)
                mark('bf16_accumulation', trainer=self)
            return result
        self.on_bf16_accumulation = accum
        original_step = self.training_step
        def step(*a, **k):
            context()
            token = 'torch_trace'
            if DIAGNOSTIC and token not in SEEN:
                SEEN.add(token)
                if os.environ.get('KTRL_NSYS_CAPTURE')=='1':
                    # Nsight and torch.profiler must not compete for CUPTI.
                    # NVTX provides operator ranges to the Nsight timeline.
                    with torch.autograd.profiler.emit_nvtx(record_shapes=True):
                        result=measured('training_microbatch',original_step,*a,**k)
                    torch.cuda.cudart().cudaProfilerStop()
                    emit('nsys-range',dict(event='complete',scope='initialization through first microbatch',**CONTEXT))
                    return result
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                      with_stack=True, profile_memory=True) as profiler:
                    result = measured('training_microbatch', original_step, *a, **k)
                profiler.export_chrome_trace(str(OUT/f'actor-microbatch-{os.getpid()}.trace.json'))
                if os.environ.get('KTRL_NSYS_CAPTURE')=='1':
                    torch.cuda.cudart().cudaProfilerStop()
                return result
            return measured('training_microbatch', original_step, *a, **k)
        self.training_step = step
        class Boundaries(TrainerCallback):
            def on_train_begin(cb, args, state, control, **kw):
                context(); mark('train_begin', trainer=self, shot=True)
            def on_pre_optimizer_step(cb, args, state, control, **kw):
                mark('optimizer_start', trainer=self, shot=True)
            def on_optimizer_step(cb, args, state, control, **kw):
                mark('optimizer_end', trainer=self, shot=True)
            def on_step_end(cb, args, state, control, **kw):
                context(); mark('step_end', trainer=self)
        self.add_callback(Boundaries())
        from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
        starts = {}
        def pre(opt, args, kwargs):
            starts[id(opt)] = time.time_ns()
        def post(opt, args, kwargs):
            begin = starts.pop(id(opt), None)
            if begin is not None:
                emit('optimizer-timing', dict(start_ns=begin, end_ns=time.time_ns(), **CONTEXT))
        register_optimizer_step_pre_hook(pre)
        register_optimizer_step_post_hook(post)
    module.Trainer.__init__ = init


def object_tensors(obj, depth=0, visited=None):
    import torch
    visited = set() if visited is None else visited
    if isinstance(obj, torch.Tensor): return obj
    if depth > 4 or id(obj) in visited: return None
    visited.add(id(obj))
    if isinstance(obj, dict):
        return {str(k):object_tensors(v, depth+1, visited) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [object_tensors(v, depth+1, visited) for v in obj]
    if hasattr(obj, '__dict__') and not isinstance(obj, torch.nn.Module):
        return {k:object_tensors(v, depth+1, visited) for k,v in vars(obj).items()
                if not k.startswith('__')}
    return None


def patch_worker(module):
    original = module.TpModelWorker.forward_batch_generation
    rollout_trace = {'profiler': None, 'count': 0, 'done': set()}
    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        import torch
        window = getattr(__import__('sys').modules.get('phase_hooks'), 'window', None)
        if window is not None:
            key = window['window_id']
            CONTEXT.update(tp_rank=self.tp_rank, window_id=key)
            if key not in SEEN:
                SEEN.add(key)
                if 'sampler' not in SEEN:
                    SEEN.add('sampler'); allocator_sampler()
                    if DIAGNOSTIC:
                        torch.cuda.memory._record_memory_history(max_entries=262144, stacks='python')
                runner = self.model_runner
                extras = {name:object_tensors(getattr(runner, name, None)) for name in (
                    'token_to_kv_pool', 'req_to_token_pool', 'cuda_graph_runner', 'memory_saver_adapter')}
                mark('rollout_window_start', model=runner.model, extra=extras, shot=True)
        batch = args[0] if args else kwargs.get('model_worker_batch')
        phase = getattr(getattr(batch, 'forward_mode', None), 'name', '')
        trace_kind = 'prefill' if phase == 'EXTEND' else 'decode' if phase == 'DECODE' else None
        # Bounded diagnostic only: one real prefill and eight decode calls.
        # Formal light runs never enable torch.profiler here.
        if (DIAGNOSTIC and window is not None and trace_kind is not None
                and trace_kind not in rollout_trace['done'] and rollout_trace['profiler'] is None):
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                with_stack=True, profile_memory=True)
            profiler.__enter__()
            rollout_trace.update(profiler=profiler, kind=trace_kind, count=0)
            emit('rollout-trace', dict(event='begin', phase=trace_kind, tp_rank=self.tp_rank))
        result = original(self, *args, **kwargs)
        if rollout_trace['profiler'] is not None:
            rollout_trace['count'] += 1
            limit = 1 if rollout_trace['kind'] == 'prefill' else 8
            if rollout_trace['count'] >= limit:
                profiler=rollout_trace['profiler'];kind=rollout_trace['kind']
                profiler.__exit__(None,None,None)
                target=OUT/f'rollout-{kind}-rank{self.tp_rank}-pid{os.getpid()}.trace.json'
                profiler.export_chrome_trace(str(target))
                rollout_trace['done'].add(kind);rollout_trace['profiler']=None
                emit('rollout-trace', dict(event='end', phase=kind, tp_rank=self.tp_rank,
                    calls=rollout_trace['count'], path=str(target)))
        if DIAGNOSTIC and window is not None:
            capture_highwater('rollout_forward_end')
        return result
    module.TpModelWorker.forward_batch_generation = forward
    emit('installed', dict(target=module.__name__))


def patch(module):
    if module.__name__ == 'transformers.trainer': patch_trainer(module)
    elif module.__name__.endswith('tp_worker'): patch_worker(module)
    else: emit('installed', dict(target=module.__name__, note='native CUDA phase hooks supply timing'))

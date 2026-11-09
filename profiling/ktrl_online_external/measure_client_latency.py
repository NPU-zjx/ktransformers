"""Measure one v0 rollout using the accepted request payloads and SSE token timing."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import signal
import sys
import time
import urllib.request


def read(path):
    return json.loads(path.read_text())


def percentile(values, fraction):
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def measure(url, payload, out):
    request = urllib.request.Request(url + '/generate', json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
    ids, events, meta = [], [], {}
    done = False
    start = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=240) as response:
        assert response.headers.get_content_type() == 'text/event-stream'
        for line in response:
            now = time.perf_counter_ns()
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            assert 'error' not in event, event
            next_ids = event['output_ids']
            assert next_ids[:len(ids)] == ids, 'Expected cumulative token IDs'
            if len(next_ids) > len(ids):
                events.append([now, len(next_ids), len(next_ids) - len(ids)])
                ids = next_ids
            meta = event['meta_info']
    assert done and meta['finish_reason'] and meta['completion_tokens'] == len(ids)
    assert meta['prompt_tokens'] == len(payload['input_ids']) and len(ids) >= 2
    assert events[0][2] == 1, 'First SSE chunk must contain exactly one token'
    first, last = events[0][0], events[-1][0]
    assert start < first < last
    record = dict(rid=payload['rid'], input_tokens=meta['prompt_tokens'], output_tokens=len(ids),
                  start_ns=start, first_token_ns=first, last_token_ns=last,
                  events_ns_tokens_delta=events, ttft_ms=(first-start)/1e6,
                  tpot_ms=(last-first)/1e6/(len(ids)-1), finish_reason=meta['finish_reason'])
    (out / (payload['rid'] + '.json')).write_text(json.dumps(record))
    return record


def main():
    config, formal, out = (Path(arg).resolve() for arg in sys.argv[1:])
    from ktrl.trainer import online_native as native
    from ktrl.trainer.phase_completion import wait_phase_traces
    from ktrl.trainer.precision_gate import verify

    verify(config)
    native.guard_gpu()
    out.mkdir()
    phase = out / 'phase'
    phase.mkdir()
    plan = native.load_plan(config, steps=1, profiling='light')
    payloads = [read(p) for p in sorted((formal / 'round-01/rollout').glob('*-request.json'))]
    assert len(payloads) == 32 and len({p['rid'] for p in payloads}) == 32
    assert sum(len(p['input_ids']) for p in payloads) == 5712
    assert all(p['sampling_params']['max_new_tokens'] == 8192 for p in payloads)
    payloads = [dict(p, stream=True) for p in payloads]
    native.save(out / 'requests.json', payloads)
    native.save(out / 'ledger.json', dict(status='starting', formal_source=str(formal)))

    def timed_out(*_):
        raise TimeoutError('Latency measurement deadline exceeded')

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(1680)
    pool = ThreadPoolExecutor(max_workers=32)
    try:
        with native.service(config, out / 'service', plan) as (client, _):
            native.control_probe(client, out / 'service', payloads[0]['input_ids'], 'v0', plan['inputs']['seed'])
            client.call('/start_profile', dict(activities=['KT_PHASE'], output_dir=str(phase),
                                               record_shapes=False, with_stack=False), timeout=20)
            native.save(out / 'ledger.json', dict(status='submitted', requests=32))
            launch = read(out / 'service/launch.json')
            url = f"http://127.0.0.1:{launch['port']}"
            futures = [pool.submit(measure, url, p, out) for p in payloads]
            records = [f.result() for f in futures]
            client.call('/stop_profile', {}, timeout=40)
            traces = wait_phase_traces(phase, timeout=40)
            summary = dict(requests=len(records), input_tokens=sum(r['input_tokens'] for r in records),
                           output_tokens=sum(r['output_tokens'] for r in records), ranks=len(traces),
                           policy='v0', scope='Independent streaming latency supplement')
            for metric in ('ttft_ms', 'tpot_ms'):
                values = [r[metric] for r in records]
                summary[metric] = dict(p50=percentile(values, .5), p95=percentile(values, .95))
            native.save(out / 'client-summary.json', summary)
        native.save(out / 'ledger.json', dict(status='complete', requests=32, optimizer_updates=0))
    except BaseException as exc:
        native.save(out / 'ledger.json', dict(status='failed', error=repr(exc), optimizer_updates=0))
        raise
    finally:
        signal.alarm(0)
        pool.shutdown(wait=False)


if __name__ == '__main__':
    main()

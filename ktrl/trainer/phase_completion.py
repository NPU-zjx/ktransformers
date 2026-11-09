"""Wait for asynchronous per-rank profiling output without weakening validation."""
import json
import time


def wait_phase_traces(folder, world_size=8, timeout=120, interval=.1):
    deadline = time.monotonic() + timeout
    expected = set(range(world_size))
    while True:
        ranks = {}
        incomplete = []
        for path in sorted(folder.glob('phase-*.json')):
            try:
                trace = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                incomplete.append(path.name)
                continue
            rank = trace.get('rank')
            if type(rank) is not int or rank not in expected or rank in ranks:
                raise ValueError(f'Unexpected or duplicate profiling rank: {rank}, {path}')
            decode = [row for row in trace['rows'] if row['phase'] == 'DECODE']
            if not decode or not all(row.get('graph_replayed') is True for row in decode):
                raise ValueError(f'Missing actual decode Graph replay: {path}')
            ranks[rank] = trace
        if set(ranks) == expected and not incomplete:
            return [ranks[rank] for rank in range(world_size)]
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Profiling completion timeout: missing ranks={sorted(expected-set(ranks))}, incomplete={incomplete}')
        time.sleep(interval)

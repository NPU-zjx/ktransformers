"""Linux cgroup/process/NVML sampler; does not create a CUDA context."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def kb_fields(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] == 'kB':
            result[fields[0].rstrip(':')] = int(fields[1])*1024
    return result


def container_sample():
    root = Path('/sys/fs/cgroup')
    values = {}
    for name in ('memory.current', 'memory.peak', 'memory.swap.current', 'memory.stat',
                 'memory.events', 'cpu.stat', 'io.stat', 'memory.pressure'):
        p = root/name
        if p.exists():
            values[name] = p.read_text()
    values['host_meminfo_bytes'] = kb_fields('/proc/meminfo')
    return values


def process_sample():
    records, errors = [], []
    # Namespace-local processes only. RSS/PSS are not interchangeable.
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            cmd = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            record = dict(pid=int(path.name), cmd=cmd,
                          status=(path/'status').read_text(), stat=(path/'stat').read_text(),
                          smaps_rollup_bytes=kb_fields(path/'smaps_rollup'))
            record['uss_bytes'] = sum(record['smaps_rollup_bytes'].get(k, 0)
                                      for k in ('Private_Clean', 'Private_Dirty', 'Private_Hugetlb'))
            records.append(record)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (OSError, ValueError) as error:
            errors.append(dict(pid=path.name, error=repr(error)))
    return dict(records=records, errors=errors)


class Sampler:
    def __init__(self, out):
        self.out = Path(out)
        self.stop = threading.Event()
        self.threads = []
        self.errors = []

    def emit(self, name, row):
        with (self.out/(name+'.jsonl')).open('a') as f:
            f.write(json.dumps(row)+'\n')

    def loop(self, name, period, function):
        while not self.stop.is_set():
            begin, mono = time.time_ns(), time.monotonic()
            try:
                result = function()
                self.emit(name, dict(start_ns=begin, end_ns=time.time_ns(), **result))
            except Exception as error:
                self.errors.append(dict(name=name, time_ns=begin, error=repr(error)))
                self.emit('sampler-errors', self.errors[-1])
            self.stop.wait(max(0, period-(time.monotonic()-mono)))

    def gpu(self):
        result = {}
        for name, flag, query in [
            ('cards', '--query-gpu=', 'index,uuid,memory.used,memory.total,utilization.gpu,power.draw'),
            ('processes', '--query-compute-apps=', 'pid,gpu_uuid,used_memory')]:
            p = subprocess.run(['nvidia-smi', flag+query, '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=8)
            if p.returncode:
                raise RuntimeError(p.stderr)
            result[name+'_csv'] = p.stdout
        return result

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        for name, period, function in [('container', 1., container_sample),
                                       ('processes', 10., process_sample), ('nvml', 1., self.gpu)]:
            t = threading.Thread(target=self.loop, args=(name, period, function), daemon=True)
            t.start()
            self.threads.append(t)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        for t in self.threads:
            t.join(12)
        alive = any(t.is_alive() for t in self.threads)
        (self.out/'sampler-status.json').write_text(json.dumps(dict(errors=self.errors,
            thread_still_alive=alive, complete=not self.errors and not alive)))

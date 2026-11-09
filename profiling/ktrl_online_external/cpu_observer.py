"""CPU-only Torch observation; no CUDA/CUPTI activity subscription."""
import json
import os
from pathlib import Path
import time


class CPUWindow:
    def __init__(self, folder, label, rank):
        import torch
        self.folder, self.label, self.rank = Path(folder), label, rank
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True,
            with_stack=False, profile_memory=False)
        self.profiler.start()
        self.region = torch.profiler.record_function(label)
        self.region.__enter__()
        self.start_ns = time.time_ns()

    def finish(self):
        self.end_ns = time.time_ns()
        self.region.__exit__(None, None, None)
        self.profiler.stop()
        stem = f'{self.label}-cpu-rank{self.rank}-{os.getpid()}'
        trace = self.folder / (stem + '.trace.json')
        self.profiler.export_chrome_trace(str(trace))
        rows = []
        for item in self.profiler.key_averages(group_by_input_shape=True):
            rows.append(dict(name=item.key, calls=item.count, input_shapes=item.input_shapes,
                             cpu_total_us=item.cpu_time_total, self_cpu_us=item.self_cpu_time_total))
        summary = dict(rank=self.rank, stage=self.label, start_ns=self.start_ns,
                       end_ns=self.end_ns, activities=['CPU'], record_shapes=True,
                       trace=trace.name, operators=rows)
        with (self.folder / (stem + '.summary.json')).open('x') as f:
            json.dump(summary, f)
        return summary

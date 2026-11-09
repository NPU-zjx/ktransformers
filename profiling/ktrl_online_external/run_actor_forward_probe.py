"""Run and verify one NSys forward-only probe in the existing profiler container."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    source, config, batch, root = (Path(x).resolve() for x in sys.argv[1:])
    sys.path.insert(0, str(source))
    from ktrl.trainer import online_native as native
    from ktrl.trainer.precision_gate import verify

    verify(config)
    native.guard_gpu()
    actor = root / 'actor'
    actor.mkdir()
    tool = '/usr/local/cuda-13.0/bin/nsys'
    worker = Path(__file__).with_name('probe_actor_forward.py')
    prefix = root / 'forward'
    command = [
        tool, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
        '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
        '--cuda-flush-interval=1000', '--wait=all', '--output=' + str(prefix),
        str(native.PYTHON), '-B', '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=8', str(worker), str(config), str(batch), str(actor),
    ]
    native.save(root / 'probe-command.json', command)
    subprocess.run(command, env=native.environment(actor, True), cwd=source, check=True, timeout=540)
    markers = [json.loads(p.read_text()) for p in actor.glob('forward-probe-rank*.json')]
    assert {row['rank'] for row in markers} == set(range(8))
    assert all(row['forward_complete'] and row['optimizer_updates'] == 0 for row in markers)
    report, = root.glob('forward*.nsys-rep')
    db = root / 'forward.sqlite'
    subprocess.run([tool, 'export', '--type=sqlite', '--output=' + str(db), str(report)],
                   check=True, timeout=90)
    subprocess.run(['/usr/bin/python3', '-B', str(Path(__file__).with_name('analyze_nsys_online.py')),
                    str(db), str(root / 'forward-analysis.json')], check=True, timeout=90)
    analysis = json.loads((root / 'forward-analysis.json').read_text())
    counts = {}
    for row in analysis['phase_kernels']:
        if row['phase'] == 'actor-forward-probe-rank0':
            counts[row['device']] = counts.get(row['device'], 0) + row['calls']
    passed = set(counts) == set(range(8)) and analysis['api_linked_kernels'] == analysis['kernel_count']
    native.save(root / 'validated.json', dict(verified=passed, per_gpu_kernel_counts=counts,
        recorded=analysis['kernel_count'], api_linked=analysis['api_linked_kernels'], optimizer_updates=0))
    if not passed:
        raise RuntimeError('NSys Actor forward coverage did not pass')


if __name__ == '__main__':
    main()

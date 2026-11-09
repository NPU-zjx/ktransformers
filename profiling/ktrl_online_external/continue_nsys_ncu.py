"""One-shot continuation: accepted NSys online run -> independent NCU online run."""
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/raid/data/zjx/kt-baseline')
NSYS = BASE/'results/ns-a03'
NCU = BASE/'results/nc-a10'
CONTROL = BASE/'checks/ktrl-profile-ncu-a10'
SOURCE = BASE/'checks/ktrl-review-005-20261108/resident-a04'


def state(phase, **details):
    (CONTROL/'state.json').write_text(json.dumps(dict(phase=phase, time=time.time(), **details), indent=2))


def main():
    assert not CONTROL.exists() and not NCU.exists(), 'Never repeat this continuation'
    CONTROL.mkdir()
    state('waiting_for_nsys', source=str(NSYS))
    deadline = time.monotonic()+7200
    while not (NSYS/'container-exit.json').exists():
        if time.monotonic()>deadline:
            raise TimeoutError('NSys did not finish within two hours; no automatic retry')
        time.sleep(15)
    assert json.loads((NSYS/'container-exit.json').read_text())['exit_code']==0
    assert json.loads((NSYS/'nsys-analysis/coverage.json').read_text())['kernel_api_and_device_coverage_passed']
    assert json.loads((NSYS/'rl-online-profile01/accepted.json').read_text())['real_online_updates']==1
    cmd=[sys.executable,'-B','-u',str(Path(__file__).with_name('launch_ncu_container.py')),
         '--profiler','ncu','--source',str(SOURCE/'source'),'--recipe',str(SOURCE/'recipe.yaml'),
         '--root',str(NCU)]
    subprocess.run(cmd+['--check'],check=True,stdout=subprocess.DEVNULL)
    state('starting_ncu', command=cmd)
    with (CONTROL/'launch.log').open('x') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0, f'NCU failed with {result.returncode}; no automatic retry'
    assert json.loads((NCU/'ncu/validated.json').read_text())['verified']
    assert json.loads((NCU/'rl-online-profile01/accepted.json').read_text())['real_online_updates']==1
    state('capture_complete_pending_delivery', source=str(NCU))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if CONTROL.exists():
            state('failed', error=repr(exc))
        raise

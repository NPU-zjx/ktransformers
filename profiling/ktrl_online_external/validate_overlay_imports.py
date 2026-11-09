"""Check native loader composition in isolated subprocesses, without models."""
import json
import os
from pathlib import Path
import subprocess
import sys

source, root = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(source))
from ktrl.trainer.online_native import environment

proofs = []
for role in ('rollout', 'actor'):
    folder = root / 'overlay-imports' / role
    folder.mkdir(parents=True)
    env = environment(folder, role == 'actor')
    env.update(KTRL_EXTERNAL_PROFILE_DIR=str(folder), KTRL_EXTERNAL_PROFILE_ROLE=role,
               KTRL_EXTERNAL_PROFILE_MODE='ncu', RANK='0')
    env['PYTHONPATH'] = str(root / 'overlay') + ':' + env['PYTHONPATH']
    code = ("from sglang.srt.model_executor.model_runner import ModelRunner; "
            "from sglang.srt.managers.scheduler import Scheduler; "
            "from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod; "
            "assert hasattr(ModelRunner.forward,'__wrapped__'), 'ModelRunner.forward observer missing'; "
            "assert hasattr(Scheduler.run_batch,'__wrapped__'), 'Scheduler.run_batch observer missing'; "
            "assert hasattr(KTEPWrapperMethod.apply,'__wrapped__'), 'KTEPWrapperMethod.apply observer missing'; "
            "print('rollout observation hooks imported; no model constructed')" if role == 'rollout' else
            "from transformers.trainer import Trainer; "
            "assert hasattr(Trainer.training_step,'__wrapped__'); "
            "print('actor observation hooks imported; no trainer/model constructed')")
    with (folder / 'import.log').open('x') as log:
        result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    proofs.append(dict(role=role, returncode=result.returncode))
    assert result.returncode == 0, (role, str(folder / 'import.log'))
(root / 'overlay-imports/validated.json').write_text(json.dumps(proofs, indent=2))

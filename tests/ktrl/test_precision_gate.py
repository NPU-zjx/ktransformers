import json
from pathlib import Path
import tempfile
import unittest
from ktrl.trainer.precision_gate import REQUIRED,digest,source_digest,verify


class GateTest(unittest.TestCase):
    def test_changed_recipe_evidence_and_source_cannot_reuse_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'recipe.yaml'
            config.write_text('frozen')
            evidence=root/'accepted.json';evidence.write_text('{"accepted":true}')
            gate=dict(accepted=True,config_sha256=digest(config),source_sha256=source_digest(),
                checks={name:dict(path=str(evidence.resolve()),sha256=digest(evidence)) for name in REQUIRED})
            path=root/'precision-gate.json';path.write_text(json.dumps(gate))
            verify(config)
            config.write_text('modified')
            with self.assertRaises(RuntimeError):verify(config)
            config.write_text('frozen');evidence.write_text('{"accepted":false}')
            with self.assertRaises(RuntimeError):verify(config)
            evidence.write_text('{"accepted":true}');gate['source_sha256']='stale'
            path.write_text(json.dumps(gate))
            with self.assertRaises(RuntimeError):verify(config)

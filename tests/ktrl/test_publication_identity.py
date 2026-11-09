import json
from pathlib import Path
import tempfile
import unittest
from ktrl.weights.identity import build_manifest,validate_manifest


class IdentityTest(unittest.TestCase):
    def test_same_size_tampering_and_incomplete_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'expert.safetensors').write_bytes(b'first')
            (path/'model.safetensors').write_bytes(b'gpu')
            (path/'trainer_state.json').write_text(json.dumps({'global_step':1}))
            (path/'kt_full_experts.index.json').write_text(json.dumps({'layers':{'0':{'file':'expert.safetensors'}}}))
            digest=build_manifest(path)
            self.assertEqual(validate_manifest(path),(1,digest))
            with self.assertRaises(FileExistsError):build_manifest(path)
            (path/'expert.safetensors').write_bytes(b'other')
            with self.assertRaisesRegex(ValueError,'content changed'):validate_manifest(path)
            (path/'expert.safetensors').write_bytes(b'first')
            manifest=json.loads((path/'publication-manifest.json').read_text())
            del manifest['files']['model.safetensors']
            (path/'publication-manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Incomplete'):validate_manifest(path)

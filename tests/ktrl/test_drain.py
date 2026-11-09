import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from urllib.error import HTTPError
from ktrl.trainer.online_native import drain_rollout


class DrainTest(unittest.TestCase):
    def test_pending_requests_must_reach_native_success(self):
        with tempfile.TemporaryDirectory() as directory:
            client=Mock()
            client.call.side_effect=[HTTPError('local',400,'pending',{},None),'Cache flushed.']
            with patch('ktrl.trainer.online_native.time.sleep'):
                drain_rollout(client,Path(directory),'test')
            proof=json.loads((Path(directory)/'test-drained.json').read_text())
            self.assertEqual(proof['attempts'],2)
            self.assertTrue(proof['native_scheduler_confirmed_empty'])

    def test_unrelated_http_error_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            client=Mock();client.call.side_effect=HTTPError('local',500,'failed',{},None)
            with self.assertRaises(HTTPError):drain_rollout(client,Path(directory),'test')
            self.assertFalse((Path(directory)/'test-drained.json').exists())

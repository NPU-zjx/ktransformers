from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from ktrl.trainer.online_native import monitor_publication_failure


class SupervisorTest(unittest.TestCase):
    def check(self,identity_matches):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            folder=Path(directory)
            (folder/'launch.json').write_text('{"name":"owned-service"}')
            (folder/'publication-failed-rank0.json').write_text('{}')
            stack.enter_context(patch.object(Path,'exists',return_value=True))
            command=b'python\0--served-model-name\0'+(b'owned-service' if identity_matches else b'unrelated')+b'\0'
            stack.enter_context(patch.object(Path,'read_bytes',return_value=command))
            stack.enter_context(patch('ktrl.trainer.online_native.os.getpgid',create=True,return_value=123))
            stack.enter_context(patch('ktrl.trainer.online_native.signal.SIGQUIT',create=True,new=3))
            kill=stack.enter_context(patch('ktrl.trainer.online_native.os.kill'))
            stop=Mock();stop.wait.return_value=False
            if identity_matches:
                monitor_publication_failure(folder,123,stop)
                kill.assert_called_once_with(123,3)
                self.assertEqual(json.loads((folder/'failure-supervisor.json').read_text())['pid'],123)
            else:
                with self.assertRaises(AssertionError):monitor_publication_failure(folder,123,stop)
                kill.assert_not_called()

    def test_owned_failed_service_is_stopped(self):self.check(True)
    def test_changed_identity_is_never_signaled(self):self.check(False)

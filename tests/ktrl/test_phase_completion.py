import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ktrl.trainer.phase_completion import wait_phase_traces


def write(folder, rank, replay=True):
    (folder/f'phase-{rank}.json').write_text(json.dumps(dict(rank=rank,
        rows=[dict(phase='DECODE', graph_replayed=replay)])))


class PhaseCompletionTest(unittest.TestCase):
    def test_delayed_rank_and_partial_json(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            write(folder,0)
            (folder/'phase-1.json').write_text('{')
            def finish(_): write(folder,1)
            with patch('ktrl.trainer.phase_completion.time.sleep',side_effect=finish):
                self.assertEqual([t['rank'] for t in wait_phase_traces(folder,2)],[0,1])

    def test_missing_rank_times_out(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);write(folder,0)
            with self.assertRaisesRegex(TimeoutError,'missing ranks='):
                wait_phase_traces(folder,2,timeout=0)

    def test_invalid_replay_is_not_retried(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);write(folder,0,False)
            with self.assertRaises(ValueError):wait_phase_traces(folder,1)

    def test_duplicate_rank_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);write(folder,0)
            (folder/'phase-copy.json').write_bytes((folder/'phase-0.json').read_bytes())
            with self.assertRaises(ValueError):wait_phase_traces(folder,1)

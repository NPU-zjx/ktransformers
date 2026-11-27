import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
from ktrl.workers.rollout.sglang.smoke import run_one


class SmokeSummaryTest(unittest.TestCase):
    def test_summary_counts_submitted_prompt_and_is_written(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=root/'config.yaml';config.write_text('{}')
            prompt=root/'prompt.json';prompt.write_text('[11,12,13]')
            client=Mock()
            def generate(payload):
                self.assertEqual(payload['input_ids'],[11,12,13])
                return {'text':'answer','output_ids':[21,22], 'meta_info':{
                    'id':payload['rid'],'prompt_tokens':3,'completion_tokens':2,
                    'finish_reason':{'type':'length'}}}
            client.generate.side_effect=generate
            @contextmanager
            def service(config, folder, plan):
                folder.mkdir()
                try:yield client,None
                finally:(folder/'exit.json').write_text('{"returncode":0}')
            plan={'rollout':{'backend':'kt-sglang','server_args':{'tp_size':1}},
                  'inputs':{'context_length':100}}
            with patch('ktrl.configs.recipe.load_plan',return_value=plan), \
                 patch('ktrl.trainer.online_native.service',service), \
                 patch('ktrl.trainer.phase_completion.wait_phase_traces'), \
                 patch('ktrl.workers.rollout.sglang.smoke.verify_replays',return_value=[{'decode_replay_batches':1}]) as replays:
                result=run_one(config,root/'out',2,prompt)
            self.assertEqual(result['input_tokens'],3)
            self.assertEqual(result['output_tokens'],2)
            self.assertEqual(json.loads((root/'out'/'rollout-summary.json').read_text()),result)
            client.generate.assert_called_once()
            replays.assert_called_once_with(root/'out'/'phase',1,3,2)

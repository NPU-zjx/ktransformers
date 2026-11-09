import json
from pathlib import Path
import tempfile
import unittest

from ktrl.workers.rollout.sglang.smoke import validate_response, verify_replays, wait_ready


class RolloutEvidenceTest(unittest.TestCase):
    def test_cold_start_checks_identity_then_opens_control_loop_without_generation(self):
        calls = []
        expected = {"served_model_name": "dedicated", "skip_server_warmup": True}

        class ColdServer:
            def call(self, path, data=None, **kwargs):
                calls.append(path)
                if path == "/v1/models":
                    return {"data": [{"id": "dedicated"}]}
                if path == "/open_session":
                    return data["session_id"]
                if path == "/server_info":
                    return expected

        with tempfile.TemporaryDirectory() as folder:
            wait_ready(ColdServer(), lambda: True, expected, Path(folder))
        self.assertEqual(calls, ["/v1/models", "/open_session", "/close_session", "/server_info"])

    def test_graph_eligibility_alone_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "phase-TP0.json"
            trace = {"rank": 0, "uuid": "gpu0", "pid": 1, "replays": 1, "rows": [
                {"index": 0, "phase": "EXTEND", "tokens": 3},
                {"index": 1, "phase": "DECODE", "tokens": 1, "graph_eligible": True,
                 "graph_replayed": False, "stream_ms": 1.0},
            ]}
            path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "Graph replay"):
                verify_replays(path.parent, 1, 3, 2)
            trace["rows"][1]["graph_replayed"] = True
            path.write_text(json.dumps(trace))
            self.assertEqual(verify_replays(path.parent, 1, 3, 2)[0]["decode_replay_batches"], 1)
            with self.assertRaisesRegex(ValueError, "Missing TP rank"):
                verify_replays(path.parent, 2, 3, 2)

    def test_response_must_match_request_and_native_token_count(self):
        payload = {"rid": "r1", "input_ids": [4], "sampling_params": {"max_new_tokens": 2}}
        response = {"output_ids": [5, 6], "meta_info": {
            "id": "r1", "prompt_tokens": 1, "completion_tokens": 2, "finish_reason": {"type": "length"},
        }}
        self.assertEqual(validate_response(payload, response), [5, 6])
        response["meta_info"]["id"] = "other"
        with self.assertRaisesRegex(ValueError, "match"):
            validate_response(payload, response)
        response["meta_info"].update(id="r1", completion_tokens=1)
        with self.assertRaisesRegex(ValueError, "count"):
            validate_response(payload, response)

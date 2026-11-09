"""Exercise KTRL's public CLI with metadata only, without any compute runtime."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
GUARD = """
import runpy, sys
class BlockCompute:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sglang', 'transformers', 'kt_kernel', 'accelerate', 'ray', 'verl'}:
            raise AssertionError(f'dry-run imported {fullname}')
def offline(event, args):
    if event in {'subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'}:
        raise AssertionError(event)
    if event == 'open' and str(args[0]).endswith(('.safetensors', '.bin', '.pt', '.parquet')):
        raise AssertionError(f'dry-run opened payload {args[0]}')
sys.meta_path.insert(0, BlockCompute())
sys.addaudithook(offline)
runpy.run_module('ktrl', run_name='__main__')
"""


class DryRunTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        model = self.folder / "model"
        model.mkdir()
        self.metadata = {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {"num_hidden_layers": 40, "num_experts": 256, "num_experts_per_tok": 8},
        }
        (model / "config.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        (self.folder / "train.parquet").touch()
        self.config = yaml.safe_load((ROOT / "examples/ktrl/qwen36_35b_grpo.yaml").read_text(encoding="utf-8"))
        self.config["actor_rollout_ref"]["model"]["path"] = "model"
        self.config["data"]["train_file"] = "train.parquet"

    def invoke(self, *flags, raw=None, command="run"):
        recipe = self.folder / "recipe.yaml"
        recipe.write_text(raw if raw is not None else yaml.safe_dump(self.config), encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-B", "-c", GUARD, command, "--config", str(recipe), *flags],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )

    def test_optional_rollout_correction(self):
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(json.loads(result.stdout)["rollout_correction"]["rollout_is"])
        self.config["algorithm"]["rollout_correction"] = {"rollout_is": "token", "rollout_is_threshold": 1.5}
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["rollout_correction"],
                         {"rollout_is": "token", "rollout_is_threshold": 1.5})
        self.config["algorithm"]["rollout_correction"]["rollout_is_threshold"] = -1
        result = self.invoke("--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("finite and positive", result.stderr)

    def test_shared_placement_and_cli_overrides(self):
        result = self.invoke("--dry-run", "--steps", "2", "--profiling", "light")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["orchestrator"], "KTRL")
        self.assertEqual(plan["active_roles"], ["rollout", "actor"])
        self.assertEqual(plan["rollout"]["backend"], "KT-SGLang")
        placement = plan["expert_placement"]
        self.assertEqual(placement["gpu"], {"count": 0, "ids_inclusive": None})
        self.assertEqual(placement["cpu"], {"count": 256, "ids_inclusive": [0, 255]})
        self.assertEqual(plan["rollout"]["expert_placement"], placement)
        self.assertEqual(plan["actor"]["expert_placement"], placement)
        self.assertFalse(plan["rollout"]["server_args"]["kt_enable_dynamic_expert_update"])
        self.assertEqual(plan["rollout"]["server_args"]["kt_gpu_prefill_token_threshold"], 0)
        self.assertEqual(plan["actor"]["strategy"], "fsdp2")
        self.assertEqual(plan["actor"]["expert_placement"]["cpu"]["count"], 256)
        self.assertEqual(plan["max_updates"], 2)
        self.assertEqual(plan["profiling"]["preset"], "light")
        self.assertEqual(plan["inputs"]["responses_per_update"], 32)
        self.assertEqual(plan['gradient_accumulation_steps'],4)
        self.assertEqual(plan['inputs']['max_response_tokens'],8192)
        self.assertEqual(plan['inputs']['context_length'],9216)
        self.assertEqual(plan["model"]["path"], str(self.folder / "model"))

    def test_legacy_recipe_and_mismatch_before_any_generation(self):
        del self.config["expert_placement"]
        group = self.config["actor_rollout_ref"]
        for name in ("actor", "rollout"):
            group[name]["kt_num_gpu_experts"] = 0
        self.assertEqual(self.invoke("--dry-run").returncode, 0)
        group["rollout"]["kt_num_gpu_experts"] = 32
        for command, flags in (("run", ["--dry-run"]), ("rollout", ["--output-dir", str(self.folder / "unused")])):
            result = self.invoke(*flags, command=command)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("same expert placement", result.stderr)
            self.assertFalse((self.folder / "unused").exists())

    def test_nonzero_shared_policy_and_different_strategy_are_rejected(self):
        self.config["expert_placement"]["gpu_experts_per_layer"] = 32
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires 0 GPU routed experts", result.stderr)
        self.config["expert_placement"]["gpu_experts_per_layer"] = 0
        self.config["actor_rollout_ref"]["rollout"]["kt_expert_placement_strategy"] = "frequency"
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("kt_expert_placement_strategy", result.stderr)

    def test_invalid_placements_and_backends(self):
        for role, key, bad in (
            ("rollout", "kt_num_gpu_experts", 257), ("rollout", "kt_num_gpu_experts", True),
            ("actor", "kt_num_gpu_experts", 1), ("actor", "strategy", "fsdp"),
            ("rollout", "name", "vllm"), ("rollout", "kt_num_gpu_exprts", 0),
        ):
            with self.subTest(role=role, key=key, bad=bad):
                options = self.config["actor_rollout_ref"][role]
                original = options.copy()
                options[key] = bad
                result = self.invoke("--dry-run")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(key, result.stderr)
                self.assertEqual(result.stdout, "")
                self.config["actor_rollout_ref"][role] = original

    def test_duplicate_keys_and_missing_metadata(self):
        result = self.invoke("--dry-run", raw="actor_rollout_ref: {}\nactor_rollout_ref: {}\n")
        self.assertEqual(result.returncode, 2)
        self.assertIn("duplicate key", result.stderr)
        (self.folder / "model/config.json").unlink()
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("config.json", result.stderr)

    def test_execution_requires_profiling_and_output(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Online execution requires --profiling light --output-dir", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_full_dataset_update_count(self):
        del self.config["trainer"]["optimizer_schedule_steps"]
        result = self.invoke("--dry-run", "--steps", "1672")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["max_updates"], 1672)
        self.assertEqual(plan["optimizer_schedule_steps"], 1672)
        self.assertEqual(plan["inputs"]["responses_per_update"], 32)

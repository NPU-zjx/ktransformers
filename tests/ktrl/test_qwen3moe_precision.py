import unittest

from ktrl.configs.recipe import architecture_spec, load_plan


class Qwen3MoePrecisionConfigTest(unittest.TestCase):
    def test_architecture_specs(self):
        q3 = architecture_spec("Qwen3MoeForCausalLM")
        self.assertEqual(q3["auto_model_cls"], "AutoModelForCausalLM")
        self.assertEqual(q3["wrap_classes"], ["Qwen3MoeDecoderLayer"])
        self.assertFalse(q3["needs_vlm_conv3d_patch"])

        q35 = architecture_spec("Qwen3_5MoeForConditionalGeneration")
        self.assertEqual(q35["auto_model_cls"], "AutoModelForImageTextToText")
        self.assertEqual(q35["wrap_classes"], ["Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock"])
        self.assertTrue(q35["needs_vlm_conv3d_patch"])

        with self.assertRaises(ValueError):
            architecture_spec("UnknownArch")

    def test_scheme_b_plan_resolves_bfloat16_reduce_and_fp32_masters(self):
        plan = load_plan("examples/ktrl/qwen3_30b_hotpotqa_grpo.yaml")
        self.assertEqual(plan["model"]["architecture"], "Qwen3MoeForCausalLM")
        self.assertEqual(plan["actor"]["nonexpert_parameter_dtype"], "float32")
        self.assertEqual(
            plan["actor"]["mixed_precision_policy"],
            {"param_dtype": "bfloat16", "reduce_dtype": "bfloat16"},
        )
        self.assertEqual(plan["actor"]["cpu_expert_optimizer"]["master_dtype"], "float32")
        self.assertEqual(plan["actor"]["cpu_expert_optimizer"]["moment_dtype"], "float32")
        self.assertEqual(plan["actor"]["cpu_expert_optimizer"]["gradient_buffer_dtype"], "bfloat16")


if __name__ == "__main__":
    unittest.main()

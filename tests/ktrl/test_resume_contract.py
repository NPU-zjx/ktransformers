import json
import tempfile
import unittest
from pathlib import Path
from ktrl.trainer.resume import contract, load_resume, retire_old_full_checkpoints

class ResumeContractTest(unittest.TestCase):
    def test_published_boundary_and_changed_config(self):
        plan={k:{} for k in ("model","expert_placement","actor","inputs","rollout_correction")}
        plan.update(gradient_accumulation_steps=2,optimizer_schedule_steps=16,max_updates=4)
        plan["actor"]={"optimizer_steps_per_rollout":4}
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for name in ("kt_optimizer.index.json","scheduler.pt"): (p/name).write_text("{}")
            (p/"trainer_state.json").write_text(json.dumps({"global_step":8}))
            (p/"ktrl-resume.json").write_text(json.dumps(dict(contract=contract(plan),optimizer_steps=8,completed_rounds=2,batches_consumed=2)))
            self.assertEqual(load_resume(p,plan)["optimizer_steps"],8)
            plan["gradient_accumulation_steps"]=4
            with self.assertRaises(ValueError):load_resume(p,plan)

    def test_retention_preserves_logs_and_newest_checkpoint(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for i in (1,2):
                p=root/f"round-{i:02}/actor/trainer/checkpoint-{i}"
                p.mkdir(parents=True)
                (p/"ktrl-resume.json").write_text(json.dumps({"optimizer_steps":i}))
                (p/"model.safetensors").write_bytes(b"weight")
                (p/"trainer_state.json").write_text("{}")
            retire_old_full_checkpoints(root,1)
            old=root/"round-01/actor/trainer/checkpoint-1"
            self.assertFalse((old/"model.safetensors").exists())
            self.assertTrue((old/"trainer_state.json").exists())
            self.assertTrue((root/"round-02/actor/trainer/checkpoint-2/model.safetensors").exists())

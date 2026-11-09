"""CPU-only adapter from KT Full checkpoints to the native inference loader."""

import json
import os
from pathlib import Path

from safetensors import safe_open


class ExpertCheckpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.index = json.loads((self.path / "kt_full_experts.index.json").read_text())
        if self.index["dtype"] != "BF16" or len(self.index["layers"]) != 40:
            raise ValueError("Expected the complete M2 BF16 expert checkpoint")

    def load_experts(self, base_key, device="cpu"):
        if device != "cpu":
            raise ValueError("CPU expert publication requires CPU loading")
        layer = base_key.rsplit(".", 1)[1]
        entry = self.index["layers"][layer]
        path = self.path / entry["file"]
        if path.parent != self.path or path.stat().st_size != entry["size_bytes"]:
            raise ValueError("Invalid expert checkpoint file")
        with safe_open(path, framework="pt", device="cpu") as source:
            weights = {}
            for short in ("gate", "up", "down"):
                key = next(k for k in source.keys() if k.endswith(short + "_proj"))
                tensor = source.get_tensor(key)
                if list(tensor.shape) != entry["shapes"][short + "_proj"]:
                    raise ValueError("Expert checkpoint shape mismatch")
                weights[short] = list(tensor.unbind(0))
                if layer == "0" and short == "gate" and os.environ.get("KTRL_PUBLICATION_OUTPUT"):
                    sample = {"layer": 0, "expert": 0, "flat_index": 294948,
                              "value": tensor.reshape(-1)[294948].item(), "device": str(tensor.device),
                              "source": str(path), "path": "CPU safetensors -> native CPU load_weights_task"}
                    (Path(os.environ["KTRL_PUBLICATION_OUTPUT"]) / "cpu-source-sample.json").write_text(json.dumps(sample))
        return weights

    def close_all_handles(self):
        pass

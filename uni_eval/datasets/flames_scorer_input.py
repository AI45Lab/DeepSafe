import json
import os
from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

@DATASETS.register_module()
class FlamesScorerInputDataset(BaseDataset):
    """
    Dataset of precomputed target responses for Flames scorer.

    Expected JSONL fields per line:
      - dimension
      - prompt
      - response
    """

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Flames scorer input JSONL not found at: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.data.append(
                    {
                        "dimension": obj.get("dimension", ""),
                        "subcomponent": obj.get("subcomponent", ""),
                        "prompt": obj.get("prompt", ""),
                        "response": obj.get("response", ""),
                        "ground_truth": "",
                        "meta": {"raw": obj},
                    }
                )


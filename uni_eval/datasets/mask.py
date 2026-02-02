import os
import datasets
from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

@DATASETS.register_module()
class MASKDataset(BaseDataset):
    """
    Dataset loader for the MASK dataset (cais/mask).
    Subset: [continuations, disinformation, doubling_down_known_facts, known_facts, provided_facts, statistics]
    Expected format: parquet where each line contains (at least):
      - prompt: chat messages [{role:system,...},{role:user,...}] for neutral run
      - ground_truth: original ground_truth (string)
      - meta: includes proposition, belief_elicit_1/2/3, delivery, formatted_ground_truth, canary, subset, task_id, type
    """

    def _load_hf(self):
        subset_files = {
            "continuations": "continuations/test-00000-of-00001.parquet", 
            "disinformation": "disinformation/test-00000-of-00001.parquet", 
            "doubling_down_known_facts": "doubling_down_known_facts/test-00000-of-00001.parquet",
            "known_facts": "known_facts/test-00000-of-00001.parquet",
            "provided_facts": "provided_facts/test-00000-of-00001.parquet",
            "statistics": "statistics/test-00000-of-00001.parquet",
        }
        raw_data = {}
        for subset, parquet_file in subset_files.items():
            dataframe = datasets.load_dataset("parquet", data_files=os.path.join(self.path, parquet_file))["train"]
            raw_data[subset] = dataframe
        return raw_data

    def load(self):
        """
        Load MASK data (parquet).
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"MASK dataset not found at: {self.path}")

        raw_data = self._load_hf()

        self.data = []
        for subset in raw_data.keys():
            for item in raw_data[subset]:
                task_id = item.get("task_id", None)
                
                prompt = [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_prompt"]},
                ]

                meta = {
                    "subset": subset,
                    "task_id": task_id,
                    "type": item.get("type", ""),
                    "proposition": item.get("proposition", ""),
                    "belief_elicit_1": item.get("belief_elicit_1", ""),
                    "belief_elicit_2": item.get("belief_elicit_2", ""),
                    "belief_elicit_3": item.get("belief_elicit_3", ""),
                    "delivery": item.get("delivery", ""),
                    "formatted_ground_truth": item.get("formatted_ground_truth", ""),
                    "canary": item.get("canary", ""),
                    "type": item.get("type", ""),
                }

                if "dd_turn_1_prompt" in item:
                    meta["dd_turn_1_prompt"] = item.get("dd_turn_1_prompt", "")
                if "dd_turn_1_response" in item:
                    meta["dd_turn_1_response"] = item.get("dd_turn_1_response", "")

                self.data.append(
                    {
                        "system_prompt": item["system_prompt"],
                        "user_prompt": item["user_prompt"],
                        "prompt": prompt,
                        "ground_truth": item["ground_truth"],
                        "meta": meta,
                    }
                )
        
        print(f"Loaded {len(self.data)} samples from MASK dataset.")

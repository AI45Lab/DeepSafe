import json
import os
import logging

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class ArgusDataset(BaseDataset):
    """Argus Dataset.

    Args:
        split (str): The dataset split, e.g., 'train', 'val', 'test'.
        **kwargs: Additional keyword arguments passed to the BaseDataset.
    """

    def __init__(self, path: str, **kwargs):
        super().__init__(path=path, **kwargs)

    def load(self) -> None:
        for idx in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
            in_file_path = f"{self.path}/Benchmark/{idx}/meta_{idx}.jsonl"

            if not os.path.exists(in_file_path):
                raise FileNotFoundError(f"Argus dataset not found at: {in_file_path}")
            
            dataset = []
            with open(in_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line.strip())
                    dataset.append(data)
            logger.info(f"The volume of domain {idx} is {len(dataset)}. ")

            for item in dataset:
                uid = item["uid"]
                image_name = item['image_name']
                dec_question = item['dec_question']
                bas_question = item['bas_question']
                standard_answer = item['standard_answer']
                trap = item['trap']
                image_path = f"{self.path}/Benchmark/{idx}/{image_name}"

                self.data.append(
                    {
                        "uid": uid,
                        "image_path": image_path,
                        "dec_question": dec_question,
                        "bas_question": bas_question,
                        "standard_answer": standard_answer,
                        "trap": trap,
                        "meta": {
                            "dataset": "Argus",
                            "domain": idx,
                            "image_name": image_name
                        }
                    }
                )
        
        logger.info(f"Total volume of Argus dataset is {len(self.data)}.")
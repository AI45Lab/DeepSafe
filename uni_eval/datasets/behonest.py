import json
import logging
import os
from glob import glob
from typing import Dict, Any, Optional

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class BeHonestDataset(BaseDataset):
    """
    Dataset loader for BeHonest benchmark.

    Supports multiple task categories:
    - Self-Knowledge: Unknowns, Knowns
    - Non-Deceptiveness: Burglar_Deception, Game, Persona_Sycophancy, Preference_Sycophancy
    - Consistency: Prompt_Format, Open_Form, Multiple_Choice, Demonstration_Format

    Usage:
        path: Can be a specific JSON file or a directory containing JSON files
        category: Task category to load (e.g., 'Unknowns', 'Knowns', 'Persona_Sycophancy')
        split: For multi-split datasets (e.g., 'no_persona', 'persona')
    """

    def __init__(
        self,
        path: str,
        category: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            path: Path to the BeHonest dataset directory or specific JSON file
            category: Task category (e.g., 'Unknowns', 'Knowns', 'Persona_Sycophancy')
            split: Dataset split for categories with multiple files (e.g., 'no_persona', 'persona')
        """
        self.category = category
        self.split = split
        super().__init__(path=path, **kwargs)

    def load(self):
        """Load BeHonest data and standardize to uni_eval format."""
                                      
        files_to_load = []
        path = self.path

        if os.path.isdir(path):
            if self.category:
                category_path = os.path.join(path, self.category)
                if os.path.isdir(category_path):
                    if self.split:
                                        
                        file_path = os.path.join(category_path, f"{self.split}.json")
                        if os.path.exists(file_path):
                            files_to_load = [file_path]
                        else:
                                                     
                            pattern = os.path.join(category_path, f"{self.split}*.json")
                            files_to_load = glob(pattern)
                    else:
                                                                   
                        files_to_load = glob(os.path.join(category_path, "*.json"))
                else:
                                                          
                    if category_path.endswith('.json') and os.path.exists(category_path):
                        files_to_load = [category_path]
            else:
                                                            
                files_to_load = glob(os.path.join(path, "*", "*.json"))
        elif os.path.isfile(path):
            files_to_load = [path]
        else:
            logger.warning(f"Path not found: {path}")
            self.data = []
            return

        if not files_to_load:
            logger.warning(f"No JSON files found for path={path}, category={self.category}, split={self.split}")
            self.data = []
            return

        self.data = []
        for file_path in sorted(files_to_load):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                if not raw_data:
                    continue

                category_name = self._detect_category(file_path, raw_data)
                standardized = self._standardize_data(raw_data, category_name, file_path)
                self.data.extend(standardized)

                logger.info(f"Loaded {len(standardized)} samples from {file_path}")

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                if self.strict:
                    raise

        logger.info(f"Total loaded samples: {len(self.data)}")

    def _detect_category(self, file_path: str, raw_data: list) -> str:
        """Detect task category from file path and data structure."""
                                
        parts = file_path.split(os.sep)
        for part in parts:
            if part in ['Unknowns', 'Knowns', 'Burglar_Deception', 'Game',
                       'Persona_Sycophancy', 'Preference_Sycophancy',
                       'Prompt_Format', 'Open_Form', 'Multiple_Choice',
                       'Demonstration_Format']:
                return part

        if raw_data and isinstance(raw_data[0], dict):
            first_item = raw_data[0]
            if 'answer' in first_item:
                return 'Knowns'
            elif 'classification question' in first_item:
                return 'Burglar_Deception'
            elif 'answer_matching' in first_item:
                return 'Persona_Sycophancy'
            elif 'question' in first_item and 'response' not in first_item:
                                              
                return 'Unknowns'

        return 'Unknown'

    def _standardize_data(self, raw_data: list, category: str, file_path: str) -> list:
        """Standardize data to uni_eval format."""
        standardized = []

        for idx, item in enumerate(raw_data):
                                 
            item_id = item.get("id", idx)
            prompt = item.get("prompt", "")

            if category == 'Multiple_Choice':
                                                                         
                prompt = item.get("prompt_1", "")
                prompt_2 = item.get("prompt_2", "")

            std_item = {
                "id": item_id,
                "prompt": prompt,
                "meta": {
                    "task_type": category,
                    "category": category,
                    "source_file": os.path.basename(file_path),
                    "raw": item,
                }
            }

            if category == 'Multiple_Choice':
                std_item["meta"]["prompt_2"] = prompt_2

            if category == 'Knowns':
                std_item["ground_truth"] = item.get("answer", [])
            elif category == 'Burglar_Deception':
                std_item["ground_truth"] = {
                    "classification_question": item.get("classification question", ""),
                    "cheap_object": item.get("cheap object", ""),
                    "expensive_item": item.get("extremely expensive item", ""),
                }
            elif category in ['Persona_Sycophancy', 'Preference_Sycophancy']:
                std_item["ground_truth"] = item.get("answer_matching", "")
            elif category == 'Unknowns':
                std_item["ground_truth"] = None                                
            else:
                                   
                std_item["ground_truth"] = item.get("answer", None)

            standardized.append(std_item)

        return standardized

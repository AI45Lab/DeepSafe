import csv
import os
import base64
import logging
from typing import Dict, Any, Optional, List, Union

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class MOSSBenchDataset(BaseDataset):
    """
    Unified dataset loader for MOSSBench dataset.
    
    Standardized item format:
        {
            "id": str,              # pid
            "prompt": List[Dict],   # multimodal prompt with image and question
            "ground_truth": None,   # open-ended generation
            "meta": {
                "pid": str,
                "image_path": str,
                "short_description": str,
                "description": str,
                "meta_data_over": str,
                "meta_data_human": str,
                "meta_data_child": str,
                "meta_data_syn": str,
                "meta_data_ocr": str,
                "meta_data_harm": str,
                "raw": Dict,
            },
        }
    """

    def load(self):
        """Load MOSSBench data and standardize."""
        logger.info(f"Loading MOSSBench from {self.path}")

        csv_dir = os.path.dirname(self.path)
        image_root = os.path.join(csv_dir, "images")
        
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"MOSSBench CSV file not found at {self.path}")
        
        self.data = []
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
                             
            header = next(reader, None)
            if header is None:
                logger.warning("CSV file is empty")
                return
            
            for row_idx, row in enumerate(reader, start=2):                                   
                if len(row) < 11:
                    logger.warning(f"Skipping row {row_idx}: insufficient columns ({len(row)} < 11)")
                    continue
                
                try:
                    pid = row[0].strip()
                    question = row[1].strip()
                    image_path_rel = row[2].strip()
                    short_description = row[3].strip()
                    description = row[4].strip()
                    meta_data_over = row[5].strip()
                    meta_data_human = row[6].strip()
                    meta_data_child = row[7].strip()
                    meta_data_syn = row[8].strip()
                    meta_data_ocr = row[9].strip()
                    meta_data_harm = row[10].strip()
                    
                    if not question:
                        logger.warning(f"Skipping row {row_idx}: empty question")
                        continue

                    if image_path_rel.startswith("images/"):
                                                            
                        image_file_name = image_path_rel.replace("images/", "", 1)
                    else:
                        image_file_name = image_path_rel
                    
                    full_image_path = os.path.join(image_root, image_file_name)
                    
                    if not os.path.exists(full_image_path):
                        logger.warning(f"Image file missing: {full_image_path} (row {row_idx})")
                        continue

                    try:
                        with open(full_image_path, "rb") as image_file:
                            image_data = image_file.read()
                            image_base64 = base64.b64encode(image_data).decode("utf-8")

                        mime_type = "image/png"
                        lower_path = full_image_path.lower()
                        if lower_path.endswith('.jpg') or lower_path.endswith('.jpeg'):
                            mime_type = "image/jpeg"
                        elif lower_path.endswith('.webp'):
                            mime_type = "image/webp"
                        elif lower_path.endswith('.gif'):
                            mime_type = "image/gif"
                    except Exception as e:
                        logger.error(f"Error loading image {full_image_path}: {e}")
                        continue

                    prompt: List[Dict[str, Any]] = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": question
                                }
                            ]
                        }
                    ]

                    standardized = {
                        "id": str(pid),
                        "prompt": prompt,
                        "ground_truth": None,                         
                        "meta": {
                            "pid": pid,
                            "image_path": full_image_path,
                            "image_path_rel": image_path_rel,
                            "short_description": short_description,
                            "description": description,
                            "meta_data_over": meta_data_over,
                            "meta_data_human": meta_data_human,
                            "meta_data_child": meta_data_child,
                            "meta_data_syn": meta_data_syn,
                            "meta_data_ocr": meta_data_ocr,
                            "meta_data_harm": meta_data_harm,
                            "raw": {
                                "pid": pid,
                                "question": question,
                                "image_path": image_path_rel,
                                "short_description": short_description,
                                "description": description,
                                "meta_data_over": meta_data_over,
                                "meta_data_human": meta_data_human,
                                "meta_data_child": meta_data_child,
                                "meta_data_syn": meta_data_syn,
                                "meta_data_ocr": meta_data_ocr,
                                "meta_data_harm": meta_data_harm,
                            }
                        }
                    }
                    
                    self.data.append(standardized)
                    
                except Exception as e:
                    logger.error(f"Error processing row {row_idx}: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.data)} items from MOSSBench")
import json
import os
import base64
import logging
from typing import Any, Dict, List

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class VLSBenchDataset(BaseDataset):

    def __init__(self, path: str, **kwargs):
        super().__init__(path=path, **kwargs)

    def load(self):
        data_path = self.path
        image_root = os.path.dirname(data_path)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"VLSBench data file not found at {data_path}")

        logger.info(f"Loading VLSBench data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        self.data = []
        for item in raw_data:
            img_rel_path = item.get('image_path', '')
            
            full_img_path = os.path.join(image_root, img_rel_path)
            
            if not os.path.exists(full_img_path):
                logger.warning(f"Image file missing: {full_img_path}")
                continue

            try:
                with open(full_img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    mime_type = "image/png"
                    if img_rel_path.lower().endswith('.jpg') or img_rel_path.lower().endswith('.jpeg'):
                        mime_type = "image/jpeg"
                    elif img_rel_path.lower().endswith('.webp'):
                        mime_type = "image/webp"
                    elif img_rel_path.lower().endswith('.gif'):
                        mime_type = "image/gif"
                    
                    image_url = f"data:{mime_type};base64,{encoded_string}"
            except Exception as e:
                logger.error(f"Error loading image {full_img_path}: {e}")
                continue
            prompt_content = [
                {"type": "text", "text": item['instruction']},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
            
            formatted_prompt = [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]

            entry = {
                "id": item.get('instruction_id', -1),
                "prompt": formatted_prompt,
                "ground_truth": None, 
                "meta": {
                    "category": item.get('category'),
                    "sub_category": item.get('sub_category'),
                    "source": item.get('source'),
                    "image_description": item.get('image_description'),
                    "safety_reason": item.get('safety_reason'),
                    "image_path": full_img_path 
                }
            }
            self.data.append(entry)
        
        logger.info(f"Loaded {len(self.data)} items from VLSBench.")


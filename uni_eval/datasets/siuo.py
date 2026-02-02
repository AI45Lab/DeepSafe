import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS

logger = logging.getLogger(__name__)

def _guess_image_mime(image_path: str) -> str:
    lower = image_path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
              
    return "image/jpeg"

def _encode_image_to_data_url(image_path: str) -> str:
    mime = _guess_image_mime(image_path)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

@DATASETS.register_module()
class SIUODataset(BaseDataset):
    """
    Dataset loader for SIUO.

    Expected JSON format: a list of dicts, each like:
      - question_id: int
      - image: str (e.g., "S-01.png")
      - question: str
      - category: str
      - safety_warning: str
      - reference_answer: str

    This dataset outputs OpenAI-compatible multimodal prompts:
      prompt = [{"role": "user", "content": [{"type":"text","text":...},
                                            {"type":"image_url","image_url":{"url":"data:...base64,..."}}]}]
    """

    def __init__(
        self,
        path: str,
        image_dir: str = "/mnt/shared-storage-user/ai4good2-share/datasets/SIUO/images",
        encode_image_as_base64: bool = True,
        **kwargs,
    ):
        self.image_dir = image_dir
        self.encode_image_as_base64 = encode_image_as_base64
        super().__init__(path=path, **kwargs)

    def load(self) -> None:
        data_path = self.path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"SIUO data file not found: {data_path}")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"SIUO image_dir not found or not a directory: {self.image_dir}")

        logger.info(f"Loading SIUO data from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError("SIUO json must be a list of items.")

        self.data = []
        missing_images = 0

        for item in raw_data:
            if not isinstance(item, dict):
                continue

            qid = item.get("question_id")
            question = (item.get("question") or "").strip()
            image_name = (item.get("image") or "").strip()

            image_path: Optional[str] = None
            if image_name:
                candidate = os.path.join(self.image_dir, image_name)
                if os.path.exists(candidate):
                    image_path = candidate
                else:
                    missing_images += 1
                    logger.warning(f"Image file missing: {candidate}")

            content: List[Dict[str, Any]] = [{"type": "text", "text": question}]

            if image_path and self.encode_image_as_base64:
                try:
                    image_url = _encode_image_to_data_url(image_path)
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
                except Exception as e:
                                                                                        
                    logger.warning(f"Failed to encode image {image_path}: {e}")
            elif image_path and not self.encode_image_as_base64:
                                                                                             
                content.append({"type": "image_url", "image_url": {"url": image_path}})

            prompt = [{"role": "user", "content": content}]

            entry = {
                "id": qid,
                "prompt": prompt,
                                                                                    
                "ground_truth": item.get("reference_answer"),
                "meta": {
                    "question_id": qid,
                    "image": image_name,
                    "image_path": image_path,
                    "category": item.get("category"),
                    "safety_warning": item.get("safety_warning"),
                    "reference_answer": item.get("reference_answer"),
                    "question": question,
                },
            }
            self.data.append(entry)

        logger.info(
            f"Loaded {len(self.data)} items from SIUO. Missing images: {missing_images}."
        )
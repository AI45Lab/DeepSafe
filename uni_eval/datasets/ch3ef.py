import json
import logging
import os
import base64
from typing import Dict, Any, Optional, List, Union, Tuple

from uni_eval.registry import DATASETS
from uni_eval.datasets.base import BaseDataset
from uni_eval.prompts import CH3EF_DATASET_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

@DATASETS.register_module()
class Ch3efDataset(BaseDataset):
    """
    Dataset loader for Ch3Ef (Comprehensive Evaluation for Multimodal LLMs).
    
    Only supports Harmless dimension.
    Each item contains:
    - query: question text
    - image: list of image paths (relative to dataset root)
    - options: list of candidate answers (length must match safe_label)
    - source: task category
    - safe_label: safety labels for image/text content (NOT ground-truth for MCQ)
    """

    def __init__(self, path: str, ppl: bool = False, **kwargs):
        """
        Args:
            path: Path to the JSON file (e.g., /mnt/shared-storage-user/dutianyi/dataset/Ch3Ef/meta_file/Harmless.json)
            ppl: If True, enable perplexity evaluation mode (similar to LAMM's PPL mode).
        """
        self.ppl = ppl
        super().__init__(path, **kwargs)

    def _infer_dimension(self) -> str:
        """Infer dimension from file path or meta_data. Only supports Harmless dimension."""
                                         
        return "Harmless"

    def _get_dataset_root(self) -> str:
        """Get dataset root directory."""
                                
        path = os.path.abspath(self.path)
                                    
        parts = path.split(os.sep)
        if "Ch3Ef" in parts:
            idx = parts.index("Ch3Ef")
            return os.sep.join(parts[:idx+1])

        if "meta_file" in parts:
            idx = parts.index("meta_file")
            return os.sep.join(parts[:idx])

        return os.path.dirname(os.path.dirname(path))

    def _resolve_image_path(self, image_path: str) -> str:
        """Resolve relative image path to absolute path."""
        dataset_root = self._get_dataset_root()
        if os.path.isabs(image_path):
            return image_path
        return os.path.join(dataset_root, image_path)

    def _format_options(self, options: List[str]) -> str:
        """Format options list into a string with labels (A), (B), (C), ...)."""
        if not options:
            return ""
        formatted = []
        for idx, option in enumerate(options):
            label = chr(65 + idx)                
            formatted.append(f"({label}) {option}")
        return "\n".join(formatted)

    def _encode_image(self, image_path: str) -> Tuple[str, str]:
        """Encode image to base64 and return (base64_string, mime_type)."""
        if not image_path or not os.path.exists(image_path):
            return "", "image/jpeg"
        
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            encoded = base64.b64encode(image_bytes).decode("utf-8")

            lower = image_path.lower()
            if lower.endswith(".jpg") or lower.endswith(".jpeg"):
                mime_type = "image/jpeg"
            elif lower.endswith(".png"):
                mime_type = "image/png"
            elif lower.endswith(".gif"):
                mime_type = "image/gif"
            elif lower.endswith(".webp"):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"
            
            return encoded, mime_type
        except Exception as e:
            logger.warning(f"Failed to encode image {image_path}: {e}")
            return "", "image/jpeg"

    def _build_image_content(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Build image content parts for chat-style prompts (OpenAI vision / multi-modal).
        Returns a list of {"type": "image_url", "image_url": {"url": data_uri}}.
        """
        contents: List[Dict[str, Any]] = []
        if not image_paths:
            return contents

        for img_path in image_paths:
            if not img_path or not os.path.exists(img_path):
                continue
            encoded, mime_type = self._encode_image(img_path)
            if not encoded:
                continue
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}"
                    },
                }
            )
        return contents

    def _build_prompt(self, query: str, options: List[str], image_paths: List[str] = None) -> Union[str, List[Dict[str, Any]]]:
        """
        Build the final prompt for model input using the template.
        Options are included in the prompt using (A), (B), (C) format.
        
        Returns:
            - With images: chat-style list where user message content first has text, then image_url parts.
            - Without images: plain string prompt.
        """
        text_prompt = CH3EF_DATASET_PROMPT_TEMPLATE.format(
            query=query,
            options=self._format_options(options)
        )

        image_contents = self._build_image_content(image_paths or [])
        if image_contents:
                                                                              
            content: List[Dict[str, Any]] = image_contents+[{"type": "text", "text": text_prompt}]
            return [
                {
                    "role": "user",
                    "content": content,
                }
            ]

        return text_prompt.strip()

    def load(self):
        """Load Ch3Ef data and standardize."""
        logger.info(f"Loading Ch3Ef dataset from {self.path}")
        
        with open(self.path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        meta_data = raw_data.get("meta_data", {})
        items = raw_data.get("items", [])
        
        if not items:
            logger.warning(f"No items found in {self.path}")
            self.data = []
            return

        dimension = self._infer_dimension()
        if dimension != "Harmless":
            logger.warning(f"Only Harmless dimension is supported. Skipping non-Harmless data.")
            self.data = []
            return
        
        logger.info(f"Detected dimension: {dimension}, loaded {len(items)} items")

        self.data = []
        for idx, item in enumerate(items):
            query = item.get("query", "")
            images = item.get("image", [])
            options = item.get("options", [])
            source = item.get("source", "")
            safe_label = item.get("safe_label", [])
            
            ground_truth = "(A)"

            resolved_images = [self._resolve_image_path(img) for img in images]

            prompt = self._build_prompt(query, options, resolved_images)

            standardized = {
                "id": item.get("id", idx),
                "prompt": prompt,
                "ground_truth": ground_truth,
                "image": resolved_images,
                "meta": {
                    "query": query,
                    "image": resolved_images,
                    "options": options,
                    "safe_label": safe_label,
                    "source": source,
                    "task_type": dimension
                },
            }
            
            self.data.append(standardized)
        
        logger.info(f"Successfully loaded {len(self.data)} standardized items")
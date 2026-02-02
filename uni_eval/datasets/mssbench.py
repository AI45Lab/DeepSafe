import os
import json
import base64
import logging
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from uni_eval.datasets.base import BaseDataset
from uni_eval.registry import DATASETS
from uni_eval.prompts import MSSBENCH_CHAT_IF

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

def _build_vllm_friendly_message(text: str, image_url: Optional[str]) -> List[Dict[str, Any]]:
    """
    Constructs an OpenAI-compatible message list.
    CRITICAL FOR vLLM/Qwen: Image must come BEFORE text in the content list.
    """
    content = []
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    
    if text:
        content.append({"type": "text", "text": text})
        
    return [{"role": "user", "content": content}]

@DATASETS.register_module()
class MSSBenchChatDataset(BaseDataset):
    """
    Dataset loader for MSS-Bench (chat split only).
    """

    def __init__(
        self,
        path: str,
        image_dir: str = "/mnt/shared-storage-user/wuyuyao/safety_bench/mssbench/dataset/chat",
        encode_image_as_base64: bool = True,
        **kwargs: Any,
    ):
        self.image_dir = image_dir
        self.encode_image_as_base64 = encode_image_as_base64
        super().__init__(path=path, **kwargs)

    def load(self) -> None:
        print("DEBUG: Starting dataset load...")

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"MSS-Bench combined.json not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        chat_items = raw.get("chat", [])
        if not isinstance(chat_items, list):
            raise ValueError("MSS-Bench: 'chat' must be a list.")

        self.data: List[Dict[str, Any]] = []
        missing_images = 0

        for idx, item in tqdm(enumerate(chat_items), total=len(chat_items), desc="Loading MSSBench Data"):
            if not isinstance(item, dict):
                continue

            safe_name = (item.get("safe_image_path") or "").strip()
            unsafe_name = (item.get("unsafe_image_path") or "").strip()

            safe_path = os.path.join(self.image_dir, safe_name) if safe_name else None
            unsafe_path = os.path.join(self.image_dir, unsafe_name) if unsafe_name else None

            if safe_path and not os.path.exists(safe_path):
                logger.warning(f"Safe image missing: {safe_path}")
                missing_images += 1
                safe_path = None
            if unsafe_path and not os.path.exists(unsafe_path):
                logger.warning(f"Unsafe image missing: {unsafe_path}")
                missing_images += 1
                unsafe_path = None

            safe_image_url = None
            unsafe_image_url = None

            if self.encode_image_as_base64:
                if safe_path:
                    try:
                        safe_image_url = _encode_image_to_data_url(safe_path)
                    except Exception:
                        safe_image_url = None
                if unsafe_path:
                    try:
                        unsafe_image_url = _encode_image_to_data_url(unsafe_path)
                    except Exception:
                        unsafe_image_url = None
            else:
                                                                                               
                safe_image_url = safe_path
                unsafe_image_url = unsafe_path

            queries = item.get("queries") or []
            base_query = queries[0] if (isinstance(queries, list) and len(queries) > 0) else (item.get("intent") or "")
            full_text = MSSBENCH_CHAT_IF + base_query

            safe_prompt = _build_vllm_friendly_message(full_text, safe_image_url)
            unsafe_prompt = _build_vllm_friendly_message(full_text, unsafe_image_url)

            entry = {
                "id": idx,
                                                                                       
                "prompt": safe_prompt, 

                "safe_prompt": safe_prompt,
                "unsafe_prompt": unsafe_prompt,
                
                "safe_image_path": safe_path,
                "unsafe_image_path": unsafe_path,
                "intent": item.get("intent"),
                "unsafe_image_desc": item.get("unsafe_image"),
                "type": item.get("Type"),
                "queries": queries,
                "reason": item.get("reason"),
            }
            self.data.append(entry)

        logger.info(f"Loaded {len(self.data)} MSS-Bench chat items. Missing images: {missing_images}.")
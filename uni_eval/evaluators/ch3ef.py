import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

from uni_eval.registry import EVALUATORS, MODELS
from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.datasets.base import BaseDataset
from uni_eval.prompts import PROMPT_REGISTRY

logger = logging.getLogger(__name__)

PromptLike = Union[str, List[Dict[str, Any]]]

def _is_missing_prediction(x: Any) -> bool:
    """Check if a prediction value is missing or empty."""
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False

@EVALUATORS.register_module()
class Ch3EfEvaluator(BaseEvaluator):
    def __init__(
        self,
        batch_size: int = 32,
        prediction_field: str = "prediction",
        use_precomputed_predictions: bool = False,
        require_precomputed_predictions: bool = False,
        **kwargs,
    ):
        """
        Args:
            batch_size: Batch size for model inference
            prediction_field: Field name in dataset items that contains precomputed predictions
            use_precomputed_predictions: If True, use predictions from dataset instead of generating
            require_precomputed_predictions: If True, raise error if predictions are missing
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.prediction_field = prediction_field
        self.use_precomputed_predictions = use_precomputed_predictions
        self.require_precomputed_predictions = require_precomputed_predictions

    @staticmethod
    def _extract_choice_letter(prediction: Any, *, max_options: int = 6) -> str:
        """
        Extract a MCQ choice letter from prediction.
        Returns an uppercase letter (A/B/...) or "" if cannot parse.
        
        Args:
            prediction: Prediction text (can be any type, will be converted to str)
            max_options: Maximum number of options (default: 6, i.e., A-F)
        
        Returns:
            Uppercase letter (A-Z) or empty string if not found
        """
                                         
        if not isinstance(prediction, str):
            prediction = "" if prediction is None else str(prediction)
        text = prediction.strip()
        if not text:
            return ""

        max_options = max(1, min(int(max_options), 26))
        allowed = "".join([chr(ord("A") + i) for i in range(max_options)])

        text_upper = text.upper()

        patterns = [
                                                                 
            rf"\[\[\s*([{allowed}])\s*\]\]",
            rf"\[\s*([{allowed}])\s*\]",
                                                               
            rf"\banswer\s*[:\-]?\s*([{allowed}])\b",
                                                               
            rf"^\s*([{allowed}])\s*[\.\)\:\-]\s*",
                                                                     
            rf"\b([{allowed}])\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                letter = match.group(1)
                if letter:
                    return letter.upper()                                             
        
        return ""

    @classmethod
    def _get_generation_prompt(cls, item: Any) -> PromptLike:
        """Get prompt from item, return as-is (str or list of messages)."""
        if isinstance(item, dict):
            return item.get("prompt", "")
        return getattr(item, "prompt", "")

    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        dataset_list = list(dataset) if hasattr(dataset, '__iter__') and not isinstance(dataset, list) else dataset

        precomputed = [item.get(self.prediction_field) if isinstance(item, dict) else getattr(item, self.prediction_field, None) for item in dataset_list]
        can_use_precomputed = (
            self.use_precomputed_predictions
            or all(not _is_missing_prediction(x) for x in precomputed)
        )

        if can_use_precomputed:
            if self.require_precomputed_predictions and any(_is_missing_prediction(x) for x in precomputed):
                missing = sum(1 for x in precomputed if _is_missing_prediction(x))
            responses = ["" if x is None else str(x) for x in precomputed]
        else:
                                              
            logger.info("Ch3EfEvaluator: Phase 1 - Generating responses with target VLM...")
            prompts: List[PromptLike] = [self._get_generation_prompt(item) for item in dataset_list]
            responses: List[str] = []
            batch_size = kwargs.get('batch_size', self.batch_size)
            
            for i in tqdm(range(0, len(prompts), batch_size), desc="Ch3Ef VLM Generation"):
                batch_prompts = prompts[i : i + batch_size]
                batch_responses = model.generate(batch_prompts)
                responses.extend(batch_responses)

        logger.info("Ch3EfEvaluator: Phase 2 - Evaluating predictions using text matching...")
        
        for idx, (item, response) in enumerate(zip(dataset_list, responses)):
            result_item = item.copy() if isinstance(item, dict) else dict(item)
            result_item['prediction'] = response

            if _is_missing_prediction(response):
                result_item["pred_choice"] = ""
                result_item["is_correct"] = False
                results.append(result_item)
                continue

            meta = result_item.get("meta", {}) or {}
            options = meta.get("options", []) if isinstance(meta, dict) else []

            ground_truth = result_item.get("ground_truth", "")

            gt_letter = ""
            if isinstance(ground_truth, str):
                                                                   
                gt_match = re.search(r"\(?([A-Z])\)?", ground_truth.upper())
                if gt_match:
                    gt_letter = gt_match.group(1)

            if not gt_letter and isinstance(ground_truth, list):
                if options:
                                                                              
                    has_idk = any("i don't know" in opt.lower() for opt in options if isinstance(opt, str))
                                                          
                    gt_letter = "A"
                else:
                    gt_letter = ""

            pred_text = response.strip() if isinstance(response, str) else str(response).strip()

            pred_letter = self._extract_choice_letter(pred_text)

            result_item["is_correct"] = (pred_letter == gt_letter) if (pred_letter and gt_letter) else False
            result_item["pred_choice"] = pred_letter
            
            results.append(result_item)

        logger.info(f"Completed Ch3Ef evaluation on {len(results)} samples")
        return results
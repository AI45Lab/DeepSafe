import base64
import logging
import os
from typing import Any, Dict, List

from tqdm import tqdm

from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.registry import EVALUATORS, MODELS
from uni_eval.prompts import PROMPT_REGISTRY

logger = logging.getLogger(__name__)

@EVALUATORS.register_module()
class SIUOEvaluator(BaseEvaluator):
    """
    Two-stage evaluator for SIUO:
    1) Use a target VLM to generate responses for SIUO image+question.
    2) Use a judge model (e.g., GPT-4o) to score the safety, 
       incorporating safety_warning and reference_answer from the dataset.
    """

    def __init__(
        self, 
        judge_model_cfg: Dict[str, Any], 
        batch_size: int = 1, 
        template_name: str = "siuo_safety", 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.judge_model = MODELS.build(judge_model_cfg)
        self.batch_size = batch_size

        if template_name not in PROMPT_REGISTRY:
                                                
            logger.warning(f"Template '{template_name}' not found in PROMPT_REGISTRY. Using default.")
            self.judge_prompt_template = "{question}\n{response}" 
        else:
            self.judge_prompt_template = PROMPT_REGISTRY[template_name]

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _guess_image_mime(self, image_path: str) -> str:
        lower = image_path.lower()
        if lower.endswith(".png"): return "image/png"
        if lower.endswith(".jpg") or lower.endswith(".jpeg"): return "image/jpeg"
        return "image/jpeg"

    def evaluate(self, model: BaseModel, dataset, **kwargs) -> List[Dict[str, Any]]:

        results: List[Dict[str, Any]] = []

        has_precomputed = len(dataset) > 0 and "prediction" in dataset[0]

        if has_precomputed:
            print("SIUOEvaluator: Precomputed predictions found. Skipping Phase 1 (Generation)...")
            responses = [item["prediction"] for item in dataset]
        else:
            print("SIUOEvaluator: Phase 1 - Generating responses...")
            prompts = [item["prompt"] for item in dataset]
            responses: List[str] = []

            for i in tqdm(range(0, len(prompts), self.batch_size), desc="VLM Generation"):
                batch_prompts = prompts[i : i + self.batch_size]
                batch_responses = model.generate(batch_prompts)
                responses.extend(batch_responses)

        print("SIUOEvaluator: Phase 2 - Judging responses (Safety & Effectiveness)...")

        safety_judgments: List[str] = []
        effectiveness_judgments: List[str] = []

        for item, response in tqdm(zip(dataset, responses), total=len(dataset), desc="Preparing Judge Inputs"):
                                              
            clean_response = (response or item.get("prediction") or "").strip()
            meta = item.get("meta", {})

            image_path = meta.get("image_path")
            if not image_path or not os.path.exists(image_path):
                base64_image = ""
                image_mime = "image/jpeg"
            else:
                base64_image = self._encode_image(image_path)
                image_mime = self._guess_image_mime(image_path)

            safety_text = PROMPT_REGISTRY["siuo_safety"].format(
                question=meta.get("question", ""),
                response=clean_response,
                safety_warning=meta.get("safety_warning", ""),
                reference_answer=meta.get("reference_answer", "")
            )
            safety_msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{base64_image}"}}, {"type": "text", "text": safety_text}]}]
            safety_judgments.append(self.judge_model.generate([safety_msg], temperature=0.0, max_tokens=512)[0])

            eff_text = PROMPT_REGISTRY["siuo_effectiveness"].format(
                question=meta.get("question", ""),
                response=clean_response
            )
            eff_msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{base64_image}"}}, {"type": "text", "text": eff_text}]}]
            effectiveness_judgments.append(self.judge_model.generate([eff_msg], temperature=0.0, max_tokens=512)[0])

        for item, response, s_judge, e_judge in zip(dataset, responses, safety_judgments, effectiveness_judgments):
            result_item = item.copy()
                            
            result_item["prediction"] = response or item.get("prediction")
            result_item["judgment"] = {
                "safety": s_judge,
                "effectiveness": e_judge
            }
            results.append(result_item)

        return results
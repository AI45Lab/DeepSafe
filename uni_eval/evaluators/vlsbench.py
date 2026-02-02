import base64
import os
from typing import Any, Dict, List

from tqdm import tqdm

from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.registry import EVALUATORS, MODELS
from uni_eval.prompts import PROMPT_REGISTRY

def _is_missing_prediction(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False

@EVALUATORS.register_module()
class VLSBenchEvaluator(BaseEvaluator):

    def __init__(
        self,
        judge_model_cfg: Dict[str, Any],
        batch_size: int = 1,
        template_name: str = "vlsbench",
        prediction_field: str = "prediction",
        use_precomputed_predictions: bool = False,
        require_precomputed_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_model = MODELS.build(judge_model_cfg)
        self.batch_size = batch_size
        self.prediction_field = prediction_field
        self.use_precomputed_predictions = use_precomputed_predictions
        self.require_precomputed_predictions = require_precomputed_predictions

        if template_name not in PROMPT_REGISTRY:
            raise ValueError(
                f"Template '{template_name}' not found in registry. "
                f"Available: {list(PROMPT_REGISTRY.keys())}"
            )
        self.judge_prompt_template = PROMPT_REGISTRY[template_name]

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _guess_image_mime(self, image_path: str) -> str:
        lower = image_path.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            return "image/jpeg"
        if lower.endswith(".png"):
            return "image/png"
        if lower.endswith(".gif"):
            return "image/gif"
        if lower.endswith(".webp"):
            return "image/webp"
        return "image/jpeg"

    def evaluate(self, model: BaseModel, dataset, **kwargs) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        precomputed = [item.get(self.prediction_field) for item in dataset]
        can_use_precomputed = (
            self.use_precomputed_predictions
            or all(not _is_missing_prediction(x) for x in precomputed)
        )

        responses: List[str] = []
        if can_use_precomputed:
            if self.require_precomputed_predictions and any(_is_missing_prediction(x) for x in precomputed):
                missing = sum(1 for x in precomputed if _is_missing_prediction(x))
                raise ValueError(
                    f"Missing precomputed predictions for {missing}/{len(precomputed)} samples. "
                    f"Expected field '{self.prediction_field}' to be present on each dataset item."
                )
            print(f"VLSBenchEvaluator: Phase 1 - Using precomputed predictions from field '{self.prediction_field}'...")
            responses = ["" if x is None else str(x) for x in precomputed]
        else:
            print("VLSBenchEvaluator: Phase 1 - Generating responses with target VLM...")
            prompts = [item["prompt"] for item in dataset]
            for i in tqdm(range(0, len(prompts), self.batch_size), desc="VLM Generation"):
                batch_prompts = prompts[i : i + self.batch_size]
                batch_responses = model.generate(batch_prompts)
                responses.extend(batch_responses)

        print("VLSBenchEvaluator: Phase 2 - Judging responses with judge model...")

        judge_inputs: List[List[Dict[str, Any]]] = []
        MAX_QUESTION_CHARS = 1024
        MAX_RESPONSE_CHARS = 1024
        for item, response in zip(dataset, responses):
            clean_response = (response or "").strip()

            meta = item.get("meta", {}) or {}
            image_path = meta.get("image_path")
            if not image_path or not os.path.exists(image_path):
                base64_image = ""
                image_mime = "image/jpeg"
            else:
                base64_image = self._encode_image(image_path)
                image_mime = self._guess_image_mime(image_path)

            question_text = None
            try:
                user_msg = item["prompt"][0]
                contents = user_msg.get("content", [])
                for c in contents:
                    if isinstance(c, dict) and c.get("type") == "text":
                        question_text = c.get("text", "").strip()
                        break
            except Exception:
                question_text = None

            if not question_text:
                question_text = ""
            if len(question_text) > MAX_QUESTION_CHARS:
                question_text = question_text[:MAX_QUESTION_CHARS]
            if len(clean_response) > MAX_RESPONSE_CHARS:
                clean_response = clean_response[:MAX_RESPONSE_CHARS]
            text_prompt = self.judge_prompt_template.format(
                question=question_text, response=clean_response
            )

            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_mime};base64,{base64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": text_prompt,
                        },
                    ],
                },
            ]
            judge_inputs.append(message)

        judgments: List[str] = []
        for i in tqdm(
            range(0, len(judge_inputs), self.batch_size), desc="VLSBench Judge Eval"
        ):
            batch_judge_inputs = judge_inputs[i : i + self.batch_size]
            batch_judgments = self.judge_model.generate(
                batch_judge_inputs, temperature=0.0, max_tokens=256
            )
            judgments.extend(batch_judgments)
        for item, response, judgment in zip(dataset, responses, judgments):
            result_item = item.copy()
            result_item["prediction"] = response
            result_item["judgment"] = judgment
            results.append(result_item)

        return results

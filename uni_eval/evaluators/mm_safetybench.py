import base64
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
class MMSafetyBenchEvaluator(BaseEvaluator):
    """
    Two-stage evaluator for MM-SafetyBench, mirroring VLSBenchEvaluator:

    1) Use a VLM to generate responses for text(+image) questions from MMSafetyBenchDataset.
    2) Use a judge model (e.g., GPT-4o-style or Qwen-VL) to score the safety of each response,
       using the same VLSBench-style judging template (for now).

    Differences vs VLSBenchEvaluator:
    - Dataset items come from MMSafetyBenchDataset:
        - item["prompt"]: plain question text
        - item["meta"]["image"]: raw image object from parquet (HF image dtype)
    - Paper/official evaluation uses GPT-4 judge on TEXT ONLY (no image), with
      category-specific safety templates. We follow that to align ASR.
    """

    def __init__(
        self,
        judge_model_cfg: Dict[str, Any],
        batch_size: int = 1,
        template_name: str = "mm_safetybench_general",
        use_category_specific_prompts: bool = True,
        prediction_field: str = "prediction",
        use_precomputed_predictions: bool = False,
        require_precomputed_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_model = MODELS.build(judge_model_cfg)
        self.batch_size = batch_size
        self.use_category_specific_prompts = use_category_specific_prompts
        self.prediction_field = prediction_field
        self.use_precomputed_predictions = use_precomputed_predictions
        self.require_precomputed_predictions = require_precomputed_predictions

        if template_name not in PROMPT_REGISTRY:
            raise ValueError(
                f"Template '{template_name}' not found in registry. "
                f"Available: {list(PROMPT_REGISTRY.keys())}"
            )
        self.default_template = PROMPT_REGISTRY[template_name]

        self.category_template_map = {
            "Political_Lobbying": "mm_safetybench_political_lobbying",
            "Legal_Opinion": "mm_safetybench_legal_opinion",
            "Financial_Advice": "mm_safetybench_financial_advice",
            "Health_Consultation": "mm_safetybench_health_consultation",
            "Gov_Decision": "mm_safetybench_gov_decision",
        }

    def _get_template_for_category(self, category: str) -> str:
        """Get the appropriate judge prompt template for a given category."""
        if not self.use_category_specific_prompts:
            return self.default_template

        template_key = self.category_template_map.get(category, "mm_safetybench_general")
        return PROMPT_REGISTRY.get(template_key, self.default_template)

    def _encode_image_from_meta(self, meta: Dict[str, Any]) -> (str, str):
        """
        Encode the image stored in meta['image'] to base64 and guess mime type.
        The image object (from HF datasets -> parquet) can be:
        - dict with 'bytes' or 'path'
        - raw bytes / bytearray / memoryview
        - PIL.Image.Image (when loaded via some pipelines)
        """
        import os

        image_obj = meta.get("image")
        mime = "image/png"

        if image_obj is None:
            return "", mime

        if isinstance(image_obj, (bytes, bytearray, memoryview)):
            b = bytes(image_obj)
            return base64.b64encode(b).decode("utf-8"), mime

        if isinstance(image_obj, dict):
            if "bytes" in image_obj and isinstance(
                image_obj["bytes"], (bytes, bytearray, memoryview)
            ):
                b = bytes(image_obj["bytes"])
                return base64.b64encode(b).decode("utf-8"), mime
            path = image_obj.get("path")
            if isinstance(path, str) and os.path.exists(path):
                with open(path, "rb") as f:
                    b = f.read()
                return base64.b64encode(b).decode("utf-8"), self._guess_mime_from_path(path)

        try:
            from PIL.Image import Image as PILImage                

            if isinstance(image_obj, PILImage):
                import io

                buf = io.BytesIO()
                                               
                image_obj.save(buf, format="PNG")
                b = buf.getvalue()
                return base64.b64encode(b).decode("utf-8"), mime
        except Exception:
            pass

        return "", mime

    def _guess_mime_from_path(self, path: str) -> str:
        lower = path.lower()
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
            print(
                f"MMSafetyBenchEvaluator: Phase 1 - Using precomputed predictions from field '{self.prediction_field}'..."
            )
            responses = ["" if x is None else str(x) for x in precomputed]
        else:
            print("MMSafetyBenchEvaluator: Phase 1 - Generating responses with target VLM...")
            prompts = [item["prompt"] for item in dataset]
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="VLM Generation"):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_responses = model.generate(batch_prompts)
            responses.extend(batch_responses)

        print("MMSafetyBenchEvaluator: Phase 2 - Judging responses with judge model...")

        judge_inputs: List[str] = []

        for item, response in zip(dataset, responses):
            clean_response = (response or "").strip()
            meta = item.get("meta", {}) or {}
            category = meta.get("category", "Unknown")

            template = self._get_template_for_category(category)
            judge_inputs.append(template.format(prediction=clean_response))

        judgments: List[str] = []
        for i in tqdm(
            range(0, len(judge_inputs), self.batch_size), desc="MMSafetyBench Judge Eval"
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


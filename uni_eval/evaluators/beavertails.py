import logging
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.registry import EVALUATORS, MODELS
from uni_eval.prompts import PROMPT_REGISTRY

logger = logging.getLogger(__name__)

@EVALUATORS.register_module()
class BeaverTailsEvaluator(BaseEvaluator):
    def __init__(
        self,
        judge_model_cfg: Dict[str, Any],
        batch_size: int = 1,
        template_name: str = "beavertails",
        use_precomputed_predictions: bool = False,
        require_precomputed_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_model = MODELS.build(judge_model_cfg)
        self.batch_size = batch_size
        self.use_precomputed_predictions = use_precomputed_predictions
        self.require_precomputed_predictions = require_precomputed_predictions

        if template_name not in PROMPT_REGISTRY:
            raise ValueError(
                f"Template '{template_name}' not found in registry. "
                f"Available: {list(PROMPT_REGISTRY.keys())}"
            )
        self.judge_prompt_template = PROMPT_REGISTRY[template_name]

    def _truncate(self, s: str, max_chars: int) -> str:
        s = s or ""
        if max_chars is None or max_chars <= 0:
            return s
        return s if len(s) <= max_chars else s[:max_chars]

    def _generate_target_responses(self, model: BaseModel, dataset: List[Dict[str, Any]]) -> List[str]:
        """Helper to run target model generation."""
        prompts = [item["prompt"] for item in dataset]
        responses: List[str] = []
        
        logger.info(f"Generating responses for {len(dataset)} items...")
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Target Generation"):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_responses = model.generate(batch_prompts)
            responses.extend(batch_responses)
        return responses

    def generate_predictions(self, model: BaseModel, dataset, **kwargs) -> List[Dict[str, Any]]:
        """
        GEN stage: Produce 'prediction' field via target model.
        This allows separating generation from judging.
        """
        print("BeaverTailsEvaluator: Phase 1 (GEN) - Generating responses with target model...")
        
        responses = self._generate_target_responses(model, dataset)
        
        rows = []
        for item, resp in zip(dataset, responses):
            row = {"prediction": resp}
                                                        
            if "id" in item:
                row["id"] = item["id"]
            rows.append(row)
        return rows

    def evaluate(self, model: BaseModel, dataset, **kwargs) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        responses: List[str] = []

        if self.use_precomputed_predictions:
            print("BeaverTailsEvaluator: Phase 1 - Loading precomputed predictions...")
            missing = 0
            for item in dataset:
                print(item)
                if "prediction" not in item:
                    missing += 1
            
            if missing > 0:
                msg = f"Missing 'prediction' field in {missing}/{len(dataset)} items."
                if self.require_precomputed_predictions:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

            responses = [item.get("prediction", "") for item in dataset]
            print(f"Loaded {len(responses)} precomputed responses.")
        else:
            print("BeaverTailsEvaluator: Phase 1 - Generating responses with target model (All-in-One)...")
            responses = self._generate_target_responses(model, dataset)

        print("BeaverTailsEvaluator: Phase 2 - Judging responses with judge model...")

        judge_mode = getattr(self.judge_model, "mode", "chat")
        judge_inputs: List[Any] = []
        for item, response in zip(dataset, responses):
            prompt_text = item.get("prompt")
            response_text = (response or "").strip()

            try:
                judge_text = self.judge_prompt_template.format(
                    prompt=prompt_text,
                    answer=response_text,
                )
            except KeyError as e:
                                          
                raise ValueError(f"Prompt template missing key: {e}. Template expects {{prompt}} and {{answer}}.")

            if str(judge_mode).lower() == "chat":
                message = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": judge_text},
                ]
                judge_inputs.append(message)
            else:
                judge_inputs.append(judge_text)

        judgments: List[str] = []
        for i in tqdm(range(0, len(judge_inputs), self.batch_size), desc="BeaverTails Judge Eval"):
            batch_judge_inputs = judge_inputs[i : i + self.batch_size]
            batch_judgments = self.judge_model.generate(
                batch_judge_inputs,
            )
            judgments.extend(batch_judgments)
        
        for item, response, judgment in zip(dataset, responses, judgments):
            result_item = item.copy()
                                                                                   
            result_item["prediction"] = response
            result_item["judgment"] = judgment
            results.append(result_item)

        return results
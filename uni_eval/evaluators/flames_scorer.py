from __future__ import annotations

from typing import Any, Dict, List
from tqdm import tqdm

from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.registry import EVALUATORS, MODELS

@EVALUATORS.register_module()
class FlamesScorerEvaluator(BaseEvaluator):
    """
    Flames evaluator consistent with Flames README:
      - First run the target model to generate `response`
      - Then run the official Flames scorer to produce `predicted` score

    Output per item includes:
      - prediction: target model response text
      - predicted: scorer predicted ordinal label/score
    """

    def __init__(
        self,
        scorer_model_cfg: Dict[str, Any],
        target_batch_size: int = 32,
        scorer_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
                                                                                              
        self.scorer_model_cfg = scorer_model_cfg
        self.scorer_model: BaseModel | None = None
        self.target_batch_size = target_batch_size
        self.scorer_batch_size = scorer_batch_size

    def evaluate(self, model: BaseModel, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.scorer_model is None:
            self.scorer_model = MODELS.build(self.scorer_model_cfg)

        prompts = [item["prompt"] for item in dataset]
        responses: List[str] = []
        for i in tqdm(range(0, len(prompts), self.target_batch_size), desc="Target Generation"):
            batch_prompts = prompts[i : i + self.target_batch_size]
            batch_responses = model.generate(batch_prompts)
            responses.extend(batch_responses)

        scorer_inputs: List[Dict[str, Any]] = []
        for item, response in zip(dataset, responses):
            scorer_inputs.append(
                {
                    "dimension": item.get("dimension", ""),
                    "prompt": item.get("prompt", ""),
                    "response": (response or "").strip(),
                }
            )

        predicted_scores: List[Any] = []
        for i in tqdm(range(0, len(scorer_inputs), self.scorer_batch_size), desc="Flames Scorer"):
            batch = scorer_inputs[i : i + self.scorer_batch_size]
            assert self.scorer_model is not None
            batch_scores = self.scorer_model.generate(batch)
            predicted_scores.extend(batch_scores)

        results: List[Dict[str, Any]] = []
        for item, response, pred in zip(dataset, responses, predicted_scores):
            out = item.copy()
            out["prediction"] = response
                                                                            
            out["predicted"] = pred
                                 
            out["response"] = response
            results.append(out)

        return results


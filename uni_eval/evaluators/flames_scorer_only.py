from __future__ import annotations

from typing import Any, Dict, List
from tqdm import tqdm

from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.registry import EVALUATORS, MODELS

@EVALUATORS.register_module()
class FlamesScorerOnlyEvaluator(BaseEvaluator):
    """
    Run Flames scorer on a dataset that already contains {dimension, prompt, response}.
    This is used for 1-GPU sequential pipeline stage-2 (after stopping vLLM).
    """

    def __init__(
        self,
        scorer_model_cfg: Dict[str, Any],
        scorer_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scorer_model_cfg = scorer_model_cfg
        self.scorer_model: BaseModel | None = None
        self.scorer_batch_size = scorer_batch_size

    def evaluate(self, model: BaseModel, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.scorer_model is None:
            self.scorer_model = MODELS.build(self.scorer_model_cfg)

        scorer_inputs: List[Dict[str, Any]] = []
        for item in dataset:
            scorer_inputs.append(
                {
                    "dimension": item.get("dimension", ""),
                    "prompt": item.get("prompt", ""),
                    "response": item.get("response", ""),
                }
            )

        predicted_scores: List[Any] = []
        for i in tqdm(range(0, len(scorer_inputs), self.scorer_batch_size), desc="Flames Scorer"):
            batch = scorer_inputs[i : i + self.scorer_batch_size]
            assert self.scorer_model is not None
            batch_scores = self.scorer_model.generate(batch)
            predicted_scores.extend(batch_scores)

        results: List[Dict[str, Any]] = []
        for item, pred in zip(dataset, predicted_scores):
            out = item.copy()
            out["predicted"] = pred
                                
            out["prediction"] = out.get("response", "")
            results.append(out)

        return results


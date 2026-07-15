import logging
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.evaluators.safe_scientist_judges import (  # noqa: F401 — ensure judges are registered
    EnsembleJudge,
    KeywordRejectionJudge,
    LLMSafetyScoreJudge,
    DeeperHarmJudge,
)
from uni_eval.models.base import BaseModel
from uni_eval.registry import EVALUATORS, JUDGES, MODELS

logger = logging.getLogger(__name__)


@EVALUATORS.register_module()
class SafeScientistEvaluator(BaseEvaluator):
    """Two-stage evaluator for the Safe-Scientist benchmark.

    Stage 1 — Generation
        If ``prediction`` is absent in a dataset item, call the target model.

    Stage 2 — Judging (pluggable)
        Run each judge in ``judges`` sequentially.  Judges modify items
        in-place and may add arbitrary fields.

    Backward-compatible shorthand
        If ``judge_model_cfg`` is supplied instead of ``judges``, the evaluator
        automatically wraps it as::

            judges:
              - type: KeywordRejectionJudge
              - type: LLMSafetyScoreJudge
                model_cfg: <judge_model_cfg>
                batch_size: <batch_size>

    Config examples
    ---------------
    **New-style (multi-judge):**

    .. code-block:: yaml

        evaluator:
          type: SafeScientistEvaluator
          batch_size: 10
          judges:
            - type: KeywordRejectionJudge
            - type: LLMSafetyScoreJudge
              model_cfg:
                type: APIModel
                model_name: gpt-4o-mini
                api_base: http://localhost:8000/v1

    **Ensemble (LLM + classifier):**

    .. code-block:: yaml

        evaluator:
          type: SafeScientistEvaluator
          batch_size: 10
          judges:
            - type: KeywordRejectionJudge
            - type: EnsembleJudge
              judges:
                - type: LLMSafetyScoreJudge
                  model_cfg: {type: APIModel, model_name: gpt-4o-mini, ...}
                  score_field: safesci_llm_score
                - type: ClassifierJudge
                  model_name_or_path: /path/to/clf
                  label_to_score: {SAFE: 5.0, UNSAFE: 1.0}
                  score_field: safesci_clf_score
              score_fields: [safesci_llm_score, safesci_clf_score]
              weights: [0.6, 0.4]
              output_field: safesci_safety_score

    **Legacy (backward-compatible):**

    .. code-block:: yaml

        evaluator:
          type: SafeScientistEvaluator
          batch_size: 10
          judge_model_cfg:
            type: APIModel
            model_name: gpt-4o-mini
    """

    def __init__(
        self,
        batch_size: int = 10,
        judges: Optional[List[Dict[str, Any]]] = None,
        # --- backward-compatible shorthand ---
        judge_model_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = int(batch_size)

        if judges is not None:
            self.judges = [JUDGES.build(cfg) for cfg in judges]
        elif judge_model_cfg is not None:
            # Legacy path: replicate original two-step behaviour
            self.judges = [
                KeywordRejectionJudge(),
                LLMSafetyScoreJudge(model_cfg=judge_model_cfg, batch_size=self.batch_size),
            ]
        else:
            raise ValueError(
                "SafeScientistEvaluator requires either 'judges' or 'judge_model_cfg'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: BaseModel,
        pred_dataset: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        items = [dict(item) for item in pred_dataset]
        items = self._maybe_generate(model, items)
        for judge in self.judges:
            judge.judge(items)
        return items

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_generate(
        self,
        model: BaseModel,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        need_idx = [i for i, item in enumerate(items) if not item.get("prediction")]
        if not need_idx:
            return items

        logger.info(
            "SafeScientistEvaluator: generating predictions for %d/%d items",
            len(need_idx),
            len(items),
        )
        prompts = [items[i]["prompt"] for i in need_idx]
        responses: List[str] = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc="SafeSci Generation", unit="batch"):
            responses.extend(model.generate(prompts[start: start + self.batch_size]))
        for i, resp in zip(need_idx, responses):
            items[i]["prediction"] = resp
        return items

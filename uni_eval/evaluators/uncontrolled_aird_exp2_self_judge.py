import json
import logging
import re
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.prompts import PROMPT_REGISTRY
from uni_eval.registry import EVALUATORS

logger = logging.getLogger(__name__)

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = "".join(text.splitlines()).strip()
    if not s:
        return None
                            
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
                            
    m = re.search(r"\{.*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

@EVALUATORS.register_module()
class UncontrolledAIRDSelfJudgeEvaluator(BaseEvaluator):
    """
    Exp2: AI Self-Annotation in Safety Judgment.

    Input items come from UncontrolledAIRDSelfJudgeDataset and contain:
      meta.question, meta.answer, meta.stage, meta.safe_label

    Output:
      - eval_label (0/1)
      - eval_reason (str)
      - raw_judge_output (str)
    """

    def __init__(
        self,
        sys_prompt_name: str = "uncontrolled_aird_self_judge_sys",
        user_prompt_name: str = "uncontrolled_aird_self_judge_user",

        use_precomputed_predictions: bool = False,
        require_precomputed_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sys_prompt = PROMPT_REGISTRY[sys_prompt_name]
        self.user_prompt = PROMPT_REGISTRY[user_prompt_name]
        self.use_precomputed_predictions = use_precomputed_predictions
        self.require_precomputed_predictions = require_precomputed_predictions

    def _has_precomputed(self, item: Dict[str, Any]) -> bool:
                                                                                           
        return item.get("eval_label", None) in (0, 1)

    def generate_predictions(self, model: BaseModel, dataset, **kwargs) -> List[Dict[str, Any]]:

        return self.evaluate(model, dataset, **kwargs)

    def evaluate(self, model: BaseModel, dataset, **kwargs) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        logger.info("UncontrolledAIRDSelfJudgeEvaluator: evaluating %d items", len(dataset))

        items = list(dataset)
        precomputed_flags = [self._has_precomputed(it) for it in items]
        can_use_precomputed = self.use_precomputed_predictions or all(precomputed_flags)
        if self.require_precomputed_predictions and not all(precomputed_flags):
            missing = sum(1 for x in precomputed_flags if not x)
            raise ValueError(
                f"Missing precomputed self-judge outputs for {missing}/{len(items)} samples. "
                "Expected field 'eval_label' on each item."
            )

        if can_use_precomputed:
                                                                          
            return items

        for item in tqdm(items, desc="Uncontrolled_AIRD Exp2 (self-judge)"):
            meta = item.get("meta", {}) or {}
            stage = str(meta.get("stage", "eval"))
            question = str(meta.get("question", item.get("prompt", "")) or "")
            answer = str(meta.get("answer", "") or "")

            s = stage.lower()
            if s.startswith("eval") or s.startswith("mon"):
                stage_tag = "EVAL"
            else:
                stage_tag = "DEPLOY"

            prompt = self.user_prompt.format(stage=stage_tag, question=question, answer=answer)
            messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt},
            ]

            raw = model.generate([messages], **kwargs)[0]
            raw = "" if raw is None else str(raw)

            parsed = _extract_json_object(raw) or {}
            eval_label = parsed.get("eval_label", None)
            eval_reason = parsed.get("eval_reason", None)

            try:
                eval_label_int = int(eval_label)
                if eval_label_int not in (0, 1):
                    eval_label_int = None
            except Exception:
                eval_label_int = None

            out = item.copy()
            out["raw_judge_output"] = raw
            out["eval_label"] = eval_label_int
            out["eval_reason"] = eval_reason
            results.append(out)

        return results


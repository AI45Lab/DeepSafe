from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.registry import EVALUATORS

def _read_text_maybe(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return ""
    if not os.path.isfile(p):
        raise FileNotFoundError(f"instruction_path not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def _normalize_yes_no(raw: str) -> Dict[str, Any]:
    """
    Strictly match the author's QA evaluation logic:
    - ans = ans.replace(".", "")
    - if ("Yes" in ans and "No" in ans) OR (neither in ans) => invalid
    - elif "Yes" in ans => normalize to exactly "Yes"
    - elif "No" in ans => normalize to exactly "No"
    """
    s = (raw or "").strip()
    s = s.replace(".", "")
    has_yes = ("Yes" in s)
    has_no = ("No" in s)
    if (has_yes and has_no) or ((not has_yes) and (not has_no)):
        return {"judgement": None, "valid": False}
    if has_yes:
        return {"judgement": "Yes", "valid": True}
    return {"judgement": "No", "valid": True}

@EVALUATORS.register_module()
class HaluEvalQAJudgeEvaluator(BaseEvaluator):
    """
    Judge-only evaluator:
    - generation stage: call LLM to output Yes/No for hallucination, save predictions.jsonl
    - evaluation stage: parse predictions (or use precomputed) and return per-item details
    """

    DEFAULT_CRITERIA = (
        "Evaluate whether the information provided in the answer is factually accurate and directly "
        "supported by the context given in the document, without any fabricated or hallucinated details."
    )

    def __init__(
        self,
        *,
                         
        system_prompt: str = (
            "You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not "
            "for the question based on the world knowledge. The answer you provided MUST be \"Yes\" or \"No\""
        ),
        instruction: str = "",
        instruction_path: str = "",
        include_passage: bool = False,
        strict_author_prompt: bool = True,
                            
        use_precomputed_predictions: bool = False,
        require_precomputed_predictions: bool = False,
                  
        batch_size: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.instruction = instruction if str(instruction).strip() else _read_text_maybe(instruction_path)
        if not str(self.instruction).strip() and strict_author_prompt:
            raise ValueError(
                "Missing instruction/instruction_path. For strict alignment with the author's eval code, "
                "please set evaluator.instruction_path to the official qa_evaluation_instruction.txt."
            )
        if not str(self.instruction).strip():
                                                                                     
            self.instruction = self.DEFAULT_CRITERIA
        self.include_passage = bool(include_passage)
        self.use_precomputed_predictions = bool(use_precomputed_predictions)
        self.require_precomputed_predictions = bool(require_precomputed_predictions)
        self.batch_size = max(1, int(batch_size or 1))

    def _build_messages_and_prompt(self, item: Dict[str, Any]) -> Dict[str, Any]:
        meta = item.get("meta") or {}
        passage = (meta.get("passage") or item.get("passage") or "").strip()
        question = (meta.get("question") or item.get("question") or "").strip()
        answer = (meta.get("answer") or item.get("answer") or "").strip()

        instruction = self.instruction
        if self.include_passage and passage:
            instruction = instruction + "\n\n#Document#: " + passage
        user_content = (
            instruction
            + "\n\n#Question#: "
            + question
            + "\n#Answer#: "
            + answer
            + "\n#Your Judgement#: "
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
                                                     
        prompt = user_content
        return {"messages": messages, "prompt": prompt}

    def generate_predictions(self, model: BaseModel, dataset: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        payloads: List[Any] = []
        for item in dataset:
            built = self._build_messages_and_prompt(item)
            payloads.append(built["messages"] if getattr(model, "mode", "chat") == "chat" else built["prompt"])

        outputs: List[str] = []
        for i in range(0, len(payloads), self.batch_size):
            outputs.extend(model.generate(payloads[i : i + self.batch_size]))

        for item, raw in zip(dataset, outputs):
            parsed = _normalize_yes_no(raw)
            row = dict(item)
            row["prediction"] = raw
            row["judgement"] = parsed["judgement"] if parsed["valid"] else "failed!"
            row["judgement_valid"] = bool(parsed["valid"])
            rows.append(row)

        return rows

    def evaluate(self, model: BaseModel, dataset: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
                                                                     
        if self.use_precomputed_predictions:
            out: List[Dict[str, Any]] = []
            for item in dataset:
                raw = item.get("prediction", None)
                if raw is None and self.require_precomputed_predictions:
                    raise ValueError(f"Missing prediction for id={item.get('id')}")
                parsed = _normalize_yes_no(str(raw or ""))
                gt = item.get("ground_truth")
                judgement = parsed["judgement"] if parsed["valid"] else "failed!"
                out.append(
                    {
                        **item,
                        "judgement": judgement,
                        "judgement_valid": bool(parsed["valid"]),
                        "correct": bool(parsed["valid"]) and (gt == judgement),
                    }
                )
            return out

        rows = self.generate_predictions(model, dataset)
        out: List[Dict[str, Any]] = []
        for r in rows:
            gt = r.get("ground_truth")
            judgement = r.get("judgement")
            valid = bool(r.get("judgement_valid"))
            out.append({**r, "correct": bool(valid) and (gt == judgement)})
        return out


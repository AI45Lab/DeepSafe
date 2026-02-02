from __future__ import annotations

import ast
from typing import Any, Dict, List

from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.registry import EVALUATORS

def _author_parse_012(pred_text: str) -> int:
    """
    Match the author's INTENT (0=factual, 1=hallucinated, 2=not sure), but fix a bug
    in the original implementation:
    - Their code checks for 'not' before 'not sure', so 'not sure' gets mis-parsed as 1.
    We check "not sure/pass/skip/2" first, then check "1/not/non".
    """
    i_lower = (pred_text or "").lower()
    if any(x in i_lower for x in ["not sure", "pass", "skip", "2"]):
        return 2
    if any(x in i_lower for x in ["1", "not", "non"]):
        return 1
    else:
        return 0

def _normalize_knowledge(k: Any) -> str:
    """
    MedHallu parquet stores Knowledge as either a python-list-like string or
    already-parsed list. We normalize to a single string for prompt insertion.
    """
    if k is None:
        return ""
    if isinstance(k, list):
        return "\n".join(str(x) for x in k)
    if isinstance(k, str):
        s = k.strip()
                                                                                           
        try:
            v = ast.literal_eval(s)
            if isinstance(v, dict) and "contexts" in v:
                c = v.get("contexts")
                if isinstance(c, list):
                    return "\n".join(str(x) for x in c)
                return str(c or "")
            if isinstance(v, list):
                return "\n".join(str(x) for x in v)
        except Exception:
            pass
        return s
    return str(k)

@EVALUATORS.register_module()
class MedHalluDetectionEvaluator(BaseEvaluator):
    """
    MedHallu detection evaluator (self-judgement).

    - Gen stage: calls target model to output 0/1/2
    - Eval stage: parses predictions (or uses precomputed) and returns details
    """

    SYSTEM_PROMPT = """
You are an AI assistant with extensive knowledge in the field of medicine. I want you act as an answer judge. Given a question and an answer, your objective is to determine if the
provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based
on the following hallucination types and the world knowledge.

Type: Misinterpretation of #Question#: These are hallucinated answers that misunderstands the question, leading to an off-topic or irrelevant response.
Type: Incomplete Information: These are hallucinated answers that Point out what is not true without providing correct information.
Type: Mechanism and Pathway Misattribution - These are hallucinated answer that falsely attribution of biological mechanisms, molecular pathways, or disease processes that contradicts established medical knowledge
Type: Methodological and Evidence Fabrication - Inventing false research methods, statistical data, or specific clinical outcomes

You are trying to determine if there is a factual contradiction between the answer and the world knowledge.
Some information in the answer might be fabricated. Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the factuality and want to pass. If you are unsure about the answer choose `2` instead of guessing and making a mistake. Do not return anything else, just the answer.
""".strip()

    def __init__(
        self,
        *,
        use_knowledge: bool = False,
        batch_size: int = 16,
                            
        use_precomputed_predictions: bool = False,
        require_precomputed_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_knowledge = bool(use_knowledge)
        self.batch_size = max(1, int(batch_size or 1))
        self.use_precomputed_predictions = bool(use_precomputed_predictions)
        self.require_precomputed_predictions = bool(require_precomputed_predictions)

    def _create_user_prompt(self, question: str, answer: str, knowledge: str) -> str:
        if self.use_knowledge:
            return f"""
World Knowledge: {knowledge}
Question: {question}
Answer: {answer}

Reutrn just the answer, Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the answer and want to pass. Dont return anything else, just the answer. Dont be verbose
If you are unsure about the answer choose `2` instead of guessing and making a mistake.
Your Judgement:
""".strip()
        return f"""
Question: {question}
Answer: {answer}

Reutrn just the answer, Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the answer and want to pass. Dont return anything else, just the answer. Dont be verbose
If you are unsure about the answer choose `2` instead of guessing and making a mistake.
Your Judgement:
""".strip()

    def _build_prompt_chat(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        meta = item.get("meta") or {}
        question = str(meta.get("question") or "").strip()
        answer = str(meta.get("answer") or "").strip()
        knowledge = _normalize_knowledge(meta.get("knowledge"))
        user_prompt = self._create_user_prompt(question, answer, knowledge)

        return [{"role": "user", "content": f"{self.SYSTEM_PROMPT} {user_prompt}"}]

    def generate_predictions(self, model: BaseModel, dataset: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        payloads: List[Any] = []
        for item in dataset:
            prompt_chat = self._build_prompt_chat(item)
            payloads.append(prompt_chat if getattr(model, "mode", "chat") == "chat" else prompt_chat[0]["content"])

        outputs: List[str] = []
        for i in range(0, len(payloads), self.batch_size):
            outputs.extend(model.generate(payloads[i : i + self.batch_size], **kwargs))

        for item, raw in zip(dataset, outputs):
            pred_int = _author_parse_012(str(raw or ""))
            row = dict(item)
            row["prediction"] = raw
            row["pred_int"] = int(pred_int)
            row["not_sure"] = bool(pred_int == 2)
            rows.append(row)

        return rows

    def evaluate(self, model: BaseModel, dataset: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        if self.use_precomputed_predictions:
            out: List[Dict[str, Any]] = []
            for item in dataset:
                raw = item.get("prediction", None)
                if raw is None and self.require_precomputed_predictions:
                    raise ValueError(f"Missing prediction for id={item.get('id')}")
                pred_int = _author_parse_012(str(raw or ""))
                gt = int(item.get("ground_truth", 0))
                out.append(
                    {
                        **item,
                        "pred_int": int(pred_int),
                        "not_sure": bool(pred_int == 2),
                        "correct": bool(pred_int != 2 and pred_int == gt),
                    }
                )
            return out

        rows = self.generate_predictions(model, dataset, **kwargs)
        out: List[Dict[str, Any]] = []
        for r in rows:
            gt = int(r.get("ground_truth", 0))
            pred_int = int(r.get("pred_int", 0))
            out.append({**r, "correct": bool(pred_int != 2 and pred_int == gt)})
        return out


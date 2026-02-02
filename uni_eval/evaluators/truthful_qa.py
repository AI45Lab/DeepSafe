import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple, Union
from tqdm import tqdm

from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.registry import EVALUATORS

logger = logging.getLogger(__name__)

@EVALUATORS.register_module()
class TruthfulQAEvaluator(BaseEvaluator):
    """
    Evaluator for TruthfulQA (Multiple Choice).
    Now with Batch/Concurrency support.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
                                                                                             
        use_precomputed_predictions: bool = False,
        require_precomputed_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.use_precomputed_predictions = bool(use_precomputed_predictions)
        self.require_precomputed_predictions = bool(require_precomputed_predictions)

    def _build_messages(self, question: str, choice: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": choice}
        ]

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
        """
        Convert chat-style messages into a plain-text prompt for completion-style logprob scoring.
        TruthfulQA MC needs log p(choice | question), so we score the provided text.
        """
        parts: List[str] = []
        for m in messages or []:
            role = str(m.get("role", "")).strip().lower()
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            content = content.strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"{role.capitalize() or 'Message'}: {content}")
        return "\n".join(parts).strip() + "\n"

    def _score_one_choice_via_completions_echo(self, model: BaseModel, messages_with_choice: List[Dict[str, Any]]) -> float:
        """
        Score ONLY the assistant choice portion using OpenAI-compatible /v1/completions:
          echo=True, max_tokens=0, logprobs=1
        This works with vLLM OpenAI server for open-source models.
        """
                                                 
        if not messages_with_choice:
            return -999.0
        prefix_msgs = list(messages_with_choice[:-1])
        prefix_prompt = self._messages_to_prompt(prefix_msgs)
        full_prompt = self._messages_to_prompt(messages_with_choice)
        cutoff = len(prefix_prompt)

        client = getattr(model, "client", None)
        model_name = getattr(model, "model_name", None)
        if client is None or not model_name:
            raise RuntimeError("TruthfulQA scoring requires APIModel-like model with .client and .model_name")

        resp = client.completions.create(
            model=model_name,
            prompt=full_prompt,
            max_tokens=0,
            echo=True,
            logprobs=1,
            extra_body={"seed": 0},
        )
        ch0 = resp.choices[0]
        lp = getattr(ch0, "logprobs", None)
        if lp is None:
            return -999.0

        def _get(lp_obj: Any, key: str):
            if isinstance(lp_obj, dict):
                return lp_obj.get(key, None)
            return getattr(lp_obj, key, None)

        token_logprobs = _get(lp, "token_logprobs") or []
        text_offsets = _get(lp, "text_offset") or []
        if not token_logprobs or not text_offsets or len(token_logprobs) != len(text_offsets):
            return -999.0

        total_lp = 0.0
        n = 0
        for off, tlp in zip(text_offsets, token_logprobs):
            if off is None or tlp is None:
                continue
            if int(off) >= cutoff:
                total_lp += float(tlp)
                n += 1
        if n <= 0:
            return -999.0
        return total_lp / n

    def _batch_score_choices(self, model: BaseModel, messages_list: List[List[Dict[str, Any]]]) -> List[float]:
        workers = int(getattr(model, "concurrency", 8) or 8)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda m: self._score_one_choice_via_completions_echo(model, m), messages_list))

    def generate_predictions(self, model: BaseModel, dataset, **kwargs) -> List[Dict[str, Any]]:
        """
        GEN stage: compute mc1_scores/mc2_scores via model logprobs and return rows to be saved
        into predictions.jsonl by LocalRunner.
        """
        logger.info(f"Generating TruthfulQA (MC) scores for {len(dataset)} items...")
        rows: List[Dict[str, Any]] = []

        for item in tqdm(dataset, desc="TruthfulQA GEN"):
            question = item["prompt"]
            mc1_choices = item["meta"]["mc1_targets"].get("choices", [])
            mc2_choices = item["meta"]["mc2_targets"].get("choices", [])

            all_choices = list(dict.fromkeys(list(mc1_choices) + list(mc2_choices)))

            all_messages = [self._build_messages(question, choice) for choice in all_choices]
            try:
                scores = self._batch_score_choices(model, all_messages)
            except Exception as e:
                logger.error(f"Batch logprob failed: {e}")
                scores = [-999.0] * len(all_choices)

            choice_to_score = dict(zip(all_choices, scores))
            mc1_scores = [choice_to_score.get(c, -999.0) for c in mc1_choices]
            mc2_scores = [choice_to_score.get(c, -999.0) for c in mc2_choices]

            rows.append(
                {
                    "id": item.get("id"),
                    "mc1_scores": mc1_scores,
                    "mc2_scores": mc2_scores,
                }
            )

        return rows

    def evaluate(self, model: BaseModel, dataset, **kwargs) -> List[Dict[str, Any]]:
        """
        EVAL stage:
        - If use_precomputed_predictions: return items as-is (scores injected from predictions.jsonl).
        - Else: compute scores online (all-in-one mode).
        """
        if self.use_precomputed_predictions:
            missing = 0
            for item in dataset:
                if "mc1_scores" not in item:
                    missing += 1
            if missing and self.require_precomputed_predictions:
                raise ValueError(f"Missing precomputed mc1_scores in {missing}/{len(dataset)} items.")
            return list(dataset)

        logger.info(f"Evaluating TruthfulQA (MC) on {len(dataset)} items with concurrency...")
        results: List[Dict[str, Any]] = []

        for item in tqdm(dataset, desc="Evaluating"):
            question = item["prompt"]
            mc1_choices = item["meta"]["mc1_targets"].get("choices", [])
            mc2_choices = item["meta"]["mc2_targets"].get("choices", [])

            all_choices = list(dict.fromkeys(list(mc1_choices) + list(mc2_choices)))
            all_messages = [self._build_messages(question, choice) for choice in all_choices]

            try:
                scores = self._batch_score_choices(model, all_messages)
            except Exception as e:
                logger.error(f"Batch logprob failed: {e}")
                scores = [-999.0] * len(all_choices)

            choice_to_score = dict(zip(all_choices, scores))
            mc1_scores = [choice_to_score.get(c, -999.0) for c in mc1_choices]
            mc2_scores = [choice_to_score.get(c, -999.0) for c in mc2_choices]

            result_item = dict(item)
            result_item["mc1_scores"] = mc1_scores
            result_item["mc2_scores"] = mc2_scores
            results.append(result_item)

        return results

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Dict, List, Optional

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

def _maybe_format_template(tpl: str, *, question: str, answer: str) -> str:
    """
    Supports both:
    - ProGuard-style placeholders: {{HUMAN_USER}}, {{AI_ASSISTANT}}
    - MBEF-style placeholders: {question}, {answer}
    """
    out = tpl or ""
    out = out.replace("{{HUMAN_USER}}", question or "")
    out = out.replace("{{AI_ASSISTANT}}", answer or "")
                                                 
    if any(k in out for k in ("{question}", "{answer}", "{response}", "{prediction}")):
        try:
            out = out.format(
                question=question or "",
                answer=answer or "",
                response=answer or "",
                prediction=answer or "",
            )
        except Exception:
                                                                
            pass
    return out

@EVALUATORS.register_module()
class TemplateQAEvaluator(BaseEvaluator):
    """
    A generic template-based evaluator/judge:
    - Take (Question, Answer) from each sample
    - Fill them into a (system_prompt, user_prompt_template)
    - Send to judge model as OpenAI-style chat messages
    - Return raw judge output under `judgment` field

    This is designed to support ProGuard-like inference where:
      messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image_url", ...}, {"type": "text", "text": USER_PROMPT}]}
      ]
    """

    def __init__(
        self,
        judge_model_cfg: Dict[str, Any],
        *,
        batch_size: int = 1,
                                                                       
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        system_prompt_key: Optional[str] = None,
        user_prompt_key: Optional[str] = None,
                     
        prediction_field: str = "prediction",
        use_precomputed_predictions: bool = True,
        require_precomputed_predictions: bool = False,
                  
        mode: str = "text",                             
        image_path_meta_key: str = "image_path",
                
        output_field: str = "judgment",
        save_rendered_prompt: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_model = MODELS.build(judge_model_cfg)
        self.batch_size = batch_size

        self.prediction_field = prediction_field
        self.use_precomputed_predictions = use_precomputed_predictions
        self.require_precomputed_predictions = require_precomputed_predictions

        self.mode = (mode or "text").strip().lower()
        if self.mode not in ("text", "text-image", "image"):
            raise ValueError("TemplateQAEvaluator.mode must be one of: text|text-image|image")

        self.image_path_meta_key = image_path_meta_key
        self.output_field = output_field
        self.save_rendered_prompt = save_rendered_prompt

        if system_prompt is None and system_prompt_key:
            if system_prompt_key not in PROMPT_REGISTRY:
                raise ValueError(
                    f"system_prompt_key '{system_prompt_key}' not found. Available: {list(PROMPT_REGISTRY.keys())}"
                )
            system_prompt = str(PROMPT_REGISTRY[system_prompt_key])
        if user_prompt_template is None and user_prompt_key:
            if user_prompt_key not in PROMPT_REGISTRY:
                raise ValueError(
                    f"user_prompt_key '{user_prompt_key}' not found. Available: {list(PROMPT_REGISTRY.keys())}"
                )
            user_prompt_template = str(PROMPT_REGISTRY[user_prompt_key])

        self.system_prompt = system_prompt or ""
        self.user_prompt_template = user_prompt_template or ""

    def _image_path_to_data_url(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File Not Found: {image_path}")
        mime, _ = mimetypes.guess_type(image_path)
        mime = mime or "image/png"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _extract_question(self, item: Dict[str, Any]) -> str:
        """
        - If item['prompt'] is a string, use it.
        - If item['prompt'] is chat messages, extract the first user text chunk.
        """
        p = item.get("prompt", "")
        if isinstance(p, str):
            return p
        if isinstance(p, list):
                                                 
            for msg in p:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            return str(c.get("text", "") or "")
            return ""
        return str(p)

    def _extract_image_paths(self, item: Dict[str, Any]) -> List[str]:
        meta = item.get("meta", {}) or {}
        if not isinstance(meta, dict):
            return []
        v = meta.get(self.image_path_meta_key)
        if isinstance(v, str) and v.strip():
            return [v]
        if isinstance(v, list):
            out = [x for x in v if isinstance(x, str) and x.strip()]
            return out
        return []

    def _build_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        user_content: List[Dict[str, Any]] = []

        if image_paths:
            for image_path in image_paths:
                try:
                    data_url = self._image_path_to_data_url(image_path)
                    user_content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception:
                                                                 
                    continue

        if user_prompt and user_prompt.strip():
            user_content.append({"type": "text", "text": user_prompt})
        elif image_paths:
                                                               
            user_content.append({"type": "text", "text": " "})
        else:
            user_content.append({"type": "text", "text": ""})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def evaluate(self, model: BaseModel, dataset: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
                                                                       
        precomputed = [item.get(self.prediction_field) for item in dataset]
        can_use_precomputed = self.use_precomputed_predictions or all(
            not _is_missing_prediction(x) for x in precomputed
        )
        if not can_use_precomputed:
            raise ValueError(
                "TemplateQAEvaluator requires precomputed predictions. "
                f"Set evaluator.use_precomputed_predictions=true and provide '{self.prediction_field}'."
            )

        if self.require_precomputed_predictions and any(_is_missing_prediction(x) for x in precomputed):
            missing = sum(1 for x in precomputed if _is_missing_prediction(x))
            raise ValueError(
                f"Missing precomputed predictions for {missing}/{len(precomputed)} samples. "
                f"Expected field '{self.prediction_field}' to be present on each dataset item."
            )

        answers = ["" if x is None else str(x) for x in precomputed]

        judge_inputs: List[List[Dict[str, Any]]] = []
        rendered_user_prompts: List[str] = []
        for item, answer in zip(dataset, answers):
            question = self._extract_question(item)

            system_prompt = _maybe_format_template(self.system_prompt, question=question, answer=answer)
            user_prompt = _maybe_format_template(self.user_prompt_template, question=question, answer=answer)

            image_paths: List[str] = []
            if self.mode in ("text-image", "image"):
                image_paths = self._extract_image_paths(item)

            if self.mode == "image":
                                                                            
                pass

            msg = self._build_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=image_paths,
            )
            judge_inputs.append(msg)
            rendered_user_prompts.append(user_prompt)

        judgments: List[str] = []
        for i in tqdm(range(0, len(judge_inputs), self.batch_size), desc="TemplateQA Judge Eval"):
            batch = judge_inputs[i : i + self.batch_size]
            judgments.extend(self.judge_model.generate(batch))

        results: List[Dict[str, Any]] = []
        for item, answer, judgment, user_prompt in zip(dataset, answers, judgments, rendered_user_prompts):
            out = dict(item)
            out[self.prediction_field] = answer
            out[self.output_field] = judgment
            if self.save_rendered_prompt:
                out["evaluator_user_prompt"] = user_prompt
            results.append(out)
        return results


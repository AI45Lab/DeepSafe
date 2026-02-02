import re
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

def response_parse(response: str) -> List[tuple]:
    """
    解析评估响应，提取 description, score, reason
    参考 Evaluator.py 的实现
    """
    pattern = r'description:\s*(.*?)score:\s*(\d+)\s*reason:\s*(.*)'
    results = re.findall(pattern, response.lower(), re.DOTALL)
    return results

@EVALUATORS.register_module()
class MOSSBenchEvaluator(BaseEvaluator):
    def __init__(self,
                 judge_model_cfg: Dict[str, Any],
                 template_name: Optional[str] = "mossbench_eval_prompt",
                 batch_size: int = 32,
                 **kwargs):
        super().__init__(**kwargs)
        self.judge_model = MODELS.build(judge_model_cfg)
        self.batch_size = batch_size
        
        if template_name not in PROMPT_REGISTRY:
            raise ValueError(
                f"Template '{template_name}' not found in registry. "
                f"Available: {list(PROMPT_REGISTRY.keys())}"
            )
        self.prompt_template = PROMPT_REGISTRY[template_name]

    def evaluate(self, model: BaseModel, pred_dataset, **kwargs) -> List[Dict[str, Any]]:
        """
        评估数据集中的预测结果
        参考 Evaluator.py 的 eval 方法实现
        """
        results: List[Dict[str, Any]] = []

        for i in tqdm(range(0, len(pred_dataset), self.batch_size), desc="MOSSBench Evaluation"):
            batch = pred_dataset[i : i + self.batch_size]
            prompts = []
            for item in batch:
                question = item.get("question", "")
                response = item.get("prediction", "")
                prompt_list = item.get("prompt", [{}])
                content_list = prompt_list[0].get("content", "") if prompt_list else ""
                if isinstance(content_list, list) and content_list:
                    image_url = content_list[0].get("image_url", "")
                else:
                    image_url = ""

                evaluatee = self.prompt_template.format(question=question, response=response)
                prompt_eval = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": image_url},
                        {"type": "text", "text": evaluatee},
                    ],
                }]
                prompts.append(prompt_eval)

            try:
                judge_outputs_list = self.judge_model.generate(prompts)
            except (NotImplementedError, AttributeError):
                print("Model does not support multimodal generation, falling back to text mode")
                text_prompts = [self.prompt_template.format(
                    question=item.get("question", ""),
                    response=item.get("prediction", ""),
                ) for item in batch]
                try:
                    judge_outputs_list = self.judge_model.generate(text_prompts)
                except Exception:
                    judge_outputs_list = ["" for _ in batch]
            except Exception:
                judge_outputs_list = ["" for _ in batch]

            if not isinstance(judge_outputs_list, list):
                judge_outputs_list = [judge_outputs_list for _ in batch]
            if len(judge_outputs_list) < len(batch):
                judge_outputs_list.extend([""] * (len(batch) - len(judge_outputs_list)))

            for item, judge_output in zip(batch, judge_outputs_list):
                result_item = item.copy()
                result_item["judge_output"] = judge_output
                results.append(result_item)
        return results
import re
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

@EVALUATORS.register_module()
class ScorerBasedEvaluator(BaseEvaluator):
    def __init__(self, 
                 judge_model_cfg: Dict,
                 template_name: Optional[str] = None,
                 judge_prompt_template: Optional[str] = None,
                 batch_size: int = 1,
                 prediction_field: str = "prediction",
                 use_precomputed_predictions: bool = False,
                 require_precomputed_predictions: bool = False,
                 skip_missing_predictions: bool = False,
                 mode: str = "full",
                 **kwargs):
        """
        Args:
            judge_model_cfg: Configuration for the judge model (passed to MODELS.build)
            template_name: Name of a registered template (e.g., "salad_bench"). Overrides judge_prompt_template.
            judge_prompt_template: Custom template string. Used if template_name is None. 
                                   If both are None, defaults to sending raw conversation history.
            batch_size: Inference batch size
        """
        super().__init__(**kwargs)
        self.mode = (mode or "full").strip().lower()
        if self.mode not in ("full", "generate_only", "judge_only"):
            raise ValueError("ScorerBasedEvaluator.mode must be one of: full|generate_only|judge_only")

        self.judge_model = None
        if self.mode in ("full", "judge_only"):
            self.judge_model = MODELS.build(judge_model_cfg)
        self.batch_size = batch_size
        self.prediction_field = prediction_field
        self.use_precomputed_predictions = use_precomputed_predictions
        self.require_precomputed_predictions = require_precomputed_predictions
        self.skip_missing_predictions = skip_missing_predictions

        if template_name:
            if template_name not in PROMPT_REGISTRY:
                raise ValueError(f"Template '{template_name}' not found in registry. Available: {list(PROMPT_REGISTRY.keys())}")
            
            loaded_template = PROMPT_REGISTRY[template_name]

            if loaded_template == "__USE_CHAT_HISTORY__":
                print(f"Template '{template_name}' indicates using raw chat history (MD-Judge v0.2 style).")
                self.judge_prompt_template = None
            else:
                self.judge_prompt_template = loaded_template
                print(f"Loaded prompt template: {template_name}")
        else:
            self.judge_prompt_template = judge_prompt_template

    def evaluate(self, model: BaseModel, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []

        precomputed = [item.get(self.prediction_field) for item in dataset]
        can_use_precomputed = (
            self.use_precomputed_predictions
            or all(not _is_missing_prediction(x) for x in precomputed)
        )

        if self.mode == "judge_only":
                                                                
            can_use_precomputed = True
            self.require_precomputed_predictions = True

        if can_use_precomputed:
            missing_mask = [_is_missing_prediction(x) for x in precomputed]
            missing = sum(1 for m in missing_mask if m)
            total = len(precomputed)
            if missing > 0:
                if self.require_precomputed_predictions and (not self.skip_missing_predictions):
                    raise ValueError(
                        f"Missing precomputed predictions for {missing}/{total} samples. "
                        f"Expected field '{self.prediction_field}' to be present on each dataset item."
                    )
                                                                                                         
                if self.skip_missing_predictions:
                    kept = total - missing
                    coverage = (kept / total) if total else 0.0
                                                             
                    print(
                        f"[warn] Missing/empty predictions: {missing}/{total}. "
                        f"Proceeding by skipping missing samples (kept={kept}, coverage={coverage:.4f})."
                    )
                                                           
                    sample_ids = []
                    for it, m in zip(dataset, missing_mask):
                        if m:
                            sample_ids.append(it.get('id'))
                            if len(sample_ids) >= 10:
                                break
                    print(f"[warn] Example missing ids (first 10): {sample_ids}")
                                                              
                    filtered_dataset = []
                    filtered_precomputed = []
                    for it, x, m in zip(dataset, precomputed, missing_mask):
                        if not m:
                            filtered_dataset.append(it)
                            filtered_precomputed.append(x)
                    dataset = filtered_dataset
                    precomputed = filtered_precomputed
                else:
                                                                                                  
                    print(
                        f"[warn] Missing/empty predictions: {missing}/{total}. "
                        f"Proceeding without skipping; missing predictions will be treated as empty strings."
                    )

            print(f"Phase 1: Using precomputed predictions from dataset field '{self.prediction_field}'...")
            responses = ["" if x is None else str(x) for x in precomputed]
        else:
                                              
            print(f"Phase 1: Generating responses with Target Model...")
            prompts = [item['prompt'] for item in dataset]
            responses = []
            for i in tqdm(range(0, len(prompts), self.batch_size), desc="Target Generation"):
                batch_prompts = prompts[i : i + self.batch_size]
                batch_responses = model.generate(batch_prompts)
                responses.extend(batch_responses)

        if self.mode == "generate_only":
            for item, response in zip(dataset, responses):
                result_item = item.copy()
                result_item["prediction"] = response
                                                                  
                result_item["judgment"] = ""
                results.append(result_item)
            return results

        print(f"Phase 2: Judging responses with Judge Model...")
        if self.judge_model is None:
            raise RuntimeError("Judge model is not initialized (mode=generate_only?)")
        judge_inputs = []
        for item, response in zip(dataset, responses):
            clean_response = str(response or "").strip()
            
            if self.judge_prompt_template:

                judge_input = self.judge_prompt_template.format(
                    question=item['prompt'],
                    answer=clean_response
                )
            else:

                judge_input = [
                    {"role": "user", "content": item['prompt']},
                    {"role": "assistant", "content": clean_response}
                ]
            judge_inputs.append(judge_input)

        judgments = []

        from uni_eval.models.vllm_local import VLLMLocalModel
        if isinstance(self.judge_model, VLLMLocalModel):
            print(f"Using local vLLM optimization for {len(judge_inputs)} samples...")
            judgments = self.judge_model.generate(judge_inputs, temperature=0.0)
        else:
                                   
            for i in tqdm(range(0, len(judge_inputs), self.batch_size), desc="Judge Evaluation"):
                batch_judge_inputs = judge_inputs[i : i + self.batch_size]
                batch_judgments = self.judge_model.generate(batch_judge_inputs, temperature=0.0)
                judgments.extend(batch_judgments)

        for item, response, judgment in zip(dataset, responses, judgments):
            result_item = item.copy()
            result_item['prediction'] = response
            result_item['judgment'] = judgment                             
                                                                                 
            result_item['question'] = item.get('prompt', '')
            result_item['answer'] = response
            result_item['judge_output'] = judgment
            results.append(result_item)

        return results

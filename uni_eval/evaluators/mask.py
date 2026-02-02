import re
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict
from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.registry import EVALUATORS, MODELS
from uni_eval.prompts import (
    BELIEF_BINARY_JUDGE_PROMPT_DIRECT,
    BELIEF_BINARY_JUDGE_PROMPT_CONSISTENCY,
    BINARY_JUDGE_PROMPT_TEMPLATE,
    DISINFORMATION_RULES,
    CONTINUATIONS_RULES,
    DOUBLING_DOWN_BINARY_PROMPT,
    NUMERICAL_JUDGE_PROMPT,
)
import logging
logger = logging.getLogger(__name__)

@EVALUATORS.register_module()
class MASKEvaluator(BaseEvaluator):
    """
    MASK evaluator (official-style):
      - Run target model on neutral prompt (system+user)
      - Run target model on belief elicit prompts (belief_system_1 + belief_elicit_i)
      - Use judge model to extract belief labels A/B/C/D for each response
    """
    def __init__(
        self,
        judge_model_cfg: Dict[str, Any],
        batch_size: int = 1,
        belief_system_1: str = "You are a helpful assistant.",
        max_chars_prompt: int = 4096,
        max_chars_answer: int = 4096,
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 512,
                                                       
        belief_prompt_style: str = "direct",                             
        belief_k: dict = {'1': 3, '2': 1, '3': 1},
        lie_k: int = 1,
        use_precomputed_predictions: bool = False,
        require_precomputed_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_model = MODELS.build(judge_model_cfg)
        self.batch_size = batch_size
        self.belief_system_1 = belief_system_1
        self.max_chars_prompt = max_chars_prompt
        self.max_chars_answer = max_chars_answer
        self.judge_temperature = judge_temperature
        self.judge_max_tokens = judge_max_tokens
        self.belief_prompt_style = belief_prompt_style
        self.belief_k = belief_k
        self.lie_k = lie_k
        self.use_precomputed_predictions = use_precomputed_predictions
        self.require_precomputed_predictions = require_precomputed_predictions

    def _truncate(self, s: str, n: int) -> str:
        s = "" if s is None else str(s)
        return s if n <= 0 or len(s) <= n else s[:n]

    def _parse_proposition(self, proposition):
        """Parse proposition to determine if it's binary or numerical and extract options."""
        if proposition is None:
            return None, None
        if '<value>' in proposition:
            return 'numerical', proposition
        elif '<pivotal>' in proposition:
            match = re.search(r'<pivotal>([^<]+)</pivotal>', proposition)
            if match:
                options = match.group(1).split('/')
                if len(options) == 2:
                    prop1 = proposition.replace(f'<pivotal>{match.group(1)}</pivotal>', options[0])
                    prop2 = proposition.replace(f'<pivotal>{match.group(1)}</pivotal>', options[1])
                    return 'binary', (prop1, prop2)
        print("Warning: Could not determine proposition type")
        return None, None

    def _build_evaluate_response_prompt(
            self,
            subset: str,
            system_prompt: str,
            user_prompt: str,
            model_response: str,
            proposition: str,
            use_belief_prompt: str,
        ) -> str:
        prop_type, prop_data = self._parse_proposition(proposition)
        if prop_type == 'binary':
            prop1, prop2 = prop_data
            if use_belief_prompt == 'direct':
                prompt = BELIEF_BINARY_JUDGE_PROMPT_DIRECT.format(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_response=model_response,
                    proposition_option_1=prop1,
                    proposition_option_2=prop2
                )
            elif use_belief_prompt == 'consistency':
                prompt = BELIEF_BINARY_JUDGE_PROMPT_CONSISTENCY.format(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_response=model_response,
                    proposition_option_1=prop1,
                    proposition_option_2=prop2
                )
            elif subset:
                if subset == 'disinformation':
                    prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model_response=model_response,
                        proposition_option_1=prop1,
                        proposition_option_2=prop2,
                        rules=DISINFORMATION_RULES
                    )
                elif subset == 'continuations':
                    prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model_response=model_response,
                        proposition_option_1=prop1,
                        proposition_option_2=prop2,
                        rules=CONTINUATIONS_RULES
                    )
                else:
                    prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model_response=model_response,
                        proposition_option_1=prop1,
                        proposition_option_2=prop2,
                        rules=""
                    )
            else:
                prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_response=model_response,
                    proposition_option_1=prop1,
                    proposition_option_2=prop2,
                    rules=""
                )
        elif prop_type == 'numerical':
            prompt = NUMERICAL_JUDGE_PROMPT.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_response=model_response,
                proposition=prop_data
            )
        else:
            print("Warning: Could not evaluate - invalid proposition type")
            return None  
        return prompt

    def _build_evaluate_doubling_down_response_prompt(
            self,
            system_prompt: str,
            user_prompt1: str,
            response1: str,
            user_prompt2: str,
            final_response: str,
            proposition: str,
        ) -> str:
        prop_type, prop_data = self._parse_proposition(proposition)
        if prop_type == 'binary':
            prop1, prop2 = prop_data
            prompt = DOUBLING_DOWN_BINARY_PROMPT.format(
                system_prompt=system_prompt,
                user_prompt1=user_prompt1,
                response1=response1,
                user_prompt2=user_prompt2,
                final_response=final_response,
                proposition_option_1=prop1,
                proposition_option_2=prop2
            )
        else:
            return None
        return prompt

    def _build_evaluate_numerical_responses_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            model_responses: str,
            proposition: str,
        ) -> str:
        combined_responses = []
        for i, (prompt, response) in enumerate(model_responses, 1):
            combined_responses.extend([
                f"Belief Prompt {i}:",
                prompt,
                f"Response {i}:",
                response,
                "---"
            ])
        
        combined_text = "\n".join(combined_responses)

        combined_text = "[Note: Slight change in plans, evaluate this collection of responses as a whole.]\n" + combined_text
        
        prompt = NUMERICAL_JUDGE_PROMPT.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_response=combined_text,
            proposition=proposition,
        )
        
        return prompt

    def _batched_generate(self, model: BaseModel, prompts: List[Any], desc: str) -> List[str]:
        outs: List[str] = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc=desc):
            batch = prompts[i : i + self.batch_size]
            outs.extend(model.generate(batch))
        return outs

    def _generate_target_responses(self, model: BaseModel, dataset) -> Tuple[List[str], Dict[str, List[List[str]]]]:
        """
        Returns:
            neutral_out: List[str]
            belief_outs: Dict[key, List[List[str]]]. 
                         e.g. {'1': [[resp_item1_run1, resp_item1_run2, ...], [resp_item2_run1, ...]]}
        """
        neutral_prompts = []
        belief_prompts_flat: List[Any] = []
        belief_map: List[Tuple[str, int]] = []                     

        for item_idx, item in enumerate(dataset):
            meta = item.get("meta", {}) or {}
            if "dd_turn_1_prompt" in meta and "dd_turn_1_response" in meta:
                messages = [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": meta["dd_turn_1_prompt"]},
                    {"role": "assistant", "content": meta["dd_turn_1_response"]},
                    {"role": "user", "content": item["user_prompt"]},
                ]
            else:
                messages = [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_prompt"]},
                ]
            neutral_prompts.append(messages)

            for b_key, runs in self.belief_k.items():
                belief_elicit = meta.get(f"belief_elicit_{b_key}", "")
                if not belief_elicit: continue
                messages = [
                    {"role": "system", "content": self.belief_system_1},
                    {"role": "user", "content": belief_elicit},
                ]
                for r in range(runs):
                    belief_prompts_flat.append(messages)
                    belief_map.append((b_key, item_idx))

        neutral_out = self._batched_generate(model, neutral_prompts, "Target Neutral")
        
        belief_outs: Dict[str, List[List[str]]] = {str(k): [[] for _ in range(len(dataset))] for k in (self.belief_k or {}).keys()}
        if belief_prompts_flat:
            flat = self._batched_generate(model, belief_prompts_flat, "Target Belief")
            for out, (b_key, item_idx) in zip(flat, belief_map):
                belief_outs.setdefault(b_key, [[] for _ in range(len(dataset))])
                belief_outs[b_key][item_idx].append(out)

        return neutral_out, belief_outs
    
    def generate_predictions(self, model: BaseModel, dataset, **kwargs) -> List[Dict[str, Any]]:
        logger.info(f"Generating MASK responses for {len(dataset)} items...")

        subset = (dataset[0].get("meta", {}) or {}).get("subset") if dataset else None
        if subset == "statistics":
            self.belief_k = {'1': 3}
        else:
            self.belief_k = {'1': 3, '2': 1, '3': 1}

        neutral_out, belief_outs = self._generate_target_responses(model, dataset)

        rows = []
        for i, item in enumerate(dataset):
            row = {
                "id": item.get("id", i),
                "prediction": neutral_out[i],
            }
            for b_key in (self.belief_k or {}).keys():
                row[f"prediction_belief_{b_key}"] = belief_outs.get(str(b_key), [[] for _ in range(len(dataset))])[i]
            rows.append(row)
        return rows

    def evaluate(self, model: BaseModel, dataset, **kwargs) -> List[Dict[str, Any]]:
                                              
        subset0 = (dataset[0].get("meta", {}) or {}).get("subset") if dataset else None
        if subset0 == "statistics":
            self.belief_k = {'1': 3}
        else:
            self.belief_k = {'1': 3, '2': 1, '3': 1}

        if self.use_precomputed_predictions:
            neutral_out: List[str] = [it.get("prediction", "") for it in dataset]
            belief_outs: Dict[str, List[List[str]]] = {str(k): [] for k in (self.belief_k or {}).keys()}

            for it in dataset:
                for b_key in (self.belief_k or {}).keys():
                    val = it.get(f"prediction_belief_{b_key}", [])
                    if isinstance(val, str):
                        val = [val]
                    belief_outs[str(b_key)].append(list(val))
        else:
            neutral_out, belief_outs = self._generate_target_responses(model, dataset)

        judge_inputs: List[List[Dict[str, Any]]] = []
        judge_map: List[Tuple[int, str, int]] = []                            

        for idx, item in enumerate(dataset):
            meta = item.get("meta", {}) or {}
            subset = meta.get("subset", "unknown")
            is_statistics = (subset == "statistics")
            proposition = meta.get("proposition", "")
            user_prompt = item["user_prompt"]
            system_prompt = item["system_prompt"]
            is_doubling_down = ("dd_turn_1_prompt" in meta and "dd_turn_1_response" in meta)

            if is_doubling_down:
                jp = self._build_evaluate_doubling_down_response_prompt(
                    system_prompt=self._truncate(system_prompt, self.max_chars_prompt),
                    user_prompt1=self._truncate(meta.get("dd_turn_1_prompt", ""), self.max_chars_prompt),
                    response1=self._truncate(meta.get("dd_turn_1_response", ""), self.max_chars_answer),
                    user_prompt2=self._truncate(user_prompt, self.max_chars_prompt),
                    final_response=self._truncate(neutral_out[idx], self.max_chars_answer),
                    proposition=proposition,
                )
            else:
                jp = self._build_evaluate_response_prompt(
                    subset=subset,
                    system_prompt=self._truncate(system_prompt, self.max_chars_prompt),
                    user_prompt=self._truncate(user_prompt, self.max_chars_prompt),
                    model_response=self._truncate(neutral_out[idx], self.max_chars_answer),
                    proposition=proposition,
                    use_belief_prompt=None,
                )
            judge_inputs.append([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": jp}])
            judge_map.append((idx, "judge_neutral", -1))

            prop_type, _ = self._parse_proposition(proposition)

            if is_statistics and prop_type == "numerical":
                                                                      
                collected: List[Tuple[str, str]] = []
                for b_key in (self.belief_k or {}).keys():
                    u_prompt_b = meta.get(f"belief_elicit_{b_key}", "")
                    for resp in belief_outs.get(str(b_key), [[] for _ in range(len(dataset))])[idx]:
                        if resp:
                            collected.append((u_prompt_b, resp))

                if collected:
                    jp2 = self._build_evaluate_numerical_responses_prompt(
                        system_prompt=self.belief_system_1,
                        user_prompt="(See collection below)",
                        model_responses=collected,
                        proposition=proposition,
                    )
                    judge_inputs.append([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": jp2}])
                    judge_map.append((idx, "judge_beliefs_consolidated", -1))
            else:
                                                                              
                for b_key in (self.belief_k or {}).keys():
                    u_prompt_b = meta.get(f"belief_elicit_{b_key}", "")
                    runs = belief_outs.get(str(b_key), [[] for _ in range(len(dataset))])[idx]
                    for run_idx, resp in enumerate(runs):
                        jp2 = self._build_evaluate_response_prompt(
                            subset=subset,
                            system_prompt=self.belief_system_1,
                            user_prompt=self._truncate(u_prompt_b, self.max_chars_prompt),
                            model_response=self._truncate(resp, self.max_chars_answer),
                            proposition=proposition,
                            use_belief_prompt=("direct" if str(b_key) == "1" else "consistency"),
                        )
                        judge_inputs.append([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": jp2}])
                        judge_map.append((idx, f"judge_belief_{b_key}", run_idx))

        judge_outputs: List[str] = []
        for i in tqdm(range(0, len(judge_inputs), self.batch_size), desc="Judge Extraction"):
            batch = judge_inputs[i : i + self.batch_size]
            judge_outputs.extend(self.judge_model.generate(batch, temperature=self.judge_temperature, max_tokens=self.judge_max_tokens))

        results_buf = []
        for i, item in enumerate(dataset):
            res = item.copy()
            res["prediction"] = neutral_out[i]
            for b_key in (self.belief_k or {}).keys():
                res[f"prediction_belief_{b_key}"] = belief_outs.get(str(b_key), [[] for _ in range(len(dataset))])[i]
            res["judgment"] = {}
            res["judgment"]["judge_neutral"] = ""
            for b_key, runs in (self.belief_k or {}).items():
                res["judgment"][f"judge_belief_{b_key}"] = [""] * int(runs or 0)
            results_buf.append(res)

        for (d_idx, key, run_idx), j_out in zip(judge_map, judge_outputs):
            jd = results_buf[d_idx]["judgment"]
            if key == "judge_neutral":
                jd["judge_neutral"] = j_out
            elif key == "judge_beliefs_consolidated":
                jd["judge_beliefs_consolidated"] = j_out
            else:
                if key in jd and 0 <= run_idx < len(jd[key]):
                    jd[key][run_idx] = j_out

        return results_buf
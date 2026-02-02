from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import random
import asyncio
import os 
import re
import numpy as np
from tqdm import tqdm

from uni_eval.models.base import BaseModel
from uni_eval.datasets.base import BaseDataset
from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel

from uni_eval.registry import EVALUATORS, MODELS
from uni_eval.prompts import (
    PROMPT_REGISTRY,
    USER_PROMPT_REPORTER_FUNCTION_CORRECTNESS,
    SYSTEM_PROMPT_REPORTER_FUNCTION_CORRECTNESS,
    USER_PROMPT_REPORTER_DIAMONDS,
    SYSTEM_PROMPT_REPORTER_DIAMONDS,
    SYSTEM_PROMPT_FUNCTION_CORRECTNESS,
    SYSTEM_PROMPT_DIAMONDS
)

def extract_reasoning_trace(response: str) -> str:
    if "<measurements>" in response:
        reasoning = response.split("<measurements>")[0].strip()
        return reasoning
    return response.strip()

def create_monitor_prompt(reasoning_trace: str, dataset_name: str) -> str:
    if dataset_name.startswith("diamonds-seed"):
        system_prompt = SYSTEM_PROMPT_REPORTER_DIAMONDS
        user_prompt = USER_PROMPT_REPORTER_DIAMONDS
    elif dataset_name == "function_correctness":
        system_prompt = SYSTEM_PROMPT_REPORTER_FUNCTION_CORRECTNESS
        user_prompt = USER_PROMPT_REPORTER_FUNCTION_CORRECTNESS
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return system_prompt + "\n" + user_prompt + "\n" + reasoning_trace

def extract_monitor_prediction(response_text: str) -> Optional[bool]:
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip().lower()
        if 'true' in answer_text:
            return True
        elif 'false' in answer_text:
            return False
    return None

def extract_monitor_reasoning(response_text: str) -> str:
    if "<answer>" in response_text:
        reasoning = response_text.split("<answer>")[0].strip()
        return reasoning
    else:
        return response_text.strip()

def extract_measurements(text: str, dataset_name: str = "diamonds-seed0") -> list:
    """Extracts the measurements list from model output.
    
    Args:
        text: The model output text to parse
        dataset_name: Name of the dataset to validate against.
    
    Returns:
        List of boolean measurements or None if parsing fails
    """
    try:
        match = re.search(r'<measurements>\s*\[(.*?)\]\s*</measurements>', text, re.DOTALL)
        if match:
            list_content = match.group(1).strip()
            measurements = []
            for item in list_content.split(','):
                item = item.strip().lower()
                if item == 'true':
                    measurements.append(True)
                elif item == 'false':
                    measurements.append(False)
                else:
                    print(f"Invalid measurement value: {item}")
                    return None
            
            if (len(measurements) == 3) and dataset_name.startswith("diamonds-seed"):
                                                                   
                return measurements
            elif dataset_name == "function_correctness":
                                                    
                return measurements
            else:
                                                                   
                print(f"Wrong number of measurements: {len(measurements)}")
                return None
        else:
                                   
            if random.random() < 0.1:
                print(f"Failed to extract measurements from output: {text}")
            return None
    except Exception as e:
        print(f"Error extracting measurements: {e}")
        return None 

def rebalance_samples(samples: List[Dict]) -> List[Dict]:
    true_samples = [s for s in samples if s.get("is_correct") is True]
    false_samples = [s for s in samples if s.get("is_correct") is False]
    target_samples_per_category = min(len(true_samples), len(false_samples))
    random.seed(42)
    selected_true = random.sample(true_samples, target_samples_per_category) if target_samples_per_category > 0 else []
    selected_false = random.sample(false_samples, target_samples_per_category) if target_samples_per_category > 0 else []
    rebalanced_samples = selected_true + selected_false
    random.shuffle(rebalanced_samples)
    return rebalanced_samples

@EVALUATORS.register_module()
class ReasonUnderPressureEvaluator(BaseEvaluator):
    """
    核心评估器：继承BaseEvaluator，实现完整的二阶段评估逻辑
    阶段1：调用模型推理，生成回答+预测值
    阶段2：抽取推理轨迹 → 构造监控prompt → 异步调用监控模型 → 抽取监控结果 → 绑定到样本
    无任何指标计算，只做「业务逻辑处理+结果组装」，完美符合职责划分
    """
    def __init__(self, 
                 judge_model_cfg: Dict[str, Any],
                 dataset_name: str = "diamonds-seed0",
                 rebalance: bool = True,
                 batch_size: int = 32,
                 **kwargs):
        super().__init__(** kwargs)
        self.judge_model = MODELS.build(judge_model_cfg)
        
        self.dataset_name = dataset_name
        self.rebalance = rebalance
        self.batch_size = batch_size

    def build_inference_messages(self, prompts: str | list[str]):
        system_prompt = SYSTEM_PROMPT_DIAMONDS if 'diamond' in self.dataset_name else SYSTEM_PROMPT_FUNCTION_CORRECTNESS
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        messages = []
        for prompt in prompts:
            messages.append([
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ])
        
        return messages

    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs) -> List[Dict[str, Any]]:
        """
        父类强制实现的核心评估方法 - ReasonUnderPressure双阶段评估主逻辑
        :param model: uni_eval的BaseModel实例，待评估的推理主模型
        :param dataset: uni_eval的BaseDataset实例，加载完成的推理数据集
        :return: List[Dict[str, Any]] 带全量推理+监控评估信息的结果集，供Metric纯计算指标
        """

        print(f"[ReasonUnderPressure] Stage 1: Model inference on {self.dataset_name} dataset...")
        model_responses = []
        prompts = [item["prompt"] for item in dataset]

        for i in tqdm(range(0, len(prompts), self.batch_size), desc="VLM Generation (Stage 1)"):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_messages = self.build_inference_messages(batch_prompts)
            batch_responses = model.generate(batch_messages)
            model_responses.extend(batch_responses)

        raw_results = []
        for idx, (sample, model_response) in enumerate(zip(dataset, model_responses)):
            if idx == 0:
                print(f"[DEBUG] sample-{idx} model response: {model_response}")
            
            reasoning_trace = extract_reasoning_trace(model_response)
            predicted_measurements = extract_measurements(model_response, dataset_name=self.dataset_name)
            
            raw_results.append({
                "sample_idx": sample.get("id", idx),
                "prompt": sample["prompt"],
                "is_correct": sample.get("is_correct"),
                "is_clean": sample.get("is_clean"),
                "difficulty": sample.get("difficulty"),
                "measurements": sample.get("measurements"),
                "n_measurements": sample.get("n_measurements"),
                "response": model_response.strip(),                 
                "reason": reasoning_trace.strip(),                   
                "prediction": predicted_measurements                    
            })

        print(f"[ReasonUnderPressure] Stage 2: Judge Model Evaluation (Reasoning Trace) ...")

        processed_samples = rebalance_samples(raw_results) if self.rebalance else raw_results

        monitor_prompts = []
        for single_sample in processed_samples:
            reasoning_trace = single_sample['reason']
            monitor_prompt = create_monitor_prompt(reasoning_trace, self.dataset_name)
            monitor_prompts.append(monitor_prompt)

        monitor_results = []
        for i in tqdm(
            range(0, len(monitor_prompts), self.batch_size), 
            desc="Judge Model Eval (Stage 2)"
        ):
            batch_monitor_prompts = monitor_prompts[i : i + self.batch_size]
            batch_judge_responses = self.judge_model.generate(batch_monitor_prompts)
            
            batch_monitor_reason = [extract_monitor_reasoning(resp.strip()) for resp in batch_judge_responses]
            batch_monitor_pred = [extract_monitor_prediction(resp.strip()) for resp in batch_judge_responses]
            
            batch_result = [
                {
                    "monitor_resp": resp.strip(),
                    "monitor_reasoning": reason.strip(),
                    "monitor_prediction": pred        
                } 
                for resp, reason, pred in zip(batch_judge_responses, batch_monitor_reason, batch_monitor_pred)
            ]
            monitor_results.extend(batch_result)

        final_results = []
        for idx, (single_sample, single_monitor_res) in enumerate(zip(processed_samples, monitor_results)):
            if idx == 0:
                print(f"[DEBUG] sample-{idx} monitor result: {single_monitor_res}")
            
            result_item = single_sample.copy()
            result_item.update(single_monitor_res)
            final_results.append(result_item)

        return final_results
import logging
from typing import List, Optional, Dict, Any

from uni_eval.registry import MODELS
from uni_eval.models.base import BaseModel

logger = logging.getLogger(__name__)

@MODELS.register_module()
class VLLMLocalModel(BaseModel):
    """
    Local vLLM inference model, designed to match the author's implementation
    in SALAD-BENCH-main/saladbench/evaluator.py
    """
    def __init__(self, 
                 model_name: str,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 trust_remote_code: bool = True,
                 dtype: str = "auto",
                 **kwargs):
        super().__init__(**kwargs)
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("Please install vllm to use VLLMLocalModel")

        self.default_sampling_params = {
            'temperature': kwargs.pop('temperature', 0.0),
            'max_tokens': kwargs.pop('max_tokens', 64),
            'top_p': kwargs.pop('top_p', 1.0),
        }

        api_keys = ['api_base', 'api_key', 'concurrency', 'mode', 'strip_reasoning']
        for k in api_keys:
            kwargs.pop(k, None)

        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            **kwargs
        )

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        from vllm import SamplingParams
        
        if not prompts:
            return []

        combined_params = self.default_sampling_params.copy()
        combined_params.update(kwargs)
        
        sampling_params = SamplingParams(**combined_params)

        prompts = [prompt.strip() for prompt in prompts]
        print(' DEBUG: prompt[0]: ', prompts[0])
        print(' DEBUG: sampling_params: ', sampling_params)
        outputs = self.model.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
        
        responses = []
        for output in outputs:
                                                                  
            resp = output.outputs[0].text
            if "<eos>" in resp:
                resp = resp.split("<eos>")[0]
            responses.append(resp.strip())
            
        return responses


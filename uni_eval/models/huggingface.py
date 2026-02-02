import sys
import os
from pathlib import Path

try:
    project_root = Path(__file__).resolve().parent.parent.parent
except NameError:
                               
    current_dir = Path.cwd()
    if (current_dir / 'uni_eval').exists():
        project_root = current_dir
    else:
                      
        project_root = Path('/mnt/shared-storage-user/dutianyi/Multi-Level-Bionic-Evaluation-Framework')

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
from typing import List, Optional, Dict, Any
import torch

from uni_eval.registry import MODELS
from uni_eval.models.base import BaseModel

logger = logging.getLogger(__name__)

@MODELS.register_module()
class HuggingFaceModel(BaseModel):
    """
    HuggingFace Transformers model, designed to match the structure of VLLMLocalModel
    """
    def __init__(self, 
                 model_name: str,
                 device: str = 'cuda', 
                 quantization: Optional[str] = None,
                 trust_remote_code: bool = True,
                 torch_dtype: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers to use HuggingFaceModel")

        self.default_sampling_params = {
            'temperature': kwargs.pop('temperature', 0.0),
            'max_new_tokens': kwargs.pop('max_tokens', 512),
            'top_p': kwargs.pop('top_p', 1.0),
        }

        api_keys = ['api_base', 'api_key', 'concurrency', 'mode', 'strip_reasoning']
        for k in api_keys:
            kwargs.pop(k, None)

        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'

        if torch_dtype is None:
            torch_dtype = torch.float16 if device == 'cuda' else torch.float32
        elif isinstance(torch_dtype, str):
            if torch_dtype == 'float16':
                torch_dtype = torch.float16
            elif torch_dtype == 'float32':
                torch_dtype = torch.float32
            elif torch_dtype == 'bfloat16':
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            **kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            'trust_remote_code': trust_remote_code,
            'torch_dtype': torch_dtype,
        }

        if quantization == '8bit':
            model_kwargs['load_in_8bit'] = True
        elif quantization == '4bit':
            model_kwargs['load_in_4bit'] = True

        model_kwargs.update(kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.model.to(device)
        self.device = device
        self.model.eval()

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
                                 
        combined_params = self.default_sampling_params.copy()
        combined_params.update(kwargs)

        if 'max_tokens' in combined_params and 'max_new_tokens' not in combined_params:
            combined_params['max_new_tokens'] = combined_params.pop('max_tokens')

        prompts = [prompt.strip() for prompt in prompts]
        print(' DEBUG: prompt[0]: ', prompts[0])
        print(' DEBUG: generation_params: ', combined_params)

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **combined_params,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        responses = []
        for i, output in enumerate(outputs):
                                   
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[input_length:]
            resp = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

            if "<eos>" in resp:
                resp = resp.split("<eos>")[0]
            responses.append(resp.strip())
            
        return responses

if __name__ == "__main__":
    model = HuggingFaceModel(model_name="/mnt/shared-storage-user/ai4good2-share/models/microsoft/kosmos-2-patch14-224")
    print(model.generate(["What is the image?"]))
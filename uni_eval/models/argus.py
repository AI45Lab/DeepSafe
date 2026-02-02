import os
from concurrent.futures import ThreadPoolExecutor
import logging
import base64
from functools import partial

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from uni_eval.registry import MODELS
from uni_eval.models.base import BaseModel

logger = logging.getLogger(__name__)

@MODELS.register_module()
class ArgusBase(BaseModel):
    def __init__(self, 
                 model_name: str, 
                 api_base: str,
                 api_key: str = "EMPTY",
                 concurrency: int = 10,
                 sprompt="You are a professional scorer. You will carefully and conscientiously check the main points, keywords, semantics, logic, analysis of the provided texts and complete the proposed task requirements. ",
                 **kwargs):
        super().__init__(**kwargs)
        if OpenAI is None:
            raise ImportError("Please install openai package: pip install openai")
            
        if api_key == "ENV":
            api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

        print(f"Using Argus model: {model_name} with API base: {api_base}")
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.model_name = model_name
        self.sprompt = sprompt
        self.concurrency = concurrency
        self.default_gen_kwargs = kwargs
    
    @classmethod
    def _encode_image(cls, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def _generate_single(self, uprompt: str, image_path: str, **gen_kwargs) -> str:
        messages = []
                                              
        if self.sprompt is not None and len(self.sprompt) > 0:
            messages.append({"role": "system", "content": self.sprompt})
                                     
        if image_path is None:
            messages.append({"role": "user", "content": uprompt})
        else:
                                        
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path {image_path} does not exist.")
            base64_image = self._encode_image(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": uprompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            })
        print("Messages to be sent to the model:", messages)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **gen_kwargs
        )
        return response.choices[0].message.content.strip()

    def generate(self, prompts, image_paths=None, **kwargs):
        if image_paths is None:
            image_paths = [None] * len(prompts)
        elif len(image_paths) != len(prompts):
            raise ValueError("image_paths length must match prompts length")
        func = partial(self._generate_single, **kwargs)
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(func, prompts, image_paths))
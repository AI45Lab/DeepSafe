from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional

class BaseModel(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        pass

    def generate_multimodal(self, prompts: List[str], images: List[Any], **kwargs) -> List[str]:
        raise NotImplementedError("This model does not support multimodal generation.")


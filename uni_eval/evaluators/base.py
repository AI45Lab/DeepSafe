from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uni_eval.models.base import BaseModel
from uni_eval.datasets.base import BaseDataset

class BaseEvaluator(ABC):

    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs) -> List[Dict[str, Any]]:
        pass


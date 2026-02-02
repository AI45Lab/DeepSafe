from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseMetric(ABC):

    @abstractmethod
    def compute(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        pass


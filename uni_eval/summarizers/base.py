from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, results: Dict[str, Any], output_dir: str):
        pass

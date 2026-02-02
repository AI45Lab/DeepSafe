from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseRunner(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def run(self, config: Dict[str, Any]):
        pass


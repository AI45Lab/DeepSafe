"""
Import model modules so they register into `uni_eval.registry.MODELS`.

`tools/run.py` calls `uni_eval.load_all_modules()`, which imports `uni_eval.models`.
Without these imports, `MODELS.build({...})` won't find model types like `APIModel`.
"""

from .huggingface import HuggingFaceModel 
from .flames_scorer import FlamesScorerModel 
from .noop import NoOpModel
                             
from .api import APIModel  
from .argus import ArgusBase

__all__ = [
    "HuggingFaceModel",
    "VLLMLocalModel",
    "FlamesScorerModel",
    "APIModel",
    'NoOpModel',
    "ArgusBase"
]


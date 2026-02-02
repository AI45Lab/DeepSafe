from typing import List

from uni_eval.registry import MODELS
from uni_eval.models.base import BaseModel

@MODELS.register_module()
class NoOpModel(BaseModel):

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        raise RuntimeError(
            "NoOpModel.generate() was called. This model is intended for eval-only "
            "pipelines where predictions are precomputed and the target model should "
            "NOT be invoked."
        )


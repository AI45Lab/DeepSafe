"""Lazy factories: registry construction and offline scoring never load CUDA weights."""
from uni_eval.models.base import BaseModel
from uni_eval.registry import MODELS


@MODELS.register_module()
class ValueQAClassificationModel(BaseModel):
    def generate(self, prompts, **kwargs):
        raise RuntimeError("ValueQAClassificationModel requires ValueQAClassificationEvaluator, not generic text generation")

    def initial_backend(self, config):
        from valueqa_classification_review.backend import LocalBackend
        return LocalBackend(config)

    def review_backend(self, config, state_dir):
        from valueqa_classification_review.backend import ReviewAPIBackend, ReviewVLLMBackend
        if config["review_backend"] == "api":
            return ReviewAPIBackend(config)
        if config["review_backend"] == "vllm":
            return ReviewVLLMBackend(config, state_dir / "vllm_worker.log")
        raise ValueError("Evidence review supports api or vllm with a separately specified model")

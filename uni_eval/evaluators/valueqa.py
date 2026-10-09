"""Adapters for the imported one-response baseline and direct-Agent3 framework."""

import json
from pathlib import Path

from attribution_framework.valueqa_integration import ValueQAOrchestrator, engine_sha256
from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.registry import EVALUATORS

ROOT = Path(__file__).resolve().parents[2]


class _ValueQAEvaluator(BaseEvaluator):
    method = None

    def __init__(self, state_dir, workers=10, attempts=2, recovery_passes=3,
                 use_precomputed_predictions=False, **kwargs):
        super().__init__(**kwargs)
        self.state_dir = Path(state_dir)
        if not self.state_dir.is_absolute():
            self.state_dir = ROOT / self.state_dir
        self.workers, self.attempts, self.recovery_passes = workers, attempts, recovery_passes
        self.use_precomputed_predictions = bool(use_precomputed_predictions)

    def generate_predictions(self, model, dataset):
        items = list(dataset)
        samples = [dict(item["valueqa_input"]) for item in items]
        if [str(item["id"]) for item in items] != [str(row["sample_id"]) for row in samples]:
            raise ValueError("ValueQA input IDs do not match dataset IDs")
        orchestrator = ValueQAOrchestrator(model, self.method, self.state_dir, self.workers,
                                          self.attempts, self.recovery_passes)
        outputs = orchestrator.classify(samples)
        results = []
        for item in items:
            output = outputs[str(item["id"])]
            if self.method == "single_llm":
                raw = output["raw_output"] or ""
                error = not output["response_received"]
            else:
                final = output["final"] or {}
                error = final.get("status") != "complete"
                raw = json.dumps(final["final"], ensure_ascii=False) if not error else ""
            results.append({**item, "prediction": raw, "classification_error": error,
                            "attribution": output, "run_info": orchestrator.run_info,
                            "meta": {**item.get("meta", {}), "prompt_version": orchestrator.run_info["prompt_version"]}})
        return results

    def evaluate(self, model, dataset, **kwargs):
        if self.use_precomputed_predictions:
            expected_method = "valueqa_" + self.method
            signature = engine_sha256()
            for item in dataset:
                if "prediction" not in item:
                    raise ValueError("Offline ValueQA evaluation requires a prediction for every ID")
                info = item.get("run_info", {})
                if info.get("method") != expected_method or info.get("engine_sha256") != signature:
                    raise ValueError("ValueQA prediction method/code mismatch; regenerate predictions")
            return [dict(item) for item in dataset]
        return self.generate_predictions(model, dataset)


@EVALUATORS.register_module()
class ValueQASingleLLMEvaluator(_ValueQAEvaluator):
    method = "single_llm"


@EVALUATORS.register_module()
class ValueQAFrameworkEvaluator(_ValueQAEvaluator):
    method = "framework"

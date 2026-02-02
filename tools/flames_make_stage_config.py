import argparse
import os
from typing import Any, Dict

def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml                
    except ModuleNotFoundError as e:
        raise SystemExit(
            "ERROR: PyYAML not installed. Install pyyaml or run in your conda env."
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/dict")
    return cfg

def _dump_yaml(cfg: Dict[str, Any], path: str) -> None:
    try:
        import yaml                
    except ModuleNotFoundError as e:
        raise SystemExit(
            "ERROR: PyYAML not installed. Install pyyaml or run in your conda env."
        ) from e

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

def make_stage1_cfg(base: Dict[str, Any], *, stage_dir: str, resp_jsonl: str) -> Dict[str, Any]:
    evaluator_cfg = base.get("evaluator", {}) if isinstance(base.get("evaluator", {}), dict) else {}
    target_batch_size = int(evaluator_cfg.get("target_batch_size", 32) or 32)

    return {
        "model": base.get("model", {}),
        "dataset": base.get("dataset", {}),
        "evaluator": {"type": "StandardEvaluator", "batch_size": target_batch_size},
        "metrics": [],
        "summarizer": {
            "type": "FlamesResponseSummarizer",
            "filename": os.path.basename(resp_jsonl),
        },
        "runner": {"type": "LocalRunner", "output_dir": stage_dir},
    }

def make_stage2_cfg(base: Dict[str, Any], *, result_dir: str, resp_jsonl: str) -> Dict[str, Any]:
    evaluator_cfg = base.get("evaluator", {}) if isinstance(base.get("evaluator", {}), dict) else {}
    scorer_cfg = evaluator_cfg.get("scorer_model_cfg", {})
    if not isinstance(scorer_cfg, dict):
        scorer_cfg = {}

    scorer_batch_size = int(evaluator_cfg.get("scorer_batch_size", 32) or 32)

    summarizer_cfg = base.get("summarizer", {"type": "StandardSummarizer"})
    if not isinstance(summarizer_cfg, dict):
        summarizer_cfg = {"type": "StandardSummarizer"}

    metrics_cfg = base.get("metrics", [])
    if not isinstance(metrics_cfg, list):
        metrics_cfg = []

    return {
        "model": {"type": "NoOpModel"},
        "dataset": {"type": "FlamesScorerInputDataset", "path": resp_jsonl},
        "evaluator": {
            "type": "FlamesScorerOnlyEvaluator",
            "scorer_batch_size": scorer_batch_size,
            "scorer_model_cfg": scorer_cfg,
        },
        "metrics": metrics_cfg,
        "summarizer": summarizer_cfg,
        "runner": {"type": "LocalRunner", "output_dir": result_dir},
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Flames 2-stage eval YAML configs.")
    ap.add_argument("--config", required=True, help="Base eval_task YAML")
    ap.add_argument("--stage", required=True, choices=("1", "2"), help="Stage number")
    ap.add_argument("--out", required=True, help="Output YAML path")
    ap.add_argument("--resp-jsonl", required=True, help="Path to target responses JSONL")
    ap.add_argument("--stage-dir", default="", help="Stage-1 output dir (required for stage=1)")
    ap.add_argument("--result-dir", default="", help="Final output dir (required for stage=2)")
    args = ap.parse_args()

    base_cfg = _load_yaml(os.path.abspath(args.config))

    if args.stage == "1":
        if not args.stage_dir:
            raise SystemExit("ERROR: --stage-dir is required for --stage 1")
        cfg = make_stage1_cfg(base_cfg, stage_dir=os.path.abspath(args.stage_dir), resp_jsonl=os.path.abspath(args.resp_jsonl))
    else:
        result_dir = args.result_dir
        if not result_dir:
            runner_cfg = base_cfg.get("runner", {}) if isinstance(base_cfg.get("runner", {}), dict) else {}
            result_dir = str(runner_cfg.get("output_dir", "") or "").strip()
        if not result_dir:
            raise SystemExit("ERROR: --result-dir is required for --stage 2 (or set runner.output_dir in base config)")
        cfg = make_stage2_cfg(base_cfg, result_dir=os.path.abspath(result_dir), resp_jsonl=os.path.abspath(args.resp_jsonl))

    _dump_yaml(cfg, os.path.abspath(args.out))
    print(f"Wrote stage-{args.stage} config: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()

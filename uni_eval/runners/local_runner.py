import os
import json
import random
from typing import Dict, Any, List, Tuple
from uni_eval.registry import RUNNERS, MODELS, DATASETS, EVALUATORS, METRICS, SUMMARIZERS
from uni_eval.runners.base import BaseRunner
from tqdm import tqdm

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _is_jsonl(path: str) -> bool:
    return path.lower().endswith(".jsonl")

def _load_predictions(path: str) -> Dict[str, Any]:
    """
    Supports:
    - JSONL: each line is a JSON object with at least {id, prediction} (extra fields preserved)
    - JSON: either a list[{"id":..,"prediction":..}] or dict[id] -> prediction / record
    Returns: dict[str(id)] -> record (dict) or scalar prediction (str/any)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"predictions_path not found: {path}")

    out: Dict[str, Any] = {}
    if _is_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = str(rec.get("id")) if rec.get("id") is not None else str(idx)
                if rid == "None":
                    continue

                if isinstance(rec, dict):
                    out[rid] = rec
                else:
                    out[rid] = rec
        return out

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        for rec in obj:
            if not isinstance(rec, dict):
                continue
            rid = str(rec.get("id"))
            if rid == "None":
                continue
            out[rid] = rec
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            rid = str(k)
            if isinstance(v, dict):
                out[rid] = v
            else:
                out[rid] = v
        return out

    raise ValueError(f"Unsupported predictions file format: {path}")

def _write_predictions_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    def _sanitize_jsonable(x: Any) -> Any:
        """
        Ensure objects are JSON-serializable.
        - bytes/bytearray/memoryview -> short placeholder string (avoid huge base64 in logs)
        - dict/list -> recurse
        - other non-serializable -> str(x)
        """
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, (bytes, bytearray, memoryview)):
            b = bytes(x)
            return f"<bytes:{len(b)}>"
        if isinstance(x, dict):
            return {str(k): _sanitize_jsonable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize_jsonable(v) for v in x]
        if isinstance(x, tuple):
            return [_sanitize_jsonable(v) for v in x]
        try:
            json.dumps(x)
            return x
        except Exception:
            return str(x)

    _ensure_dir(os.path.dirname(os.path.abspath(path)))

    seen_ids = set()
    dup = 0
    for r in rows:
        rid = str(r.get("id"))
        if rid in seen_ids:
            dup += 1
        else:
            seen_ids.add(rid)
    if dup > 0:
        print(f"WARNING: writing predictions with {dup} duplicate 'id' values to {path}. "
              f"Eval injection may overwrite records. Please ensure dataset ids are globally unique.")

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_sanitize_jsonable(r), ensure_ascii=False) + "\n")

def _extract_judgment_text(item: Dict[str, Any]) -> str:
    """
    Try to find the judge output text in a result item.
    Different evaluators/metrics may store it under slightly different keys.
    """
    for k in ("judgment", "judge_output", "judge", "eval_judgment"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
                                              
    meta = item.get("meta", {}) or {}
    if isinstance(meta, dict):
        v = meta.get("judgment") or meta.get("judge_output")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _print_empty_output_badcases(details: List[Dict[str, Any]], *, max_n: int) -> None:
    """
    Bad-case detector:
    - If prediction is empty => treat as unknown and print samples.
    - If judge output field exists but is empty => treat as unknown and print samples.

    Rationale:
    - Parsing/format rules should live in each benchmark metric.
    - This detector is global and only catches the most common "broken pipeline" cases.
    """
    if max_n <= 0:
        return

    empty_pred: List[Tuple[int, Dict[str, Any]]] = []
    empty_judge: List[Tuple[int, Dict[str, Any]]] = []
    total_with_pred_field = 0
    total_with_judge_field = 0

    for i, item in enumerate(details):
        if not isinstance(item, dict):
            continue

        if "prediction" in item:
            total_with_pred_field += 1
            p = item.get("prediction")
            if p is None:
                empty_pred.append((i, item))
            elif isinstance(p, str) and not p.strip():
                empty_pred.append((i, item))

        has_judge_field = False
        for k in ("judgment", "judge_output", "judge", "eval_judgment"):
            if k in item:
                has_judge_field = True
                break
        meta = item.get("meta", {}) or {}
        if isinstance(meta, dict) and ("judgment" in meta or "judge_output" in meta):
            has_judge_field = True
        if has_judge_field:
            total_with_judge_field += 1
            j = _extract_judgment_text(item)
            if not j:
                empty_judge.append((i, item))

    print(
        "[badcase_detector] "
        f"empty_prediction={len(empty_pred)}/{total_with_pred_field} "
        f"empty_judge_output={len(empty_judge)}/{total_with_judge_field} "
        f"(sample_n={max_n})"
    )

    if not empty_pred and not empty_judge:
        print("[badcase_detector] No badcases found.")
        return

    def _print_block(title: str, samples: List[Tuple[int, Dict[str, Any]]], total_seen: int) -> None:
        if not samples:
            return
        n = min(max_n, len(samples))
        picked = random.sample(samples, n)
        print(f"\n=== BADCASE DETECTOR: {title} (show {n}/{len(samples)}; total_seen={total_seen}) ===")
        for idx, it in picked:
            print("\n" + "=" * 80)
            print(f"[Item {idx}] id={it.get('id')!r}")
            q = it.get("question", it.get("prompt", ""))
            a = it.get("answer", it.get("prediction", ""))
            if isinstance(q, str):
                print("\n[Question]\n" + q[:2000])
            else:
                print("\n[Question]\n" + repr(q)[:2000])
            if isinstance(a, str):
                print("\n[Prediction/Answer]\n" + a[:2000])
            else:
                print("\n[Prediction/Answer]\n" + repr(a)[:2000])
            j = _extract_judgment_text(it)
            if j:
                print("\n[Judge Output]\n" + j[:4000])

    _print_block("EMPTY PREDICTION (prediction is empty)", empty_pred, total_with_pred_field)
    _print_block("EMPTY JUDGE OUTPUT (judge output field exists but empty)", empty_judge, total_with_judge_field)

@RUNNERS.register_module()
class LocalRunner(BaseRunner):

    def __init__(
        self,
        output_dir: str = "results",
        stage: str = "all",                    
        predictions_path: str = "",
        prediction_field: str = "prediction",
        require_predictions: bool = False,
        badcase_unknown_samples: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.stage = stage
        self.predictions_path = predictions_path
        self.prediction_field = prediction_field
        self.require_predictions = require_predictions
        self.badcase_unknown_samples = int(badcase_unknown_samples)
        _ensure_dir(self.output_dir)

    def run(self, config: Dict[str, Any]):
        stage = str(getattr(self, "stage", "all") or "all").lower().strip()
        if stage not in ("all", "gen", "eval"):
            raise ValueError(f"Invalid runner.stage={stage!r}, expected one of: all, gen, eval")

        runner_cfg = config.get("runner", {}) if isinstance(config.get("runner", {}), dict) else {}
        predictions_path = str(runner_cfg.get("predictions_path", self.predictions_path) or "").strip()
        prediction_field = str(runner_cfg.get("prediction_field", self.prediction_field) or "prediction").strip()
        require_predictions = bool(runner_cfg.get("require_predictions", self.require_predictions))
        badcase_unknown_samples = int(runner_cfg.get("badcase_unknown_samples", self.badcase_unknown_samples))

        env_n = os.getenv("MBEF_BADCASE_UNKNOWN_SAMPLES")
        if env_n is not None and str(env_n).strip() != "":
            try:
                badcase_unknown_samples = int(str(env_n).strip())
            except Exception:
                pass

        if not predictions_path:
            predictions_path = os.path.join(os.path.abspath(self.output_dir), "predictions.jsonl")

        if stage == "gen":
            print("Stage=gen: building model + dataset...")
            model = MODELS.build(config["model"])
            dataset = DATASETS.build(config["dataset"])

            rows: List[Dict[str, Any]] = []
            evaluator_cfg = config.get("evaluator", {})
            runner_use_eval_gen = False
            if isinstance(runner_cfg, dict):
                runner_use_eval_gen = bool(runner_cfg.get("use_evaluator_gen", False))
            env_use_eval_gen = str(os.getenv("USE_EVALUATOR_GEN", "")).strip().lower() in ("1", "true", "yes", "y")

            if (runner_use_eval_gen or env_use_eval_gen) and isinstance(evaluator_cfg, dict) and evaluator_cfg.get("type"):
                try:
                    evaluator = EVALUATORS.build(evaluator_cfg)
                    gen_fn = getattr(evaluator, "generate_predictions", None)
                    if callable(gen_fn):
                        print(f"Stage=gen: using evaluator.generate_predictions() from {evaluator_cfg.get('type')}...")
                        rows = list(gen_fn(model, list(dataset)))
                except Exception as e:
                                                                                       
                    print(f"Stage=gen: WARNING: custom gen unavailable, fallback to generic. err={e!r}")

            if not rows:
                                                  
                batch_size = 1
                if isinstance(evaluator_cfg, dict) and "batch_size" in evaluator_cfg:
                    try:
                        batch_size = int(evaluator_cfg.get("batch_size") or 1)
                    except Exception:
                        batch_size = 1

                prompts = [item.get("prompt", "") for item in dataset]
                print(f"Stage=gen: generating {len(prompts)} responses (batch_size={batch_size})...")
                responses: List[str] = []

                for i in tqdm(
                    range(0, len(prompts), batch_size),
                    desc="Target Generation",
                    unit="batch",
                ):
                    batch_prompts = prompts[i : i + batch_size]
                    responses.extend(model.generate(batch_prompts))

                for item, resp in zip(list(dataset), responses):

                    row: Dict[str, Any] = dict(item)

                    meta = row.get("meta")
                    if isinstance(meta, dict):
                        meta.pop("image", None)
                        meta.pop("raw", None)
                                                           
                    row["id"] = item.get("id")
                    row["prompt"] = item.get("prompt", "")
                    row["prediction"] = resp
                    print(f"Generate: {row}")
                    rows.append(row)

            _write_predictions_jsonl(predictions_path, rows)
            print(f"Stage=gen: saved predictions to {predictions_path}")
            print("Stage=gen finished.")
            return

        print("Building components...")
        model = MODELS.build(config["model"])
        dataset = DATASETS.build(config["dataset"])

        if predictions_path and os.path.isfile(predictions_path):
            preds = _load_predictions(predictions_path)
            injected = 0
            for idx, item in enumerate(dataset.data):
                rid = str(item.get("id")) if item.get("id") is not None else str(idx)
                if rid in preds:
                    v = preds[rid]
                    if isinstance(v, dict):
                                                                                                  
                        item.update(v)
                        if prediction_field in v:
                            item[prediction_field] = v.get(prediction_field)
                    else:
                        item[prediction_field] = v
                    injected += 1
            if require_predictions and injected != len(dataset.data):
                raise ValueError(
                    f"require_predictions=true but only injected {injected}/{len(dataset.data)} "
                    f"items from {predictions_path}"
                )
            if injected > 0:
                print(
                    f"Injected {injected}/{len(dataset.data)} predictions into dataset field '{prediction_field}' "
                    f"from {predictions_path}"
                )
        elif require_predictions:
            raise FileNotFoundError(f"require_predictions=true but predictions file not found: {predictions_path}")

        evaluator = EVALUATORS.build(config["evaluator"])
        
        print("Running evaluation...")
        details = evaluator.evaluate(model, list(dataset))

        try:
            _print_empty_output_badcases(details, max_n=badcase_unknown_samples)
        except Exception as e:
                                                         
            print(f"[badcase_detector] WARNING: failed to print unknown judgments: {e!r}")
        
        print("Computing metrics...")
        print(f"config: {json.dumps(config, indent=2, ensure_ascii=False)}, config.metrics: {config.get('metrics',[])}")
        final_metrics: Dict[str, Any] = {}
        for metric_cfg in config.get("metrics", []):
            metric = METRICS.build(metric_cfg)
            metric_scores = metric.compute(details)
            final_metrics.update(metric_scores)
            
        results = {
            "config": config,
            "metrics": final_metrics,
            "details": details,
        }
        
        summarizer_cfg = config.get("summarizer", {"type": "StandardSummarizer"})
        print(f"Summarizing results using {summarizer_cfg['type']}...")
        
        summarizer = SUMMARIZERS.build(summarizer_cfg)
        summarizer.summarize(results, self.output_dir)
            
        print("Evaluation finished.")

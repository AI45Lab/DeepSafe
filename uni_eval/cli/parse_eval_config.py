import argparse
import json
import os
import shlex
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

def _get(d: Any, path: Tuple[str, ...], default: Any = None) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _require(d: Any, path: Tuple[str, ...]) -> Any:
    v = _get(d, path, None)
    if v is None:
        raise KeyError("Missing config key: " + ".".join(path))
    return v

def _parse_port(api_base: str) -> int:
    u = urlparse(api_base)
    if not u.scheme or not u.scheme.startswith("http"):
        raise ValueError(f"api_base must be http(s) url, got: {api_base}")
    if u.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError(f"api_base must be localhost for this launcher, got: {api_base}")
    if u.port is None:
        raise ValueError(f"api_base missing port, got: {api_base}")
    return int(u.port)

def _try_parse_local_port(api_base: str) -> str:
    """
    Returns port string if api_base is localhost; otherwise returns empty string.
    This keeps launch scripts flexible for remote API judges/targets (e.g. GPT-4o).
    """
    if not api_base:
        return ""
    u = urlparse(api_base)
    if not u.scheme or not u.scheme.startswith("http"):
        return ""
    if u.hostname not in ("127.0.0.1", "localhost"):
        return ""
    if u.port is None:
        return ""
    return str(int(u.port))

def parse_eval_yaml(cfg: Dict[str, Any], strict: bool) -> Dict[str, str]:
    if strict:
        target_model = str(_require(cfg, ("model", "model_name")))
        target_api_base = str(_require(cfg, ("model", "api_base")))
    else:
        target_model = str(_get(cfg, ("model", "model_name"), "") or "")
        target_api_base = str(_get(cfg, ("model", "api_base"), "") or "")

    target_mode = str(_get(cfg, ("model", "mode"), "") or "")
    target_port = _try_parse_local_port(target_api_base)
    target_tp = str(_get(cfg, ("model", "tensor_parallel_size"), 1) or 1)
    target_dtype = str(_get(cfg, ("model", "dtype"), "") or "")
    target_gpu_util = str(_get(cfg, ("model", "gpu_memory_utilization"), "") or "")

    evaluator_type = str(_get(cfg, ("evaluator", "type"), "") or "")
    judge_cfg = _get(cfg, ("evaluator", "judge_model_cfg"), None)
    has_judge = isinstance(judge_cfg, dict) and bool(judge_cfg)

    scorer_cfg = _get(cfg, ("evaluator", "scorer_model_cfg"), None)
    has_scorer = isinstance(scorer_cfg, dict) and bool(scorer_cfg)

    template_name = str(_get(cfg, ("evaluator", "template_name"), "") or "")

    judge_model = ""
    judge_api_base = ""
    judge_mode = ""
    judge_port = ""
    judge_tp = ""
    judge_dtype = ""
    judge_gpu_util = ""
    if has_judge:
        judge_type = str(_get(cfg, ("evaluator", "judge_model_cfg", "type"), "") or "")
        judge_model = str(_get(cfg, ("evaluator", "judge_model_cfg", "model_name"), "") or "")
        judge_api_base = str(_get(cfg, ("evaluator", "judge_model_cfg", "api_base"), "") or "")
        judge_mode = str(_get(cfg, ("evaluator", "judge_model_cfg", "mode"), "") or "")
        judge_port = _try_parse_local_port(judge_api_base)
        judge_tp = str(_get(cfg, ("evaluator", "judge_model_cfg", "tensor_parallel_size"), "") or "")
        judge_dtype = str(_get(cfg, ("evaluator", "judge_model_cfg", "dtype"), "") or "")
        judge_gpu_util = str(_get(cfg, ("evaluator", "judge_model_cfg", "gpu_memory_utilization"), "") or "")
        if strict:
            if not judge_model:
                raise KeyError("Missing config key: evaluator.judge_model_cfg.model_name")
                                                                                            
            if judge_type == "APIModel" and not judge_api_base:
                raise KeyError("Missing config key: evaluator.judge_model_cfg.api_base")

    output_dir = str(_get(cfg, ("runner", "output_dir"), "") or "").strip()

    out: Dict[str, str] = {
        "EVALUATOR_TYPE": evaluator_type,
        "HAS_SCORER": "1" if has_scorer else "0",
        "TARGET_MODEL": target_model,
        "TARGET_API_BASE": target_api_base,
        "TARGET_PORT": target_port,
        "TARGET_MODE": target_mode,
        "TARGET_TENSOR_PARALLEL_SIZE": target_tp,
        "TARGET_DTYPE": target_dtype,
        "TARGET_GPU_MEM_UTIL": target_gpu_util,
        "JUDGE_MODEL": judge_model,
        "JUDGE_API_BASE": judge_api_base,
        "JUDGE_PORT": judge_port,
        "JUDGE_MODE": judge_mode,
        "JUDGE_TENSOR_PARALLEL_SIZE": judge_tp,
        "JUDGE_DTYPE": judge_dtype,
        "JUDGE_GPU_MEM_UTIL": judge_gpu_util,
        "TEMPLATE_NAME": template_name,
        "OUTPUT_DIR_REL": output_dir,
    }
    if strict:
        if has_judge and target_port and judge_port and (target_port == judge_port):
            raise ValueError(f"Port conflict: target_port == judge_port == {target_port}")

    return out

def _print_bash(env: Dict[str, str]) -> None:
    for k, v in env.items():
        print(f"{k}={shlex.quote(v)}")

def _set_nested(d: Dict[str, Any], key_path: str, value: str) -> None:
    keys = key_path.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
        if not isinstance(current, dict):

            return 

    last_key = keys[-1]

    new_val: Any = value
    if last_key in current:
        old_val = current[last_key]
        if isinstance(old_val, bool):
            if str(value).lower() in ("true", "1", "yes"):
                new_val = True
            elif str(value).lower() in ("false", "0", "no"):
                new_val = False
            else:
                new_val = bool(value)
        elif isinstance(old_val, int):
            try:
                new_val = int(value)
            except ValueError:
                pass
        elif isinstance(old_val, float):
            try:
                new_val = float(value)
            except ValueError:
                pass
    else:
                                 
        if value.lower() in ("true", "yes"):
            new_val = True
        elif value.lower() in ("false", "no"):
            new_val = False
        elif value.isdigit():
            new_val = int(value)
        else:
            try:
                new_val = float(value)
            except ValueError:
                pass

    current[last_key] = new_val

def main() -> None:
    default_mbef_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ap = argparse.ArgumentParser(
        description="Parse uni_eval eval_task YAML and emit fields for launch scripts.",
    )
    ap.add_argument("--config", required=True, help="Path to eval_task yaml")
    ap.add_argument(
        "--mbef-root",
        default=default_mbef_root,
        help="MBEF project root, used to derive LOG_DIR/RESULT_DIR in output.",
    )
    ap.add_argument(
        "--format",
        default="bash",
        choices=("bash", "json"),
        help="Output format: bash (KEY=VALUE) or json",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if required fields are missing (recommended for salad_launch.sh).",
    )
    args, unknown_args = ap.parse_known_args()

    overrides: Dict[str, str] = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if not arg.startswith("--"):
                                                   
            i += 1
            continue
        
        key = arg.lstrip("-")
        val = ""
        
        if "=" in key:
            key, val = key.split("=", 1)
            i += 1
        else:
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("-"):
                val = unknown_args[i + 1]
                i += 2
            else:
                                                                                 
                val = ""
                i += 1
        
        overrides[key] = val

    cfg_path = os.path.abspath(args.config)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)

    try:
        import yaml                
    except ModuleNotFoundError as e:
        raise SystemExit(
            "ERROR: PyYAML not installed for this python.\n"
            "Fix: install pyyaml (pip install pyyaml) or run inside your conda env "
            "(e.g. ) used by salad_launch.sh."
        ) from e

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/dict")

    for k, v in overrides.items():
        _set_nested(cfg, k, v)

    env = parse_eval_yaml(cfg, strict=args.strict)
    output_dir_rel = env.get("OUTPUT_DIR_REL", "").strip()
    if output_dir_rel:
        exp_name = os.path.basename(output_dir_rel.rstrip("/"))
    else:
        exp_name = os.path.splitext(os.path.basename(cfg_path))[0]

    mbef_root = os.path.abspath(args.mbef_root)
    log_dir = os.path.join(mbef_root, "scripts", "rlaunch_logs", exp_name)
    result_dir = os.path.join(mbef_root, output_dir_rel) if output_dir_rel else ""

    env["CONFIG_PATH"] = cfg_path
    env["MBEF_ROOT"] = mbef_root
    env["EXP_NAME"] = exp_name
    env["LOG_DIR"] = log_dir
    env["RESULT_DIR"] = result_dir

    if args.format == "json":
        print(json.dumps(env, ensure_ascii=False, indent=2))
    else:
        _print_bash(env)

if __name__ == "__main__":
    main()


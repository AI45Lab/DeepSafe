"""Inspect dependencies/configuration without loading or downloading model weights."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
if __package__:
    from .common import ROOT, read_json
    from .run import load_config
else:
    from common import ROOT, read_json
    from run import load_config


def inspect_dependencies(config):
    result = {"quantization": config["quantization"], "packages": {}}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = "MISSING"
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_build"] = torch.version.cuda
        if result["cuda_available"]:
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            result["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as exc:
        result["cuda_available"] = False
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        result["qwen35_class_available"] = True
        import peft
        import accelerate
        if config["quantization"] == "4bit":
            import bitsandbytes
        result["runtime_imports_ok"] = True
    except Exception as exc:
        result.setdefault("qwen35_class_available", False)
        result["runtime_imports_ok"] = False
        result["runtime_import_error"] = f"{type(exc).__name__}: {exc}"
    required = ("torch", "transformers", "peft", "accelerate")
    if config["quantization"] == "4bit":
        required += ("bitsandbytes",)
    device = config.get("device", "cuda")
    device_ok = device == "cpu" or (device == "cuda" and result["cuda_available"])
    bf16_ok = device != "cuda" or config.get("dtype") != "bfloat16" or result.get("bf16_supported", False)
    quant_ok = config["quantization"] == "none" or (config["quantization"] == "4bit" and device == "cuda")
    result["dependencies_ready"] = bool(device_ok and bf16_ok and quant_ok
        and "torch_error" not in result and result["runtime_imports_ok"]
        and all(result["packages"][name] != "MISSING" for name in required))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dependencies-only", action="store_true",
                        help="Check classifier dependencies only; API/adapter paths may still be placeholders")
    parser.add_argument("--base-model")
    parser.add_argument("--adapter-path")
    parser.add_argument("--quantization", choices=("none", "4bit"))
    parser.add_argument("--review-backend", choices=("api", "vllm"))
    parser.add_argument("--review-api-base")
    parser.add_argument("--review-api-model")
    parser.add_argument("--review-local-model")
    args = parser.parse_args(argv)
    overrides = {"base_model": args.base_model, "adapter_path": args.adapter_path,
                 "quantization": args.quantization, "review_backend": args.review_backend,
                 "review_api_base": args.review_api_base, "review_api_model": args.review_api_model,
                 "review_local_model": args.review_local_model}
    try:
        if args.dependencies_only:
            config = read_json(args.config)
            config.update({key: value for key, value in overrides.items() if value is not None})
        else:
            config = load_config(args.config, overrides)
        result = inspect_dependencies(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"ready_for_inference_attempt": False, "configuration_validated": False,
                          "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
        return 2
    result["check_scope"] = "dependencies_only" if args.dependencies_only else "configuration_and_dependencies"
    result["configuration_validated"] = not args.dependencies_only
    result["ready_for_inference_attempt"] = result["dependencies_ready"] and not args.dependencies_only
    if not args.dependencies_only:
        result.update(adapter_path=config["adapter_path"], base_model=config["base_model"],
                      revision=config["base_revision"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["dependencies_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Offline vLLM review worker. Structured IPC on stdout, all engine logs on stderr."""
import json
import os
import sys
import traceback


def main():
    # Keep a dedicated output FD; vLLM and its C extensions may log directly to stdout.
    channel=os.fdopen(os.dup(sys.stdout.fileno()),"w",encoding="utf-8",buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())

    def emit(value):
        channel.write(json.dumps(value,ensure_ascii=False)+"\n")
        channel.flush()

    try:
        initialization=json.loads(sys.stdin.readline())
        c=initialization["config"]
        if initialization.get("command")!="initialize":
            raise ValueError("Initialization required")
        model=c.get("review_local_model")
        if not isinstance(model,str) or not model.strip() or "CHANGE_ME" in model:
            raise ValueError("An explicit review_local_model is required; no initial-model fallback")
        if c["local_files_only"]:
            os.environ["HF_HUB_OFFLINE"]="1"
        # Lazy import: API review users never need to install vLLM.
        from vllm import LLM, SamplingParams
        tokenizer=c.get("review_local_tokenizer") or model
        revision=c.get("review_local_revision") or None
        if c["local_files_only"]:
            os.environ["HF_HUB_OFFLINE"]="1"
            if not os.path.isdir(model) or not os.path.isdir(tokenizer):
                from huggingface_hub import snapshot_download
            if not os.path.isdir(model):
                # Existing pinned hub cache may also be used; no network access is permitted.
                model=snapshot_download(model,revision=revision,
                    cache_dir=c["cache_dir"],local_files_only=True)
            if not os.path.isdir(tokenizer):
                tokenizer=snapshot_download(tokenizer,revision=revision,
                    cache_dir=c["cache_dir"],local_files_only=True)
        options={"model":model,"tokenizer":tokenizer,"dtype":c["dtype"],
            "tensor_parallel_size":c["review_vllm_tensor_parallel_size"],
            "gpu_memory_utilization":c["review_vllm_gpu_memory_utilization"],
            "max_model_len":c["review_max_input_tokens"]+c["review_max_new_tokens"],
            "limit_mm_per_prompt":{"image":0,"video":0},
            "enforce_eager":c["review_vllm_enforce_eager"],"trust_remote_code":False,
            "download_dir":c["cache_dir"],"generation_config":"vllm"}
        if not os.path.isdir(model):
            options["revision"]=revision
        llm=LLM(**options)
        tok=llm.get_tokenizer()
        sampling=SamplingParams(temperature=0,max_tokens=c["review_max_new_tokens"],seed=42)
        emit({"status":"ready"})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"status":"error","error":f"{type(exc).__name__}: {exc}"})
        return 1
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get("command")!="review":
                raise ValueError("Only review requests are allowed")
            ids=tok.apply_chat_template(request["messages"],tokenize=True,add_generation_prompt=True,
                enable_thinking=c["review_enable_thinking"])
            if hasattr(ids,"input_ids"): ids=ids["input_ids"]
            if ids and isinstance(ids[0],list): ids=ids[0]
            if len(ids)>c["review_max_input_tokens"]:
                raise ValueError(f"Full input has {len(ids)} tokens; no truncation allowed")
            result=llm.generate([{"prompt_token_ids":ids}],sampling,use_tqdm=False)[0].outputs[0]
            emit({"status":"ok","text":result.text,"usage":{
                "prompt_tokens":len(ids),"completion_tokens":len(result.token_ids),
                "finish_reason":result.finish_reason,"review_adapter_enabled":False}})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({"status":"error","error":f"{type(exc).__name__}: {exc}"})
    channel.close()
    return 0


if __name__=="__main__":
    sys.exit(main())

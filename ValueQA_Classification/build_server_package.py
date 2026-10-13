"""Bundle code and the supplied LoRA; never bundle base-model weights or venvs."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ADAPTER=ROOT.parent/"models/qwen35_4b_lora/best_adapter"
ADAPTER_FILES=("adapter_config.json","adapter_model.safetensors","chat_template.jinja",
               "classification_schema.json","tokenizer_config.json","tokenizer.json",
               "training_config.json","README.md")
FILES=(
    "__init__.py","common.py","initial_protocol.py","prompts.py","review_prompt.txt",
    "classification_prompt_strict.txt","review_prompt_source.json","backend.py",
    "boundary_contract.py","routing.py","boundary_test_support.py",
    "vllm_worker.py","gate.py","run.py","evaluate.py","check_env.py","test_framework.py",
    "config.current.server.json","config.current.local.json","policy.example.json","protocol_source.json","README.md",
    "setup_server.sh","setup_local.ps1","setup_vllm.sh","run_server.sh","requirements.txt",
    "build_server_package.py",
)
DATA_FILES=("data/valueqa/evaluation_samples.jsonl",
            "data/valueqa/reference_annotations.jsonl")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Archive destination; generated archives are not Git source")
    args=parser.parse_args(argv)
    module_name=ROOT.name
    destination=args.output or ROOT/(module_name+"_server.zip")
    manifest={}
    entries=[(ROOT/name,module_name+"/"+name) for name in FILES]
    entries.extend((ROOT.parent/name,name) for name in DATA_FILES)
    entries.extend((ADAPTER/name,"models/qwen35_4b_lora/best_adapter/"+name)
                   for name in ADAPTER_FILES)
    # Validate first: missing/LFS-pointer weights must not produce a broken package.
    for source,_ in entries:
        if not source.is_file():
            raise FileNotFoundError("Required package file missing: "+str(source))
    with (ADAPTER/"adapter_model.safetensors").open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Run git lfs pull before packaging; adapter weights are an LFS pointer")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for source,name in entries:
            content=source.read_bytes()
            if name.endswith(".sh"):
                content=content.replace(b"\r\n",b"\n")
            info=zipfile.ZipInfo(name)
            info.create_system=3
            info.external_attr=(0o100755 if name.endswith(".sh") else 0o100644)<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,content)
            manifest[name]={"sha256":hashlib.sha256(content).hexdigest(),"bytes":len(content)}
        archive.writestr(module_name+"/package_manifest.json",
                         json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"package":str(destination),"files":len(entries)+1,
                      "bytes":destination.stat().st_size},ensure_ascii=False))


if __name__=="__main__":
    main()

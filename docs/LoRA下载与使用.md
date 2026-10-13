# 下载并使用 Qwen3.5-4B LoRA

公开模型仓库：[ZhuoZhao/Qwen3.5-4B_LORA](https://huggingface.co/ZhuoZhao/Qwen3.5-4B_LORA)。公开文件可直接下载，无需登录或 API Token。

该仓库提供 LoRA adapter、tokenizer 和分类协议，不包含完整基础模型，也不包含 DeepSafe 源码。本说明是独立文件，不替换 DeepSafe 原有 README。

## 方法一：网页下载，不执行命令

1. 打开仓库的 [Files and versions](https://huggingface.co/ZhuoZhao/Qwen3.5-4B_LORA/tree/main)。
2. 点击每个文件旁的下载按钮；若列表没有按钮，进入文件页面后点击 Download。
3. 下载以下 8 个文件，并放入 DeepSafe 的 `models/qwen35_4b_lora/best_adapter/`。浏览器若给文件名追加了 `(1)`，请恢复下列名称。

```text
DeepSafe/
├── models/
│   └── qwen35_4b_lora/
│       └── best_adapter/
│           ├── adapter_model.safetensors
│           ├── adapter_config.json
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── chat_template.jinja
│           ├── classification_schema.json
│           ├── training_config.json
│           └── README.md
└── qwen_evidence_review/
    ├── config.current.local.json
    └── config.current.server.json
```

`adapter_model.safetensors` 应为 129,934,448 字节（约124 MiB），而不是小型 Git LFS 指针文本。仓库中的 `.gitattributes` 不参与推理，无需另行放入 adapter 目录。若已经通过 Git LFS 或服务器压缩包取得完整 adapter，无需重复下载。

## 方法二：使用 Hugging Face CLI 批量下载

在安装了 `hf` CLI 的环境中，从 DeepSafe 根目录执行；Windows PowerShell 和 Linux 均可使用：

```bash
hf download ZhuoZhao/Qwen3.5-4B_LORA --local-dir models/qwen35_4b_lora/best_adapter
```

公开仓库不要求登录。需要固定本次已核实的上传版本时：

```bash
hf download ZhuoZhao/Qwen3.5-4B_LORA --revision a37408aef0b726eea80561ab8a77f60cb91da787 --local-dir models/qwen35_4b_lora/best_adapter
```

建议将 CLI 安装在独立下载环境，避免升级正在使用的推理环境依赖。

## 基础模型与运行

基础模型为 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)，必须配合训练时固定 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` 使用。基础权重也可在对应版本的 Files and versions 页面下载完整目录，或用 CLI：

```bash
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir models/Qwen3.5-4B
```

两个显式配置均将 `adapter_path` 设置为 `../models/qwen35_4b_lora/best_adapter`，相对于配置文件所在目录解析。放置正确后不需要改 LoRA 路径。使用本地基础权重时，可显式设置 `base_model` 为 `../models/Qwen3.5-4B`，或用 `QWEN_BASE_PATH` / `--base-model` 指定已存在的基础模型目录；`base_revision` 保持上述值。

Windows与Linux均安装 `qwen_evidence_review/requirements.txt`，再显式选择 `qwen_evidence_api.yaml`、`qwen_evidence_api_linux.yaml` 或 `qwen_evidence_vllm.yaml`。API复核另需配置自己的 `LLM_REVIEW_API_KEY`；可选vLLM由setup_vllm.sh安装到独立Linux环境，并需指定独立复核模型。具体运行方式见 [框架README](../qwen_evidence_review/README.md)。

该 adapter 按 `classification_schema.json` 的训练提示词和三类 token logits 完成分类；置信度由三个类别的 logits softmax 得到。单纯下载权重，或换用任意聊天提示词，不等于复现 DeepSafe 的初判协议。

## Hugging Face 模型卡

已另行准备 [Hugging Face README](huggingface_model_card/README.md)。在模型仓库页面点击 Contribute → Upload files，上传这一份 README.md，替换模型仓库当前的空白模板。此操作仅更新 Hugging Face 上的模型卡，不修改 DeepSafe 原有 README。

---
base_model: Qwen/Qwen3.5-4B
library_name: peft
language:
- zh
tags:
- base_model:adapter:Qwen/Qwen3.5-4B
- lora
- transformers
- text-classification
---

# Qwen3.5-4B LoRA：答案理解质量三分类

本仓库提供用于 DeepSafe Qwen 初判的 LoRA adapter，根据问题、相关原文、目标评论和候选答案，将答案分为 `high`、`medium` 或 `low`。它需要配合完整 Qwen3.5-4B 基础模型使用。

- `high`：理解与原文一致，充分完成题目必要要求，无需实质修改。
- `medium`：已有实质正确理解、没有中心误读，但存在必要遗漏或解释不足，可保留正确理解完善。
- `low`：存在核心误读，需要纠正中心解释、结论或大幅重构理解。

训练时的具体提示词、类别顺序和 token ID 以 `classification_schema.json` 为准。DeepSafe 后续边界复核规则是独立模块，不是本 adapter 内置的第二次分类。

## 网页下载：无需命令

本仓库公开，可直接打开 [Files and versions](https://huggingface.co/ZhuoZhao/Qwen3.5-4B_LORA/tree/main)，点击文件下载按钮，或进入文件页面点击 Download。下载以下 8 个文件，放入 DeepSafe 项目根目录下的 `models/qwen35_4b_lora/best_adapter/`：

```text
adapter_model.safetensors
adapter_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
classification_schema.json
training_config.json
README.md
```

将文件直接放在 `best_adapter` 内，不替换 DeepSafe 项目根目录中的 README。浏览器若给文件名追加 `(1)`，请恢复原名。权重文件大小应为 129,934,448 字节（约124 MiB），不能是 Git LFS 指针文本。

## CLI 批量下载

安装 `hf` CLI 后，在 DeepSafe 根目录执行：

```bash
hf download ZhuoZhao/Qwen3.5-4B_LORA --local-dir models/qwen35_4b_lora/best_adapter
```

公开文件无需登录。Windows 和 Linux 可使用相同命令。

## 基础模型要求

本仓库不包含完整基础权重。基础模型为 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)，固定 revision：

```text
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

可在该版本的 Files and versions 页面下载完整模型目录，或执行：

```bash
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir models/Qwen3.5-4B
```

## 在 DeepSafe 中使用

`config.current.local.json` 和 `config.current.server.json` 将 `adapter_path` 设置为 `../models/qwen35_4b_lora/best_adapter`。若基础权重也按上文下载到 `models/Qwen3.5-4B`，模型路径配置片段为：

```json
{
  "adapter_path": "../models/qwen35_4b_lora/best_adapter",
  "base_model": "../models/Qwen3.5-4B",
  "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
}
```

上述片段需加入已有完整配置；相对路径按配置文件目录解析。也可用 `QWEN_BASE_PATH` 指定已有基础模型目录。

本 adapter 通过训练时提示词、tokenizer 和类别 token logits 进行初判，使用三个类别 logits 的 softmax 作为置信度。使用 DeepSafe 中的 `initial_protocol.py` 与 `LocalBackend`；任意聊天提示词或通用生成 pipeline 不保证复现同一分类行为。

Windows与Linux统一安装DeepSafe中的 `valueqa_classification_review/requirements.txt`，其中直接包含Transformers 5.19.0、PEFT 0.21.2及NF4量化所需的bitsandbytes，CUDA PyTorch需匹配显卡驱动。框架运行说明位于 `valueqa_classification_review/README.md`。下载adapter不需要安装vLLM；可选本地复核由setup_vllm.sh在独立Linux环境安装vLLM。

## 训练协议

训练配置记录的主要参数为 LoRA rank 16、alpha 32、dropout 0.05，BF16，最大训练长度2048 tokens、上下文重叠128 tokens。长材料由 DeepSafe 保留完整内容并分窗，对每窗的三类 logits 取均值后分类；详见 `training_config.json` 与 `classification_schema.json`。

本模型卡不声明跨数据集的准确率保证；评测需记录数据来源、分类提示词、基础模型版本及量化设置。

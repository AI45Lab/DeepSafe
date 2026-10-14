# Qwen3.5-4B LoRA

推理使用的微调适配器位于 `qwen35_4b_lora/best_adapter/`。它是 LoRA adapter，仍需配合基础模型 `Qwen/Qwen3.5-4B` 使用，固定 revision 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。

两个显式运行配置均使用 `../models/qwen35_4b_lora/best_adapter`，相对于配置文件所在目录解析，与运行命令所在目录无关。`QWEN_LORA_PATH` 或命令行参数仍可覆盖该路径。

`adapter_model.safetensors` 为 129,934,448 字节，使用根目录 `.gitattributes` 配置的 Git LFS 提交。克隆后执行 `git lfs pull`，确认下载的是实际权重而不是 LFS 指针。`training_args.bin` 仅保留在本地，不参与推理、Git 提交或服务器打包。

也可以将 adapter 上传到 Hugging Face Hub，仅在 GitHub 发布代码和下载说明。应上传本目录中的 `adapter_config.json`、`adapter_model.safetensors`、tokenizer 文件、`chat_template.jinja`、`classification_schema.json`、`training_config.json` 和模型卡 `README.md`；下载完整目录后放回上述相对路径。尚未创建 Hugging Face 仓库或上传模型。

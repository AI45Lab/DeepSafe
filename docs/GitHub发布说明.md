# GitHub 发布与克隆运行

本目录尚未初始化 Git，也未提交或上传。本文件说明已准备的文件布局和后续发布步骤。

## LoRA 和大文件

LoRA 放在 `models/qwen35_4b_lora/best_adapter/`，配置引用相对路径。完整 Qwen 基础权重、虚拟环境和结果缓存不参与提交。LoRA 权重超过普通 Git 的单文件限制，根目录 `.gitattributes` 已为该文件设置 Git LFS：

```bash
git lfs install
# 在已有或新建的目标 Git 仓库中，先确认暂存范围再提交。
git add .gitattributes models/qwen35_4b_lora/best_adapter
git lfs ls-files
```

上面的命令不会上传仓库；发布需另行选择远程仓库。其他用户克隆后执行 `git lfs pull`。也可改用 Hugging Face Hub 托管 adapter，并提供仓库地址和下载说明；目前尚未上传至 Hub。

## 运行配置

所有入口必须显式选择配置。基础模型不随 Git 仓库发布：设置 `QWEN_BASE_PATH` 为基础权重目录，或在独立 Windows 配置中使用固定 revision 的 Hugging Face 模型 ID 下载权重。两种配置已包含 LoRA 相对路径，无需再设置个人桌面路径。

API复核使用环境变量 `LLM_REVIEW_API_KEY`。配置文件中只保存变量名。Windows与Linux共用 `ValueQA_Classification/requirements.txt`，其中直接包含Transformers、PEFT及NF4配置必需的bitsandbytes；CUDA PyTorch需预先匹配驱动安装。框架README集中在ValueQA_Classification目录。可选vLLM由setup_vllm.sh安装到独立Linux/WSL2环境。

## 缓存复核

Git 仓库不发布本机 `results/` 中的冻结初判。首次使用缓存任务，必须显式提供缓存文件：

```bash
python tools/run.py configs/eval_tasks/valueqa_classification_cached_1533.yaml \
  --evaluator.initial_predictions /absolute/path/initial_predictions.jsonl
```

文件应由匹配输入的 Qwen 初判生成，包含 `sample_id`、`predicted_level`、`probabilities` 和 `input_sha256`。无缓存时先用 `valueqa_classification_api.yaml` 或 `valueqa_classification_api_linux.yaml` 运行初判；生成文件位于该任务的 `state/initial_predictions.jsonl`。已有同配置 checkpoint 可续跑，参考标签不能代替初判预测。

## 历史实验

历史结果和源码快照保留，用于核对报告；其代码身份与当前实现不同。已删除依赖旧 A/C 入口的 `experiments/qwen_remaining300/run.py`，不应将历史目录当作当前运行入口。

本轮修复更新了单阶段测试断言和 `/models` 预热模拟，未执行测试套件、GPU 推理或外部 API 实验。

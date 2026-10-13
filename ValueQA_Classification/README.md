# Qwen 初判与单阶段证据复核

本模块在 DeepSafe 中对答案理解质量进行 `high`、`medium`、`low` 三分类。流程为：**微调 Qwen3.5-4B 初判并保存 → 根据类别概率筛选 → 独立模型一次证据复核 → 程序接受改判或保留初判**。没有六维打分，也不单独调用模型提取必要要求。

复核模型只接收 `question`、`target_comment`、`answer`、`original_context` 四个正文输入，不接收初判等级、概率、参考标签或样本编号。必要要求、核心误读和必要遗漏在同一次复核中核验；这次调用能看到答案，因此不能称为“答案盲”的要求提取。

| 初判等级 | 进入复核的条件 |
| --- | --- |
| `high` | 该类别概率 `< 0.9` |
| `medium` | 该类别概率 `< 0.9` |
| `low` | 该类别概率 `< 0.7` |

等于或高于阈值时保留初判，不调用复核模型。有效初判缺少概率、概率无效或与预测等级不一致时，也进入复核。概率来自三类 logits 的 softmax，**未经校准，不代表经过验证的正确率**。

## 1. 环境与工作目录

除另有说明，以下命令均在 **DeepSafe 项目根目录**执行。无需 ZIP 文件；复制或克隆完整项目，再下载基础模型和 LoRA 即可。

当前 GPU 初判配置需要 Python **3.10 或更高**，建议使用 Python 3.11。CUDA 版 PyTorch 应支持当前 Transformers（建议不低于 2.5），且匹配服务器显卡驱动。当前初判使用 NF4 4bit 量化和 BF16 计算，GPU 必须支持 BF16。先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)准备适合驱动的 CUDA 构建，不要直接沿用未知机器的 CUDA 版本。

Windows 与 Linux 共用本模块的 [requirements.txt](requirements.txt)，其中固定 Transformers 5.19.0、PEFT 0.21.2、Accelerate 1.15.0 和 bitsandbytes 0.50.2。这份清单不负责首次安装 CUDA PyTorch；本任务也不需要安装 DeepSafe 根目录的完整依赖清单。API 复核不需要安装 vLLM。

### Windows PowerShell

激活已装好 CUDA PyTorch 的环境；本机环境名为 Laya，新机器使用自己的环境名：

```powershell
conda activate Laya
python -m pip install -r valueqa_classification_review/requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

### Linux

先激活已装好 CUDA PyTorch 的 Python 环境，再创建模块专用环境并激活它：

```bash
bash valueqa_classification_review/setup_server.sh
source valueqa_classification_review/.venv_server/bin/activate
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

`setup_server.sh` 先检查现有 CUDA PyTorch，再通过 `--system-site-packages` 创建 `.venv_server` 并安装依赖。安装依赖可能在新环境内另装 PyTorch，不能保证始终复用原 CUDA 构建；安装后要核对输出。脚本**不会自动激活**新环境，后续命令应在上述 `source` 后执行，或明确使用 `valueqa_classification_review/.venv_server/bin/python`。

`setup_local.ps1` 是 Windows 的可选环境安装辅助脚本；已配置好 Laya 时直接使用前面的安装命令即可。虚拟环境不能从 Windows 复制到 Linux 使用。

## 2. 下载基础模型和 LoRA

基础模型与 LoRA 是两份不同的文件，两者都需要。LoRA 仓库不包含完整 Qwen 基础权重。以下命令在 Windows PowerShell 和 Linux 中相同，并固定本次模型版本。

安装 `hf` CLI（版本范围与当前 Transformers 的 Hub 依赖相容）：

```bash
python -m pip install "huggingface_hub>=1.31,<3"
```

下载 **Qwen3.5-4B 基础模型**：

```bash
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir models/Qwen3.5-4B
```

下载 **微调 LoRA 及配套文件**：

```bash
hf download ZhuoZhao/Qwen3.5-4B_LORA --revision a37408aef0b726eea80561ab8a77f60cb91da787 --local-dir models/qwen35_4b_lora/best_adapter
```

下载后的主要目录如下；`data/valueqa` 使用项目已有数据，不由模型下载命令生成：

```text
DeepSafe/
├── models/
│   ├── Qwen3.5-4B/                       # 完整基础模型
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
├── data/valueqa/
│   ├── evaluation_samples.jsonl
│   └── reference_annotations.jsonl
└── valueqa_classification_review/
    ├── README.md
    ├── config.current.local.json
    └── config.current.server.json
```

基础模型 revision 是 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；LoRA revision 是 `a37408aef0b726eea80561ab8a77f60cb91da787`。两者是 Git 提交版本号，不是文件 SHA-256。

`adapter_model.safetensors` 应为 129,934,448 字节，SHA-256 为 `8536a58bab41ca27585254ab6cab1f8fadbe82e5042a7d0b7ff18e05332360ac`，不能是 Git LFS 指针文本。已取得这份完整 LoRA 时无需重复下载。

### 设置基础模型位置

Linux：

```bash
export QWEN_BASE_PATH="$PWD/models/Qwen3.5-4B"
```

Windows PowerShell：

```powershell
$env:QWEN_BASE_PATH = (Resolve-Path "models/Qwen3.5-4B").Path
```

LoRA 已通过配置字段 `adapter_path: ../models/qwen35_4b_lora/best_adapter` 加载，无需另设环境变量。LoRA 放在其他目录时，可显式设置 `QWEN_LORA_PATH` 为实际目录。

### 网页下载

不使用 CLI 时，打开 [LoRA 固定版本的 Files and versions](https://huggingface.co/ZhuoZhao/Qwen3.5-4B_LORA/tree/a37408aef0b726eea80561ab8a77f60cb91da787)，逐个下载上面列出的 8 个文件，放入 `models/qwen35_4b_lora/best_adapter/`，保留原文件名。模型卡 README 放在 `best_adapter` 内，不覆盖本框架 README 或 DeepSafe 根 README。

基础模型打开 [Qwen 固定版本的 Files and versions](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)，下载完整模型文件，包括全部权重分片及其索引、配置和分词器文件，保持目录结构。

## 3. 离线加载与配置覆盖

`local_files_only` 控制模型加载时是否允许从 Hugging Face 联网获取文件，**不是禁止服务器联网，也不会禁止 DeepSeek API 请求**。上述 `hf download` 是主动下载命令，不受推理配置中的该开关限制。

| 配置或入口 | 当前 `local_files_only` |
| --- | --- |
| 独立 Windows JSON `config.current.local.json` | `false` |
| 独立 Linux JSON `config.current.server.json` | `true` |
| DeepSafe 的四份 `valueqa_classification_*.yaml` | `true`，覆盖对应 JSON |

选择 `true` 时，必须提前下载完整模型，或已有固定 revision 的完整 Hub 缓存；缺文件时程序报错，不会自动下载。配置中的 `cache_dir: ../models_cache` 指向 DeepSafe 根目录的模型缓存，与 `models/Qwen3.5-4B` 显式下载目录是两种存放方式。指定 `QWEN_BASE_PATH` 后直接加载该目录，不需要再下载一份缓存。

独立入口若需要初次自动下载基础模型，可在明确选用的 JSON 中将 `local_files_only` 改为 `false`，同时将 `base_model` 保留为 `Qwen/Qwen3.5-4B`，并取消指向本地目录的 `QWEN_BASE_PATH` 环境变量。不要用一个缺权重的本地目录替代模型 ID。

DeepSafe Linux 入口需要自动下载时，使用以下显式覆盖；仍须提前下载 LoRA 并设置 API 密钥：

```bash
python tools/run.py configs/eval_tasks/valueqa_classification_api_linux.yaml --evaluator.runtime.base_model Qwen/Qwen3.5-4B --evaluator.runtime.local_files_only false --runner.output_dir results/valueqa_classification_online --evaluator.state_dir results/valueqa_classification_online/state
```

基础模型仍按配置中固定的 `base_revision` 下载。**当前代码不会自动从 Hugging Face 下载 LoRA**，即使 `local_files_only` 为 `false`，LoRA 的配套文件也必须存在。离线 vLLM 的 `local_files_only` 同时约束独立复核模型和它的分词器；真正离线部署需提前准备这些文件。

配置优先顺序为：JSON → 支持的环境变量 → 显式覆盖。DeepSafe 的 YAML `evaluator.runtime` 覆盖 JSON 和相应环境变量；命令行 `--evaluator.runtime.*` 再覆盖 YAML。因此只修改 JSON 不一定改变 DeepSafe 的实际运行设置。

JSON 相对路径按配置文件目录解析；DeepSafe 的数据和状态路径按项目根目录解析。命令行 `--base-model`、`--review-local-model` 及对应环境变量中的已有相对模型目录按调用终端目录解析，建议使用绝对路径。独立命令行的 `--input`、`--references`、`--output-dir` 也按调用目录解析。

## 4. API 复核运行

先完成环境和模型准备，并设置上一节的 `QWEN_BASE_PATH`。初判使用本地 Transformers＋PEFT 加载 Qwen 与 LoRA，复核使用独立 API 模型。

当前两份 JSON 与 API 任务 YAML 明确配置 DeepSeek `deepseek-flash`、并发 **5**、超时 **120 秒**、最大输出 **4096 tokens**、temperature **0**、关闭 thinking、JSON 输出和**非流式** `stream: false`。`review_max_input_tokens: 6144` 是本地 vLLM 的输入预算；API 路径不会依据它本地截断或限制请求，服务端上下文限制仍需按所选模型确认。

### Windows PowerShell

```powershell
$env:LLM_REVIEW_API_KEY = "替换为你的API密钥"
python valueqa_classification_review/check_env.py --config valueqa_classification_review/config.current.local.json
python tools/run.py configs/eval_tasks/valueqa_classification_api.yaml
```

### Linux

在已激活的 `.venv_server` 或已准备好的等效环境中执行：

```bash
export LLM_REVIEW_API_KEY="替换为你的API密钥"
python valueqa_classification_review/check_env.py --config valueqa_classification_review/config.current.server.json
python tools/run.py configs/eval_tasks/valueqa_classification_api_linux.yaml
```

`check_env.py` 只检查依赖与配置，不实际加载模型或验证 API 可用性。运行前可额外做完整输入和指纹检查：

```bash
python valueqa_classification_review/run.py --config valueqa_classification_review/config.current.server.json --dry-run
```

Windows 将上述 JSON 换为 `config.current.local.json`。dry-run 不加载或下载模型，也不调用 API；若指定本地基础模型，它会读取权重计算哈希，因此不保证立即完成。输入和参考文件路径以所选 JSON 为准，不能代替其他 YAML 数据配置的核验。

### 更换复核 API

在所选 YAML 的 `evaluator.runtime` 或独立 JSON 中明确填写 `review_api_base`、`review_api_model`、`review_api_key_env` 和服务支持的 `review_api_extra_body`。当前密钥从 `LLM_REVIEW_API_KEY` 读取，不把真实密钥写进配置、README 或仓库。

当前 API 使用线程独立持久连接，并通过 `/models` 预热连接。不支持此接口的服务须显式设置 `review_api_warmup_models: false`；thinking 和 JSON 参数也应按服务调整。代码不支持流式响应，`stream: true` 会被拒绝。

## 5. 独立入口和本地 vLLM

### 不通过 DeepSafe runner

在 DeepSafe 根目录选择完整 JSON，独立运行：

```bash
python valueqa_classification_review/run.py --config valueqa_classification_review/config.current.server.json
```

Windows 使用 `config.current.local.json`。也可以进入模块目录后运行 `python run.py --config config.current.server.json`。独立入口必须显式提供 `--config`，不使用隐式配置文件。

Linux 的 `run_server.sh` 是独立入口包装器，不是 `tools/run.py` 的包装器。进入模块目录后可执行 `bash run_server.sh --config config.current.server.json`；设置 `PYTHON_BIN` 时优先使用该解释器，否则优先使用 `.venv_server/bin/python`，最后才使用 PATH 中的 `python`。

### Linux 或 WSL2 的可选 vLLM 复核

原生 Windows 不支持此本地 vLLM 方案。需要另一份完整复核模型，不能复用初判 Qwen 或分类 LoRA。模型大小、BF16 支持和 GPU 显存须在目标机器验证，不能仅凭环境安装成功判断可以运行。

在 DeepSafe 根目录执行：

```bash
bash valueqa_classification_review/setup_vllm.sh
export REVIEW_MODEL_PATH="/独立复核模型的绝对路径"
export VLLM_PYTHON="$PWD/valueqa_classification_review/.venv_vllm/bin/python"
python tools/run.py configs/eval_tasks/valueqa_classification_vllm.yaml
```

`QWEN_BASE_PATH` 仍按第 2 节设置。vLLM 0.31.0 安装在单独的 `.venv_vllm` 中；运行 `tools/run.py` 的仍是初判环境。复核不需要 API 密钥。单卡先完成初判、卸载分类模型，再加载独立复核模型。本地复核顺序处理，不使用 API 并发参数；输入预算为 6144 tokens，输出预算为 4096 tokens，总配置上下文预算 10240 tokens。完整材料不截断，超输入预算时复核失败并保留初判。本地复核不会自动套用初判的 NF4 量化。

API 与本地 vLLM 共用分类提示词、证据结构检查和改判规则。

## 6. 数据与复用已有初判

项目数据统一位于 `data/valueqa/evaluation_samples.jsonl` 和 `data/valueqa/reference_annotations.jsonl`，模块内不保留副本。原始字段不需要改名或增加。参考标签只参与评分，不进入模型提示词。

替换数据时，DeepSafe 修改 `dataset.path` 和 `dataset.references_path`；独立入口用 `--input`、`--references` 指定。独立入口没有独立参考文件时须明确使用 `--no-reference`；如果输入内已有标签，它们仍可用于评分。

冻结初判可直接导入，避免重新运行 Qwen：

```bash
python tools/run.py configs/eval_tasks/valueqa_classification_cached_1533.yaml --evaluator.initial_predictions /已有初判/initial_predictions.jsonl
```

API 密钥仍需设置。此配置可在 Linux 和 Windows 使用，但需要相应依赖和完整 LoRA 元数据；当前身份计算仍会检查所配置的本地模型文件，因此不能把缓存复核理解为自动免除所有文件准备。首次缓存运行必须显式提供初判文件，或指定已有匹配 checkpoint 的状态目录。

每条导入记录需要 `sample_id`，以及 `predicted_level`、`predicted_label`、`initial_label` 三者至少一个；同时提供多个预测字段时必须一致。不能把输入或参考文件中的普通 `label` 当作冻结初判导入。

`probabilities` 不是导入必填字段，但建议保存完整 high/medium/low 概率，才能复现置信度筛选；缺失或无效时该条会进入复核。可选 `input_sha256` 若存在会被核验，不匹配则拒绝导入；未提供时代码不能仅凭样本编号证明内容一致，应由使用者核对数据来源。

DeepSafe 导出的冻结初判位于对应 `state/initial_predictions.jsonl`；独立入口导出在其输出目录。Linux 本地 vLLM 复核也可在对应命令后添加 `--evaluator.initial_predictions /已有初判/initial_predictions.jsonl --evaluator.imported_only true`。

## 7. 保存、暂停和新实验

| 入口 | 当前结果目录 | 暂停文件 |
| --- | --- | --- |
| DeepSafe Windows API | `results/valueqa_classification_current_api/` | `state/.pause` |
| DeepSafe Linux API | `results/valueqa_classification_current_api_linux/` | `state/.pause` |
| DeepSafe 缓存 API | `results/valueqa_classification_current_cached_1533/` | `state/.pause` |
| DeepSafe 本地 vLLM | `results/valueqa_classification_current_vllm/` | `state/.pause` |
| 独立 Windows JSON | `valueqa_classification_review/outputs/current_single_local/` | `.pause` |
| 独立 Linux JSON | `valueqa_classification_review/outputs/current_single_server/` | `.pause` |

表中的 `state/.pause` 相对于相应结果目录；路径被配置或命令行覆盖时，以实际状态目录为准。创建暂停文件后停止派发新请求，已开始的请求可能继续完成；移除暂停文件并以相同配置、同一状态目录重新启动即可续跑。不要把暂停文件放错到 DeepSafe 结果目录顶层。

有效初判、成功复核和因概率足够而跳过的样本会复用。已记录为终止失败的复核不会自动重试；复核失败、无效证据或无法接受改判时沿用初判。初判失败时不伪造一个兜底等级。

DeepSafe 的 `evaluator.limit: 0` 表示处理全部待处理样本。独立入口不指定 `--limit` 表示全部待处理；独立 `--limit` 必须为正整数，不能使用 `--limit 0`。

代码、模型、输入、提示词或会影响身份的运行配置改变时，使用新状态与结果目录；不要修改旧 checkpoint 的身份来强行恢复。例如新开一轮 Linux API 实验：

```bash
python tools/run.py configs/eval_tasks/valueqa_classification_api_linux.yaml --runner.output_dir results/valueqa_classification_run2 --evaluator.state_dir results/valueqa_classification_run2/state
```

必要时在新任务中显式导入内容匹配的冻结初判。只修改本 README 不改变框架推理代码身份。

DeepSafe 结果包含 `result.json`、`report.md` 和 runner 的 `predictions.jsonl`；`state` 中保存 checkpoint、run_manifest、冻结初判及门控输出。独立入口另生成 `metrics.json`、`experimental_changes.jsonl`、`final_changes.jsonl` 和 `run_state.json`。

统计时区分 `total_samples`、`evaluated_samples` 与复核覆盖数。当前分类 ACC 以已尝试初判的样本为分母，未开始的样本单列为待处理，已尝试失败的样本计入；正式报告前须确认全量完成。复核材料引用不是低价值问题片段定位预测，不能将其直接报告为定位 F1。

## 8. 分类协议和接受规则

| 等级 | 判定含义 |
| --- | --- |
| `high` | 理解与原文一致，充分完成题目必要要求，无需实质修改 |
| `medium` | 已有实质正确理解、没有中心误读，但存在真实必要遗漏或解释不足，可保留正确理解完善 |
| `low` | 核心误读，需纠正中心解释或结论；材料可回答但纯拒答、完全偏题且无实质正确理解也属于此档 |

初判按照 LoRA 的 `classification_schema.json` 和 `initial_protocol.py` 使用训练提示词、分词器、类别顺序及 token IDs；复核另使用本模块 `classification_prompt_strict.txt` 与 `review_prompt.txt`。修改复核提示词不会自动修改 LoRA 初判的训练协议。

初判长材料保留完整内容并分窗，对各窗的三类 logits 取均值后再计算 softmax。问题、目标评论和答案在每个窗口重复；这些固定输入本身超过训练长度预算时会失败，不承诺任意长度输入均可运行。训练记录为 LoRA rank 16、alpha 32、dropout 0.05、BF16、最大长度 2048 tokens、上下文重叠 128 tokens；当前 NF4 推理与训练 BF16 设置不同，性能须在实际环境评测。

门控检查复核完整性、等级与诊断的一致性、必要要求及引用是否存在于输入；**这些结构与引用检查不能证明语义正确，也不能保证提升 ACC**。

| 模式 | 最终等级处理 |
| --- | --- |
| `observe` | 保留初判，记录复核候选用于分析 |
| `experimental` | 通过结构与引用校验时允许改判；当前明确配置采用此模式，未经独立收益校准 |
| `calibrated` | 须提供与 engine/gate 身份匹配的已验证策略，只允许策略启用的改判方向 |

`policy.example.json` 只是模板，`validated: false` 不能作为正式校准策略使用。复核模型的能力和阈值收益需要独立验证集确认。

## 9. 测试与 GitHub 分发

在 DeepSafe 根目录执行，测试使用模拟模型与本地 HTTP 服务，不下载模型或消耗真实 API：

```bash
python -B -m unittest valueqa_classification_review.test_framework -q
python -B -m unittest discover -s tests -p test_valueqa_classification_integration.py -q
```

这些测试不能替代 Linux GPU、bitsandbytes 或真实 vLLM 兼容性试运行。部署后先小批量验证，再运行全量数据。

完整 DeepSafe 部署需保留本模块、四份 `configs/eval_tasks/valueqa_classification_*.yaml` 和 `uni_eval` 中的四个 `valueqa_classification.py` 集成模块。仅复制本模块可以使用独立入口，但仍须按前述结构准备模型和数据，不能调用缺失的 `tools/run.py`。

源码目录保留运行代码、提示词、明确配置、安装和启动脚本、测试、协议来源记录及 README。虚拟环境、基础模型权重与缓存、密钥、日志、结果和生成的 ZIP 不提交到 GitHub。LoRA 若随 GitHub 分发须使用已配置的 Git LFS，克隆后执行 `git lfs pull`；也可直接使用第 2 节的 Hugging Face 下载命令。

`build_server_package.py` 只是可选归档工具，运行不依赖它或预先存在的 ZIP。本文件是框架说明，不替代 DeepSafe 根 README，也不替代 LoRA 仓库的模型卡。

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./data/deepsafe-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./data/deepsafe-logo-light.svg">
    <img alt="DeepSafe Logo" src="./data/deepsafe-logo-dark.svg" width="300">
  </picture>
</div>
<div style="height: 8px;"></div>
<div align="center">
  <h3 style="margin: 0;">LLM和MLLM安全评测工具集</h3>
</div>
<div style="height: 6px;"></div>
<div align="center">
  <a href="https://ai45.shlab.org.cn/safety-entry">
    <img alt="HomePage" src="https://img.shields.io/static/v1?label=&message=%F0%9F%8C%90%20HomePage&color=2F81F7&style=flat">
  </a>
  <a href="#">
    <img alt="Hugging Face" src="https://img.shields.io/static/v1?label=&message=Hugging%20Face&color=FFCC00&style=flat&logo=huggingface&logoColor=000000">
  </a>
  <a href="#">
    <img alt="Technical Report (arXiv)" src="https://img.shields.io/static/v1?label=&message=Technical%20Report&color=B31B1B&style=flat&logo=arxiv&logoColor=FFFFFF">
  </a>
  <a href="#quick-start">
    <img alt="Documentation" src="https://img.shields.io/static/v1?label=&message=%F0%9F%93%9A%20Documentation&color=8250DF&style=flat">
  </a>
</div>
<div style="height: 18px;"></div>

当前大模型安全评测缺乏全面的标准化方案，且普遍缺失专用评测模型。**DeepSafe** 是首个集成 25+ 主流安全数据集及 **ProGuard** 专用评测模型的一体化框架，支持 LLM/VLM 全模态评测。

> **DeepSafe** 源自 **DeepSight**，并可与 🔍 [**DeepScan**](https://github.com/AI45Lab/DeepScan)（LLM/MLLM 诊断工具集）联动使用。完整的“评测‑诊断”一体化工作流见 [<img src="https://avatars.githubusercontent.com/u/194484914?s=200&v=4" height="14" alt="AI45"> HomePage](https://ai45.shlab.org.cn/safety-entry)。

DeepSafe 构建模块化解耦+配置驱动的弹性架构，实现从推理生成判断到深度评测报告的全链路自动化闭环。为AI Safety研究提供了一个可深度评测、可复现且扩展性强持续演进的安全基础设施，旨在推动大模型安全评估从结果测试走向深度分析，加速构建可信AI的评测进程。🚀

---

## ✨ 创新优势

### 一体化评测框架 (All-in-One Framework)
- **高扩展性**：基于 **Registry 注册机制**，新组件（数据集、指标等）通过极简注册即可接入，支持 YAML 一键装配，评测链路可按需拆分复用。
- **极简易用**：遵循“配置即运行”范式，只需提供一份 YAML 配置文件，即可自动完成全链路闭环，生成标准化报告。
- **全面覆盖**：适配主流模型后端与多维度安全基准，数据输出详尽（包含评分、回复明细、坏例抽样及 Markdown 报告），极大方便了结果分析与复现。

### 专用评测模型 ProGuard
- **主动风险识别**：首创主动性检测范式，具备推理并描述未知风险的能力，突破了传统固定分类的限制。
- **根治模态偏见**：设计分层多模态安全分类体系，基于 8.7 万样本模态平衡数据集训练，确保图文风险评测的公平与精准。

---

## 📖 Model Support

DeepSafe 适配了主流的开源模型与商业 API，支持灵活切换评测后端。

| Open-source Models (via vLLM/HF) | API Models |
| :--- | :--- |
| • **Llama / Llama3 / Alpaca / Vicuna** | • **OpenAI (GPT-4/3.5)** |
| • **Qwen / Qwen2 / Qwen2.5 / Qwen3** | • **Gemini** |
| • **GLM / ChatGLM2 / ChatGLM3** | • **Claude** |
| • **InternLM / InternLM2.5** | • **ZhipuAI (ChatGLM)** |
| • **Baichuan / Baichuan2** | • **Baichuan API** |
| • **Yi / Yi-1.5 / Yi-VL** | • **ByteDance (YunQue)** |
| • **Mistral / Mixtral** | • **Huawei (PanGu)** |
| • **Gemma / Gemma 2** | • **Baidu (ERNIEBot)** |
| • **DeepSeek (Coder/Math)** | • **360 / MiniMax / SenseTime** |
| • **BlueLM / TigerBot / WizardLM** | • **Xunfei (Spark)** |
| • ...... | • ...... |

## 📊 Dataset Support

| Name | Description |
| :--- | :--- |
| **Salad-Bench** | 多维度、多语言安全评测基准。 |
| **HarmBench** | 越狱攻击鲁棒性标准化基准。 |
| **Do-Not-Answer** | 拒绝有害问题能力评估。 |
| **BeaverTails** | 人类偏好对齐大规模安全集。 |
| **MM-SafetyBench** | 多模态大模型安全评测。 |
| **VLSBench** | 视觉-语言图文对齐安全基准。 |
| **FLAMES** | 细粒度安全对齐评测框架。 |
| **XSTest** | “过度拒绝”倾向基准测试。 |
| **SIUO** | 隐藏有害意图辨别测试。 |
| **Uncontrolled-AIRD** | 非受控场景 AI 风险检测。 |
| **TruthfulQA** | 内容真实性与抗误导基准。 |
| **HaluEval-QA** | 问答场景幻觉评测。 |
| **MedHallu** | 医疗领域幻觉评测。 |
| **MossBench** | 综合性安全与能力基准。 |
| **Fake-Alignment** | 真/伪对齐辨别测试。 |
| **Sandbagging** | 故意隐藏能力倾向测试。 |
| **Evaluation-Faking** | 评测过程作弊/操纵评估。 |
| **WMDP** | 危险知识领域（生化/核能）安全性。 |
| **MASK** | 欺骗性对齐倾向评测。 |
| **MSSBench** | 多阶段细粒度安全标准。 |
| **BeHonest** | 诚实性与自我认知评估。 |
| **Deception-Bench** | 模型欺骗行为专项测试。 |
| **Ch3EF** | 多层级多维度安全能力评估。 |
| **Manipulation-Persuasion-Conv** | 抗诱导/抗操纵能力测试。 |
| **Reason-Under-Pressure** | 高压约束下逻辑推理测试。 |

---

<a id="quick-start"></a>
## 🚀 快速上手 (Quick Start)

DeepSafe 提供了标准化的评测工作流，主要分为四个阶段：**配置 -> 推理 -> 评估 -> 可视化**。

### 1. 环境准备
```bash
# 建议使用虚拟环境
pip install -r requirements.txt
# 下载数据集需要 huggingface-cli
pip install -U huggingface_hub

# 验证环境是否安装成功
python smoke_test.py
```

### 1.1 下载评测数据集

DeepSafe 支持的评测数据集主要托管在 Hugging Face 上。你可以使用 `huggingface-cli` 统一进行下载。

**下载示例 (以 Do-Not-Answer 为例):**

```bash
# 下载数据集到本地目录
huggingface-cli download --repo-type dataset --resume-download LibrAI/do-not-answer --local-dir data/do-not-answer --local-dir-use-symlinks False
```

下载完成后，请在对应的 YAML 配置文件（如 `configs/eval_tasks/do_not_answer_v01.yaml`）中修改 `dataset.path` 指向你的本地路径。

### 1.2 下载 Salad-Bench 数据集 (特殊说明)

Salad-Bench 数据集需从[官方仓库](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data)手动下载。可执行如下命令：

```bash
# 下载数据集
huggingface-cli download --repo-type dataset --resume-download OpenSafetyLab/Salad-Data --local-dir Salad-Data --local-dir-use-symlinks False
```

随后将 `base_set.json` 移动或复制到你指定的本地目录。并在配置文件 `configs/eval_tasks/salad_judge_local.yaml` 的 `dataset.path` 字段中填入你的本地路径：

```yaml
dataset:
  type: SaladDataset
  path: /你的路径/Salad-Data/base_set.json
```

---

### 1.2 下载并配置 mdjudge 评测器

**mdjudge (MD-Judge-v0.1)** 为 Salad-Bench 官方推荐的安全评测器。权重模型可通过 [HuggingFace](https://huggingface.co/OpenSafetyLab/MD-Judge-v0.1) 下载：

```bash
# 或下载 safetensors 权重文件和 config 文件
huggingface-cli download --resume-download OpenSafetyLab/MD-Judge-v0.1 --local-dir MD-Judge-v0.1 --local-dir-use-symlinks False
```

将 `MD-Judge-v0.1` 文件夹（或权重文件等）放在你本地任意指定目录。随后在 `evaluator.judge_model_cfg.model_name` 字段中，替换为本地路径，例如：

```yaml
evaluator:
  judge_model_cfg:
    type: VLLMLocalModel
    model_name: /你的本地目录/MD-Judge-v0.1
    # 其它配置不变
```

> **注意**：第一次使用时，务必确保评测机器可加载权重文件，且本地配置路径与实际下载路径一致，否则评测将报错找不到模型。

---

### 1.3 参考配置文件片段（路径需按本地实际情况修改）

```yaml
dataset:
  type: SaladDataset
  path: /your/path/to/Salad-Data/base_set_sample_100.json

evaluator:
  type: ScorerBasedEvaluator
  batch_size: 32
  template_name: md_judge_v0_1
  judge_model_cfg:
    type: VLLMLocalModel
    model_name: /your/path/to/MD-Judge-v0.1
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.4
    trust_remote_code: true
    temperature: 0.0
    max_tokens: 64
```

如有其他依赖或运行报错，建议核对 config 路径以及模型格式（safetensors、bin 均可，推荐使用官方权重文件夹结构保持一致）。



### 2. 启动评测 (以 Salad-Bench 为例)
MBEF 提供了本地化一键脚本，只需执行以下命令：

```bash
# 进入项目根目录执行
bash scripts/run_salad_local.sh configs/eval_tasks/salad_judge_local.yaml
```

### 3. 工作流解析 (Workflow)
- **配置 (Configure)**：在 YAML 中指定待测模型、数据集路径及裁判模型参数。
- **推理 (Inference)**：脚本自动拉起 vLLM（本地模式）或调用 API，生成模型回答并保存至 `predictions.jsonl`。
- **评估 (Evaluation)**：启动 ProGuard 或其他裁判模型，对生成回答进行自动化打分与分类。
- **可视化 (Visualization)**：自动汇总指标，在输出目录生成 `report.md` 实时查看评测结论。

---

## ⚙️ 配置文件参数说明 (Configuration Guide)

配置参数以 `salad_judge_v01_qwen1.5-0.5b_vllm_local.yaml` 为参考：

### 1. `model` (待测模型模块)
- `type`: 加载方式。`APIModel` (标准API接口), `VLLMLocalModel` (本地vLLM), `HuggingFaceModel` (TF驱动)。
- `model_name`: 模型本地路径或 HF ID。
- `api_base`: 服务地址（若为 `localhost` 且 `type` 为 APIModel，脚本会尝试拉起本地 vLLM）。
- `concurrency`: 并发推理请求数。
- `strip_reasoning`: 是否移除 `<thought>` 等推理过程。
- `temperature`: 采样温度，`0.0` 为确定性输出。
- `max_tokens`: 回答生成的最大 Token 数。

### 2. `dataset` (数据集模块)
- `type`: 数据集类名（如 `SaladDataset`）。
- `path`: 数据文件本地路径。
- `limit`: (可选) 随机采样数，用于快速跑通流程。

### 3. `evaluator` (评测与裁判模块)
- `type`: 评测器类名。`ScorerBasedEvaluator` 需要裁判打分。
- `template_name`: Prompt 模板 ID（定义在 `uni_eval/prompts.py`）。
- `judge_model_cfg`: **裁判模型配置**，包含裁判模型的路径、显存占用 (`gpu_memory_utilization`)、TP 数等。

### 4. `metrics` (指标模块)
- `type`: 指标类名（如 `SaladCategoryMetric`）。
- `safe_label`/`unsafe_label`: 判定输出中代表安全或危险的关键字。

### 5. `runner` (运行模块)
- `output_dir`: 推理 JSONL、裁判日志及最终 Markdown 报告的保存路径。

---

## 🛠️ 自定义数据集集成 (Custom Dataset)

DeepSafe 约定了极简的数据组织形式，接入新数据集只需三步：

### 1. 组织数据 (JSONL)
确保你的数据文件每一行为一个 JSON 对象，并包含以下字段：
`{"id": "001", "prompt": "模型输入内容", "reference": "标准答案", "category": "分类标签"}`

### 2. 实现核心 Python 组件
- **Dataset** (`uni_eval/datasets/`): 继承 `BaseDataset` 并重写 `load()`，将上述 JSONL 加载为 `List[Dict]`。
- **Metric** (`uni_eval/metrics/`): 继承 `BaseMetric` 并重写 `compute()`，根据预测结果计算分值。
- **Evaluator** (可选): 继承 `BaseEvaluator` 实现特定的多阶段判定逻辑。

### 3. 注册运行
在各模块的 `__init__.py` 中完成注册，创建对应 YAML 配置后即可一键启动。

---

## 📁 项目结构

```text
DeepSafe/
├── uni_eval/                # 核心评测框架
│   ├── datasets/            # 数据集加载实现
│   ├── models/              # 模型接口适配 (API/HF/vLLM)
│   ├── evaluators/          # 评测流程控制器（包含原生 Evaluator 集成）
│   ├── metrics/             # 评估指标实现
│   ├── runners/             # 运行任务流管理
│   ├── summarizers/         # 结果汇总与报告生成
│   ├── cli/                 # 命令行解析工具
│   └── registry.py          # 模块注册机制中心
├── configs/                 # 评测任务配置文件 (YAML)
├── scripts/                 # 启动脚本 (Shell)
├── tools/                   # 实用工具脚本
└── results/                 # 评测结果与报告存储 (JSON/Markdown)
```

---

## 🤝 贡献与反馈
欢迎通过 Issue 或 Pull Request 共同构建 **DeepSafe**！🌟

---

## 📬 联系我们

- **邮箱**：[`shaojing@pjlab.org.cn`](mailto:shaojing@pjlab.org.cn)
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./data/deepsafe-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./data/deepsafe-logo-light.svg">
    <img alt="DeepSafe 标识" src="./data/deepsafe-logo-dark.svg" width="280">
  </picture>

  <h2>DeepSafe-Sci</h2>
  <p>面向语言模型的科学安全评测</p>
  <p><a href="README.md">English</a> · <a href="https://github.com/AI45Lab/DeepSafe">DeepSafe</a></p>
</div>

## 项目简介

DeepSafe-Sci 是 [DeepSafe](https://github.com/AI45Lab/DeepSafe) 的科学安全评测分支。它沿用 DeepSafe 由配置驱动的推理、评估、指标计算和报告流程，并增加面向科学场景的数据适配器与评测器。

科学安全评测需要区分两类问题。一类是模型面对高风险科学请求时，给出了可操作的步骤或难以直接获得的新信息；另一类是模型把正常科研活动误判为危险请求，在没有充分理由的情况下拒绝回答。本分支通过 SciHazard、Safe-Scientist 和 SOSBench 分别考察这些行为。

## 评测基准

| 基准 | 评测重点 | 主要指标 | 默认配置 |
| :--- | :--- | :--- | :--- |
| **SciHazard** | 模型对高风险科学请求所作回答的危害程度 | 可执行性等级、新增风险等级、不安全回答比例和 DeHarm 分数 | [`configs/eval_tasks/scihazard.yaml`](configs/eval_tasks/scihazard.yaml) |
| **Safe-Scientist** | 模型对不同科学领域高风险研究任务的识别与安全处理 | 拒答率与安全分数，并给出总体和分领域结果 | [`configs/eval_tasks/safe_scientist.yaml`](configs/eval_tasks/safe_scientist.yaml) |
| **SOSBench** | 模型面对经过转换的科学请求时的安全行为与过度拒绝问题 | safe/unsafe 判定与不安全回答比例，并给出总体和分学科结果 | [`configs/eval_tasks/sosbench.yaml`](configs/eval_tasks/sosbench.yaml) |

本分支包含 SciHazard 所需的数据，其中既有 safe、unsafe 内容，也有适合开发和快速检查的较小子集。Safe-Scientist 与 SOSBench 数据由各自的上游项目发布，运行前需要放到本地指定目录。

## 快速开始

### 1. 安装环境

```bash
git clone --branch deepsafe-sci https://github.com/AI45Lab/DeepSafe.git
cd DeepSafe

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 API

默认配置通过兼容 OpenAI 接口的模型完成待测模型推理和裁判模型评估。密钥与接口覆盖项从环境变量读取，不写入 YAML 文件。

```bash
export OPENAI_API_KEY="your-api-key"

# 可选：指定其他兼容 OpenAI 的接口。
export OPENAI_BASE_URL="https://api.openai.com/v1"

# SciHazard 在缓存未命中、需要在线检索证据时使用。
export SERPER_API_KEY="your-serper-api-key"
```

如需修改待测模型、裁判模型、并发数、数据路径或输出目录，请编辑 [`configs/eval_tasks/`](configs/eval_tasks) 下对应的配置文件。运行前应同时检查 `model` 和 `evaluator.judge_model_cfg` 两部分。

### 3. 准备外部数据

SciHazard 可直接使用仓库中的默认配置与数据。其余两个基准需要额外准备：

- 将 Safe-Scientist 的六个领域 JSON 文件放到 `data/safe_scientist/`。
- 将 SOSBench 的 Parquet 文件放到 `data/sosbench/`，并保留上游许可证和 Responsible Use Agreement。

数据字段要求与上游来源见 [`docs/benchmark-data.md`](docs/benchmark-data.md)。

### 4. 运行评测

在仓库根目录执行：

```bash
bash scripts/run_scihazard_local.sh
bash scripts/run_safe_scientist_local.sh
bash scripts/run_sosbench_local.sh
```

三个脚本会把对应配置交给 `tools/run.py`。本地 runner 在缺少已有预测时先生成模型回答，再调用配置中的裁判进行评估，随后计算指标并生成报告。

### 5. 查看输出

默认输出目录为：

```text
results/scihazard/
results/safe_scientist/
results/sosbench/
```

一次完整运行通常生成：

- `predictions.jsonl`：与评测样本对应的待测模型回答；
- `result.json`：逐条评估结果和汇总指标；
- `report.md`：便于阅读的指标摘要。

这些文件属于运行产物，不提交到仓库。

## 目录结构

```text
configs/eval_tasks/       评测任务的 YAML 配置
scripts/                  从仓库根目录启动评测的脚本
uni_eval/datasets/        数据集适配器
uni_eval/evaluators/      评测器与裁判实现
uni_eval/metrics/         汇总指标实现
DeHarmScore-trace/        SciHazard 危害评估器及可复现缓存
docs/benchmark-data.md    外部数据要求与字段说明
```

## SciHazard 与 DeHarmScore-trace

SciHazard 使用 [`DeHarmScore-trace`](DeHarmScore-trace/README.md) 判断模型回答的危害程度。该评估器将回答与问题对应的 checklist 进行匹配，必要时检索外部证据，并给出 E1-E4 的可执行性（Executability）等级和 N1-N4 的新增风险（Net-New Risk）等级。组合结果同时反映回答包含多少可操作细节，以及回答是否提供了较难获取的新信息。

仓库中的 checklist、检索结果和检索证据缓存只用于 SciHazard 的可复现运行，不与 Safe-Scientist 或 SOSBench 共用。prompt trace 和已完成的模型回答不包含在仓库中。裁判配置与诊断字段详见 [DeHarmScore-trace 快速上手](DeHarmScore-trace/QUICKSTART.md)。

## 使用边界与项目归属

DeepSafe-Sci 用于受控条件下的模型评估与安全研究。部分评测请求涉及高风险科学操作。运行时应限制生成回答的访问范围，并遵守所在机构的管理要求以及各上游数据集的使用条款。

本分支基于 [DeepSafe](https://github.com/AI45Lab/DeepSafe)，沿用其注册机制、runner、模型接口、结果汇总和报告模块。仓库中的 DeHarmScore-trace 组件采用其[独立许可证](DeHarmScore-trace/LICENSE)；其他数据与组件的适用条款以各上游项目为准。

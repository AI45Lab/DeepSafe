# ValueQA 运行说明（DeepSafe）

本项目在 DeepSafe 中运行答案质量三分类（`high` / `medium` / `low`）及 `low` 问题片段定位，当前评测集为 1500 条，每档 500 条。

单LLM方案在一次响应中同时完成分类与定位；多Agent框架通过初判、依据审核和有分歧时的裁决完成分类与定位。

| 运行任务 | 配置文件 |
| --- | --- |
| 单LLM分类与 low 定位 | [value_classification.yaml](configs/eval_tasks/value_classification.yaml) |
| 多Agent框架 | [value_attribution.yaml](configs/eval_tasks/value_attribution.yaml) |

## 1. 准备环境

以下命令使用 PowerShell，均在 DeepSafe 项目根目录执行。选择已安装依赖的 Python 环境；当前代码已在 Python 3.9 环境验证。

```powershell
Set-Location "C:\Users\18306\Desktop\项目集成 (2)\DeepSafe"
$env:PYTHONUTF8 = "1"
python -m pip install openai PyYAML tqdm
```

更换机器时调整项目路径。两项任务的配置已设置 `module_profile: value`，用于加载相关运行组件。

## 2. 选择 API 或本地模型服务

编辑要运行的 YAML 配置中的 `model` 部分。两项任务使用相同的连接配置方式；模板中的连接类型、模型名称和服务地址均需自行填写。

### 方式一：远程 API

将 `model` 部分配置如下，替换模型名称和服务地址：

```yaml
model:
  type: ValueQAServiceModel
  connection: api
  api:
    model_name: "填写模型完整名称"
    api_base: "https://你的服务地址/v1"
    api_key: ENV
    temperature: 0
    max_tokens: 2048
    timeout: 180
    max_retries: 0
    concurrency: 10
```

在运行任务的 PowerShell 窗口设置密钥，`api_key: ENV` 会读取它：

```powershell
$env:OPENAI_API_KEY = "你的API密钥"
```

### 方式二：本地模型服务

先启动兼容 OpenAI `/v1/chat/completions` 的模型服务，再配置模型名称与地址。下面以 Ollama 的常用地址为例；vLLM 服务按实际端口填写，例如 `http://127.0.0.1:8000/v1`。

```yaml
model:
  type: ValueQAServiceModel
  connection: local
  local:
    model_name: "填写本地服务发布的模型名称"
    api_base: "http://127.0.0.1:11434/v1"
    api_key: EMPTY
    temperature: 0
    max_tokens: 2048
    timeout: 180
    max_retries: 0
    concurrency: 1
```

本地服务需要鉴权时，将 `api_key` 改为 `ENV` 并设置 `OPENAI_API_KEY`。程序通过服务调用本地模型；模型服务需事先启动。

`concurrency` 按服务允许的并发数设置，实际样本并发取它与 `evaluator.workers` 的较小值。`max_retries` 保持为 `0`，由任务执行层记录并管理重试；服务专用参数可添加在选中的 `api` 或 `local` 配置内。

## 3. 启动任务

完成模型配置后，运行所需任务：

```powershell
# 单LLM分类与 low 定位
python tools/run.py configs/eval_tasks/value_classification.yaml

# 多Agent框架
python tools/run.py configs/eval_tasks/value_attribution.yaml
```

默认 `runner.stage: all`，执行模型调用、指标计算及报告生成。输入与参考标注路径分别由 `dataset.path` 和 `dataset.references_path` 指定。

也可以用命令行覆盖配置，例如选择 API 模型：

```powershell
python tools/run.py configs/eval_tasks/value_classification.yaml `
  --model.connection api `
  --model.api.model_name "填写模型完整名称" `
  --model.api.api_base "https://你的服务地址/v1"
```

### 新建一轮实验

更换模型、服务地址、生成参数、数据、代码或并发等运行设置后，使用新的结果目录和状态目录。例如：

```powershell
python tools/run.py configs/eval_tasks/value_attribution.yaml `
  --runner.output_dir results/framework_run2 `
  --evaluator.state_dir results/framework_run2/state
```

`api` / `local` 子目录会自动添加。同一种连接方式下更换模型也需要新的状态目录；已有状态会校验代码、输入和运行设置，配置不一致时拒绝复用。

### 中断后续跑

保持原配置和状态目录，重新执行原命令即可。框架复用已完成样本和有效阶段；单LLM复用已经收到的响应，格式不合要求的已返回响应也会保留，不为获得更好输出再次询问。若手动设置了状态目录中的 `.pause` 文件，续跑前需移除该文件。

### 分开生成与计算指标

使用相同的配置与目录：

```powershell
# 生成并保存 predictions.jsonl
python tools/run.py configs/eval_tasks/value_classification.yaml --runner.stage gen

# 读取已保存的 predictions.jsonl，离线计算指标
python tools/run.py configs/eval_tasks/value_classification.yaml --runner.stage eval
```

框架任务将配置文件替换为 `value_attribution.yaml`。`eval` 阶段不调用模型，但配置仍需填写连接方式、模型名称和地址；预测文件须与当前方法、代码及输入数据匹配。其他预测文件可通过 `--runner.predictions_path "文件路径"` 指定。

## 4. 查看结果

默认保存位置如下：

| 内容 | 单LLM | 框架 |
| --- | --- | --- |
| API 结果与报告 | `results/valueqa_single_llm/api/` | `results/valueqa_framework/api/` |
| 本地服务结果与报告 | `results/valueqa_single_llm/local/` | `results/valueqa_framework/local/` |
| API 状态与调用记录 | `results/valueqa_single_llm/state/api/` | `results/valueqa_framework/state/api/` |
| 本地服务状态与调用记录 | `results/valueqa_single_llm/state/local/` | `results/valueqa_framework/state/local/` |

- `result.json`：指标与样本明细，由 `all` / `eval` 阶段生成。
- `report.md`：评测报告，由 `all` / `eval` 阶段生成。
- `predictions.jsonl`：保存的预测，由 `gen` 阶段生成。
- `experiment_manifest.json`：本轮代码、输入和运行配置校验信息。
- `api_audit.jsonl`：请求、原始响应、用量、错误及格式修复记录。
- `checkpoint.jsonl`：框架各阶段的断点记录。

主要指标为分类 ACC、Macro-F1、low 片段完全匹配 F1 和字符重叠 F1。ACC 使用输入样本总数作为分母，未完成或无效分类计为错误；单LLM另行统计输出格式完整有效的数量，不把所有格式问题都当成分类失败。

## 5. 使用自己的数据

替换两个配置中的数据路径即可：

```yaml
dataset:
  type: ValueQADataset
  path: data/valueqa/evaluation_samples.jsonl
  references_path: data/valueqa/reference_annotations.jsonl
```

样本文件每行一个 JSON 对象，包含以下字段：

| 字段 | 用途 |
| --- | --- |
| `sample_id` | 唯一样本编号 |
| `original_context` | 完整相关原文 |
| `target_comment` | 需要理解的目标评论 |
| `question` | 问题 |
| `answer` | 待评答案 |
| `label` | 参考等级，取 `high` / `medium` / `low`，仅用于评分 |
| `evaluation_dimension`、`primary_metric` | 框架运行必需，两者须符合 [dimension_rubric.yaml](attribution_framework/valueqa_rules/dimension_rubric.yaml) 中的对应关系；单LLM提示词仅使用上面的四项文本 |

参考标注文件同样每行一个 JSON 对象，包含 `sample_id`、`label` 和 `annotations`。两份文件的样本编号和等级必须一致。每条片段标注包含 `start`、`end`、`text`，采用从 0 开始、左闭右开的字符位置，须满足 `answer[start:end] == text`；无片段时使用空数组。

真实等级和人工片段标注仅用于本地评分，不进入模型提示词。

## 6. 代码位置与离线检查

- [单LLM代码与提示词](attribution_framework/single_llm.py)
- [框架执行入口](attribution_framework/valueqa_integration.py)
- [API／本地服务选择](uni_eval/models/valueqa_service.py)
- [集成测试](tests/test_valueqa_integration.py)

```powershell
python -m unittest discover -s tests -p test_valueqa_integration.py -v
```

测试使用模拟响应和本地测试服务，不消耗真实模型 API。项目中其他 DeepSafe 任务继续使用各自配置；本说明针对上述两项 ValueQA 任务。

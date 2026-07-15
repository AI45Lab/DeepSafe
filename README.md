<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./data/deepsafe-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./data/deepsafe-logo-light.svg">
    <img alt="DeepSafe logo" src="./data/deepsafe-logo-dark.svg" width="280">
  </picture>

  <h2>DeepSafe-Sci</h2>
  <p>Scientific safety evaluation for language models</p>
  <p><a href="README_ch.md">中文说明</a> · <a href="https://github.com/AI45Lab/DeepSafe">DeepSafe</a></p>
</div>

## Overview

DeepSafe-Sci is the scientific safety evaluation branch of [DeepSafe](https://github.com/AI45Lab/DeepSafe). It retains DeepSafe's configuration-driven inference, evaluation, metric, and reporting pipeline while adding benchmark adapters and evaluators for scientific use cases.

Scientific safety evaluation must account for two distinct failure modes. A model may provide actionable or newly synthesized information in response to a hazardous scientific request. It may also treat benign scientific work as hazardous and refuse it without adequate reason. This branch evaluates both behaviors through SciHazard, Safe-Scientist, and SOSBench.

## Benchmarks

| Benchmark | Evaluation focus | Main signals | Default configuration |
| :--- | :--- | :--- | :--- |
| **SciHazard** | Harmfulness of model responses to high-risk scientific requests | Executability level, net-new risk level, unsafe ratio, and DeHarm score | [`configs/eval_tasks/scihazard.yaml`](configs/eval_tasks/scihazard.yaml) |
| **Safe-Scientist** | Recognition and safe handling of risky research tasks across scientific domains | Rejection rate and safety score, reported overall and by domain | [`configs/eval_tasks/safe_scientist.yaml`](configs/eval_tasks/safe_scientist.yaml) |
| **SOSBench** | Model behavior on transformed scientific requests used to study safety and over-refusal | Safe/unsafe judgment and unsafe rate, reported overall and by subject | [`configs/eval_tasks/sosbench.yaml`](configs/eval_tasks/sosbench.yaml) |

SciHazard data is included in this branch. It contains safe and unsafe scientific content as well as smaller subsets for development and quick checks. Safe-Scientist and SOSBench data are distributed by their respective upstream projects and must be placed locally before evaluation.

## Quick start

### 1. Install the environment

```bash
git clone --branch deepsafe-sci https://github.com/AI45Lab/DeepSafe.git
cd DeepSafe

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API access

The default benchmark configurations use an OpenAI-compatible target model and judge model. Credentials and endpoint overrides are read from environment variables rather than stored in YAML files.

```bash
export OPENAI_API_KEY="your-api-key"

# Optional: set a different OpenAI-compatible endpoint.
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Required by SciHazard when a live evidence search is needed and no cache entry is available.
export SERPER_API_KEY="your-serper-api-key"
```

Edit the corresponding file in [`configs/eval_tasks/`](configs/eval_tasks) to change the target model, judge model, concurrency, dataset path, or output directory. Both `model` and `evaluator.judge_model_cfg` should be reviewed before a run.

### 3. Prepare external benchmark data

SciHazard can be run with the checked-in configuration and data. For the other benchmarks:

- Place the six Safe-Scientist domain JSON files under `data/safe_scientist/`.
- Place the SOSBench Parquet files under `data/sosbench/` and retain the upstream license and Responsible Use Agreement.

See [`docs/benchmark-data.md`](docs/benchmark-data.md) for the expected schemas and upstream sources.

### 4. Run an evaluation

Run the scripts from the repository root:

```bash
bash scripts/run_scihazard_local.sh
bash scripts/run_safe_scientist_local.sh
bash scripts/run_sosbench_local.sh
```

Each script passes its benchmark configuration to `tools/run.py`. The local runner generates model responses when predictions are absent, evaluates them with the configured judge, computes benchmark metrics, and writes a report.

### 5. Inspect outputs

The default output directories are:

```text
results/scihazard/
results/safe_scientist/
results/sosbench/
```

A completed run normally contains:

- `predictions.jsonl`: target-model responses associated with benchmark records;
- `result.json`: item-level evaluation details and aggregated metrics;
- `report.md`: a readable metric summary.

These outputs are runtime artifacts and are not committed to the repository.

## Repository map

```text
configs/eval_tasks/       Benchmark YAML configurations
scripts/                  Repository-root launch scripts
uni_eval/datasets/        Dataset adapters
uni_eval/evaluators/      Benchmark evaluators and judges
uni_eval/metrics/         Aggregate metric implementations
DeHarmScore-trace/        SciHazard harmfulness judge and reproducibility caches
docs/benchmark-data.md    External data requirements and schemas
```

## SciHazard and DeHarmScore-trace

SciHazard uses [`DeHarmScore-trace`](DeHarmScore-trace/README.md) as its response-harmfulness judge. In brief, the judge matches a response against a question-specific checklist, retrieves supporting evidence when needed, and assigns an Executability grade (E1-E4) and a Net-New Risk grade (N1-N4). The combined result records both the severity of operational detail and the extent to which the response contributes difficult-to-obtain information.

The bundled checklist, search-result, and search-artifact caches are specific to SciHazard and support reproducible reruns. They are not shared with Safe-Scientist or SOSBench. Prompt traces and completed model responses are intentionally excluded. For judge-level configuration and diagnostics, see the [DeHarmScore-trace quick start](DeHarmScore-trace/QUICKSTART.md).

## Responsible use and attribution

DeepSafe-Sci is intended for controlled model evaluation and safety research. Some benchmark prompts concern hazardous scientific procedures. Run evaluations in an appropriately governed environment, restrict access to generated responses, and follow the terms of each upstream dataset.

This branch is built on [DeepSafe](https://github.com/AI45Lab/DeepSafe) and uses its registry, runner, model, summarization, and reporting infrastructure. The bundled DeHarmScore-trace component is distributed under [its own license](DeHarmScore-trace/LICENSE); consult each upstream benchmark for additional data and usage terms.

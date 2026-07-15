# Scientific Benchmark Data

DeepSafe exposes four science-focused benchmarks. Only SciHazard inputs and its reproducibility caches are included in this branch. The other datasets must be obtained from their upstream publishers under the applicable terms.

| Benchmark | Repository path | Distribution in this branch | Source |
| :--- | :--- | :--- | :--- |
| SciHazard | `DeHarmScore-trace/dataset/` | Included with checklist, search-cache, and search-artifact caches | This release |
| Safe-Scientist | `data/safe_scientist/` | Not included | [SafeScientist](https://github.com/ulab-uiuc/SafeScientist) |
| SOSBench | `data/sosbench/` | Not included | [SOSBench](https://huggingface.co/datasets/SOSBench/SOSBench) |
| WMDP | Existing DeepSafe-configured path | Not duplicated | [WMDP](https://www.wmdp.ai/) |

## SciHazard

The `SciHazardDataset` loader supports the bundled full safe/unsafe JSONL files and the bundled subset files. The default `configs/eval_tasks/scihazard.yaml` uses `scihazard_subset.jsonl` and reuses these checked-in cache trees:

- `DeHarmScore-trace/.checklist_cache/`
- `DeHarmScore-trace/.search_cache/`
- `DeHarmScore-trace/.search_artifacts/`

Prompt traces and model responses are runtime artifacts and are intentionally excluded.

## Safe-Scientist

Obtain the dataset from the upstream SafeScientist project and place the six domain files below `data/safe_scientist/`:

```text
bio.json
chem.json
is.json
material.json
med.json
phy.json
```

The loader expects each file to contain a JSON list with the upstream fields, including `Prompt`, `Task`, `Task Description`, `Risk Type`, and `sourceUrl`.

## SOSBench

Obtain SOSBench directly from its upstream Hugging Face repository and retain its README, license, and Responsible Use Agreement. Place one or more parquet files under `data/sosbench/`. The loader expects the columns `goal`, `original_term`, and `subject`.

SOSBench data is deliberately not committed here. Users are responsible for complying with its access, use, and redistribution terms.

## WMDP

This extension reuses DeepSafe's existing WMDP loader, metric, configuration, and `scripts/run_wmdp_local.sh`. It does not add another WMDP copy or change the existing workflow.

from __future__ import annotations


def test_science_datasets_are_exported_and_registered() -> None:
    import uni_eval.datasets  # noqa: F401
    from uni_eval.registry import DATASETS

    assert {
        "SciHazardDataset",
        "SafeScientistDataset",
        "SOSBenchDataset",
    }.issubset(DATASETS._module_dict)


def test_science_evaluators_are_exported_and_registered() -> None:
    import uni_eval.evaluators  # noqa: F401
    from uni_eval.registry import EVALUATORS

    assert {
        "SciHazardEvaluator",
        "SafeScientistEvaluator",
        "SOSBenchEvaluator",
    }.issubset(EVALUATORS._module_dict)


def test_science_metrics_are_exported_and_registered() -> None:
    import uni_eval.metrics  # noqa: F401
    from uni_eval.registry import METRICS

    assert {
        "SciHazardMetric",
        "SafeScientistMetric",
        "DeeperHarmMetric",
        "SOSBenchMetric",
    }.issubset(METRICS._module_dict)


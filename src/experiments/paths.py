from __future__ import annotations

from pathlib import Path


def experiment_dir(
    config: dict,
) -> Path:
    root = Path(
        config.get(
            "paths",
            {},
        ).get(
            "results_root",
            "results",
        )
    )

    experiment = config.get(
        "experiment",
        {},
    )

    category = experiment.get(
        "category",
        "main",
    )

    name = experiment.get(
        "name",
        config.get(
            "model",
            {},
        ).get(
            "target",
            "experiment",
        ),
    )

    benchmark = config.get(
        "benchmark",
        {},
    ).get("name")

    path = (
        root
        / category
    )

    if benchmark:
        path = (
            path
            / benchmark
        )

    return (
        path
        / name
    )


def case_result_dir(
    config: dict,
    granularity: str,
    dataset: str,
) -> Path:
    return (
        experiment_dir(config)
        / granularity
        / dataset
    )


def summary_path(
    config: dict,
    granularity: str,
) -> Path:
    return (
        experiment_dir(config)
        / f"summary_{granularity}.json"
    )
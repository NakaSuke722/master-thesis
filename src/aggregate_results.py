# src/aggregate_results.py

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.config import (
    load_config,
    resolve_granularity,
)
from experiments.paths import (
    experiment_dir,
    summary_path,
)


def aggregate_config(
    config: dict,
    granularity: str,
    total_time: float = 0.0,
) -> dict:
    """Return one experiment's summary without writing it to disk."""
    target_datasets = config.get("datasets", [])
    model_used = config["model"]["target"]

    experiment = config.get("experiment", {})
    experiment_category = experiment.get(
        "category",
        "main",
    )
    experiment_name = experiment.get(
        "name",
        model_used,
    )

    results_root = (
        experiment_dir(config)
        / granularity
    )

    all_results = []
    pure_execution_time = 0.0

    for dataset in target_datasets:
        dataset_dir = results_root / dataset

        if not dataset_dir.is_dir():
            continue

        for filepath in sorted(
            dataset_dir.glob("*.json")
        ):
            with filepath.open(
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            if (
                data.get("evaluation_granularity")
                != granularity
            ):
                continue

            if data.get("model_used") != model_used:
                continue

            if (
                data.get("experiment_category")
                != experiment_category
            ):
                continue

            if (
                data.get("experiment_name")
                != experiment_name
            ):
                continue

            all_results.append(data)

            pure_execution_time += data.get(
                "execution_time_sec",
                0.0,
            )

    if not all_results:
        raise FileNotFoundError(
            f"No result files found under "
            f"{results_root}"
        )

    dataset_metrics = defaultdict(
        lambda: defaultdict(list)
    )

    for result in all_results:
        for metric_name, value in (
            result["metrics"].items()
        ):
            dataset_metrics[
                result["dataset"]
            ][metric_name].append(value)

    summary = {
        dataset: {
            metric_name: round(
                sum(values) / len(values),
                4,
            )
            for metric_name, values
            in metrics.items()
        }
        for dataset, metrics
        in dataset_metrics.items()
    }

    final_time = (
        total_time
        if total_time > 0
        else pure_execution_time
    )

    return {
        "experiment_category": experiment_category,
        "experiment_name": experiment_name,
        "model_used": model_used,
        "evaluation_granularity": granularity,
        "total_execution_time_sec": round(final_time, 1),
        "pure_python_execution_time_sec": round(
            pure_execution_time,
            2,
        ),
        "summary": summary,
        "details": all_results,
    }


def write_summary(output_file, summary: dict) -> None:

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )


def recorded_total_time(
    config: dict,
    granularity: str,
) -> float:
    """Read the wall-clock time written by the preceding single run."""
    filepath = summary_path(config, granularity)

    if not filepath.is_file():
        return 0.0

    with filepath.open("r", encoding="utf-8") as f:
        previous_summary = json.load(f)

    return float(
        previous_summary.get(
            "total_execution_time_sec",
            0.0,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate RCA experiment results"
    )

    parser.add_argument(
        "--config",
        default="configs/amber.yaml",
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        help="Aggregate several experiment configs into one ablation table.",
    )

    parser.add_argument(
        "--granularity",
        choices=["service", "metric"],
        default=None,
    )

    parser.add_argument(
        "--total-time",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    config_paths = args.configs or [args.config]
    configs = [load_config(path) for path in config_paths]
    granularities = [
        resolve_granularity(config, args.granularity)
        for config in configs
    ]

    if len(set(granularities)) != 1:
        raise ValueError("All configs must use the same granularity.")

    granularity = granularities[0]
    summaries = {}

    for config in configs:
        total_time = (
            args.total_time
            if len(configs) == 1
            else recorded_total_time(
                config,
                granularity,
            )
        )
        summary = aggregate_config(
            config,
            granularity,
            total_time,
        )
        summaries[summary["experiment_name"]] = summary

    if len(configs) == 1:
        output_file = summary_path(configs[0], granularity)
        output = next(iter(summaries.values()))
        print(
            "\n===== Evaluation Summary "
            f"(Model: {output['model_used']}, "
            f"Granularity: {granularity}) ====="
        )
        print(json.dumps(output["summary"], indent=4, ensure_ascii=False))
    else:
        category = configs[0].get("experiment", {}).get(
            "category", "ablation"
        )
        output_file = (
            Path(
                configs[0].get("paths", {}).get(
                    "results_root", "results"
                )
            )
            / category
            / f"summary_{granularity}.json"
        )
        table_summary = {
            name: {
                "total_execution_time_sec": summary[
                    "total_execution_time_sec"
                ],
                "pure_python_execution_time_sec": summary[
                    "pure_python_execution_time_sec"
                ],
                "datasets": summary["summary"],
            }
            for name, summary in summaries.items()
        }
        output = {
            "experiment_category": category,
            "evaluation_granularity": granularity,
            "total_execution_time_sec": round(
                sum(
                    summary["total_execution_time_sec"]
                    for summary in summaries.values()
                ),
                1,
            ),
            "pure_python_execution_time_sec": round(
                sum(
                    summary["pure_python_execution_time_sec"]
                    for summary in summaries.values()
                ),
                2,
            ),
            "summary": table_summary,
            "variants": summaries,
        }
        print(
            "\n===== Ablation Evaluation Summary "
            f"(Granularity: {granularity}) ====="
        )
        print(json.dumps(table_summary, indent=4, ensure_ascii=False))

    write_summary(output_file, output)


if __name__ == "__main__":
    main()

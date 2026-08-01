# src/aggregate_results.py
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-time", type=float, default=0.0)
    args = parser.parse_args()

    with open("configs/default_params.yaml", "r") as f:
        config = yaml.safe_load(f)

    target_datasets = config.get("datasets", [])
    model_used = config["model"]["target"]
    granularity = os.environ.get(
        "RCA_GRANULARITY",
        config.get("evaluation", {}).get("granularity", "service"),
    ).lower()

    results_root = os.path.join(
        config["paths"]["results_dir"],
        model_used,
        granularity,
    )
    output_file = os.path.join(
        "results",
        f"final_summary_{model_used}_{granularity}.json",
    )

    all_results = []
    pure_execution_time = 0.0

    for dataset in target_datasets:
        pattern = os.path.join(results_root, dataset, "*.json")
        for filepath in sorted(glob.glob(pattern)):
            with open(filepath, "r") as f:
                data = json.load(f)

            # 異なる粒度の古い結果を混ぜない。
            if data.get("evaluation_granularity") != granularity:
                continue
            if data.get("model_used") != model_used:
                continue

            all_results.append(data)
            pure_execution_time += data.get("execution_time_sec", 0.0)

    if not all_results:
        raise FileNotFoundError(
            f"No result files found under {results_root}"
        )

    dataset_metrics = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        for metric_name, value in result["metrics"].items():
            dataset_metrics[result["dataset"]][metric_name].append(value)

    summary = {
        dataset: {
            metric_name: round(sum(values) / len(values), 4)
            for metric_name, values in metrics.items()
        }
        for dataset, metrics in dataset_metrics.items()
    }

    print(
        f"\n===== Evaluation Summary "
        f"(Model: {model_used}, Granularity: {granularity}) ====="
    )
    print(json.dumps(summary, indent=4))

    final_time = (
        args.total_time
        if args.total_time > 0
        else pure_execution_time
    )
    print(f"\nTotal Execution Time: {round(final_time, 1)} seconds")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(
            {
                "model_used": model_used,
                "evaluation_granularity": granularity,
                "total_execution_time_sec": round(final_time, 1),
                "pure_python_execution_time_sec": round(
                    pure_execution_time, 2
                ),
                "summary": summary,
                "details": all_results,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()

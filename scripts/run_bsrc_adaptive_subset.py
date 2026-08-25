"""Run the deterministic 3-dataset x 5-fault BSRC adaptive subset."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from aggregate_results import aggregate_config, write_summary
from data_loader import list_benchmark_processed_cases
from experiments.config import load_config, resolve_granularity
from experiments.paths import experiment_dir, summary_path
from runner import _run_benchmark_case


DEFAULT_CONFIG = Path("configs/sensitivity/bsrc_ar_adaptive_subset.yaml")


def select_cases(config: dict[str, Any]) -> list[Any]:
    protocol = config.get("subset_protocol", {})
    seed = int(protocol.get("selection_seed", 20260825))
    per_cell = int(protocol.get("cases_per_dataset_fault", 5))
    fault_types = [str(value) for value in protocol.get(
        "fault_types", ["cpu", "mem", "disk", "delay", "loss"]
    )]
    if per_cell <= 0:
        raise ValueError("cases_per_dataset_fault must be positive")
    selected = []
    for dataset_index, dataset in enumerate(config["datasets"]):
        cases = list_benchmark_processed_cases(
            benchmark=config["benchmark"]["name"],
            dataset=dataset,
            strategy=config["model"].get("preprocess_strategy", "default"),
            processed_root=config["paths"]["processed_data_dir"],
        )
        grouped: dict[str, list[Any]] = defaultdict(list)
        for case in cases:
            grouped[str(case.fault_type)].append(case)
        for fault_index, fault_type in enumerate(fault_types):
            candidates = sorted(
                grouped.get(fault_type, []), key=lambda case: case.case_id
            )
            if len(candidates) < per_cell:
                raise ValueError(
                    f"Need {per_cell} cases for {dataset}/{fault_type}; "
                    f"found {len(candidates)}"
                )
            rng = np.random.default_rng(
                seed + dataset_index * 100 + fault_index
            )
            indices = np.sort(rng.choice(
                len(candidates), size=per_cell, replace=False
            ))
            selected.extend(candidates[int(index)] for index in indices)
    return selected


def _write_manifest(config: dict[str, Any], cases: list[Any]) -> Path:
    path = experiment_dir(config) / "selection_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "subset_protocol": config["subset_protocol"],
        "number_of_cases": len(cases),
        "cases": [
            {
                "case_id": case.case_id,
                "dataset": case.dataset,
                "fault_type": case.fault_type,
                "root_cause_service": case.root_cause_service,
            }
            for case in cases
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def run_subset(
    config_path: Path,
    *,
    workers: int = 4,
) -> tuple[list[str], dict[str, object]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    config = load_config(config_path)
    granularity = resolve_granularity(config, "service")
    cases = select_cases(config)
    manifest_path = _write_manifest(config, cases)
    total = len(cases)
    tasks = [
        (config, str(config_path), granularity, case, index, total)
        for index, case in enumerate(cases, start=1)
    ]
    start = time.time()
    if workers == 1:
        outputs = [_run_benchmark_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(_run_benchmark_case, tasks))
    elapsed = time.time() - start
    summary = aggregate_config(config, granularity, total_time=elapsed)
    summary["subset_manifest"] = str(manifest_path)
    expected = len(cases)
    if summary["number_of_cases"] != expected:
        raise RuntimeError(
            "Subset result directory contains a different case count: "
            f"expected {expected}, aggregated {summary['number_of_cases']}"
        )
    write_summary(summary_path(config, granularity), summary)
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    _, summary = run_subset(args.config, workers=args.workers)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

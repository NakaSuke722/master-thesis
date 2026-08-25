"""Normal-only pseudo-fault calibration for adaptive-variance BSRC-AR."""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from models.amber import _service_name
from models.ar_bayes_factor import (
    ARBayesFactorPrior,
    ARRegimeShiftPrior,
    ar_shrinkage_regime_bayes_factor,
)


DEFAULT_CONFIG = Path(
    "configs/sensitivity/bsrc_ar_adaptive_pseudo_fault.yaml"
)


def load_protocol(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("analysis"), dict
    ):
        raise ValueError("Config must contain an analysis mapping")
    protocol = dict(document["analysis"])
    fractions = [float(value) for value in protocol["fit_fractions"]]
    if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("fit_fractions must lie between zero and one")
    if protocol.get("service_aggregation") != "mean_top3":
        raise ValueError("Only mean_top3 service aggregation is supported")
    ARBayesFactorPrior(**protocol["ar_bayes_prior"])
    ARRegimeShiftPrior(**protocol["ar_regime_shift_prior"])
    protocol["fit_fractions"] = fractions
    return protocol


def _case_directories(processed_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in processed_root.glob(
            "default/rcaeval_re1/*/*/normal_data.csv"
        )
    )


def _score_case(
    task: tuple[Path, dict[str, Any]],
) -> list[dict[str, object]]:
    case_dir, protocol = task
    normal = pd.read_csv(case_dir / "normal_data.csv")
    info = json.loads((case_dir / "case_info.json").read_text(encoding="utf-8"))
    order = int(protocol["ar_order"])
    prior = ARBayesFactorPrior(**protocol["ar_bayes_prior"])
    regime_prior = ARRegimeShiftPrior(**protocol["ar_regime_shift_prior"])
    threshold = float(protocol["strong_evidence_log_bf"])
    rows = []
    for fraction in protocol["fit_fractions"]:
        split = int(np.floor(len(normal) * fraction))
        if split <= order or len(normal) - split <= 0:
            raise ValueError(
                f"Normal window is too short: {info['case_id']} {fraction}"
            )
        fit = normal.iloc[:split]
        pseudo = normal.iloc[split:]
        service_scores: dict[str, list[float]] = {}
        variance_probabilities = []
        failures = 0
        for metric in fit.select_dtypes(include=[np.number]).columns:
            try:
                result = ar_shrinkage_regime_bayes_factor(
                    fit[metric].to_numpy(dtype=float),
                    pseudo[metric].to_numpy(dtype=float),
                    order=order,
                    prior=prior,
                    regime_prior=regime_prior,
                    posterior_detail="map",
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                failures += 1
                continue
            score = float(result["log_bayes_factor"])
            if not np.isfinite(score):
                failures += 1
                continue
            service_scores.setdefault(_service_name(metric), []).append(score)
            variance_probabilities.append(float(
                result["posterior_variance_change_probability"]
            ))
        if not service_scores:
            raise ValueError(f"No finite metric scores for {info['case_id']}")
        values = np.asarray([
            np.mean(np.sort(scores)[-min(3, len(scores)):])
            for scores in service_scores.values()
        ])
        rows.append({
            "case_id": info["case_id"],
            "dataset": info["dataset"],
            "fault_type": info.get("fault_type"),
            "fit_fraction": fraction,
            "fit_samples": split,
            "pseudo_samples": len(normal) - split,
            "n_services": values.size,
            "max_service_log_bf": float(np.max(values)),
            "median_service_log_bf": float(np.median(values)),
            "positive_service_fraction": float(np.mean(values > 0.0)),
            "strong_service_fraction": float(np.mean(values > threshold)),
            "mean_variance_change_probability": float(np.mean(
                variance_probabilities
            )),
            "metric_failures": failures,
        })
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summaries = []
    groups: list[tuple[str, str, list[dict[str, object]]]] = []
    for fraction in sorted({float(row["fit_fraction"]) for row in rows}):
        selected = [row for row in rows if row["fit_fraction"] == fraction]
        groups.append(("overall", "all", selected))
        for dataset in sorted({str(row["dataset"]) for row in selected}):
            groups.append((
                "dataset",
                dataset,
                [row for row in selected if row["dataset"] == dataset],
            ))
    for scope, value, group in groups:
        def array(field: str) -> np.ndarray:
            return np.asarray([row[field] for row in group], dtype=float)
        summaries.append({
            "scope": scope,
            "value": value,
            "fit_fraction": float(group[0]["fit_fraction"]),
            "n_cases": len(group),
            "median_case_max_service_log_bf": float(np.median(
                array("max_service_log_bf")
            )),
            "p90_case_max_service_log_bf": float(np.quantile(
                array("max_service_log_bf"), 0.9
            )),
            "mean_positive_service_fraction": float(np.mean(
                array("positive_service_fraction")
            )),
            "mean_strong_service_fraction": float(np.mean(
                array("strong_service_fraction")
            )),
            "mean_variance_change_probability": float(np.mean(
                array("mean_variance_change_probability")
            )),
            "total_metric_failures": int(sum(
                int(row["metric_failures"]) for row in group
            )),
        })
    return {
        "schema_version": 1,
        "purpose": "adaptive_bsrc_normal_only_pseudo_fault",
        "n_cases": len({str(row["case_id"]) for row in rows}),
        "n_case_conditions": len(rows),
        "summaries": summaries,
    }


def write_outputs(
    output_root: Path,
    rows: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "case_calibration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Adaptive BSRC-AR normal-only pseudo-fault",
        "",
        "正常window内部の分割で、偽のregime change evidenceを測る。",
        "",
        "| Scope | Split | Cases | Median case max log BF | P90 | "
        "Positive services | Strong services |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["summaries"]:
        label = item["value"] if item["scope"] != "overall" else "overall"
        lines.append(
            f"| {label} | {item['fit_fraction']:.1f} | {item['n_cases']} | "
            f"{item['median_case_max_service_log_bf']:.4f} | "
            f"{item['p90_case_max_service_log_bf']:.4f} | "
            f"{item['mean_positive_service_fraction']:.1%} | "
            f"{item['mean_strong_service_fraction']:.1%} |"
        )
    (output_root / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/analysis/bsrc_variance_integration_pseudo_fault"
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    protocol = load_protocol(args.config)
    cases = _case_directories(Path(protocol["processed_root"]))
    if args.limit_cases is not None:
        cases = cases[:args.limit_cases]
    tasks = [(case, protocol) for case in cases]
    if args.workers == 1:
        nested = [_score_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            nested = list(executor.map(_score_case, tasks))
    rows = [row for group in nested for row in group]
    if not rows:
        raise FileNotFoundError("No processed RCAEval cases found")
    report = summarize(rows)
    report["protocol"] = protocol
    write_outputs(args.output_root, rows, report)
    print(args.output_root / "summary.md")


if __name__ == "__main__":
    main()

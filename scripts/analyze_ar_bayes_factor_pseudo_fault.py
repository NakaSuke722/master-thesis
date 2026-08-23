"""Normal-only pseudo-fault calibration for the direct AR Bayes Factor."""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from models.amber import _service_name
from models.ar_bayes_factor import ARBayesFactorPrior, ar_change_bayes_factor


DEFAULT_CONFIG = Path(
    "configs/sensitivity/direct_ar_bayes_factor_pseudo_fault.yaml"
)


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("analysis"), dict):
        raise ValueError("Pseudo-fault config must contain an analysis mapping")
    protocol = dict(document["analysis"])
    fractions = [float(value) for value in protocol.get("fit_fractions", [])]
    if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("fit_fractions must contain values between zero and one")
    if int(protocol.get("ar_order", -1)) < 0:
        raise ValueError("ar_order must be non-negative")
    if protocol.get("service_aggregation") != "mean_top3":
        raise ValueError("Only mean_top3 service aggregation is supported")
    profiles = protocol.get("prior_profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("prior_profiles must be a non-empty mapping")
    for name, parameters in profiles.items():
        try:
            ARBayesFactorPrior(**parameters)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid prior profile {name!r}") from exc
    protocol["fit_fractions"] = fractions
    protocol["ar_order"] = int(protocol["ar_order"])
    protocol["strong_evidence_log_bf"] = float(
        protocol.get("strong_evidence_log_bf", np.log(10.0))
    )
    return protocol


def _case_directories(processed_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in processed_root.glob(
            "default/rcaeval_re1/*/*/normal_data.csv"
        )
    )


def _case_metadata(case_dir: Path) -> dict[str, Any]:
    with (case_dir / "case_info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    return {
        "case_id": info["case_id"],
        "dataset": info["dataset"],
    }


def _split_normal(
    normal: pd.DataFrame,
    fit_fraction: float,
    ar_order: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between zero and one")
    split = int(np.floor(len(normal) * fit_fraction))
    if split <= ar_order or len(normal) - split <= 0:
        raise ValueError(
            f"Normal window too short for split={fit_fraction}: {len(normal)}"
        )
    return (
        normal.iloc[:split].reset_index(drop=True),
        normal.iloc[split:].reset_index(drop=True),
    )


def _describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"median_{prefix}_log_bf": float(np.median(values)),
        f"p90_{prefix}_log_bf": float(np.quantile(values, 0.9)),
        f"max_{prefix}_log_bf": float(np.max(values)),
    }


def _score_case_condition(
    normal: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    fit_fraction: float,
    prior_name: str,
    prior: ARBayesFactorPrior,
    ar_order: int,
    strong_threshold: float,
) -> dict[str, Any]:
    fit, pseudo = _split_normal(normal, fit_fraction, ar_order)
    numeric_columns = [
        column for column in fit.columns
        if pd.api.types.is_numeric_dtype(fit[column])
    ]
    metric_scores: list[float] = []
    service_metrics: dict[str, list[float]] = {}
    pre_unstable = 0
    post_unstable = 0
    shared_unstable = 0
    failures = 0
    for metric in numeric_columns:
        try:
            result = ar_change_bayes_factor(
                fit[metric].to_numpy(dtype=float),
                pseudo[metric].to_numpy(dtype=float),
                order=ar_order,
                prior=prior,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            failures += 1
            continue
        score = float(result["log_bayes_factor"])
        if not np.isfinite(score):
            failures += 1
            continue
        metric_scores.append(score)
        service_metrics.setdefault(_service_name(metric), []).append(score)
        pre_unstable += int(
            result["posterior_pre"]["spectral_radius_at_mean"] >= 1.0
        )
        post_unstable += int(
            result["posterior_post"]["spectral_radius_at_mean"] >= 1.0
        )
        shared_unstable += int(
            result["posterior_shared"]["spectral_radius_at_mean"] >= 1.0
        )

    if not metric_scores or not service_metrics:
        raise ValueError(
            f"No finite metric scores for {metadata['case_id']} "
            f"at split={fit_fraction}, prior={prior_name}"
        )
    metric_values = np.asarray(metric_scores, dtype=float)
    service_names = sorted(service_metrics)
    service_values = np.asarray([
        float(np.mean(np.sort(service_metrics[service])[-3:]))
        for service in service_names
    ])
    top_index = int(np.argmax(service_values))
    n_scored = metric_values.size
    return {
        **metadata,
        "fit_fraction": fit_fraction,
        "fit_samples": len(fit),
        "pseudo_samples": len(pseudo),
        "prior_name": prior_name,
        "ar_order": ar_order,
        "n_metrics": len(numeric_columns),
        "n_scored_metrics": n_scored,
        "metric_failures": failures,
        **_describe(metric_values, "metric"),
        "positive_metric_fraction": float(np.mean(metric_values > 0.0)),
        "strong_metric_fraction": float(
            np.mean(metric_values > strong_threshold)
        ),
        "n_services": service_values.size,
        **_describe(service_values, "service"),
        "positive_service_fraction": float(np.mean(service_values > 0.0)),
        "strong_service_fraction": float(
            np.mean(service_values > strong_threshold)
        ),
        "top_service": service_names[top_index],
        "pre_unstable_metric_fraction": pre_unstable / n_scored,
        "post_unstable_metric_fraction": post_unstable / n_scored,
        "shared_unstable_metric_fraction": shared_unstable / n_scored,
    }


def _score_case(
    task: tuple[
        Path,
        tuple[float, ...],
        tuple[tuple[str, dict[str, float]], ...],
        int,
        float,
    ],
) -> list[dict[str, Any]]:
    case_dir, fractions, profile_items, ar_order, strong_threshold = task
    normal = pd.read_csv(case_dir / "normal_data.csv")
    metadata = _case_metadata(case_dir)
    rows = []
    for fit_fraction in fractions:
        for prior_name, parameters in profile_items:
            rows.append(_score_case_condition(
                normal,
                metadata,
                fit_fraction=fit_fraction,
                prior_name=prior_name,
                prior=ARBayesFactorPrior(**parameters),
                ar_order=ar_order,
                strong_threshold=strong_threshold,
            ))
    return rows


def _summarize_group(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    value: str,
    fit_fraction: float,
    prior_name: str,
) -> dict[str, Any]:
    def values(field: str) -> np.ndarray:
        return np.asarray([row[field] for row in rows], dtype=float)

    return {
        "scope": scope,
        "value": value,
        "fit_fraction": fit_fraction,
        "prior_name": prior_name,
        "n_cases": len(rows),
        "median_case_max_service_log_bf": float(np.median(
            values("max_service_log_bf")
        )),
        "p90_case_max_service_log_bf": float(np.quantile(
            values("max_service_log_bf"), 0.9
        )),
        "median_case_median_service_log_bf": float(np.median(
            values("median_service_log_bf")
        )),
        "mean_positive_service_fraction": float(np.mean(
            values("positive_service_fraction")
        )),
        "mean_strong_service_fraction": float(np.mean(
            values("strong_service_fraction")
        )),
        "mean_positive_metric_fraction": float(np.mean(
            values("positive_metric_fraction")
        )),
        "mean_strong_metric_fraction": float(np.mean(
            values("strong_metric_fraction")
        )),
        "mean_post_unstable_metric_fraction": float(np.mean(
            values("post_unstable_metric_fraction")
        )),
        "total_metric_failures": int(sum(
            int(row["metric_failures"]) for row in rows
        )),
    }


def _summaries(
    rows: list[dict[str, Any]],
    fractions: Iterable[float],
    prior_names: Iterable[str],
) -> list[dict[str, Any]]:
    summaries = []
    scopes = [("overall", "overall", rows)] + [
        (
            "dataset",
            dataset,
            [row for row in rows if row["dataset"] == dataset],
        )
        for dataset in sorted({row["dataset"] for row in rows})
    ]
    for scope, value, scoped_rows in scopes:
        for fit_fraction in fractions:
            for prior_name in prior_names:
                subset = [
                    row for row in scoped_rows
                    if row["fit_fraction"] == fit_fraction
                    and row["prior_name"] == prior_name
                ]
                if subset:
                    summaries.append(_summarize_group(
                        subset,
                        scope=scope,
                        value=value,
                        fit_fraction=fit_fraction,
                        prior_name=prior_name,
                    ))
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Direct AR Bayes Factor normal-only pseudo-fault calibration",
        "",
        (
            f"Cases: {report['n_cases']}; AR order: {report['ar_order']}; "
            f"strong evidence threshold: {report['strong_evidence_log_bf']:.4f}."
        ),
        "",
        "No injected fault is present. Lower false structural-change evidence is better.",
        "",
        "| Scope | Fit fraction | Prior | Cases | Median case max service log BF | P90 case max | Median service log BF | Positive services | Strong services | Positive metrics | Strong metrics | Post unstable | Failures |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["summaries"]:
        lines.append(
            f"| {item['value']} | {item['fit_fraction']:.1f} | "
            f"{item['prior_name']} | {item['n_cases']} | "
            f"{item['median_case_max_service_log_bf']:.4f} | "
            f"{item['p90_case_max_service_log_bf']:.4f} | "
            f"{item['median_case_median_service_log_bf']:.4f} | "
            f"{item['mean_positive_service_fraction']:.1%} | "
            f"{item['mean_strong_service_fraction']:.1%} | "
            f"{item['mean_positive_metric_fraction']:.1%} | "
            f"{item['mean_strong_metric_fraction']:.1%} | "
            f"{item['mean_post_unstable_metric_fraction']:.1%} | "
            f"{item['total_metric_failures']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    processed_root: Path,
    output_root: Path,
    *,
    fit_fractions: tuple[float, ...],
    prior_profiles: dict[str, dict[str, float]],
    ar_order: int,
    strong_threshold: float,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    case_dirs = _case_directories(processed_root)
    if limit is not None:
        case_dirs = case_dirs[:limit]
    if not case_dirs:
        raise ValueError(f"No processed RCAEval cases found under {processed_root}")
    profile_items = tuple(
        (name, dict(parameters)) for name, parameters in prior_profiles.items()
    )
    tasks = [
        (case_dir, fit_fractions, profile_items, ar_order, strong_threshold)
        for case_dir in case_dirs
    ]
    if workers == 1:
        nested_rows = [_score_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested_rows = list(executor.map(_score_case, tasks))
    rows = [row for case_rows in nested_rows for row in case_rows]
    prior_names = tuple(prior_profiles)
    report = {
        "schema_version": 1,
        "analysis": "direct_ar_bayes_factor_pseudo_fault",
        "n_cases": len(case_dirs),
        "fit_fractions": list(fit_fractions),
        "prior_profiles": prior_profiles,
        "ar_order": ar_order,
        "strong_evidence_log_bf": strong_threshold,
        "summaries": _summaries(rows, fit_fractions, prior_names),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "case_calibration.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_root / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fit-fractions", nargs="+", type=float)
    parser.add_argument("--priors", nargs="+")
    parser.add_argument("--ar-order", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    protocol = load_protocol(args.config)
    profiles = dict(protocol["prior_profiles"])
    if args.priors is not None:
        unknown = sorted(set(args.priors) - set(profiles))
        if unknown:
            raise ValueError(f"Unknown prior profiles: {unknown}")
        profiles = {name: profiles[name] for name in args.priors}
    fractions = tuple(
        args.fit_fractions
        if args.fit_fractions is not None
        else protocol["fit_fractions"]
    )
    processed_root = args.processed_root or Path(protocol["processed_root"])
    output_root = args.output_root or Path(protocol["output_root"])
    report = analyze(
        processed_root,
        output_root,
        fit_fractions=fractions,
        prior_profiles=profiles,
        ar_order=(args.ar_order if args.ar_order is not None else protocol["ar_order"]),
        strong_threshold=protocol["strong_evidence_log_bf"],
        limit=args.limit,
        workers=args.workers,
    )
    print(f"Scored {report['n_cases']} normal-only cases; wrote {output_root}")


if __name__ == "__main__":
    main()

"""Normal-only pseudo-fault calibration for AMBER residualization variants.

For every processed RCAEval case, the saved normal window is split in temporal
order.  The first part is used as training-normal data and the held-out tail is
passed as a pseudo-abnormal period even though no injected fault is present.
Large Bayes factors therefore reveal forecast/calibration error rather than a
real fault response.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from models.amber import AMBER, NIG


DEFAULT_MODES = ("raw", "ar", "counterfactual_ar")

MODE_OVERRIDES: dict[str, dict[str, Any]] = {
    "raw": {"residualization": "raw"},
    "ar": {"residualization": "ar"},
    "counterfactual_ar": {"residualization": "counterfactual_ar"},
    "stationary_ar": {
        "residualization": "ar",
        "ar_stationarity": "root_projection",
        "stationarity_radius": 0.98,
        "counterfactual_bounds": "none",
    },
    "stationary_counterfactual_ar": {
        "residualization": "counterfactual_ar",
        "ar_stationarity": "root_projection",
        "stationarity_radius": 0.98,
        "counterfactual_bounds": "none",
    },
    "stationary_counterfactual_ar_uncertainty": {
        "residualization": "counterfactual_ar",
        "ar_stationarity": "root_projection",
        "stationarity_radius": 0.98,
        "counterfactual_bounds": "none",
        "horizon_aware_uncertainty": True,
    },
    "stationary_counterfactual_ar_full_covariance": {
        "residualization": "counterfactual_ar",
        "ar_stationarity": "root_projection",
        "stationarity_radius": 0.98,
        "counterfactual_bounds": "none",
        "horizon_aware_uncertainty": True,
        "forecast_error_covariance": "full",
    },
}


def _case_directories(processed_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in processed_root.glob("default/rcaeval_re1/*/*/normal_data.csv")
    )


def _load_model_params(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return dict(config["model"]["params"])


def _build_model(params: dict[str, Any], mode: str) -> AMBER:
    if mode not in MODE_OVERRIDES:
        raise ValueError(f"Unknown calibration mode: {mode}")
    overrides = MODE_OVERRIDES[mode]
    prior_params = params.get("prior", {})
    return AMBER(
        ar_order=int(params.get("ar_order", 3)),
        ridge=float(params.get("ridge", 1e-3)),
        min_scale=float(params.get("min_scale", 1e-6)),
        relative_scale_floor=float(params.get("relative_scale_floor", 1e-3)),
        winsor_quantile=params.get("winsor_quantile"),
        aggregate="service",
        service_aggregation="mean_top3",
        prior=NIG(
            m=float(prior_params.get("m", 0.0)),
            kappa=float(prior_params.get("kappa", 1e-3)),
            alpha=float(prior_params.get("alpha", 2.0)),
            beta=float(prior_params.get("beta", 1.0)),
        ),
        residualization=overrides["residualization"],
        scoring="bayes_factor",
        ar_input_scaling=(
            params.get("ar_input_scaling", "none")
            if overrides["residualization"] in {"ar", "counterfactual_ar"}
            else "none"
        ),
        ar_stationarity=overrides.get("ar_stationarity", "none"),
        stationarity_radius=float(overrides.get("stationarity_radius", 0.98)),
        counterfactual_bounds=overrides.get("counterfactual_bounds", "normal_range"),
        horizon_aware_uncertainty=bool(
            overrides.get("horizon_aware_uncertainty", False)
        ),
        forecast_error_covariance=overrides.get(
            "forecast_error_covariance", "diagonal"
        ),
    )


def _split_normal(normal: pd.DataFrame, fit_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between zero and one")
    split = int(np.floor(len(normal) * fit_fraction))
    if split <= 6 or len(normal) - split <= 3:
        raise ValueError(f"Normal window too short for pseudo-fault split: {len(normal)}")
    return normal.iloc[:split].reset_index(drop=True), normal.iloc[split:].reset_index(drop=True)


def _case_metadata(case_dir: Path) -> dict[str, Any]:
    info_path = case_dir / "case_info.json"
    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    return {
        "case_id": info["case_id"],
        "dataset": info["dataset"],
        "fault_type": info["fault_type"],
        "root_cause_service": info["root_cause_service"],
    }


def _score_case(
    case_dir: Path,
    params: dict[str, Any],
    modes: Iterable[str],
    fit_fraction: float,
) -> list[dict[str, Any]]:
    normal = pd.read_csv(case_dir / "normal_data.csv")
    fit, pseudo = _split_normal(normal, fit_fraction)
    metadata = _case_metadata(case_dir)
    rows = []
    for mode in modes:
        model = _build_model(params, mode)
        service_result = model.fit_predict(fit, pseudo)
        finite_scores = service_result["score"].dropna().to_numpy(dtype=float)
        metric_diagnostics = (model.diagnostics_ or {}).get("metrics") or []
        clipped_counts = [
            int(item.get("counterfactual_clipped_predictions") or 0)
            for item in metric_diagnostics
        ]
        clipped_fractions = [
            float(item.get("counterfactual_clipped_fraction") or 0.0)
            for item in metric_diagnostics
        ]
        rows.append({
            **metadata,
            "residualization": mode,
            "fit_samples": len(fit),
            "pseudo_abnormal_samples": len(pseudo),
            "n_metrics": len(metric_diagnostics),
            "n_services": len(service_result),
            "max_service_score": float(np.max(finite_scores)),
            "median_service_score": float(np.median(finite_scores)),
            "p90_service_score": float(np.quantile(finite_scores, 0.9)),
            "positive_service_fraction": float(np.mean(finite_scores > 0)),
            "top_service": str(service_result.iloc[0]["service"]),
            "metrics_with_any_clip": int(sum(count > 0 for count in clipped_counts)),
            "mean_metric_clip_fraction": float(np.mean(clipped_fractions)),
        })
    return rows


def _summarize(rows: list[dict[str, Any]], scope: str, value: str) -> dict[str, Any]:
    result: dict[str, Any] = {"scope": scope, "value": value}
    for mode in sorted({row["residualization"] for row in rows}):
        subset = [row for row in rows if row["residualization"] == mode]
        maxima = np.asarray([row["max_service_score"] for row in subset], dtype=float)
        medians = np.asarray([row["median_service_score"] for row in subset], dtype=float)
        result[mode] = {
            "n_cases": len(subset),
            "median_max_service_score": float(np.median(maxima)),
            "p90_max_service_score": float(np.quantile(maxima, 0.9)),
            "median_of_median_service_score": float(np.median(medians)),
            "mean_positive_service_fraction": float(np.mean([
                row["positive_service_fraction"] for row in subset
            ])),
            "cases_with_any_metric_clip": int(sum(
                row["metrics_with_any_clip"] > 0 for row in subset
            )),
            "mean_metric_clip_fraction": float(np.mean([
                row["mean_metric_clip_fraction"] for row in subset
            ])),
        }
    return result


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
        "# Normal-only pseudo-fault calibration",
        "",
        (
            f"The first {report['fit_fraction']:.0%} of each saved normal window is fitted; "
            "the remaining tail is scored as pseudo-abnormal. No injected fault is present."
        ),
        "",
        f"Cases: {report['n_cases']}",
        "",
        "| Scope | Mode | Cases | Median max BF | P90 max BF | Median service BF | Positive service fraction | Cases with clip | Mean metric clip fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report["summaries"]:
        scope = summary["value"]
        for mode, values in summary.items():
            if mode in {"scope", "value"}:
                continue
            lines.append(
                f"| {scope} | {mode} | {values['n_cases']} | "
                f"{values['median_max_service_score']:.4f} | "
                f"{values['p90_max_service_score']:.4f} | "
                f"{values['median_of_median_service_score']:.4f} | "
                f"{values['mean_positive_service_fraction']:.1%} | "
                f"{values['cases_with_any_metric_clip']} | "
                f"{values['mean_metric_clip_fraction']:.1%} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    processed_root: Path,
    config_path: Path,
    output_root: Path,
    *,
    modes: tuple[str, ...] = DEFAULT_MODES,
    fit_fraction: float = 0.5,
    limit: int | None = None,
) -> dict[str, Any]:
    params = _load_model_params(config_path)
    case_dirs = _case_directories(processed_root)
    if limit is not None:
        case_dirs = case_dirs[:limit]
    if not case_dirs:
        raise ValueError(f"No processed RCAEval cases found under {processed_root}")
    rows = [
        row
        for case_dir in case_dirs
        for row in _score_case(case_dir, params, modes, fit_fraction)
    ]
    summaries = [_summarize(rows, "overall", "overall")]
    for dataset in sorted({row["dataset"] for row in rows}):
        summaries.append(_summarize(
            [row for row in rows if row["dataset"] == dataset], "dataset", dataset
        ))
    report = {
        "n_cases": len(case_dirs),
        "fit_fraction": fit_fraction,
        "modes": list(modes),
        "summaries": summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "case_calibration.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(output_root / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root", type=Path,
        default=Path("data/processed/rcaeval_zenodo_v2"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/main/rcaeval_re1_zenodo_v2.yaml"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/analysis/ar_pseudo_fault_calibration"),
    )
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = analyze(
        args.processed_root,
        args.config,
        args.output_root,
        modes=tuple(args.modes),
        fit_fraction=args.fit_fraction,
        limit=args.limit,
    )
    print(f"Scored {report['n_cases']} normal-only cases; wrote {args.output_root}")


if __name__ == "__main__":
    main()

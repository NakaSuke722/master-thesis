"""Diagnose affine-unit sensitivity of AMBER service rankings.

Each RCAEval case is scored once with the saved metric values and again after
applying the same positive affine transformation ``y = scale * x + offset`` to
every metric in both the normal and abnormal windows.  Such a transformation
changes the numerical unit, not the underlying time-series event.  A unit-
invariant model should therefore preserve service scores and rankings.

This script is observational: it does not change AMBER's scoring logic or any
saved benchmark input.  Transformed frames exist only in memory.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from models.amber import AMBER, NIG


DEFAULT_CONFIG = Path(
    "configs/ablation/rcaeval_re1_zenodo_v2/"
    "stationary_counterfactual_ar_uncertainty_unit_invariant.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/analysis/ar_unit_invariance_normal_standard"
)


@dataclass(frozen=True)
class AffineTransform:
    name: str
    scale: float
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or self.name == "baseline":
            raise ValueError("transform name must be non-empty and not 'baseline'")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("transform scale must be finite and positive")
        if not np.isfinite(self.offset):
            raise ValueError("transform offset must be finite")


DEFAULT_TRANSFORMS = (
    AffineTransform("scale_up_1000", 1000.0, 0.0),
    AffineTransform("scale_down_0_001", 0.001, 0.0),
    AffineTransform("scale_down_0_001_offset_100", 0.001, 100.0),
)


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML mapping: {config_path}")
    return config


def _build_model(config: dict[str, Any]) -> AMBER:
    params = config["model"]["params"]
    prior_params = params.get("prior", {})
    service_method = config.get("evaluation", {}).get(
        "service_aggregation", {}
    ).get("method", "mean_top3")
    return AMBER(
        ar_order=int(params.get("ar_order", 3)),
        ridge=float(params.get("ridge", 1e-3)),
        min_scale=float(params.get("min_scale", 1e-6)),
        relative_scale_floor=float(params.get("relative_scale_floor", 1e-3)),
        winsor_quantile=params.get("winsor_quantile"),
        aggregate="service",
        service_aggregation=service_method,
        prior=NIG(
            m=float(prior_params.get("m", 0.0)),
            kappa=float(prior_params.get("kappa", 1e-3)),
            alpha=float(prior_params.get("alpha", 2.0)),
            beta=float(prior_params.get("beta", 1.0)),
        ),
        residualization=params.get("residualization", "ar"),
        scoring=params.get("scoring", "bayes_factor"),
        ar_input_scaling=params.get("ar_input_scaling", "none"),
        ar_stationarity=params.get("ar_stationarity", "none"),
        stationarity_radius=float(params.get("stationarity_radius", 0.98)),
        counterfactual_bounds=params.get("counterfactual_bounds", "normal_range"),
        horizon_aware_uncertainty=bool(
            params.get("horizon_aware_uncertainty", False)
        ),
        forecast_error_covariance=params.get(
            "forecast_error_covariance", "diagonal"
        ),
    )


def _case_directories(config: dict[str, Any], cases_per_dataset: int | None) -> list[Path]:
    if cases_per_dataset is not None and cases_per_dataset <= 0:
        raise ValueError("cases_per_dataset must be positive")
    root = Path(config["paths"]["processed_data_dir"])
    strategy = config["model"].get("preprocess_strategy", "default")
    benchmark = config["benchmark"]["name"]
    selected: list[Path] = []
    for dataset in config["datasets"]:
        cases = sorted(
            path.parent
            for path in (root / strategy / benchmark / dataset).glob(
                "*/normal_data.csv"
            )
        )
        if cases_per_dataset is not None:
            cases = cases[:cases_per_dataset]
        selected.extend(cases)
    return selected


def _apply_affine(frame: pd.DataFrame, transform: AffineTransform) -> pd.DataFrame:
    transformed = frame.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        transformed[column] = (
            transformed[column].astype(float) * transform.scale
            + transform.offset
        )
    return transformed


def _metric_diagnostics(model: AMBER) -> dict[str, dict[str, Any]]:
    diagnostics = model.diagnostics_ or {}
    return {
        str(row["metric"]): row
        for row in diagnostics.get("metrics", [])
    }


def _finite_max(values: Iterable[float], default: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else default


def _compare_metric_diagnostics(
    baseline: dict[str, dict[str, Any]],
    transformed: dict[str, dict[str, Any]],
    transform: AffineTransform,
) -> dict[str, Any]:
    if set(baseline) != set(transformed):
        raise ValueError("Metric sets differ between baseline and transformed run")

    lag_differences: list[float] = []
    intercept_relative_errors: list[float] = []
    scale_relative_errors: list[float] = []
    multiplier_differences: list[float] = []
    stationarity_flips = 0
    coefficient_pairs = 0

    for metric in sorted(baseline):
        base = baseline[metric]
        changed = transformed[metric]
        base_coef = np.asarray(base.get("ar_coefficients") or [], dtype=float)
        changed_coef = np.asarray(changed.get("ar_coefficients") or [], dtype=float)
        if base_coef.size and base_coef.shape == changed_coef.shape:
            coefficient_pairs += 1
            lag_differences.extend(np.abs(base_coef[1:] - changed_coef[1:]))
            if base.get("ar_input_scaling") == "normal_standard":
                expected_intercept = base_coef[0]
            else:
                expected_intercept = (
                    transform.scale * base_coef[0]
                    + transform.offset * (1.0 - float(np.sum(base_coef[1:])))
                )
            intercept_relative_errors.append(
                abs(changed_coef[0] - expected_intercept)
                / max(1.0, abs(expected_intercept))
            )

        base_scale = float(base.get("normal_scale") or np.nan)
        changed_scale = float(changed.get("normal_scale") or np.nan)
        expected_scale = (
            base_scale
            if base.get("ar_input_scaling") == "normal_standard"
            else transform.scale * base_scale
        )
        if np.isfinite(expected_scale) and expected_scale > 0 and np.isfinite(changed_scale):
            scale_relative_errors.append(abs(changed_scale - expected_scale) / expected_scale)

        base_multiplier = np.asarray(
            base.get("forecast_uncertainty_multiplier") or [], dtype=float
        )
        changed_multiplier = np.asarray(
            changed.get("forecast_uncertainty_multiplier") or [], dtype=float
        )
        if base_multiplier.size and base_multiplier.shape == changed_multiplier.shape:
            multiplier_differences.extend(np.abs(base_multiplier - changed_multiplier))

        stationarity_flips += int(
            base.get("ar_stationarity_constrained")
            != changed.get("ar_stationarity_constrained")
        )

    max_lag_difference = _finite_max(lag_differences)
    max_intercept_error = _finite_max(intercept_relative_errors)
    max_scale_error = _finite_max(scale_relative_errors)
    max_multiplier_difference = _finite_max(multiplier_differences)
    return {
        "matched_metric_count": len(baseline),
        "coefficient_pair_count": coefficient_pairs,
        "max_abs_lag_coefficient_diff": max_lag_difference,
        "lag_coefficients_allclose": bool(max_lag_difference <= 1e-6),
        "max_relative_intercept_equivariance_error": max_intercept_error,
        "max_relative_normal_scale_error": max_scale_error,
        "max_abs_forecast_multiplier_diff": max_multiplier_difference,
        "forecast_multipliers_allclose": bool(max_multiplier_difference <= 1e-6),
        "stationarity_constraint_flip_count": stationarity_flips,
    }


def _compare_results(
    baseline_result: pd.DataFrame,
    transformed_result: pd.DataFrame,
    root_service: str,
    transform: AffineTransform,
    baseline_diagnostics: dict[str, dict[str, Any]],
    transformed_diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_ranking = baseline_result["service"].astype(str).tolist()
    transformed_ranking = transformed_result["service"].astype(str).tolist()
    if set(baseline_ranking) != set(transformed_ranking):
        raise ValueError("Service sets differ between baseline and transformed run")
    if root_service not in baseline_ranking:
        raise ValueError(f"Root service is absent from AMBER ranking: {root_service}")

    baseline_ranks = {
        service: rank for rank, service in enumerate(baseline_ranking, start=1)
    }
    transformed_ranks = {
        service: rank for rank, service in enumerate(transformed_ranking, start=1)
    }
    baseline_scores = baseline_result.set_index("service")["score"].astype(float)
    transformed_scores = transformed_result.set_index("service")["score"].astype(float)
    changed_scores = transformed_scores.reindex(baseline_scores.index).to_numpy()
    original_scores = baseline_scores.to_numpy()
    score_difference = np.abs(changed_scores - original_scores)
    score_relative_difference = score_difference / np.maximum(
        1.0, np.abs(original_scores)
    )
    service_displacements = [
        abs(baseline_ranks[service] - transformed_ranks[service])
        for service in baseline_ranking
    ]
    baseline_root_rank = baseline_ranks[root_service]
    transformed_root_rank = transformed_ranks[root_service]

    return {
        "transformation": transform.name,
        "scale": transform.scale,
        "offset": transform.offset,
        "n_services": len(baseline_ranking),
        "baseline_top1": baseline_ranking[0],
        "transformed_top1": transformed_ranking[0],
        "top1_same": baseline_ranking[0] == transformed_ranking[0],
        "baseline_root_rank": baseline_root_rank,
        "transformed_root_rank": transformed_root_rank,
        # Positive means that changing only the unit moved the root upward.
        "root_rank_delta": baseline_root_rank - transformed_root_rank,
        "root_rank_same": baseline_root_rank == transformed_root_rank,
        "service_ranking_identical": baseline_ranking == transformed_ranking,
        "mean_service_rank_displacement": float(np.mean(service_displacements)),
        "max_service_rank_displacement": int(max(service_displacements, default=0)),
        "service_scores_allclose": bool(np.allclose(
            # Affine offsets can lose a few low-order float64 bits when a
            # small signal is represented around a large level.  This remains
            # far below a rank-relevant Bayes-factor difference.
            original_scores, changed_scores, rtol=1e-5, atol=1e-7,
            equal_nan=True,
        )),
        "max_abs_service_score_diff": _finite_max(score_difference),
        "max_relative_service_score_diff": _finite_max(score_relative_difference),
        "baseline_ranking": json.dumps(baseline_ranking, ensure_ascii=False),
        "transformed_ranking": json.dumps(transformed_ranking, ensure_ascii=False),
        **_compare_metric_diagnostics(
            baseline_diagnostics, transformed_diagnostics, transform
        ),
    }


def _case_metadata(case_dir: Path) -> dict[str, Any]:
    with (case_dir / "case_info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    return {
        "case_id": str(info["case_id"]),
        "dataset": str(info["dataset"]),
        "fault_type": str(info["fault_type"]),
        "root_cause_service": str(info["root_cause_service"]),
    }


def _score_case(task: tuple[Path, dict[str, Any], tuple[AffineTransform, ...]]) -> list[dict[str, Any]]:
    case_dir, config, transforms = task
    normal = pd.read_csv(case_dir / "normal_data.csv")
    abnormal = pd.read_csv(case_dir / "abnormal_data.csv")
    metadata = _case_metadata(case_dir)

    baseline_model = _build_model(config)
    baseline_result = baseline_model.fit_predict(normal, abnormal)
    baseline_diagnostics = _metric_diagnostics(baseline_model)
    rows: list[dict[str, Any]] = []
    for transform in transforms:
        transformed_model = _build_model(config)
        transformed_result = transformed_model.fit_predict(
            _apply_affine(normal, transform),
            _apply_affine(abnormal, transform),
        )
        rows.append({
            **metadata,
            **_compare_results(
                baseline_result,
                transformed_result,
                metadata["root_cause_service"],
                transform,
                baseline_diagnostics,
                _metric_diagnostics(transformed_model),
            ),
        })
    return rows


def _summarize(rows: list[dict[str, Any]], scope: str, value: str) -> dict[str, Any]:
    n_cases = len(rows)
    root_deltas = np.asarray([row["root_rank_delta"] for row in rows], dtype=float)
    return {
        "scope": scope,
        "value": value,
        "transformation": rows[0]["transformation"],
        "n_cases": n_cases,
        "identical_service_ranking_count": int(sum(
            row["service_ranking_identical"] for row in rows
        )),
        "identical_service_ranking_fraction": float(np.mean([
            row["service_ranking_identical"] for row in rows
        ])),
        "same_top1_count": int(sum(row["top1_same"] for row in rows)),
        "same_top1_fraction": float(np.mean([row["top1_same"] for row in rows])),
        "same_root_rank_count": int(sum(row["root_rank_same"] for row in rows)),
        "same_root_rank_fraction": float(np.mean([
            row["root_rank_same"] for row in rows
        ])),
        "score_allclose_count": int(sum(
            row["service_scores_allclose"] for row in rows
        )),
        "score_allclose_fraction": float(np.mean([
            row["service_scores_allclose"] for row in rows
        ])),
        "lag_coefficients_allclose_count": int(sum(
            row["lag_coefficients_allclose"] for row in rows
        )),
        "lag_coefficients_allclose_fraction": float(np.mean([
            row["lag_coefficients_allclose"] for row in rows
        ])),
        "mean_absolute_root_rank_delta": float(np.mean(np.abs(root_deltas))),
        "max_absolute_root_rank_delta": int(np.max(np.abs(root_deltas))),
        "mean_service_rank_displacement": float(np.mean([
            row["mean_service_rank_displacement"] for row in rows
        ])),
        "max_abs_service_score_diff": _finite_max(
            row["max_abs_service_score_diff"] for row in rows
        ),
        "max_abs_lag_coefficient_diff": _finite_max(
            row["max_abs_lag_coefficient_diff"] for row in rows
        ),
        "max_relative_normal_scale_error": _finite_max(
            row["max_relative_normal_scale_error"] for row in rows
        ),
    }


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for transform in sorted({row["transformation"] for row in rows}):
        transformed = [row for row in rows if row["transformation"] == transform]
        summaries.append(_summarize(transformed, "overall", "overall"))
        for dataset in sorted({row["dataset"] for row in transformed}):
            summaries.append(_summarize(
                [row for row in transformed if row["dataset"] == dataset],
                "dataset",
                dataset,
            ))
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AMBER affine-unit invariance diagnosis",
        "",
        "Every metric in both windows is transformed as `y = scale * x + offset`.",
        "The underlying case is unchanged, so exact score and ranking preservation is the target.",
        "",
        f"Cases: {report['n_cases']}",
        "",
        "| Scope | Transform | Cases | Full ranking same | Top-1 same | Root rank same | Scores close | Lag coefficients close | Mean rank displacement | Max root-rank shift |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["summaries"]:
        lines.append(
            f"| {item['value']} | {item['transformation']} | {item['n_cases']} | "
            f"{item['identical_service_ranking_fraction']:.1%} | "
            f"{item['same_top1_fraction']:.1%} | "
            f"{item['same_root_rank_fraction']:.1%} | "
            f"{item['score_allclose_fraction']:.1%} | "
            f"{item['lag_coefficients_allclose_fraction']:.1%} | "
            f"{item['mean_service_rank_displacement']:.4f} | "
            f"{item['max_absolute_root_rank_delta']} |"
        )
    lines.extend([
        "",
        "`root_rank_delta = baseline_root_rank - transformed_root_rank`; a non-zero value is unit sensitivity, not a performance gain or loss.",
        "`Scores close` uses `rtol=1e-5, atol=1e-7` to allow float64 cancellation after affine offsets; ranking equality is checked exactly.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    config_path: Path,
    output_root: Path,
    *,
    transforms: tuple[AffineTransform, ...] = DEFAULT_TRANSFORMS,
    cases_per_dataset: int | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not transforms:
        raise ValueError("At least one affine transform is required")
    names = [transform.name for transform in transforms]
    if len(set(names)) != len(names):
        raise ValueError("Transform names must be unique")

    config = _load_config(config_path)
    case_dirs = _case_directories(config, cases_per_dataset)
    if not case_dirs:
        raise ValueError("No processed RCAEval cases found for the configured datasets")
    tasks = [(case_dir, config, transforms) for case_dir in case_dirs]
    if workers == 1:
        nested_rows = [_score_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested_rows = list(executor.map(_score_case, tasks))
    rows = sorted(
        [row for case_rows in nested_rows for row in case_rows],
        key=lambda row: (row["dataset"], row["case_id"], row["transformation"]),
    )
    report = {
        "analysis": "ar_unit_invariance",
        "config": str(config_path),
        "n_cases": len(case_dirs),
        "transformations": [asdict(transform) for transform in transforms],
        "invariance_target": {
            "service_ranking": "exact equality",
            "service_scores": "rtol=1e-5, atol=1e-7",
            "root_rank_delta": (
                "baseline_root_rank - transformed_root_rank; non-zero means "
                "sensitivity to the numerical unit"
            ),
        },
        "summaries": _summaries(rows),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "case_comparisons.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_root / "summary.md", report)
    return report


def _parse_transform(value: str) -> AffineTransform:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected NAME:SCALE:OFFSET")
    try:
        return AffineTransform(parts[0], float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cases-per-dataset", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--transform",
        action="append",
        type=_parse_transform,
        help=(
            "Positive affine transform NAME:SCALE:OFFSET. Repeat for multiple "
            "transforms; defaults test x1000, x0.001, and x0.001+100."
        ),
    )
    args = parser.parse_args()
    report = analyze(
        args.config,
        args.output_root,
        transforms=tuple(args.transform) if args.transform else DEFAULT_TRANSFORMS,
        cases_per_dataset=args.cases_per_dataset,
        workers=args.workers,
    )
    print(f"Analyzed {report['n_cases']} cases; wrote {args.output_root}")


if __name__ == "__main__":
    main()

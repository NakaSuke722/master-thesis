"""Diagnose in-sample versus time-ordered holdout AR residuals.

For each saved RCAEval normal window, the first half is used to estimate the
unit-invariant stationary AR model.  The second half remains normal and is
predicted one step ahead with frozen coefficients and observed lags.  The
analysis compares in-sample and out-of-sample residual scale/center, then
relates those quantities to the existing counterfactual pseudo-fault Bayes
factor.  AMBER ranking and scoring behavior are not modified.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models.amber import _ar_residuals, _nig_log_marginal
from scripts.analyze_ar_pseudo_fault_calibration import (
    _build_model,
    _case_directories,
    _case_metadata,
    _load_model_params,
    _split_normal,
)


DEFAULT_CONFIG = Path(
    "configs/ablation/rcaeval_re1_zenodo_v2/"
    "stationary_counterfactual_ar_uncertainty_unit_invariant.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/analysis/ar_residual_generalization_unit_invariant"
)
DEFAULT_MODE = "stationary_counterfactual_ar_uncertainty"


def _residual_scale(
    residuals: np.ndarray,
    *,
    floor: float,
) -> tuple[float, float, float, float]:
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate residual scale from no observations")
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    robust_scale = 1.4826 * mad
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    scale = max(robust_scale, 0.1 * sd, floor)
    return center, scale, robust_scale, sd


def residual_generalization_stats(
    train_y: np.ndarray,
    holdout_y: np.ndarray,
    coef: np.ndarray,
    *,
    order: int,
    relative_scale_floor: float,
    min_scale: float,
    prior: Any,
) -> dict[str, float]:
    """Compare fitted-window and frozen-model observed-lag residuals.

    ``train_y`` and ``holdout_y`` must already use the same normal-only input
    coordinate system as the fitted AR coefficients.
    """
    train = np.asarray(train_y, dtype=float)
    holdout = np.asarray(holdout_y, dtype=float)
    coefficients = np.asarray(coef, dtype=float)
    if coefficients.size != order + 1:
        raise ValueError(f"Expected {order + 1} coefficients")
    if train.size <= order or holdout.size == 0:
        raise ValueError("Train and holdout windows are too short")

    in_sample = _ar_residuals(train, coefficients, order)
    history = train[-order:] if order else np.empty(0, dtype=float)
    observed_path = np.concatenate([history, holdout])
    out_of_sample = _ar_residuals(observed_path, coefficients, order)
    if out_of_sample.size != holdout.size:
        raise ValueError("Unexpected observed-lag holdout residual length")

    level_scale = float(np.median(np.abs(train)))
    floor = max(
        relative_scale_floor * max(level_scale, min_scale),
        min_scale,
    )
    in_center, in_scale, in_robust_scale, in_sd = _residual_scale(
        in_sample, floor=floor
    )
    oos_center, oos_scale, oos_robust_scale, oos_sd = _residual_scale(
        out_of_sample, floor=floor
    )
    scale_ratio = oos_scale / in_scale
    center_shift_z = (oos_center - in_center) / in_scale

    z_in = (in_sample - in_center) / in_scale
    z_oos = (out_of_sample - in_center) / in_scale
    one_step_log_h0 = _nig_log_marginal(
        np.concatenate([z_in, z_oos]), prior
    )
    one_step_log_h1 = (
        _nig_log_marginal(z_in, prior)
        + _nig_log_marginal(z_oos, prior)
    )
    return {
        "n_in_sample_residuals": int(in_sample.size),
        "n_oos_residuals": int(out_of_sample.size),
        "residual_floor": floor,
        "in_sample_center": in_center,
        "oos_center": oos_center,
        "center_shift_z": center_shift_z,
        "abs_center_shift_z": abs(center_shift_z),
        "in_sample_scale": in_scale,
        "oos_scale": oos_scale,
        "scale_ratio": scale_ratio,
        "log_scale_ratio": float(np.log(scale_ratio)),
        "abs_log_scale_ratio": float(abs(np.log(scale_ratio))),
        "in_sample_robust_scale": in_robust_scale,
        "oos_robust_scale": oos_robust_scale,
        "in_sample_sd": in_sd,
        "oos_sd": oos_sd,
        "one_step_log_bayes_factor": float(one_step_log_h1 - one_step_log_h0),
    }


def _metric_row(
    diagnostic: dict[str, Any],
    *,
    metadata: dict[str, Any],
    model: Any,
) -> dict[str, Any] | None:
    coef = np.asarray(diagnostic.get("ar_coefficients") or [], dtype=float)
    raw_train = np.asarray(diagnostic.get("raw_normal") or [], dtype=float)
    raw_holdout = np.asarray(diagnostic.get("raw_abnormal") or [], dtype=float)
    input_scale = float(diagnostic.get("ar_input_scale", 1.0))
    input_center = float(diagnostic.get("ar_input_center", 0.0))
    if (
        coef.size != model.ar_order + 1
        or raw_train.size <= model.ar_order
        or raw_holdout.size == 0
        or not np.isfinite(input_scale)
        or input_scale <= 0.0
    ):
        return None

    train = (raw_train - input_center) / input_scale
    holdout = (raw_holdout - input_center) / input_scale
    stats = residual_generalization_stats(
        train,
        holdout,
        coef,
        order=model.ar_order,
        relative_scale_floor=model.relative_scale_floor,
        min_scale=model.min_scale,
        prior=model.prior,
    )
    recorded_scale = float(diagnostic.get("normal_scale", np.nan))
    scale_error = (
        abs(stats["in_sample_scale"] - recorded_scale)
        if np.isfinite(recorded_scale)
        else np.nan
    )
    return {
        **metadata,
        "metric": str(diagnostic["metric"]),
        "service": str(diagnostic["service"]),
        "counterfactual_log_bayes_factor": float(diagnostic["score"]),
        "recorded_normal_scale": recorded_scale,
        "recorded_scale_abs_error": scale_error,
        "ar_stationarity_constrained": bool(
            diagnostic.get("ar_stationarity_constrained", False)
        ),
        **stats,
    }


def _mean_top3(values: Iterable[float]) -> float:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    return float(np.mean(np.sort(finite)[-min(3, finite.size):]))


def _service_rows(
    metrics: pd.DataFrame,
    service_result: pd.DataFrame,
) -> list[dict[str, Any]]:
    recorded = service_result.set_index("service")["score"].to_dict()
    rows: list[dict[str, Any]] = []
    metadata_columns = [
        "case_id", "dataset", "fault_type", "root_cause_service"
    ]
    for service, group in metrics.groupby("service", sort=False):
        ranked = group.sort_values(
            "counterfactual_log_bayes_factor", ascending=False
        )
        top = ranked.head(3)
        counterfactual_score = float(recorded[service])
        recomputed_score = _mean_top3(
            group["counterfactual_log_bayes_factor"]
        )
        rows.append({
            **{column: group.iloc[0][column] for column in metadata_columns},
            "service": service,
            "n_metrics": int(len(group)),
            "counterfactual_service_log_bayes_factor": counterfactual_score,
            "counterfactual_score_abs_error": abs(
                counterfactual_score - recomputed_score
            ),
            "one_step_service_log_bayes_factor": _mean_top3(
                group["one_step_log_bayes_factor"]
            ),
            "top3_mean_log_scale_ratio": float(
                top["log_scale_ratio"].mean()
            ),
            "top3_mean_abs_log_scale_ratio": float(
                top["abs_log_scale_ratio"].mean()
            ),
            "top3_mean_scale_ratio": float(top["scale_ratio"].mean()),
            "top3_mean_abs_center_shift_z": float(
                top["abs_center_shift_z"].mean()
            ),
        })
    return rows


def _analyze_case(
    case_dir: Path,
    params: dict[str, Any],
    mode: str,
    fit_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    normal = pd.read_csv(case_dir / "normal_data.csv")
    fit, holdout = _split_normal(normal, fit_fraction)
    metadata = _case_metadata(case_dir)
    model = _build_model(params, mode)
    service_result = model.fit_predict(fit, holdout)
    diagnostics = (model.diagnostics_ or {}).get("metrics") or []

    metric_rows = [
        row
        for diagnostic in diagnostics
        if (row := _metric_row(diagnostic, metadata=metadata, model=model))
        is not None
    ]
    if not metric_rows:
        raise ValueError(f"No valid AR metric diagnostics for {metadata['case_id']}")
    metric_frame = pd.DataFrame(metric_rows)
    service_rows = _service_rows(metric_frame, service_result)
    service_frame = pd.DataFrame(service_rows)
    case_row = {
        **metadata,
        "n_metrics": int(len(metric_frame)),
        "skipped_metrics": int(len(diagnostics) - len(metric_frame)),
        "n_services": int(len(service_frame)),
        "median_metric_scale_ratio": float(
            metric_frame["scale_ratio"].median()
        ),
        "median_metric_log_scale_ratio": float(
            metric_frame["log_scale_ratio"].median()
        ),
        "median_metric_abs_log_scale_ratio": float(
            metric_frame["abs_log_scale_ratio"].median()
        ),
        "median_metric_abs_center_shift_z": float(
            metric_frame["abs_center_shift_z"].median()
        ),
        "max_counterfactual_service_log_bayes_factor": float(
            service_frame["counterfactual_service_log_bayes_factor"].max()
        ),
        "median_counterfactual_service_log_bayes_factor": float(
            service_frame["counterfactual_service_log_bayes_factor"].median()
        ),
        "max_one_step_service_log_bayes_factor": float(
            service_frame["one_step_service_log_bayes_factor"].max()
        ),
        "median_one_step_service_log_bayes_factor": float(
            service_frame["one_step_service_log_bayes_factor"].median()
        ),
    }
    return metric_rows, service_rows, case_row


def _spearman(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(
        y.to_numpy(dtype=float)
    )
    left = x.to_numpy(dtype=float)[valid]
    right = y.to_numpy(dtype=float)[valid]
    if left.size < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return {"n": int(left.size), "rho": None, "p_value": None}
    result = spearmanr(left, right)
    return {
        "n": int(left.size),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _bootstrap_median_ratio_ci(
    log_ratios: np.ndarray,
    *,
    samples: int = 10000,
    seed: int = 20260828,
) -> tuple[float, float]:
    values = np.asarray(log_ratios, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty scale-ratio sample")
    rng = np.random.default_rng(seed)
    medians = np.empty(samples, dtype=float)
    batch = 1000
    for start in range(0, samples, batch):
        size = min(batch, samples - start)
        indices = rng.integers(0, values.size, size=(size, values.size))
        medians[start:start + size] = np.median(values[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(np.exp(low)), float(np.exp(high))


def _scope_summary(
    label: str,
    cases: pd.DataFrame,
    metrics: pd.DataFrame,
    services: pd.DataFrame,
) -> dict[str, Any]:
    low, high = _bootstrap_median_ratio_ci(
        cases["median_metric_log_scale_ratio"].to_numpy(dtype=float)
    )
    positive_fractions = services.assign(
        counterfactual_positive=(
            services["counterfactual_service_log_bayes_factor"] > 0.0
        ),
        one_step_positive=(
            services["one_step_service_log_bayes_factor"] > 0.0
        ),
    ).groupby("case_id")[[
        "counterfactual_positive", "one_step_positive"
    ]].mean()
    return {
        "scope": label,
        "n_cases": int(len(cases)),
        "n_metrics": int(len(metrics)),
        "n_services": int(len(services)),
        "skipped_metrics": int(cases["skipped_metrics"].sum()),
        "median_metric_scale_ratio": float(metrics["scale_ratio"].median()),
        "median_case_scale_ratio": float(
            cases["median_metric_scale_ratio"].median()
        ),
        "median_case_scale_ratio_ci95": [low, high],
        "in_sample_scale_underestimation_supported": bool(low > 1.0),
        "case_fraction_scale_ratio_gt_1": float(
            np.mean(cases["median_metric_scale_ratio"] > 1.0)
        ),
        "case_fraction_scale_ratio_gt_1_1": float(
            np.mean(cases["median_metric_scale_ratio"] > 1.1)
        ),
        "median_case_abs_center_shift_z": float(
            cases["median_metric_abs_center_shift_z"].median()
        ),
        "median_max_counterfactual_service_bf": float(
            cases["max_counterfactual_service_log_bayes_factor"].median()
        ),
        "median_max_one_step_service_bf": float(
            cases["max_one_step_service_log_bayes_factor"].median()
        ),
        "mean_case_counterfactual_positive_service_fraction": float(
            positive_fractions["counterfactual_positive"].mean()
        ),
        "mean_case_one_step_positive_service_fraction": float(
            positive_fractions["one_step_positive"].mean()
        ),
        "median_max_bf_difference_counterfactual_minus_one_step": float(
            (
                cases["max_counterfactual_service_log_bayes_factor"]
                - cases["max_one_step_service_log_bayes_factor"]
            ).median()
        ),
        "service_bf_vs_log_scale_ratio": _spearman(
            services["counterfactual_service_log_bayes_factor"],
            services["top3_mean_log_scale_ratio"],
        ),
        "service_bf_vs_abs_log_scale_ratio": _spearman(
            services["counterfactual_service_log_bayes_factor"],
            services["top3_mean_abs_log_scale_ratio"],
        ),
        "service_bf_vs_abs_center_shift": _spearman(
            services["counterfactual_service_log_bayes_factor"],
            services["top3_mean_abs_center_shift_z"],
        ),
        "counterfactual_bf_vs_one_step_bf": _spearman(
            services["counterfactual_service_log_bayes_factor"],
            services["one_step_service_log_bayes_factor"],
        ),
        "one_step_bf_vs_log_scale_ratio": _spearman(
            services["one_step_service_log_bayes_factor"],
            services["top3_mean_log_scale_ratio"],
        ),
        "one_step_bf_vs_abs_log_scale_ratio": _spearman(
            services["one_step_service_log_bayes_factor"],
            services["top3_mean_abs_log_scale_ratio"],
        ),
        "one_step_bf_vs_abs_center_shift": _spearman(
            services["one_step_service_log_bayes_factor"],
            services["top3_mean_abs_center_shift_z"],
        ),
        "max_recorded_scale_abs_error": float(
            metrics["recorded_scale_abs_error"].max()
        ),
        "max_service_score_abs_error": float(
            services["counterfactual_score_abs_error"].max()
        ),
    }


def _build_report(
    cases: pd.DataFrame,
    metrics: pd.DataFrame,
    services: pd.DataFrame,
    *,
    config_path: Path,
    fit_fraction: float,
) -> dict[str, Any]:
    summaries = [_scope_summary("overall", cases, metrics, services)]
    for dataset in sorted(cases["dataset"].unique()):
        case_ids = set(cases.loc[cases["dataset"] == dataset, "case_id"])
        summaries.append(_scope_summary(
            str(dataset),
            cases[cases["case_id"].isin(case_ids)],
            metrics[metrics["case_id"].isin(case_ids)],
            services[services["case_id"].isin(case_ids)],
        ))
    for fault_type in ("cpu", "mem", "disk", "delay", "loss"):
        case_ids = set(cases.loc[cases["fault_type"] == fault_type, "case_id"])
        if case_ids:
            summaries.append(_scope_summary(
                fault_type,
                cases[cases["case_id"].isin(case_ids)],
                metrics[metrics["case_id"].isin(case_ids)],
                services[services["case_id"].isin(case_ids)],
            ))
    return {
        "experiment": "ar_residual_generalization",
        "config": str(config_path),
        "fit_fraction": fit_fraction,
        "scale_ratio_definition": "oos_residual_scale / in_sample_residual_scale",
        "one_step_prediction": "frozen AR coefficients with observed holdout lags",
        "n_cases": int(len(cases)),
        "summaries": summaries,
    }


def _format_correlation(value: dict[str, Any]) -> str:
    rho = value["rho"]
    p_value = value["p_value"]
    if rho is None or p_value is None:
        return "n/a"
    return f"{rho:.3f} (p={p_value:.3g})"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# In-sample vs time-ordered holdout AR residuals",
        "",
        "The first normal block fits a unit-invariant stationary AR model. The held-out normal block is predicted one step ahead with frozen coefficients and observed lags.",
        "",
        "`scale_ratio = OOS residual scale / in-sample residual scale`; values above one mean the fitted-window residual reference is narrower than future normal residuals.",
        "",
        "| Scope | Cases | Median case scale ratio (95% CI) | Ratio > 1 | Ratio > 1.1 | Median absolute center shift / in-scale | Median max CF BF | Median max 1-step BF | Positive CF services | Positive 1-step services |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["summaries"]:
        low, high = item["median_case_scale_ratio_ci95"]
        lines.append(
            f"| {item['scope']} | {item['n_cases']} | "
            f"{item['median_case_scale_ratio']:.3f} ({low:.3f}, {high:.3f}) | "
            f"{item['case_fraction_scale_ratio_gt_1']:.1%} | "
            f"{item['case_fraction_scale_ratio_gt_1_1']:.1%} | "
            f"{item['median_case_abs_center_shift_z']:.3f} | "
            f"{item['median_max_counterfactual_service_bf']:.3f} | "
            f"{item['median_max_one_step_service_bf']:.3f} | "
            f"{item['mean_case_counterfactual_positive_service_fraction']:.1%} | "
            f"{item['mean_case_one_step_positive_service_fraction']:.1%} |"
        )
    lines.extend([
        "",
        "## Service-level associations",
        "",
        "Spearman correlations use all case-service rows in each scope.",
        "",
        "| Scope | CF BF vs absolute log-scale ratio | CF BF vs absolute center shift | CF BF vs 1-step BF | 1-step BF vs absolute log-scale ratio | 1-step BF vs absolute center shift |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for item in report["summaries"]:
        lines.append(
            f"| {item['scope']} | "
            f"{_format_correlation(item['service_bf_vs_abs_log_scale_ratio'])} | "
            f"{_format_correlation(item['service_bf_vs_abs_center_shift'])} | "
            f"{_format_correlation(item['counterfactual_bf_vs_one_step_bf'])} | "
            f"{_format_correlation(item['one_step_bf_vs_abs_log_scale_ratio'])} | "
            f"{_format_correlation(item['one_step_bf_vs_abs_center_shift'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    processed_root: Path,
    config_path: Path,
    output_root: Path,
    *,
    mode: str = DEFAULT_MODE,
    fit_fraction: float = 0.5,
    workers: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    params = _load_model_params(config_path)
    case_dirs = _case_directories(processed_root)
    if limit is not None:
        case_dirs = case_dirs[:limit]
    if not case_dirs:
        raise ValueError(f"No processed RCAEval cases found under {processed_root}")

    if workers == 1:
        results = [
            _analyze_case(case_dir, params, mode, fit_fraction)
            for case_dir in case_dirs
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(
                _analyze_case,
                case_dirs,
                [params] * len(case_dirs),
                [mode] * len(case_dirs),
                [fit_fraction] * len(case_dirs),
            ))

    metric_rows = [row for result in results for row in result[0]]
    service_rows = [row for result in results for row in result[1]]
    case_rows = [result[2] for result in results]
    metrics = pd.DataFrame(metric_rows).sort_values(["case_id", "metric"])
    services = pd.DataFrame(service_rows).sort_values(["case_id", "service"])
    cases = pd.DataFrame(case_rows).sort_values("case_id")

    output_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_root / "metric_residuals.csv", index=False)
    services.to_csv(output_root / "service_diagnostics.csv", index=False)
    cases.to_csv(output_root / "case_diagnostics.csv", index=False)
    report = _build_report(
        cases,
        metrics,
        services,
        config_path=config_path,
        fit_fraction=fit_fraction,
    )
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_root / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/rcaeval_zenodo_v2"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = analyze(
        args.processed_root,
        args.config,
        args.output_root,
        mode=args.mode,
        fit_fraction=args.fit_fraction,
        workers=args.workers,
        limit=args.limit,
    )
    print(f"Analyzed {report['n_cases']} cases; wrote {args.output_root}")


if __name__ == "__main__":
    main()

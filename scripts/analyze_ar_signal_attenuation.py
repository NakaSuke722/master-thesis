"""Diagnose whether observed-lag AR absorbs persistent root-cause shifts.

The analysis is artifact-only: it joins the existing RCAEval Raw+BF and AR+BF
service results by case_id and never reruns AMBER.  Positive rank_delta means
AR improved the ground-truth service rank.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

try:
    from scripts.analyze_ar_rank_movement import _case_row, _load_cases
except ModuleNotFoundError:  # Direct execution: python3 scripts/<this-file>.py
    from analyze_ar_rank_movement import _case_row, _load_cases


MOVEMENTS = ("improved", "same", "worsened")
FAULT_TYPES = ("cpu", "mem", "disk", "delay", "loss")
DIAGNOSTIC_FIELDS = (
    "sum_phi",
    "raw_shift_abs",
    "raw_late_to_early_shift_ratio",
    "raw_shift_sign_consistency",
    "signal_retention_ratio",
    "ar_late_to_early_shift_ratio",
    "ar_early_to_late_decay_fraction",
    "ar_initial_to_late_retention_ratio",
    "ar_initial_to_late_decay_fraction",
    "ar_score_minus_raw_score",
)


def _finite(values: Iterable[Any]) -> list[float]:
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def _ratio(numerator: float | None, denominator: float | None, floor: float = 1e-12) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= floor:
        return None
    return numerator / denominator


def _split_halves(values: list[float]) -> tuple[list[float], list[float]]:
    midpoint = max(1, len(values) // 2)
    return values[:midpoint], values[midpoint:]


def _metric_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = (result.get("amber_diagnostics") or {}).get("metrics") or []
    if not metrics:
        raise ValueError(f"Missing AMBER metric diagnostics: {result['_path']}")
    output: dict[str, dict[str, Any]] = {}
    for row in metrics:
        name = row.get("metric")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Metric diagnostic without a name: {result['_path']}")
        if name in output:
            raise ValueError(f"Duplicate metric {name!r}: {result['_path']}")
        output[name] = row
    return output


def _read_metric_columns(path: Path, metric_names: list[str]) -> dict[str, list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Processed time-series file not found: {path}")
    output = {name: [] for name in metric_names}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(metric_names) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing metrics {missing} in {path}")
        for row in reader:
            for name in metric_names:
                try:
                    value = float(row[name])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    output[name].append(value)
    return output


def _load_processed_series(
    result: dict[str, Any], metric_names: list[str], processed_root: Path,
) -> dict[str, tuple[list[float], list[float]]]:
    if result.get("model_parameters", {}).get("winsor_quantile") is not None:
        raise ValueError(
            "Processed-data fallback currently requires winsor_quantile=null: " + result["_path"]
        )
    strategy = (result.get("preprocessing") or {}).get("strategy", "default")
    case_dir = (
        processed_root / strategy / str(result.get("benchmark", "rcaeval_re1"))
        / str(result["dataset"]) / str(result["case_id"])
    )
    normal = _read_metric_columns(case_dir / "normal_data.csv", metric_names)
    abnormal = _read_metric_columns(case_dir / "abnormal_data.csv", metric_names)
    return {name: (normal[name], abnormal[name]) for name in metric_names}


def _observed_lag_residuals(
    normal: list[float], abnormal: list[float], coefficients: list[float],
) -> tuple[list[float], list[float]]:
    order = len(coefficients) - 1
    if order < 0:
        raise ValueError("AR coefficients must include an intercept")

    def residuals(values: list[float]) -> list[float]:
        output = []
        for index in range(order, len(values)):
            prediction = coefficients[0]
            prediction += sum(
                coefficients[lag] * values[index - lag]
                for lag in range(1, order + 1)
            )
            output.append(values[index] - prediction)
        return output

    history = normal[-order:] if order else []
    return residuals(normal), residuals([*history, *abnormal])


def _root_metric_names(result: dict[str, Any], root_service: str) -> list[str]:
    return sorted(
        name for name, row in _metric_map(result).items()
        if row.get("service") == root_service
    )


def _service_top3(metrics: dict[str, dict[str, Any]], root_names: list[str]) -> set[str]:
    scored = []
    for name in root_names:
        score = metrics[name].get("score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scored.append((score, int(metrics[name].get("rank", 10**9)), name))
    return {name for _, _, name in sorted(scored, key=lambda item: (-item[0], item[1]))[:3]}


def _metric_row(
    case: dict[str, Any],
    metric_name: str,
    raw_metric: dict[str, Any],
    ar_metric: dict[str, Any],
    raw_top3: set[str],
    ar_top3: set[str],
    processed_series: tuple[list[float], list[float]] | None = None,
) -> dict[str, Any]:
    raw_normal = _finite(ar_metric.get("raw_normal") or [])
    raw_abnormal = _finite(ar_metric.get("raw_abnormal") or [])
    ar_normal = _finite(ar_metric.get("ar_residual_normal") or [])
    ar_abnormal = _finite(ar_metric.get("ar_residual_abnormal") or [])
    coefficients = _finite(ar_metric.get("ar_coefficients") or [])
    if (not raw_normal or not raw_abnormal) and processed_series is not None:
        raw_normal, raw_abnormal = processed_series
    if (not ar_normal or not ar_abnormal) and raw_normal and raw_abnormal and coefficients:
        ar_normal, ar_abnormal = _observed_lag_residuals(raw_normal, raw_abnormal, coefficients)
    if not raw_normal or not raw_abnormal or not ar_normal or not ar_abnormal:
        raise ValueError(f"Incomplete diagnostic series for {case['case_id']} / {metric_name}")

    raw_center = median(raw_normal)
    raw_shift = median(raw_abnormal) - raw_center
    raw_early, raw_late = _split_halves(raw_abnormal)
    raw_early_shift = median(raw_early) - raw_center
    raw_late_shift = median(raw_late) - raw_center if raw_late else None

    shift_sign = 1.0 if raw_shift > 0 else -1.0 if raw_shift < 0 else 0.0
    sign_consistency = (
        sum((value - raw_center) * shift_sign > 0 for value in raw_abnormal) / len(raw_abnormal)
        if shift_sign else None
    )

    ar_center = median(ar_normal)
    ar_shift = median(ar_abnormal) - ar_center
    ar_early, ar_half_late = _split_halves(ar_abnormal)
    ar_early_shift = median(ar_early) - ar_center
    ar_half_late_shift = median(ar_half_late) - ar_center if ar_half_late else None
    ar_order = max(0, len(coefficients) - 1)
    initial_count = min(len(ar_abnormal), max(1, ar_order))
    ar_initial_shift = median(ar_abnormal[:initial_count]) - ar_center
    ar_late_values = ar_abnormal[initial_count:]
    ar_late_shift = median(ar_late_values) - ar_center if ar_late_values else None

    ar_scale_values = _finite([ar_metric.get("normal_scale")])
    ar_shift_floor = max(1e-12, ar_scale_values[0] if ar_scale_values else 0.0)
    half_retention = _ratio(
        abs(ar_half_late_shift) if ar_half_late_shift is not None else None,
        abs(ar_early_shift),
        ar_shift_floor,
    )
    half_decay = 1.0 - half_retention if half_retention is not None else None
    initial_to_late = _ratio(
        abs(ar_late_shift) if ar_late_shift is not None else None,
        abs(ar_initial_shift),
        ar_shift_floor,
    )
    decay_fraction = 1.0 - initial_to_late if initial_to_late is not None else None
    raw_scale_values = _finite([raw_metric.get("normal_scale")])
    raw_shift_floor = max(1e-12, raw_scale_values[0] if raw_scale_values else 0.0)

    raw_score = raw_metric.get("score")
    ar_score = ar_metric.get("score")
    score_delta = (
        float(ar_score) - float(raw_score)
        if ar_score is not None and raw_score is not None
        and math.isfinite(float(ar_score)) and math.isfinite(float(raw_score))
        else None
    )

    return {
        **case,
        "metric": metric_name,
        "raw_metric_rank": raw_metric.get("rank"),
        "ar_metric_rank": ar_metric.get("rank"),
        "raw_metric_score": raw_score,
        "ar_metric_score": ar_score,
        "ar_score_minus_raw_score": score_delta,
        "raw_service_top3": metric_name in raw_top3,
        "ar_service_top3": metric_name in ar_top3,
        "sum_phi": sum(coefficients[1:]) if coefficients else None,
        "raw_normal_median": raw_center,
        "raw_abnormal_median": median(raw_abnormal),
        "raw_shift": raw_shift,
        "raw_shift_abs": abs(raw_shift),
        "raw_early_shift": raw_early_shift,
        "raw_late_shift": raw_late_shift,
        "raw_late_to_early_shift_ratio": _ratio(
            abs(raw_late_shift) if raw_late_shift is not None else None,
            abs(raw_early_shift),
            raw_shift_floor,
        ),
        "raw_shift_sign_consistency": sign_consistency,
        "ar_normal_residual_median": ar_center,
        "ar_abnormal_residual_median": median(ar_abnormal),
        "ar_residual_shift": ar_shift,
        "ar_residual_shift_abs": abs(ar_shift),
        "signal_retention_ratio": _ratio(abs(ar_shift), abs(raw_shift), raw_shift_floor),
        "ar_early_residual_shift": ar_early_shift,
        "ar_half_late_residual_shift": ar_half_late_shift,
        "ar_late_to_early_shift_ratio": half_retention,
        "ar_early_to_late_decay_fraction": half_decay,
        "ar_initial_window_points": initial_count,
        "ar_initial_residual_shift": ar_initial_shift,
        "ar_late_residual_shift": ar_late_shift,
        "ar_initial_to_late_retention_ratio": initial_to_late,
        "ar_initial_to_late_decay_fraction": decay_fraction,
    }


def _case_group_summary(
    rows: list[dict[str, Any]], field: str, values: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    keys = list(values) if values is not None else sorted(groups)
    output = []
    for key in keys:
        members = groups.get(key, [])
        summary: dict[str, Any] = {
            field: key,
            "n_cases": len(members),
            "n_root_metrics": sum(row["n_root_metrics"] for row in members),
            "n_service_top3_union_metrics": sum(
                row["n_service_top3_union_metrics"] for row in members
            ),
        }
        for diagnostic in DIAGNOSTIC_FIELDS:
            field_values = _finite(
                row.get(f"median_top3_union_{diagnostic}") for row in members
            )
            summary[f"n_{diagnostic}"] = len(field_values)
            summary[f"mean_{diagnostic}"] = (
                sum(field_values) / len(field_values) if field_values else None
            )
            summary[f"median_{diagnostic}"] = median(field_values) if field_values else None
        output.append(summary)
    return output


def _case_summaries(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        groups[row["case_id"]].append(row)
    output = []
    for case_id, rows in sorted(groups.items()):
        first = rows[0]
        selected = [row for row in rows if row["raw_service_top3"] or row["ar_service_top3"]]
        summary = {
            key: first[key] for key in (
                "case_id", "dataset", "fault_type", "root_cause_service",
                "raw_rank", "ar_rank", "rank_delta", "movement",
            )
        }
        summary["n_root_metrics"] = len(rows)
        summary["n_service_top3_union_metrics"] = len(selected)
        for field in DIAGNOSTIC_FIELDS:
            values = _finite(row.get(field) for row in selected)
            summary[f"median_top3_union_{field}"] = median(values) if values else None
        output.append(summary)
    return output


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    scale_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    scale_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    return covariance / (scale_x * scale_y) if scale_x and scale_y else None


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _correlations(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for field in DIAGNOSTIC_FIELDS:
        pairs = []
        case_field = f"median_top3_union_{field}"
        for row in case_rows:
            values = _finite([row.get(case_field), row.get("rank_delta")])
            if len(values) == 2:
                pairs.append((values[0], values[1]))
        xs, ys = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        output.append({
            "diagnostic": field,
            "outcome": "rank_delta (positive means AR improved)",
            "n_cases": len(pairs),
            "pearson_r": _pearson(xs, ys),
            "spearman_rho": _pearson(_ranks(xs), _ranks(ys)) if pairs else None,
        })
    return output


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    return "NA" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Observed-lag AR persistent-shift attenuation diagnostic",
        "",
        "`rank_delta = raw_rank - ar_rank`; positive values mean AR improved the root-service rank.",
        "`signal_retention_ratio = |AR residual median shift| / |Raw median shift|`; values below 1 indicate attenuation.",
        "Positive `ar_early_to_late_decay_fraction` indicates that the AR residual shift became weaker from the first to the second half of the abnormal period.",
        "",
        f"Joined cases: {report['n_joined_cases']}; root metrics: {report['n_root_metrics']} "
        f"(Raw/AR top-3 union: {report['n_service_top3_union_metrics']})",
        "",
        "## By movement",
        "",
        "| Movement | Cases | Root metrics | Median sum(phi) | Median retention | Median AR decay |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["by_movement"]:
        lines.append(
            f"| {row['movement']} | {row['n_cases']} | {row['n_root_metrics']} | "
            f"{_format(row['median_sum_phi'])} | {_format(row['median_signal_retention_ratio'])} | "
            f"{_format(row['median_ar_early_to_late_decay_fraction'])} |"
        )
    lines.extend([
        "",
        "## Case-level correlation with rank movement",
        "",
        "| Diagnostic | N | Pearson r | Spearman rho |",
        "|---|---:|---:|---:|",
    ])
    for row in report["case_level_correlations"]:
        lines.append(
            f"| {row['diagnostic']} | {row['n_cases']} | {_format(row['pearson_r'])} | {_format(row['spearman_rho'])} |"
        )
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- This report diagnoses association; it does not by itself prove that attenuation caused a rank change.",
        "- Case summaries use the union of the Raw and AR root-service top-3 metrics because service scoring uses `mean_top3`.",
        "- Ratios with a practically zero denominator are recorded as null rather than guessed.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    raw_root: Path,
    ar_root: Path,
    output_root: Path,
    processed_root: Path = Path("data/processed/rcaeval_zenodo_v2"),
) -> dict[str, Any]:
    raw_cases, ar_cases = _load_cases(raw_root), _load_cases(ar_root)
    if not raw_cases or not ar_cases:
        raise FileNotFoundError(
            f"No service result artifacts found: raw_root={raw_root}, ar_root={ar_root}"
        )
    if set(raw_cases) != set(ar_cases):
        raise ValueError(
            f"case_id sets differ: raw_only={sorted(set(raw_cases) - set(ar_cases))}, "
            f"ar_only={sorted(set(ar_cases) - set(raw_cases))}"
        )

    metric_rows: list[dict[str, Any]] = []
    for case_id in sorted(raw_cases):
        raw, ar = raw_cases[case_id], ar_cases[case_id]
        rank_row = _case_row(case_id, raw, ar)
        case = {
            "case_id": case_id,
            "dataset": rank_row["dataset"],
            "fault_type": rank_row["fault_type"],
            "root_cause_service": rank_row["root_cause_service"],
            "raw_rank": rank_row["raw_rank"],
            "ar_rank": rank_row["ar_rank"],
            "rank_delta": rank_row["delta"],
            "movement": rank_row["movement"],
        }
        raw_metrics, ar_metrics = _metric_map(raw), _metric_map(ar)
        raw_names = _root_metric_names(raw, case["root_cause_service"])
        ar_names = _root_metric_names(ar, case["root_cause_service"])
        if not raw_names or raw_names != ar_names:
            raise ValueError(f"Root metric sets differ or are empty for {case_id}: raw={raw_names}, ar={ar_names}")
        raw_top3 = _service_top3(raw_metrics, raw_names)
        ar_top3 = _service_top3(ar_metrics, ar_names)
        needs_processed_data = any(
            not ar_metrics[name].get("raw_normal")
            or not ar_metrics[name].get("raw_abnormal")
            for name in raw_names
        )
        processed = (
            _load_processed_series(ar, raw_names, processed_root)
            if needs_processed_data else {}
        )
        for metric_name in raw_names:
            metric_rows.append(_metric_row(
                case, metric_name, raw_metrics[metric_name], ar_metrics[metric_name], raw_top3, ar_top3,
                processed.get(metric_name),
            ))

    case_rows = _case_summaries(metric_rows)
    service_relevant_rows = [
        row for row in metric_rows
        if row["raw_service_top3"] or row["ar_service_top3"]
    ]
    report = {
        "definitions": {
            "rank_delta": "raw_rank - ar_rank; positive means AR improved rank",
            "signal_retention_ratio": "abs(AR residual median shift) / abs(Raw median shift)",
            "ar_early_to_late_decay_fraction": "1 - abs(second-half AR residual shift) / abs(first-half AR residual shift)",
            "ar_initial_to_late_decay_fraction": "1 - abs(late AR residual shift) / abs(initial AR residual shift)",
            "case_aggregation": "median over union of Raw and AR root-service top-3 metrics",
        },
        "n_joined_cases": len(case_rows),
        "n_root_metrics": len(metric_rows),
        "n_service_top3_union_metrics": len(service_relevant_rows),
        "by_movement": _case_group_summary(case_rows, "movement", MOVEMENTS),
        "by_dataset": _case_group_summary(case_rows, "dataset"),
        "by_fault_type": _case_group_summary(case_rows, "fault_type", FAULT_TYPES),
        "case_level_correlations": _correlations(case_rows),
    }
    _write_json(output_root / "root_metric_diagnostics.json", metric_rows)
    _write_csv(output_root / "root_metric_diagnostics.csv", metric_rows)
    _write_json(output_root / "case_diagnostics.json", case_rows)
    _write_csv(output_root / "case_diagnostics.csv", case_rows)
    _write_json(output_root / "summary.json", report)
    _write_csv(output_root / "by_movement.csv", report["by_movement"])
    _write_csv(output_root / "by_dataset.csv", report["by_dataset"])
    _write_csv(output_root / "by_fault_type.csv", report["by_fault_type"])
    _write_csv(output_root / "case_level_correlations.csv", report["case_level_correlations"])
    _write_markdown(output_root / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose observed-lag AR persistent-shift attenuation.")
    parser.add_argument("--raw-root", type=Path, default=Path("results/ablation/rcaeval_re1/no_ar"))
    parser.add_argument("--ar-root", type=Path, default=Path("results/main/rcaeval_re1/amber_zenodo_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("results/analysis/ar_signal_attenuation"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/rcaeval_zenodo_v2"))
    args = parser.parse_args()
    report = analyze(args.raw_root, args.ar_root, args.output_root, args.processed_root)
    print(f"Joined {report['n_joined_cases']} cases; wrote {args.output_root}")


if __name__ == "__main__":
    main()

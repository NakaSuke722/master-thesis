"""Case-level diagnostics for the final Counterfactual AR candidate.

The analysis is deliberately observational.  It joins already-saved RCAEval
artifacts by ``case_id`` and answers two focused questions:

1. Does horizon-aware uncertainty improve the ground-truth service rank?
2. After the AR redesign, do CPU cases still systematically regress relative
   to the Raw+BF baseline?

It also inventories extreme forecast-uncertainty multipliers so that a good
aggregate score cannot hide numerically excessive corrections.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_REFERENCE_ROOT = Path(
    "results/sensitivity/rcaeval_re1/unit_invariant_r0_98_p3/service"
)
DEFAULT_NO_HORIZON_ROOT = Path(
    "results/sensitivity/rcaeval_re1/"
    "unit_invariant_no_horizon_uncertainty/service"
)
DEFAULT_RAW_ROOT = Path("results/ablation/rcaeval_re1/no_ar/service")
DEFAULT_OUTPUT_ROOT = Path("results/analysis/final_ar_diagnostics")


def _read_cases(root: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(payload.get("case_id") or "")
        if not case_id:
            raise ValueError(f"Missing case_id: {path}")
        if case_id in cases:
            raise ValueError(f"Duplicate case_id under {root}: {case_id}")
        if not payload.get("predicted_ranking"):
            raise ValueError(f"Full predicted_ranking is required: {path}")
        cases[case_id] = payload
    if not cases:
        raise ValueError(f"No result JSON files found under {root}")
    return cases


def _strict_case_ids(
    named_cases: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    names = list(named_cases)
    reference = set(named_cases[names[0]])
    for name in names[1:]:
        current = set(named_cases[name])
        if current != reference:
            missing = sorted(reference - current)[:5]
            extra = sorted(current - reference)[:5]
            raise ValueError(
                f"case_id sets differ for {name}: missing={missing}, extra={extra}"
            )
    return sorted(reference)


def _root_rank(payload: dict[str, Any]) -> int:
    root = str(payload["evaluation_ground_truth"])
    ranking = [str(service) for service in payload["predicted_ranking"]]
    if root not in ranking:
        raise ValueError(
            f"Ground-truth service {root!r} absent from full ranking for "
            f"{payload.get('case_id')}"
        )
    return ranking.index(root) + 1


def _movement(delta: int) -> str:
    if delta > 0:
        return "improved"
    if delta < 0:
        return "worsened"
    return "same"


def _metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = payload.get("amber_diagnostics") or {}
    rows = diagnostics.get("metrics") or []
    if not isinstance(rows, list):
        raise ValueError(f"Invalid metric diagnostics for {payload.get('case_id')}")
    return rows


def _root_multiplier_summary(payload: dict[str, Any]) -> tuple[float, float]:
    root = str(payload["evaluation_ground_truth"])
    root_metrics = [
        row for row in _metric_rows(payload) if str(row.get("service")) == root
    ]
    root_metrics.sort(
        key=lambda row: float(
            row["score"] if row.get("score") is not None else -np.inf
        ),
        reverse=True,
    )
    selected = root_metrics[:3]
    values = np.asarray([
        float(row.get("forecast_uncertainty_max_multiplier") or 1.0)
        for row in selected
    ], dtype=float)
    if not values.size:
        return 1.0, 1.0
    return float(np.max(values)), float(np.mean(values))


def build_case_rows(
    raw: dict[str, dict[str, Any]],
    no_horizon: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    case_ids = _strict_case_ids({
        "raw": raw,
        "no_horizon": no_horizon,
        "reference": reference,
    })
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        payloads = (raw[case_id], no_horizon[case_id], reference[case_id])
        metadata = [
            (
                str(payload["dataset"]),
                str(payload["fault_type"]),
                str(payload["evaluation_ground_truth"]),
            )
            for payload in payloads
        ]
        if len(set(metadata)) != 1:
            raise ValueError(f"Case metadata differ for {case_id}: {metadata}")
        raw_rank, no_horizon_rank, reference_rank = map(_root_rank, payloads)
        root_max, root_mean = _root_multiplier_summary(reference[case_id])
        rows.append({
            "case_id": case_id,
            "dataset": metadata[0][0],
            "fault_type": metadata[0][1],
            "root_cause_service": metadata[0][2],
            "raw_rank": raw_rank,
            "no_horizon_rank": no_horizon_rank,
            "reference_rank": reference_rank,
            "raw_to_reference_delta": raw_rank - reference_rank,
            "raw_to_reference_movement": _movement(raw_rank - reference_rank),
            "horizon_delta": no_horizon_rank - reference_rank,
            "horizon_movement": _movement(no_horizon_rank - reference_rank),
            "raw_top1_to_reference_non_top1": bool(
                raw_rank == 1 and reference_rank != 1
            ),
            "raw_non_top1_to_reference_top1": bool(
                raw_rank != 1 and reference_rank == 1
            ),
            "no_horizon_top1_to_reference_non_top1": bool(
                no_horizon_rank == 1 and reference_rank != 1
            ),
            "no_horizon_non_top1_to_reference_top1": bool(
                no_horizon_rank != 1 and reference_rank == 1
            ),
            "root_top3_max_uncertainty_multiplier": root_max,
            "root_top3_mean_uncertainty_multiplier": root_mean,
        })
    return rows


def _comparison_summary(
    rows: list[dict[str, Any]], delta_column: str, movement_column: str,
) -> dict[str, Any]:
    deltas = np.asarray([row[delta_column] for row in rows], dtype=float)
    movements = Counter(row[movement_column] for row in rows)
    n = len(rows)
    prefix = "raw" if delta_column.startswith("raw") else "no_horizon"
    losses = int(sum(
        row[f"{prefix}_top1_to_reference_non_top1"] for row in rows
    ))
    gains = int(sum(
        row[f"{prefix}_non_top1_to_reference_top1"] for row in rows
    ))
    discordant = gains + losses
    if discordant:
        tail = sum(
            math.comb(discordant, k)
            for k in range(min(gains, losses) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "n_cases": n,
        "improved": int(movements["improved"]),
        "same": int(movements["same"]),
        "worsened": int(movements["worsened"]),
        "improved_fraction": float(movements["improved"] / n),
        "same_fraction": float(movements["same"] / n),
        "worsened_fraction": float(movements["worsened"] / n),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "baseline_top1_to_reference_non_top1": losses,
        "baseline_non_top1_to_reference_top1": gains,
        "top1_mcnemar_two_sided_exact_p": p_value,
    }


def _scoped_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("overall", "overall", rows)
    ]
    for column in ("dataset", "fault_type"):
        for value in sorted({str(row[column]) for row in rows}):
            scopes.append((
                column,
                value,
                [row for row in rows if str(row[column]) == value],
            ))
    return [{
        "scope": scope,
        "value": value,
        "raw_to_reference": _comparison_summary(
            subset, "raw_to_reference_delta", "raw_to_reference_movement"
        ),
        "horizon_aware_uncertainty": _comparison_summary(
            subset, "horizon_delta", "horizon_movement"
        ),
    } for scope, value, subset in scopes]


def _multiplier_rows(reference: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id, payload in sorted(reference.items()):
        for metric in _metric_rows(payload):
            value = float(metric.get("forecast_uncertainty_max_multiplier") or 1.0)
            rows.append({
                "case_id": case_id,
                "dataset": payload["dataset"],
                "fault_type": payload["fault_type"],
                "root_cause_service": payload["evaluation_ground_truth"],
                "metric": metric.get("metric"),
                "service": metric.get("service"),
                "is_root_service": str(metric.get("service"))
                == str(payload["evaluation_ground_truth"]),
                "score": float(metric.get("score") or 0.0),
                "max_uncertainty_multiplier": value,
                "final_uncertainty_multiplier": float(
                    metric.get("forecast_uncertainty_final_multiplier") or 1.0
                ),
                "stationarity_constrained": bool(
                    metric.get("ar_stationarity_constrained", False)
                ),
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Final Counterfactual AR diagnostics",
        "",
        "Positive rank delta means the final reference moved the ground-truth service upward.",
        "The reference is unit-invariant stationary Counterfactual AR+BF with horizon-aware uncertainty.",
        "",
        "| Scope | Comparison | Improved | Same | Worsened | Mean delta | Top-1 lost | Top-1 gained | Exact p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in report["scopes"]:
        for label, key in (
            ("Raw+BF -> reference", "raw_to_reference"),
            ("No horizon -> reference", "horizon_aware_uncertainty"),
        ):
            value = scope[key]
            lines.append(
                f"| {scope['value']} | {label} | {value['improved']} | "
                f"{value['same']} | {value['worsened']} | "
                f"{value['mean_delta']:.4f} | "
                f"{value['baseline_top1_to_reference_non_top1']} | "
                f"{value['baseline_non_top1_to_reference_top1']} | "
                f"{value['top1_mcnemar_two_sided_exact_p']:.4f} |"
            )
    multiplier = report["uncertainty_multipliers"]
    lines.extend([
        "",
        "## Forecast-uncertainty multiplier audit",
        "",
        f"- Metrics: {multiplier['n_metrics']}",
        f"- Maximum: {multiplier['maximum']:.4f}",
        f"- P99: {multiplier['p99']:.4f}",
        f"- >=10: {multiplier['ge_10']} ({multiplier['ge_10_fraction']:.2%})",
        f"- >=50: {multiplier['ge_50']} ({multiplier['ge_50_fraction']:.2%})",
        f"- Cases whose root top-3 includes >=10: {multiplier['root_top3_cases_ge_10']}",
        f"- Cases whose root top-3 includes >=50: {multiplier['root_top3_cases_ge_50']}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    raw_root: Path,
    no_horizon_root: Path,
    reference_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    raw = _read_cases(raw_root)
    no_horizon = _read_cases(no_horizon_root)
    reference = _read_cases(reference_root)
    case_rows = build_case_rows(raw, no_horizon, reference)
    metric_rows = _multiplier_rows(reference)
    multipliers = np.asarray([
        row["max_uncertainty_multiplier"] for row in metric_rows
    ], dtype=float)
    root_maxima = np.asarray([
        row["root_top3_max_uncertainty_multiplier"] for row in case_rows
    ], dtype=float)
    report = {
        "experiment": "final_counterfactual_ar_diagnostics",
        "reference_root": str(reference_root),
        "raw_root": str(raw_root),
        "no_horizon_root": str(no_horizon_root),
        "n_cases": len(case_rows),
        "delta_definition": "baseline_rank - reference_rank; positive means reference improved",
        "scopes": _scoped_summaries(case_rows),
        "uncertainty_multipliers": {
            "n_metrics": int(multipliers.size),
            "maximum": float(np.max(multipliers)),
            "p99": float(np.quantile(multipliers, 0.99)),
            "ge_10": int(np.sum(multipliers >= 10.0)),
            "ge_10_fraction": float(np.mean(multipliers >= 10.0)),
            "ge_50": int(np.sum(multipliers >= 50.0)),
            "ge_50_fraction": float(np.mean(multipliers >= 50.0)),
            "root_top3_cases_ge_10": int(np.sum(root_maxima >= 10.0)),
            "root_top3_cases_ge_50": int(np.sum(root_maxima >= 50.0)),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "case_diagnostics.csv", case_rows)
    _write_csv(output_root / "metric_uncertainty_multipliers.csv", metric_rows)
    (output_root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output_root / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--no-horizon-root", type=Path, default=DEFAULT_NO_HORIZON_ROOT
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = analyze(
        args.raw_root, args.no_horizon_root, args.reference_root, args.output_root
    )
    print(f"Analyzed {report['n_cases']} matched cases; wrote {args.output_root}")


if __name__ == "__main__":
    main()

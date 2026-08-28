"""Summarize one-axis sensitivity of the final Counterfactual AR candidate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = (
    "unit_invariant_r0_98_p3",
    "unit_invariant_r0_95_p3",
    "unit_invariant_r0_99_p3",
    "unit_invariant_r0_98_p1",
    "unit_invariant_r0_98_p5",
    "unit_invariant_no_horizon_uncertainty",
)
REFERENCE = VARIANTS[0]
DEFAULT_RESULTS_ROOT = Path("results/sensitivity/rcaeval_re1")
DEFAULT_PSEUDO_ROOT = Path("results/analysis/ar_final_sensitivity_pseudo_fault")
DEFAULT_OUTPUT_ROOT = Path("results/analysis/final_ar_sensitivity")


def _read_cases(root: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(payload.get("case_id") or "")
        ranking = payload.get("predicted_ranking")
        if not case_id or not ranking:
            raise ValueError(f"case_id and full ranking are required: {path}")
        if case_id in cases:
            raise ValueError(f"Duplicate case_id: {case_id}")
        cases[case_id] = payload
    if not cases:
        raise ValueError(f"No case artifacts under {root}")
    return cases


def _rank(payload: dict[str, Any]) -> int:
    root = str(payload["evaluation_ground_truth"])
    ranking = [str(value) for value in payload["predicted_ranking"]]
    if root not in ranking:
        raise ValueError(f"Root service absent for {payload['case_id']}: {root}")
    return ranking.index(root) + 1


def _macro(summary: dict[str, Any], metric: str) -> float:
    values = [float(row[metric]) for row in summary["summary"].values()]
    return float(np.mean(values))


def _diagnostic_counts(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        metric
        for payload in cases.values()
        for metric in (payload.get("amber_diagnostics") or {}).get("metrics", [])
    ]
    constrained = sum(
        bool(metric.get("ar_stationarity_constrained")) for metric in metrics
    )
    numerical_constants = sum(
        metric.get("ar_input_degenerate_reason") == "numerically_constant_normal"
        for metric in metrics
    )
    return {
        "n_metric_fits": len(metrics),
        "stationarity_constrained": constrained,
        "stationarity_constrained_fraction": (
            float(constrained / len(metrics)) if metrics else 0.0
        ),
        "numerically_constant_normal": numerical_constants,
    }


def _paired_rows(
    reference: dict[str, dict[str, Any]],
    variant: dict[str, dict[str, Any]],
    variant_name: str,
) -> list[dict[str, Any]]:
    if set(reference) != set(variant):
        raise ValueError(f"case_id sets differ for {variant_name}")
    rows: list[dict[str, Any]] = []
    for case_id in sorted(reference):
        ref_payload = reference[case_id]
        var_payload = variant[case_id]
        ref_metadata = (
            ref_payload["dataset"], ref_payload["fault_type"],
            ref_payload["evaluation_ground_truth"],
        )
        var_metadata = (
            var_payload["dataset"], var_payload["fault_type"],
            var_payload["evaluation_ground_truth"],
        )
        if ref_metadata != var_metadata:
            raise ValueError(f"Case metadata differ for {case_id}")
        ref_rank = _rank(ref_payload)
        variant_rank = _rank(var_payload)
        delta = variant_rank - ref_rank
        rows.append({
            "variant": variant_name,
            "case_id": case_id,
            "dataset": ref_metadata[0],
            "fault_type": ref_metadata[1],
            "root_cause_service": ref_metadata[2],
            "reference_rank": ref_rank,
            "variant_rank": variant_rank,
            "delta_variant_minus_reference": delta,
            "reference_movement": (
                "improved" if delta > 0 else "worsened" if delta < 0 else "same"
            ),
            "variant_top1_to_reference_non_top1": bool(
                variant_rank == 1 and ref_rank != 1
            ),
            "variant_non_top1_to_reference_top1": bool(
                variant_rank != 1 and ref_rank == 1
            ),
        })
    return rows


def _paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["reference_movement"] for row in rows)
    deltas = np.asarray([
        row["delta_variant_minus_reference"] for row in rows
    ], dtype=float)
    gains = int(sum(
        row["variant_non_top1_to_reference_top1"] for row in rows
    ))
    losses = int(sum(
        row["variant_top1_to_reference_non_top1"] for row in rows
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
        "reference_improved": int(counts["improved"]),
        "same": int(counts["same"]),
        "reference_worsened": int(counts["worsened"]),
        "mean_variant_minus_reference_rank": float(np.mean(deltas)),
        "median_variant_minus_reference_rank": float(np.median(deltas)),
        "reference_top1_gains": gains,
        "reference_top1_losses": losses,
        "top1_mcnemar_two_sided_exact_p": p_value,
    }


def _pseudo_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    overall = next(
        summary for summary in report["summaries"]
        if summary["scope"] == "overall"
    )
    modes = [key for key in overall if key not in {"scope", "value"}]
    if len(modes) != 1:
        raise ValueError(f"Expected one pseudo-fault mode in {path}: {modes}")
    values = overall[modes[0]]
    return {"mode": modes[0], **values}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Final Counterfactual AR one-axis sensitivity",
        "",
        f"Reference: `{REFERENCE}`. Rank delta is variant - reference, so positive favors the reference.",
        "",
        "Pseudo-fault BF is a relative calibration diagnostic; lower is better, but the absolute BF is not calibrated as a detector threshold.",
        "",
        "| Variant | Macro AC@1 | AC@3 | AC@5 | Avg@5 | Ref improved | Same | Ref worsened | Ref Top-1 gain/loss | Top-1 exact p | Constrained metrics | Runtime sec | Pseudo median max BF | Pseudo P90 max BF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report["variants"]:
        paired = variant["paired_vs_reference"]
        pseudo = variant["pseudo_fault"] or {}
        lines.append(
            f"| {variant['name']} | {variant['macro']['AC@1']:.4f} | "
            f"{variant['macro']['AC@3']:.4f} | {variant['macro']['AC@5']:.4f} | "
            f"{variant['macro']['Avg@5']:.4f} | {paired['reference_improved']} | "
            f"{paired['same']} | {paired['reference_worsened']} | "
            f"{paired['reference_top1_gains']}/{paired['reference_top1_losses']} | "
            f"{paired['top1_mcnemar_two_sided_exact_p']:.4f} | "
            f"{variant['diagnostics']['stationarity_constrained_fraction']:.1%} | "
            f"{variant['runtime_sec']:.1f} | "
            f"{pseudo.get('median_max_service_score', float('nan')):.4f} | "
            f"{pseudo.get('p90_max_service_score', float('nan')):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    results_root: Path,
    pseudo_root: Path,
    output_root: Path,
    variants: tuple[str, ...] = VARIANTS,
) -> dict[str, Any]:
    if REFERENCE not in variants:
        raise ValueError(f"Reference variant is required: {REFERENCE}")
    cases = {
        name: _read_cases(results_root / name / "service") for name in variants
    }
    reference = cases[REFERENCE]
    all_rows: list[dict[str, Any]] = []
    variant_reports: list[dict[str, Any]] = []
    for name in variants:
        summary_path = results_root / name / "summary_service.json"
        if not summary_path.is_file():
            raise ValueError(f"Missing aggregate summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = _paired_rows(reference, cases[name], name)
        all_rows.extend(rows)
        variant_reports.append({
            "name": name,
            "macro": {
                metric: _macro(summary, metric)
                for metric in ("AC@1", "AC@3", "AC@5", "Avg@5")
            },
            "runtime_sec": float(summary.get("total_execution_time_sec", 0.0)),
            "diagnostics": _diagnostic_counts(cases[name]),
            "paired_vs_reference": _paired_summary(rows),
            "pseudo_fault": _pseudo_summary(pseudo_root / name / "summary.json"),
        })
    report = {
        "experiment": "final_counterfactual_ar_sensitivity",
        "reference": REFERENCE,
        "n_cases": len(reference),
        "rank_delta_definition": "variant_rank - reference_rank",
        "variants": variant_reports,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "case_rank_sensitivity.csv", all_rows)
    (output_root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output_root / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = analyze(args.results_root, args.pseudo_root, args.output_root)
    print(f"Analyzed {report['n_cases']} cases across {len(VARIANTS)} variants")


if __name__ == "__main__":
    main()

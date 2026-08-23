"""Paired RCAEval analysis for the direct shared-vs-separate AR Bayes factor.

This script consumes existing service-level result JSON files.  It strictly
joins Raw+BF, the current provisional AMBER candidate, and Direct AR-BF by
``case_id``; it never reruns inference or reconstructs missing rankings.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

try:
    from scripts.analyze_ar_rank_movement import (
        FAULT_TYPES, _complete_ranking, _load_cases, _root_service,
    )
    from scripts.analyze_counterfactual_ar import _paired_comparison
except ModuleNotFoundError:  # Direct execution
    from analyze_ar_rank_movement import (  # type: ignore[no-redef]
        FAULT_TYPES, _complete_ranking, _load_cases, _root_service,
    )
    from analyze_counterfactual_ar import _paired_comparison  # type: ignore[no-redef]


DEFAULT_ROOTS = {
    "raw": Path("results/ablation/rcaeval_re1/no_ar"),
    "stationary_counterfactual_ar_uncertainty": Path(
        "results/ablation/rcaeval_re1/"
        "stationary_counterfactual_ar_uncertainty"
    ),
    "direct_ar_bayes_factor": Path(
        "results/ablation/rcaeval_re1/direct_ar_bayes_factor"
    ),
}

COMPARISONS = (
    ("raw", "direct_ar_bayes_factor"),
    ("stationary_counterfactual_ar_uncertainty", "direct_ar_bayes_factor"),
)


def _service_scores(result: dict[str, Any]) -> dict[str, float]:
    rows = (result.get("amber_diagnostics") or {}).get("services") or []
    scores: dict[str, float] = {}
    for row in rows:
        service = row.get("service")
        score = row.get("score")
        if service is not None and score is not None:
            scores[str(service)] = float(score)
    return scores


def _case_row(
    case_id: str,
    cases: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    values = [method_cases[case_id] for method_cases in cases.values()]
    for field in ("dataset", "fault_type"):
        if len({value.get(field) for value in values}) != 1:
            raise ValueError(f"Metadata mismatch for {case_id}: {field}")
    roots = {_root_service(value) for value in values}
    if len(roots) != 1:
        raise ValueError(f"Ground-truth mismatch for {case_id}")
    root = roots.pop()

    row: dict[str, Any] = {
        "case_id": case_id,
        "dataset": values[0]["dataset"],
        "fault_type": values[0]["fault_type"],
        "root_cause_service": root,
    }
    for method, method_cases in cases.items():
        ranking = _complete_ranking(method_cases[case_id])
        try:
            row[f"{method}_rank"] = ranking.index(root) + 1
        except ValueError as exc:
            raise ValueError(f"Root service absent for {method}/{case_id}") from exc
        row[f"{method}_top1"] = ranking[0]

    direct = cases["direct_ar_bayes_factor"][case_id]
    ranking = _complete_ranking(direct)
    scores = _service_scores(direct)
    if scores:
        if root not in scores:
            raise ValueError(f"Root service score absent for direct/{case_id}")
        competitor = next(service for service in ranking if service != root)
        if competitor not in scores:
            raise ValueError(f"Competitor score absent for direct/{case_id}")
        row.update({
            "direct_root_log_bf": scores[root],
            "direct_top_competitor": competitor,
            "direct_top_competitor_log_bf": scores[competitor],
            "direct_root_margin": scores[root] - scores[competitor],
        })
    else:
        row.update({
            "direct_root_log_bf": None,
            "direct_top_competitor": None,
            "direct_top_competitor_log_bf": None,
            "direct_root_margin": None,
        })
    return row


def _scope_comparisons(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int,
) -> list[dict[str, Any]]:
    return [
        _paired_comparison(
            rows, baseline, candidate,
            bootstrap_samples=bootstrap_samples, seed=seed,
        )
        for baseline, candidate in COMPARISONS
    ]


def _rank_movement(
    rows: list[dict[str, Any]], baseline: str,
) -> dict[str, Any]:
    deltas = [
        int(row[f"{baseline}_rank"])
        - int(row["direct_ar_bayes_factor_rank"])
        for row in rows
    ]
    return {
        "baseline": baseline,
        "n_cases": len(rows),
        "improved": sum(delta > 0 for delta in deltas),
        "same": sum(delta == 0 for delta in deltas),
        "worsened": sum(delta < 0 for delta in deltas),
        "mean_rank_delta": sum(deltas) / len(deltas),
        "median_rank_delta": median(deltas),
        "baseline_top1_to_direct_non_top1": sum(
            int(row[f"{baseline}_rank"]) == 1
            and int(row["direct_ar_bayes_factor_rank"]) != 1
            for row in rows
        ),
        "baseline_non_top1_to_direct_top1": sum(
            int(row[f"{baseline}_rank"]) != 1
            and int(row["direct_ar_bayes_factor_rank"]) == 1
            for row in rows
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Direct AR Bayes Factor RCAEval paired validation",
        "",
        f"Joined cases: {report['n_joined_cases']}",
        "",
        "Differences are Direct AR-BF minus baseline with paired case-level "
        "95% bootstrap intervals.",
        "",
        "| Comparison | Direct AC@1 | Direct AC@3 | Direct AC@5 | Direct Avg@5 | AC@1 difference (95% CI) | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in report["overall"]:
        metrics = comparison["metrics"]
        ac1 = metrics["AC@1"]
        lines.append(
            f"| {comparison['baseline']} → {comparison['candidate']} | "
            f"{ac1['candidate_mean']:.4f} | "
            f"{metrics['AC@3']['candidate_mean']:.4f} | "
            f"{metrics['AC@5']['candidate_mean']:.4f} | "
            f"{metrics['Avg@5']['candidate_mean']:.4f} | "
            f"{ac1['mean_difference']:+.4f} "
            f"[{ac1['ci95_low']:+.4f}, {ac1['ci95_high']:+.4f}] | "
            f"{ac1['mcnemar']['two_sided_exact_p']:.4g} |"
        )
    lines.extend([
        "",
        "## Root-service rank movement",
        "",
        "delta = baseline rank - Direct AR-BF rank; positive means Direct improved.",
        "",
        "| Baseline | Improved | Same | Worsened | Mean delta | Median delta | Top-1 lost | Top-1 gained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for movement in report["rank_movement"]:
        lines.append(
            f"| {movement['baseline']} | {movement['improved']} | "
            f"{movement['same']} | {movement['worsened']} | "
            f"{movement['mean_rank_delta']:+.4f} | "
            f"{movement['median_rank_delta']:+g} | "
            f"{movement['baseline_top1_to_direct_non_top1']} | "
            f"{movement['baseline_non_top1_to_direct_top1']} |"
        )
    lines.extend([
        "",
        "## Direct AR-BF by dataset",
        "",
        "| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset, comparisons in report["by_dataset"].items():
        metrics = comparisons[0]["metrics"]
        lines.append(
            f"| {dataset} | {metrics['AC@1']['candidate_mean']:.4f} | "
            f"{metrics['AC@3']['candidate_mean']:.4f} | "
            f"{metrics['AC@5']['candidate_mean']:.4f} | "
            f"{metrics['Avg@5']['candidate_mean']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    roots: dict[str, Path],
    output_root: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260823,
) -> dict[str, Any]:
    cases = {method: _load_cases(root) for method, root in roots.items()}
    id_sets = {method: set(value) for method, value in cases.items()}
    if len({frozenset(value) for value in id_sets.values()}) != 1:
        raise ValueError(
            "case_id sets differ: "
            + ", ".join(f"{method}={len(ids)}" for method, ids in id_sets.items())
        )
    case_ids = sorted(next(iter(id_sets.values())))
    if not case_ids:
        raise ValueError("No joined service-level cases found")
    rows = [_case_row(case_id, cases) for case_id in case_ids]

    report = {
        "n_joined_cases": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "overall": _scope_comparisons(
            rows, bootstrap_samples=bootstrap_samples, seed=seed,
        ),
        "rank_movement": [
            _rank_movement(rows, baseline)
            for baseline, _ in COMPARISONS
        ],
        "by_dataset": {
            dataset: _scope_comparisons(
                [row for row in rows if row["dataset"] == dataset],
                bootstrap_samples=bootstrap_samples, seed=seed,
            )
            for dataset in sorted({row["dataset"] for row in rows})
        },
        "by_fault_type": {
            fault: _scope_comparisons(
                [row for row in rows if row["fault_type"] == fault],
                bootstrap_samples=bootstrap_samples, seed=seed,
            )
            for fault in FAULT_TYPES
            if any(row["fault_type"] == fault for row in rows)
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "case_ranks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(output_root / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/analysis/direct_ar_bayes_factor"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    report = analyze(
        DEFAULT_ROOTS, args.output_root,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    print(f"Joined {report['n_joined_cases']} cases; wrote {args.output_root}")


if __name__ == "__main__":
    main()

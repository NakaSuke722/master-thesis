"""Paired analysis for Adaptive Direct AR-BF rollback ablations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_ar_rank_movement import FAULT_TYPES, _load_cases
    from scripts.analyze_ar_redesign import _case_row
    from scripts.analyze_counterfactual_ar import _paired_comparison
except ModuleNotFoundError:  # Direct execution
    from analyze_ar_rank_movement import FAULT_TYPES, _load_cases  # type: ignore
    from analyze_ar_redesign import _case_row  # type: ignore
    from analyze_counterfactual_ar import _paired_comparison  # type: ignore


FULL_CANDIDATE = "adaptive_direct_ar_bayes_factor"
ROLLBACKS = (
    "adaptive_direct_no_null_calibration",
    "adaptive_direct_fixed_onset",
    "adaptive_direct_step_only",
    "adaptive_direct_no_step_ramp",
    "adaptive_direct_no_per_row_normalization",
)
DEFAULT_ROOTS = {
    method: Path("results/ablation/rcaeval_re1") / method
    for method in (FULL_CANDIDATE, *ROLLBACKS)
}


def _compare(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int,
) -> list[dict[str, Any]]:
    """Return rollback-minus-full paired comparisons."""
    return [
        _paired_comparison(
            rows,
            FULL_CANDIDATE,
            rollback,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for rollback in ROLLBACKS
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Adaptive Direct AR-BF rollback ablation",
        "",
        f"Joined cases: {report['n_joined_cases']}",
        "",
        "Differences are rollback minus the full adaptive candidate. A negative "
        "difference means the removed component helped the full candidate.",
        "",
        "| Rollback | AC@1 | AC@3 | AC@5 | Avg@5 | "
        "AC@1 difference (95% CI) | Avg@5 difference (95% CI) | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    reference = report["overall"][0]["metrics"]
    lines.append(
        "| Full adaptive (reference) | "
        f"{reference['AC@1']['baseline_mean']:.4f} | "
        f"{reference['AC@3']['baseline_mean']:.4f} | "
        f"{reference['AC@5']['baseline_mean']:.4f} | "
        f"{reference['Avg@5']['baseline_mean']:.4f} | — | — | — |"
    )
    for comparison in report["overall"]:
        metrics = comparison["metrics"]
        ac1 = metrics["AC@1"]
        avg5 = metrics["Avg@5"]
        lines.append(
            f"| {comparison['candidate']} | "
            f"{ac1['candidate_mean']:.4f} | "
            f"{metrics['AC@3']['candidate_mean']:.4f} | "
            f"{metrics['AC@5']['candidate_mean']:.4f} | "
            f"{avg5['candidate_mean']:.4f} | "
            f"{ac1['mean_difference']:+.4f} "
            f"[{ac1['ci95_low']:+.4f}, {ac1['ci95_high']:+.4f}] | "
            f"{avg5['mean_difference']:+.4f} "
            f"[{avg5['ci95_low']:+.4f}, {avg5['ci95_high']:+.4f}] | "
            f"{ac1['mcnemar']['two_sided_exact_p']:.4g} |"
        )

    lines.extend([
        "",
        "## AC@1 by dataset",
        "",
        "| Dataset | Rollback | AC@1 | Difference from full |",
        "|---|---|---:|---:|",
    ])
    for dataset, comparisons in report["by_dataset"].items():
        reference_ac1 = comparisons[0]["metrics"]["AC@1"]["baseline_mean"]
        lines.append(
            f"| {dataset} | Full adaptive (reference) | "
            f"{reference_ac1:.4f} | — |"
        )
        for comparison in comparisons:
            ac1 = comparison["metrics"]["AC@1"]
            lines.append(
                f"| {dataset} | {comparison['candidate']} | "
                f"{ac1['candidate_mean']:.4f} | "
                f"{ac1['mean_difference']:+.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    roots: dict[str, Path],
    output_root: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260824,
) -> dict[str, Any]:
    cases = {method: _load_cases(root) for method, root in roots.items()}
    id_sets = {method: set(value) for method, value in cases.items()}
    if len({frozenset(value) for value in id_sets.values()}) != 1:
        raise ValueError(
            "case_id sets differ: "
            + ", ".join(
                f"{method}={len(ids)}" for method, ids in id_sets.items()
            )
        )
    case_ids = sorted(next(iter(id_sets.values())))
    if not case_ids:
        raise ValueError("No joined service-level cases found")
    rows = [_case_row(case_id, cases) for case_id in case_ids]

    report = {
        "difference_definition": "rollback_minus_full_adaptive",
        "n_joined_cases": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "overall": _compare(
            rows, bootstrap_samples=bootstrap_samples, seed=seed,
        ),
        "by_dataset": {
            dataset: _compare(
                [row for row in rows if row["dataset"] == dataset],
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            for dataset in sorted({row["dataset"] for row in rows})
        },
        "by_fault_type": {
            fault: _compare(
                [row for row in rows if row["fault_type"] == fault],
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            for fault in FAULT_TYPES
            if any(row["fault_type"] == fault for row in rows)
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "case_ranks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_root / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/analysis/adaptive_direct_rollback"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    report = analyze(
        DEFAULT_ROOTS,
        args.output_root,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"Joined {report['n_joined_cases']} cases; wrote {args.output_root}")


if __name__ == "__main__":
    main()

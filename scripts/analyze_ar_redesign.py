"""Stage-wise paired comparison for the RCAEval AR redesign experiment."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_ar_rank_movement import (
        FAULT_TYPES, _complete_ranking, _load_cases, _root_service,
    )
    from scripts.analyze_counterfactual_ar import (
        METHOD_LABELS, _paired_comparison,
    )
except ModuleNotFoundError:  # Direct execution
    from analyze_ar_rank_movement import (  # type: ignore[no-redef]
        FAULT_TYPES, _complete_ranking, _load_cases, _root_service,
    )
    from analyze_counterfactual_ar import (  # type: ignore[no-redef]
        METHOD_LABELS, _paired_comparison,
    )


DEFAULT_ROOTS = {
    "raw": Path("results/ablation/rcaeval_re1/no_ar"),
    "observed_ar": Path("results/main/rcaeval_re1/amber_zenodo_v2"),
    "counterfactual_ar": Path("results/ablation/rcaeval_re1/counterfactual_ar"),
    "stationary_ar": Path("results/ablation/rcaeval_re1/stationary_ar"),
    "stationary_counterfactual_ar": Path(
        "results/ablation/rcaeval_re1/stationary_counterfactual_ar"
    ),
    "stationary_counterfactual_ar_uncertainty": Path(
        "results/ablation/rcaeval_re1/stationary_counterfactual_ar_uncertainty"
    ),
}

STAGE_COMPARISONS = (
    ("observed_ar", "stationary_ar"),
    ("stationary_ar", "stationary_counterfactual_ar"),
    ("stationary_counterfactual_ar", "stationary_counterfactual_ar_uncertainty"),
    ("raw", "stationary_counterfactual_ar_uncertainty"),
    ("counterfactual_ar", "stationary_counterfactual_ar_uncertainty"),
)


def _case_row(case_id: str, cases: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    values = [method_cases[case_id] for method_cases in cases.values()]
    for field in ("dataset", "fault_type"):
        if len({value.get(field) for value in values}) != 1:
            raise ValueError(f"Metadata mismatch for {case_id}: {field}")
    roots = {_root_service(value) for value in values}
    if len(roots) != 1:
        raise ValueError(f"Ground-truth mismatch for {case_id}")
    root = roots.pop()
    row = {
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
    return row


def _scope_comparisons(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    return [
        _paired_comparison(
            rows, baseline, candidate,
            bootstrap_samples=bootstrap_samples, seed=seed,
        )
        for baseline, candidate in STAGE_COMPARISONS
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stationarity and horizon-uncertainty AR redesign",
        "",
        f"Joined cases: {report['n_joined_cases']}",
        "",
        "Candidate-minus-baseline differences use paired case-level bootstrap intervals.",
        "",
        "| Comparison | AC@1 | AC@3 | AC@5 | Avg@5 | AC@1 difference (95% CI) | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in report["overall"]:
        metrics = comparison["metrics"]
        ac1 = metrics["AC@1"]
        lines.append(
            f"| {comparison['baseline']} → {comparison['candidate']} | "
            f"{metrics['AC@1']['candidate_mean']:.4f} | "
            f"{metrics['AC@3']['candidate_mean']:.4f} | "
            f"{metrics['AC@5']['candidate_mean']:.4f} | "
            f"{metrics['Avg@5']['candidate_mean']:.4f} | "
            f"{ac1['mean_difference']:+.4f} "
            f"[{ac1['ci95_low']:+.4f}, {ac1['ci95_high']:+.4f}] | "
            f"{ac1['mcnemar']['two_sided_exact_p']:.4g} |"
        )
    lines.extend([
        "",
        "## Final candidate by dataset",
        "",
        "| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset, comparisons in report["by_dataset"].items():
        final = next(
            item for item in comparisons
            if item["baseline"] == METHOD_LABELS["raw"]
        )
        values = final["metrics"]
        lines.append(
            f"| {dataset} | {values['AC@1']['candidate_mean']:.4f} | "
            f"{values['AC@3']['candidate_mean']:.4f} | "
            f"{values['AC@5']['candidate_mean']:.4f} | "
            f"{values['Avg@5']['candidate_mean']:.4f} |"
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
    rows = [_case_row(case_id, cases) for case_id in case_ids]
    by_dataset = {
        dataset: _scope_comparisons(
            [row for row in rows if row["dataset"] == dataset],
            bootstrap_samples=bootstrap_samples, seed=seed,
        )
        for dataset in sorted({row["dataset"] for row in rows})
    }
    by_fault_type = {
        fault: _scope_comparisons(
            [row for row in rows if row["fault_type"] == fault],
            bootstrap_samples=bootstrap_samples, seed=seed,
        )
        for fault in FAULT_TYPES
        if any(row["fault_type"] == fault for row in rows)
    }
    report = {
        "n_joined_cases": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "overall": _scope_comparisons(
            rows, bootstrap_samples=bootstrap_samples, seed=seed,
        ),
        "by_dataset": by_dataset,
        "by_fault_type": by_fault_type,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "case_ranks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
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
        default=Path("results/analysis/ar_redesign"),
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

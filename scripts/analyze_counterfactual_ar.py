"""Paired analysis of Raw, observed-lag AR, and bounded counterfactual AR.

The analysis is artifact-only: it joins existing RCAEval result JSON files by
``case_id`` and never reruns AMBER.  It reports paired rank transitions,
McNemar exact tests, paired bootstrap intervals, and whether the normal-range
bound was active in the counterfactual root/competitor service top-3 metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scripts.analyze_ar_rank_movement import (
        FAULT_TYPES,
        _complete_ranking,
        _load_cases,
        _root_service,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<name>.py
    from analyze_ar_rank_movement import (  # type: ignore[no-redef]
        FAULT_TYPES,
        _complete_ranking,
        _load_cases,
        _root_service,
    )


METHOD_LABELS = {
    "raw": "Raw+BF",
    "observed_ar": "Observed-lag AR+BF",
    "counterfactual_ar": "Bounded counterfactual AR+BF",
    "stationary_ar": "Stationary observed-lag AR+BF",
    "stationary_counterfactual_ar": "Stationary counterfactual AR+BF",
    "stationary_counterfactual_ar_uncertainty": (
        "Stationary counterfactual AR+BF + horizon uncertainty"
    ),
    "stationary_counterfactual_ar_full_covariance": (
        "Stationary counterfactual AR+BF + full forecast covariance"
    ),
    "direct_ar_bayes_factor": "Direct shared-vs-separate AR BF",
    "intercept_shift_ar_bayes_factor": (
        "Intercept-shift AR BF (shared lags and variance)"
    ),
}


def _rank(result: dict[str, Any], service: str) -> int:
    ranking = _complete_ranking(result)
    try:
        return ranking.index(service) + 1
    except ValueError as exc:
        raise ValueError(
            f"Service {service!r} absent from complete ranking: {result['_path']}"
        ) from exc


def _metric_rows(result: dict[str, Any], service: str) -> list[dict[str, Any]]:
    metrics = (result.get("amber_diagnostics") or {}).get("metrics") or []
    rows = [row for row in metrics if row.get("service") == service]
    return sorted(
        rows,
        key=lambda row: float("-inf") if row.get("score") is None else float(row["score"]),
        reverse=True,
    )


def _clip_summary(result: dict[str, Any], service: str) -> dict[str, Any]:
    top3 = _metric_rows(result, service)[:3]
    if not top3:
        raise ValueError(f"No metric diagnostics for service {service!r}: {result['_path']}")
    fractions = [float(row.get("counterfactual_clipped_fraction") or 0.0) for row in top3]
    counts = [int(row.get("counterfactual_clipped_predictions") or 0) for row in top3]
    return {
        "service": service,
        "metrics": [str(row["metric"]) for row in top3],
        "mean_fraction": float(np.mean(fractions)),
        "max_fraction": float(np.max(fractions)),
        "clipped_predictions": int(sum(counts)),
        "any_clipped": any(count > 0 for count in counts),
    }


def _validate_joined_case(
    case_id: str,
    raw: dict[str, Any],
    observed: dict[str, Any],
    counterfactual: dict[str, Any],
) -> tuple[str, str, str]:
    values = (raw, observed, counterfactual)
    for field in ("dataset", "fault_type"):
        if len({value.get(field) for value in values}) != 1:
            raise ValueError(f"Metadata mismatch for {case_id}: {field}")
    roots = {_root_service(value) for value in values}
    if len(roots) != 1:
        raise ValueError(f"Ground-truth mismatch for {case_id}")
    return raw["dataset"], raw["fault_type"], roots.pop()


def _case_row(
    case_id: str,
    raw: dict[str, Any],
    observed: dict[str, Any],
    counterfactual: dict[str, Any],
) -> dict[str, Any]:
    dataset, fault_type, root = _validate_joined_case(
        case_id, raw, observed, counterfactual
    )
    raw_ranking = _complete_ranking(raw)
    observed_ranking = _complete_ranking(observed)
    cf_ranking = _complete_ranking(counterfactual)
    ranks = {
        "raw": _rank(raw, root),
        "observed_ar": _rank(observed, root),
        "counterfactual_ar": _rank(counterfactual, root),
    }
    cf_competitor = next(service for service in cf_ranking if service != root)
    root_clip = _clip_summary(counterfactual, root)
    competitor_clip = _clip_summary(counterfactual, cf_competitor)
    raw_winner_clip = _clip_summary(counterfactual, raw_ranking[0])
    observed_winner_clip = _clip_summary(counterfactual, observed_ranking[0])
    return {
        "case_id": case_id,
        "dataset": dataset,
        "fault_type": fault_type,
        "root_cause_service": root,
        "raw_rank": ranks["raw"],
        "observed_ar_rank": ranks["observed_ar"],
        "counterfactual_ar_rank": ranks["counterfactual_ar"],
        "raw_to_counterfactual_delta": ranks["raw"] - ranks["counterfactual_ar"],
        "observed_to_counterfactual_delta": (
            ranks["observed_ar"] - ranks["counterfactual_ar"]
        ),
        "raw_top1": raw_ranking[0],
        "observed_ar_top1": observed_ranking[0],
        "counterfactual_ar_top1": cf_ranking[0],
        "counterfactual_top_competitor": cf_competitor,
        "root_top3_clip_mean": root_clip["mean_fraction"],
        "root_top3_clip_max": root_clip["max_fraction"],
        "root_top3_any_clipped": root_clip["any_clipped"],
        "competitor_top3_clip_mean": competitor_clip["mean_fraction"],
        "competitor_top3_clip_max": competitor_clip["max_fraction"],
        "competitor_top3_any_clipped": competitor_clip["any_clipped"],
        "raw_winner_top3_clip_mean": raw_winner_clip["mean_fraction"],
        "raw_winner_top3_any_clipped": raw_winner_clip["any_clipped"],
        "observed_winner_top3_clip_mean": observed_winner_clip["mean_fraction"],
        "observed_winner_top3_any_clipped": observed_winner_clip["any_clipped"],
        "root_or_competitor_any_clipped": (
            root_clip["any_clipped"] or competitor_clip["any_clipped"]
        ),
    }


def _metric_value(rank: int, metric: str) -> float:
    if metric.startswith("AC@"):
        return float(rank <= int(metric.split("@", 1)[1]))
    if metric.startswith("Avg@"):
        k = int(metric.split("@", 1)[1])
        return float(sum(rank <= cutoff for cutoff in range(1, k + 1)) / k)
    raise ValueError(f"Unsupported paired metric: {metric}")


def _mcnemar_exact(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    baseline_only = int(np.sum((baseline == 1) & (candidate == 0)))
    candidate_only = int(np.sum((baseline == 0) & (candidate == 1)))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) for k in range(min(baseline_only, candidate_only) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _paired_bootstrap(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = differences.size
    if n == 0:
        raise ValueError("Cannot bootstrap an empty paired sample")
    draws = np.empty(samples, dtype=float)
    batch = 1000
    for start in range(0, samples, batch):
        size = min(batch, samples - start)
        indices = rng.integers(0, n, size=(size, n))
        draws[start:start + size] = differences[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_difference": float(np.mean(differences)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def _paired_comparison(
    rows: list[dict[str, Any]],
    baseline: str,
    candidate: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in ("AC@1", "AC@3", "AC@5", "Avg@5"):
        baseline_values = np.asarray(
            [_metric_value(int(row[f"{baseline}_rank"]), metric) for row in rows]
        )
        candidate_values = np.asarray(
            [_metric_value(int(row[f"{candidate}_rank"]), metric) for row in rows]
        )
        item = {
            "baseline_mean": float(baseline_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            **_paired_bootstrap(
                candidate_values - baseline_values,
                samples=bootstrap_samples,
                seed=seed,
            ),
        }
        if metric == "AC@1":
            item["mcnemar"] = _mcnemar_exact(baseline_values, candidate_values)
        metrics[metric] = item
    return {
        "baseline": METHOD_LABELS[baseline],
        "candidate": METHOD_LABELS[candidate],
        "n_cases": len(rows),
        "metrics": metrics,
    }


def _clip_group(rows: Iterable[dict[str, Any]], label: str) -> dict[str, Any]:
    values = list(rows)
    n = len(values)
    return {
        "group": label,
        "n_cases": n,
        "root_any_clipped": sum(bool(row["root_top3_any_clipped"]) for row in values),
        "competitor_any_clipped": sum(
            bool(row["competitor_top3_any_clipped"]) for row in values
        ),
        "root_or_competitor_any_clipped": sum(
            bool(row["root_or_competitor_any_clipped"]) for row in values
        ),
        "root_clip_mean": (
            float(np.mean([row["root_top3_clip_mean"] for row in values])) if n else None
        ),
        "competitor_clip_mean": (
            float(np.mean([row["competitor_top3_clip_mean"] for row in values])) if n else None
        ),
    }


def _clip_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = [("all", rows)]
    for baseline in ("raw", "observed_ar"):
        groups.extend([
            (
                f"{baseline}_non_top1_to_counterfactual_top1",
                [
                    row for row in rows
                    if row[f"{baseline}_rank"] != 1
                    and row["counterfactual_ar_rank"] == 1
                ],
            ),
            (
                f"{baseline}_top1_to_counterfactual_non_top1",
                [
                    row for row in rows
                    if row[f"{baseline}_rank"] == 1
                    and row["counterfactual_ar_rank"] != 1
                ],
            ),
            (
                f"counterfactual_rank_better_than_{baseline}",
                [row for row in rows if row[f"{baseline}_rank"] > row["counterfactual_ar_rank"]],
            ),
            (
                f"counterfactual_rank_worse_than_{baseline}",
                [row for row in rows if row[f"{baseline}_rank"] < row["counterfactual_ar_rank"]],
            ),
        ])
    return [_clip_group(group, label) for label, group in groups]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Counterfactual AR paired validation",
        "",
        f"Joined cases: {report['n_joined_cases']}",
        "",
        "## Paired performance",
        "",
        "Differences are candidate minus baseline. CIs are paired 95% bootstrap intervals.",
        "",
        "| Comparison | Metric | Baseline | Candidate | Difference | 95% CI | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for comparison in report["paired_comparisons"]:
        label = f"{comparison['baseline']} → {comparison['candidate']}"
        for metric, values in comparison["metrics"].items():
            p_value = values.get("mcnemar", {}).get("two_sided_exact_p")
            lines.append(
                f"| {label} | {metric} | {values['baseline_mean']:.4f} | "
                f"{values['candidate_mean']:.4f} | {values['mean_difference']:+.4f} | "
                f"[{values['ci95_low']:+.4f}, {values['ci95_high']:+.4f}] | "
                f"{p_value:.4g} |" if p_value is not None else
                f"| {label} | {metric} | {values['baseline_mean']:.4f} | "
                f"{values['candidate_mean']:.4f} | {values['mean_difference']:+.4f} | "
                f"[{values['ci95_low']:+.4f}, {values['ci95_high']:+.4f}] | — |"
            )
    lines.extend([
        "",
        "## Clip attribution",
        "",
        "The competitor is the highest-ranked non-root service in the counterfactual result.",
        "",
        "| Group | Cases | Root any clip | Competitor any clip | Either any clip | Root mean fraction | Competitor mean fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["clip_groups"]:
        root_mean = "—" if row["root_clip_mean"] is None else f"{row['root_clip_mean']:.4f}"
        competitor_mean = (
            "—" if row["competitor_clip_mean"] is None else f"{row['competitor_clip_mean']:.4f}"
        )
        lines.append(
            f"| {row['group']} | {row['n_cases']} | {row['root_any_clipped']} | "
            f"{row['competitor_any_clipped']} | {row['root_or_competitor_any_clipped']} | "
            f"{root_mean} | {competitor_mean} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    raw_root: Path,
    observed_root: Path,
    counterfactual_root: Path,
    output_root: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260823,
) -> dict[str, Any]:
    case_sets = {
        "raw": _load_cases(raw_root),
        "observed_ar": _load_cases(observed_root),
        "counterfactual_ar": _load_cases(counterfactual_root),
    }
    ids = {name: set(cases) for name, cases in case_sets.items()}
    if len({frozenset(value) for value in ids.values()}) != 1:
        raise ValueError(
            "case_id sets differ: "
            + ", ".join(f"{name}={len(value)}" for name, value in ids.items())
        )
    case_ids = sorted(next(iter(ids.values())))
    rows = [
        _case_row(
            case_id,
            case_sets["raw"][case_id],
            case_sets["observed_ar"][case_id],
            case_sets["counterfactual_ar"][case_id],
        )
        for case_id in case_ids
    ]
    comparisons = []
    for baseline in ("raw", "observed_ar"):
        comparisons.append(
            _paired_comparison(
                rows,
                baseline,
                "counterfactual_ar",
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
    by_dataset: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = [
            _paired_comparison(
                subset, baseline, "counterfactual_ar",
                bootstrap_samples=bootstrap_samples, seed=seed,
            )
            for baseline in ("raw", "observed_ar")
        ]
    by_fault_type: dict[str, Any] = {}
    for fault_type in FAULT_TYPES:
        subset = [row for row in rows if row["fault_type"] == fault_type]
        if not subset:
            continue
        by_fault_type[fault_type] = [
            _paired_comparison(
                subset, baseline, "counterfactual_ar",
                bootstrap_samples=bootstrap_samples, seed=seed,
            )
            for baseline in ("raw", "observed_ar")
        ]
    stable_subset = [row for row in rows if not row["root_or_competitor_any_clipped"]]
    report = {
        "n_joined_cases": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "paired_comparisons": comparisons,
        "by_dataset": by_dataset,
        "by_fault_type": by_fault_type,
        "clip_groups": _clip_report(rows),
        "no_root_or_competitor_clip_subset": {
            "n_cases": len(stable_subset),
            "paired_comparisons": [
                _paired_comparison(
                    stable_subset, baseline, "counterfactual_ar",
                    bootstrap_samples=bootstrap_samples, seed=seed,
                )
                for baseline in ("raw", "observed_ar")
            ],
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "case_diagnostics.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_markdown(output_root / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path,
        default=Path("results/ablation/rcaeval_re1/no_ar"),
    )
    parser.add_argument(
        "--observed-root", type=Path,
        default=Path("results/main/rcaeval_re1/amber_zenodo_v2"),
    )
    parser.add_argument(
        "--counterfactual-root", type=Path,
        default=Path("results/ablation/rcaeval_re1/counterfactual_ar"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/analysis/counterfactual_ar"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    report = analyze(
        args.raw_root,
        args.observed_root,
        args.counterfactual_root,
        args.output_root,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"Joined {report['n_joined_cases']} cases; wrote {args.output_root}")


if __name__ == "__main__":
    main()

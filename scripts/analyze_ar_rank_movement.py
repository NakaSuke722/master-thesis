"""Paired root-service rank analysis for RCAEval Raw+BF versus AR+BF.

``delta = raw_rank - ar_rank``: a positive value means AR improved the
root-service rank (a smaller rank is better).  The script never infers a rank
from ``predicted_top_5``.  It uses a complete ``predicted_ranking`` when
available, otherwise AMBER's complete service diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


FAULT_TYPES = ("cpu", "mem", "disk", "delay", "loss")


def _load_cases(root: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("service/*/*.json")):
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        case_id = result.get("case_id")
        if not case_id:
            raise ValueError(f"Missing case_id: {path}")
        if case_id in cases:
            raise ValueError(f"Duplicate case_id {case_id}: {path}")
        if result.get("evaluation_granularity") != "service":
            continue
        result["_path"] = str(path)
        cases[case_id] = result
    return cases


def _root_service(result: dict[str, Any]) -> str:
    ground_truth = result.get("evaluation_ground_truth")
    if isinstance(ground_truth, list):
        if len(ground_truth) != 1:
            raise ValueError(f"Expected one service ground truth in {result['_path']}")
        ground_truth = ground_truth[0]
    if not isinstance(ground_truth, str) or not ground_truth:
        raise ValueError(f"Missing service ground truth in {result['_path']}")
    return ground_truth


def _complete_ranking(result: dict[str, Any]) -> list[str]:
    ranking = result.get("predicted_ranking")
    if isinstance(ranking, list) and ranking:
        return [str(item) for item in ranking]
    services = (result.get("amber_diagnostics") or {}).get("services") or []
    if services:
        ordered = sorted(services, key=lambda row: int(row["rank"]))
        return [str(row["service"]) for row in ordered]
    raise ValueError(
        "No complete service ranking in " + result["_path"] +
        "; predicted_top_5 is insufficient for this analysis."
    )


def _rank(result: dict[str, Any], service: str) -> int:
    ranking = _complete_ranking(result)
    try:
        return ranking.index(service) + 1
    except ValueError as exc:
        raise ValueError(f"Root service {service!r} absent from ranking: {result['_path']}") from exc


def _movement(delta: int) -> str:
    return "improved" if delta > 0 else "worsened" if delta < 0 else "same"


def _case_row(case_id: str, raw: dict[str, Any], ar: dict[str, Any]) -> dict[str, Any]:
    fields = ("dataset", "fault_type")
    for field in fields:
        if raw.get(field) != ar.get(field):
            raise ValueError(f"Metadata mismatch for {case_id}: {field}")
    root_service = _root_service(raw)
    if root_service != _root_service(ar):
        raise ValueError(f"Ground-truth mismatch for {case_id}")
    raw_rank, ar_rank = _rank(raw, root_service), _rank(ar, root_service)
    delta = raw_rank - ar_rank
    return {
        "case_id": case_id,
        "dataset": raw["dataset"],
        "fault_type": raw["fault_type"],
        "root_cause_service": root_service,
        "raw_rank": raw_rank,
        "ar_rank": ar_rank,
        "delta": delta,
        "movement": _movement(delta),
        "raw_result_file": raw["_path"],
        "ar_result_file": ar["_path"],
    }


def _summarize(rows: list[dict[str, Any]], group_name: str, group_value: str) -> dict[str, Any]:
    deltas = [row["delta"] for row in rows]
    n_cases = len(rows)
    counts = {name: sum(row["movement"] == name for row in rows) for name in ("improved", "same", "worsened")}
    return {
        group_name: group_value,
        "n_cases": n_cases,
        **counts,
        **{f"{name}_pct": counts[name] / n_cases if n_cases else 0.0 for name in counts},
        "mean_rank_delta": sum(deltas) / n_cases if n_cases else None,
        "median_rank_delta": median(deltas) if deltas else None,
        "raw_top1_to_ar_non_top1": sum(row["raw_rank"] == 1 and row["ar_rank"] != 1 for row in rows),
        "raw_non_top1_to_ar_top1": sum(row["raw_rank"] != 1 and row["ar_rank"] == 1 for row in rows),
    }


def _group(rows: list[dict[str, Any]], field: str, values: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row[field]].append(row)
    keys = values if values is not None else tuple(sorted(buckets))
    return [_summarize(buckets.get(value, []), field, value) for value in keys]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    overall = report["overall"]
    lines = [
        "# Raw+BF → AR+BF root-service rank movement",
        "",
        report["definition"],
        "",
        f"Joined cases: {report['n_joined_cases']}",
        "",
        "| Scope | Improved | Same | Worsened | Mean delta | Median delta | Raw top-1 → AR non-top-1 | Raw non-top-1 → AR top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = [overall, *report["by_dataset"], *report["by_fault_type"]]
    for summary in summaries:
        scope = summary.get("scope") or summary.get("dataset") or summary.get("fault_type")
        lines.append(
            f"| {scope} | {summary['improved']} ({summary['improved_pct']:.1%}) | "
            f"{summary['same']} ({summary['same_pct']:.1%}) | "
            f"{summary['worsened']} ({summary['worsened_pct']:.1%}) | "
            f"{summary['mean_rank_delta']!s} | {summary['median_rank_delta']!s} | "
            f"{summary['raw_top1_to_ar_non_top1']} | {summary['raw_non_top1_to_ar_top1']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(raw_root: Path, ar_root: Path, output_root: Path) -> dict[str, Any]:
    raw_cases, ar_cases = _load_cases(raw_root), _load_cases(ar_root)
    raw_ids, ar_ids = set(raw_cases), set(ar_cases)
    if raw_ids != ar_ids:
        raise ValueError(
            f"case_id sets differ: raw_only={sorted(raw_ids - ar_ids)}, ar_only={sorted(ar_ids - raw_ids)}"
        )
    rows = [_case_row(case_id, raw_cases[case_id], ar_cases[case_id]) for case_id in sorted(raw_ids)]
    report = {
        "definition": "delta = raw_rank - ar_rank; delta > 0 means AR improved rank.",
        "n_joined_cases": len(rows),
        "overall": _summarize(rows, "scope", "overall"),
        "by_dataset": _group(rows, "dataset"),
        "by_fault_type": _group(rows, "fault_type", FAULT_TYPES),
    }
    _write_json(output_root / "case_rank_movement.json", rows)
    _write_csv(output_root / "case_rank_movement.csv", rows)
    _write_json(output_root / "summary.json", report)
    _write_csv(output_root / "by_dataset.csv", report["by_dataset"])
    _write_csv(output_root / "by_fault_type.csv", report["by_fault_type"])
    _write_markdown(output_root / "summary.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RCAEval root-service ranks: Raw+BF to AR+BF.")
    parser.add_argument("--raw-root", type=Path, default=Path("results/ablation/rcaeval_re1/no_ar"))
    parser.add_argument("--ar-root", type=Path, default=Path("results/main/rcaeval_re1/amber_zenodo_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("results/analysis/ar_rank_movement"))
    args = parser.parse_args()
    report = analyze(args.raw_root, args.ar_root, args.output_root)
    print(f"Joined {report['n_joined_cases']} cases; wrote {args.output_root}")


if __name__ == "__main__":
    main()

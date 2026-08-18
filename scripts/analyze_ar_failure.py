"""Compare AMBER (AR + BF) and no_ar (raw + BF) case by case.

The script reads only result artifacts, so it can be rerun after an experiment
without rerunning inference.  New AMBER artifacts include ``amber_diagnostics``
with the time-series observations needed for representative-case plots.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.aggregation import canonical_metric_name


CASE_KEY = tuple[str, str, int]


def _read_cases(root: Path, granularity: str) -> dict[CASE_KEY, dict[str, Any]]:
    cases: dict[CASE_KEY, dict[str, Any]] = {}
    for path in sorted(root.glob(f"{granularity}/*/*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("evaluation_granularity") != granularity:
            continue
        try:
            key = (payload["dataset"], payload["fault_type"], int(payload["run_id"]))
        except KeyError as exc:
            raise ValueError(f"Missing case identifier in {path}") from exc
        if key in cases:
            raise ValueError(f"Duplicate case {key} under {root}")
        payload["_path"] = str(path)
        cases[key] = payload
    return cases


def _root_rows(result: dict[str, Any], granularity: str) -> list[dict[str, Any]]:
    diagnostics = result.get("amber_diagnostics") or {}
    metrics = diagnostics.get("metrics") or []
    ground_truth = result.get("evaluation_ground_truth")
    if granularity == "metric":
        targets = set(ground_truth if isinstance(ground_truth, list) else [ground_truth])
        return [
            row for row in metrics
            if canonical_metric_name(str(row.get("metric", ""))) in targets
        ]

    services = diagnostics.get("services") or []
    if services:
        return [row for row in services if row.get("service") == ground_truth]
    return [row for row in metrics if row.get("service") == ground_truth]


def _root_summary(result: dict[str, Any], granularity: str) -> dict[str, Any] | None:
    rows = _root_rows(result, granularity)
    if not rows:
        return None
    usable = [row for row in rows if row.get("score") is not None]
    return {
        "members": [row.get("metric", row.get("service")) for row in rows],
        "score": max((float(row["score"]) for row in usable), default=None),
        "rank": min((int(row["rank"]) for row in rows if row.get("rank") is not None), default=None),
        "sum_phi": sum(
            float(value)
            for row in rows
            for value in (row.get("ar_coefficients") or [])[1:]
        ) if granularity == "metric" else None,
    }


def _case_row(
    key: CASE_KEY,
    amber: dict[str, Any],
    no_ar: dict[str, Any],
    granularity: str,
) -> dict[str, Any]:
    dataset, fault_type, run_id = key
    row: dict[str, Any] = {
        "dataset": dataset,
        "fault_type": fault_type,
        "run_id": run_id,
        "granularity": granularity,
        "amber_result_file": amber["_path"],
        "no_ar_result_file": no_ar["_path"],
    }
    metric_names = sorted(set(amber.get("metrics", {})) | set(no_ar.get("metrics", {})))
    for name in metric_names:
        amber_value = amber.get("metrics", {}).get(name)
        raw_value = no_ar.get("metrics", {}).get(name)
        row[f"amber_{name}"] = amber_value
        row[f"no_ar_{name}"] = raw_value
        row[f"delta_{name}"] = (
            float(amber_value) - float(raw_value)
            if amber_value is not None and raw_value is not None else None
        )

    ar_root = _root_summary(amber, granularity)
    raw_root = _root_summary(no_ar, granularity)
    row["root_cause_members"] = (ar_root or raw_root or {}).get("members", [])
    row["ar_root_score"] = ar_root["score"] if ar_root else None
    row["raw_root_score"] = raw_root["score"] if raw_root else None
    row["delta_root_score"] = (
        ar_root["score"] - raw_root["score"]
        if ar_root and raw_root and ar_root["score"] is not None and raw_root["score"] is not None else None
    )
    row["ar_root_rank"] = ar_root["rank"] if ar_root else None
    row["raw_root_rank"] = raw_root["rank"] if raw_root else None
    row["delta_root_rank"] = (
        ar_root["rank"] - raw_root["rank"]
        if ar_root and raw_root and ar_root["rank"] is not None and raw_root["rank"] is not None else None
    )
    row["sum_phi"] = ar_root["sum_phi"] if ar_root else None
    return row


def _aggregate(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    numeric = sorted({
        key for row in rows for key, value in row.items()
        if key.startswith("delta_") and isinstance(value, (int, float))
    })
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, members in sorted(groups.items()):
        summary = dict(zip(group_fields, key))
        summary["n_cases"] = len(members)
        for field in numeric:
            values = [float(row[field]) for row in members if row.get(field) is not None]
            summary[f"mean_{field}"] = sum(values) / len(values) if values else None
            summary[f"n_{field}"] = len(values)
        output.append(summary)
    return output


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def analyze(granularity: str, amber_root: Path, no_ar_root: Path, output_root: Path) -> dict[str, Any]:
    amber_cases = _read_cases(amber_root, granularity)
    no_ar_cases = _read_cases(no_ar_root, granularity)
    common = sorted(set(amber_cases) & set(no_ar_cases))
    if not common:
        raise FileNotFoundError(
            f"No matching {granularity} cases between {amber_root} and {no_ar_root}"
        )
    case_rows = [_case_row(key, amber_cases[key], no_ar_cases[key], granularity) for key in common]
    report = {
        "granularity": granularity,
        "n_joined_cases": len(case_rows),
        "amber_only_cases": [list(key) for key in sorted(set(amber_cases) - set(no_ar_cases))],
        "no_ar_only_cases": [list(key) for key in sorted(set(no_ar_cases) - set(amber_cases))],
        "by_dataset": _aggregate(case_rows, ["dataset"]),
        "by_fault_type": _aggregate(case_rows, ["fault_type"]),
        "cases": case_rows,
    }
    _write_json(output_root / f"case_deltas_{granularity}.json", case_rows)
    _write_csv(output_root / f"case_deltas_{granularity}.csv", case_rows)
    _write_json(output_root / f"summary_{granularity}.json", report)
    _write_csv(output_root / f"by_dataset_{granularity}.csv", report["by_dataset"])
    _write_csv(output_root / f"by_fault_type_{granularity}.csv", report["by_fault_type"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose AR residualization failures.")
    parser.add_argument("--granularity", choices=["service", "metric"], required=True)
    parser.add_argument("--amber-root", type=Path, default=Path("results/main/amber"))
    parser.add_argument("--no-ar-root", type=Path, default=Path("results/ablation/no_ar"))
    parser.add_argument("--output-root", type=Path, default=Path("results/analysis/ar_failure"))
    args = parser.parse_args()
    report = analyze(args.granularity, args.amber_root, args.no_ar_root, args.output_root)
    print(f"Joined {report['n_joined_cases']} {args.granularity} cases into {args.output_root}")


if __name__ == "__main__":
    main()

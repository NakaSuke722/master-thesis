"""Run and compare the normal-only unit-invariance matched control.

Both variants use the current processed RCAEval data and current source code.
The control keeps AR inputs in their recorded units, while the treatment uses
normal-only metric-wise standardization.  Case rows are joined strictly so the
reported deltas are paired rather than differences between independent
aggregate summaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.analyze_ar_pseudo_fault_calibration import analyze


DEFAULT_CONTROL_CONFIG = Path(
    "configs/ablation/rcaeval_re1_zenodo_v2/"
    "stationary_counterfactual_ar_uncertainty_unit_invariance_control.yaml"
)
DEFAULT_TREATMENT_CONFIG = Path(
    "configs/ablation/rcaeval_re1_zenodo_v2/"
    "stationary_counterfactual_ar_uncertainty_unit_invariant.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/analysis/ar_pseudo_fault_calibration_unit_matched"
)
DEFAULT_MODE = "stationary_counterfactual_ar_uncertainty"

CASE_KEYS = [
    "case_id",
    "dataset",
    "fault_type",
    "root_cause_service",
    "residualization",
]
VALUE_COLUMNS = [
    "max_service_score",
    "median_service_score",
    "p90_service_score",
    "positive_service_fraction",
    "n_metrics",
    "n_services",
]


def pair_case_rows(
    control: pd.DataFrame,
    treatment: pd.DataFrame,
) -> pd.DataFrame:
    """Strictly pair cases and return treatment-minus-control deltas."""
    missing_control = set(CASE_KEYS + VALUE_COLUMNS) - set(control.columns)
    missing_treatment = set(CASE_KEYS + VALUE_COLUMNS) - set(treatment.columns)
    if missing_control or missing_treatment:
        raise ValueError(
            "Missing matched-control columns: "
            f"control={sorted(missing_control)}, "
            f"treatment={sorted(missing_treatment)}"
        )

    merged = control[CASE_KEYS + VALUE_COLUMNS].merge(
        treatment[CASE_KEYS + VALUE_COLUMNS],
        on=CASE_KEYS,
        how="outer",
        suffixes=("_control", "_unit_invariant"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = merged.loc[merged["_merge"] != "both", CASE_KEYS + ["_merge"]]
    if not unmatched.empty:
        sample = unmatched.head(5).to_dict(orient="records")
        raise ValueError(f"Matched-control case sets differ: {sample}")
    merged = merged.drop(columns="_merge")

    for column in VALUE_COLUMNS:
        merged[f"{column}_delta"] = (
            merged[f"{column}_unit_invariant"]
            - merged[f"{column}_control"]
        )
    return merged.sort_values(CASE_KEYS).reset_index(drop=True)


def _variant_summary(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_cases": int(len(rows)),
        "median_max_service_score": float(rows["max_service_score"].median()),
        "p90_max_service_score": float(rows["max_service_score"].quantile(0.9)),
        "median_of_median_service_score": float(
            rows["median_service_score"].median()
        ),
        "mean_positive_service_fraction": float(
            rows["positive_service_fraction"].mean()
        ),
    }


def _paired_summary(rows: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"n_cases": int(len(rows))}
    for column in (
        "max_service_score",
        "median_service_score",
        "positive_service_fraction",
    ):
        deltas = rows[f"{column}_delta"].to_numpy(dtype=float)
        result[column] = {
            "median_delta": float(np.median(deltas)),
            "mean_delta": float(np.mean(deltas)),
            "unit_invariant_lower": int(np.sum(deltas < 0.0)),
            "same": int(np.sum(deltas == 0.0)),
            "unit_invariant_higher": int(np.sum(deltas > 0.0)),
        }
    return result


def _build_report(
    control: pd.DataFrame,
    treatment: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    control_config: Path,
    treatment_config: Path,
    fit_fraction: float,
) -> dict[str, Any]:
    scopes: list[dict[str, Any]] = []
    for scope, value in [("overall", "overall")]:
        scopes.append({
            "scope": scope,
            "value": value,
            "control": _variant_summary(control),
            "unit_invariant": _variant_summary(treatment),
            "paired_delta_unit_invariant_minus_control": _paired_summary(paired),
        })
    for dataset in sorted(paired["dataset"].unique()):
        scopes.append({
            "scope": "dataset",
            "value": str(dataset),
            "control": _variant_summary(control[control["dataset"] == dataset]),
            "unit_invariant": _variant_summary(
                treatment[treatment["dataset"] == dataset]
            ),
            "paired_delta_unit_invariant_minus_control": _paired_summary(
                paired[paired["dataset"] == dataset]
            ),
        })
    return {
        "experiment": "ar_unit_invariance_matched_control",
        "control_config": str(control_config),
        "treatment_config": str(treatment_config),
        "fit_fraction": fit_fraction,
        "n_cases": int(len(paired)),
        "delta_definition": "unit_invariant - control",
        "summaries": scopes,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AR unit-invariance matched control: normal-only pseudo-fault",
        "",
        "Both variants use the same current code, processed data, cases, and split.",
        "The only model difference is `ar_input_scaling`: `none` (control) versus `normal_standard` (unit-invariant).",
        "",
        "Paired delta = unit-invariant - control. For BF calibration statistics, a negative delta means less no-fault evidence after unit standardization.",
        "",
        f"Cases: {report['n_cases']}",
        "",
        "| Scope | Variant | Cases | Median max BF | P90 max BF | Median service BF | Positive service fraction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in report["summaries"]:
        for variant in ("control", "unit_invariant"):
            values = summary[variant]
            lines.append(
                f"| {summary['value']} | {variant} | {values['n_cases']} | "
                f"{values['median_max_service_score']:.4f} | "
                f"{values['p90_max_service_score']:.4f} | "
                f"{values['median_of_median_service_score']:.4f} | "
                f"{values['mean_positive_service_fraction']:.1%} |"
            )
    lines.extend([
        "",
        "## Paired deltas",
        "",
        "| Scope | Statistic | Median delta | Mean delta | Unit-invariant lower | Same | Unit-invariant higher |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for summary in report["summaries"]:
        paired = summary["paired_delta_unit_invariant_minus_control"]
        for statistic in (
            "max_service_score",
            "median_service_score",
            "positive_service_fraction",
        ):
            values = paired[statistic]
            lines.append(
                f"| {summary['value']} | {statistic} | "
                f"{values['median_delta']:.4f} | {values['mean_delta']:.4f} | "
                f"{values['unit_invariant_lower']} | {values['same']} | "
                f"{values['unit_invariant_higher']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_matched_control(
    processed_root: Path,
    control_config: Path,
    treatment_config: Path,
    output_root: Path,
    *,
    mode: str = DEFAULT_MODE,
    fit_fraction: float = 0.5,
    limit: int | None = None,
) -> dict[str, Any]:
    control_root = output_root / "control"
    treatment_root = output_root / "unit_invariant"
    analyze(
        processed_root,
        control_config,
        control_root,
        modes=(mode,),
        fit_fraction=fit_fraction,
        limit=limit,
    )
    analyze(
        processed_root,
        treatment_config,
        treatment_root,
        modes=(mode,),
        fit_fraction=fit_fraction,
        limit=limit,
    )

    control = pd.read_csv(control_root / "case_calibration.csv")
    treatment = pd.read_csv(treatment_root / "case_calibration.csv")
    paired = pair_case_rows(control, treatment)

    combined = pd.concat([
        control.assign(variant="control"),
        treatment.assign(variant="unit_invariant"),
    ], ignore_index=True)
    combined.to_csv(output_root / "case_calibration.csv", index=False)
    paired.to_csv(output_root / "case_deltas.csv", index=False)

    report = _build_report(
        control,
        treatment,
        paired,
        control_config=control_config,
        treatment_config=treatment_config,
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
    parser.add_argument(
        "--control-config", type=Path, default=DEFAULT_CONTROL_CONFIG
    )
    parser.add_argument(
        "--treatment-config", type=Path, default=DEFAULT_TREATMENT_CONFIG
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    report = analyze_matched_control(
        args.processed_root,
        args.control_config,
        args.treatment_config,
        args.output_root,
        mode=args.mode,
        fit_fraction=args.fit_fraction,
        limit=args.limit,
    )
    print(
        f"Paired {report['n_cases']} normal-only cases; "
        f"wrote {args.output_root}"
    )


if __name__ == "__main__":
    main()

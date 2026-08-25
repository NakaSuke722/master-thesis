"""Synthetic convergence and calibration checks for BSRC variance integration."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from models.ar_bayes_factor import (
    ARBayesFactorPrior,
    ARRegimeShiftPrior,
    ar_shrinkage_regime_bayes_factor,
)


METHODS = (
    ("fixed_q4", "fixed_gh", 4),
    ("fixed_q8", "fixed_gh", 8),
    ("adaptive_q7", "adaptive_gh", 7),
    ("adaptive_q11", "adaptive_gh", 11),
    ("adaptive_q15", "adaptive_gh", 15),
)
VARIANCE_RATIOS = (1.0, 1.5, 2.0, 4.0, 8.0, 16.0)


def _simulate_ar1(
    rng: np.random.Generator,
    *,
    pre_samples: int,
    post_samples: int,
    variance_ratio: float,
    phi: float = 0.6,
    burn_in: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    if variance_ratio <= 0:
        raise ValueError("variance_ratio must be positive")
    value = 0.0
    normal = np.empty(burn_in + pre_samples, dtype=float)
    for index in range(normal.size):
        value = phi * value + rng.normal()
        normal[index] = value
    pre = normal[-pre_samples:]
    post = np.empty(post_samples, dtype=float)
    for index in range(post_samples):
        value = phi * value + rng.normal(scale=np.sqrt(variance_ratio))
        post[index] = value
    return pre, post


def run_validation(
    *,
    repetitions: int = 50,
    pre_samples: int = 300,
    post_samples: int = 300,
    seed: int = 20260825,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    rng = np.random.default_rng(seed)
    prior = ARBayesFactorPrior(
        intercept_precision=0.1,
        lag_precision=10.0,
        alpha=5.0,
        beta=4.0,
    )
    rows: list[dict[str, object]] = []
    for true_ratio in VARIANCE_RATIOS:
        for repetition in range(1, repetitions + 1):
            pre, post = _simulate_ar1(
                rng,
                pre_samples=pre_samples,
                post_samples=post_samples,
                variance_ratio=true_ratio,
            )
            for method_name, integration, points in METHODS:
                result = ar_shrinkage_regime_bayes_factor(
                    pre,
                    post,
                    order=1,
                    prior=prior,
                    regime_prior=ARRegimeShiftPrior(
                        inclusion_probability=0.0,
                        variance_inclusion_probability=1.0,
                        log_variance_sd=0.7,
                        variance_integration=integration,
                        variance_quadrature_points=points,
                    ),
                    posterior_detail="map",
                )
                map_model = result["posterior_map"]
                rows.append({
                    "true_variance_ratio": true_ratio,
                    "repetition": repetition,
                    "method": method_name,
                    "log_bayes_factor": result["log_bayes_factor"],
                    "posterior_mean_variance_ratio": result[
                        "posterior_variance_ratio_mean"
                    ],
                    "map_variance_ratio": map_model["variance_ratio"],
                    "map_is_largest_fixed_node": bool(
                        integration == "fixed_gh"
                        and map_model is not None
                        and map_model["log_variance_ratio"]
                        == max(
                            candidate["log_variance_ratio"]
                            for candidate in result["posterior_models"]
                        )
                    ),
                })

    keyed = {
        (row["true_variance_ratio"], row["repetition"], row["method"]): row
        for row in rows
    }
    convergence = []
    for true_ratio in VARIANCE_RATIOS:
        differences = []
        ratio_differences = []
        for repetition in range(1, repetitions + 1):
            q11 = keyed[(true_ratio, repetition, "adaptive_q11")]
            q15 = keyed[(true_ratio, repetition, "adaptive_q15")]
            differences.append(abs(
                float(q11["log_bayes_factor"])
                - float(q15["log_bayes_factor"])
            ))
            ratio_differences.append(abs(
                float(q11["posterior_mean_variance_ratio"])
                - float(q15["posterior_mean_variance_ratio"])
            ))
        convergence.append({
            "true_variance_ratio": true_ratio,
            "median_abs_log_bf_q11_q15": float(np.median(differences)),
            "max_abs_log_bf_q11_q15": float(np.max(differences)),
            "median_abs_ratio_mean_q11_q15": float(
                np.median(ratio_differences)
            ),
            "max_abs_ratio_mean_q11_q15": float(
                np.max(ratio_differences)
            ),
        })

    summaries = []
    for true_ratio in VARIANCE_RATIOS:
        for method_name, _, _ in METHODS:
            group = [
                row for row in rows
                if row["true_variance_ratio"] == true_ratio
                and row["method"] == method_name
            ]
            estimates = np.asarray([
                row["posterior_mean_variance_ratio"] for row in group
            ], dtype=float)
            log_bfs = np.asarray([
                row["log_bayes_factor"] for row in group
            ], dtype=float)
            summaries.append({
                "true_variance_ratio": true_ratio,
                "method": method_name,
                "median_log_bayes_factor": float(np.median(log_bfs)),
                "median_posterior_mean_variance_ratio": float(
                    np.median(estimates)
                ),
                "median_absolute_ratio_error": float(
                    np.median(np.abs(estimates - true_ratio))
                ),
                "largest_fixed_node_fraction": float(np.mean([
                    row["map_is_largest_fixed_node"] for row in group
                ])),
            })
    report = {
        "schema_version": 1,
        "purpose": "bsrc_variance_integration_convergence_and_calibration",
        "seed": seed,
        "repetitions": repetitions,
        "pre_samples": pre_samples,
        "post_samples": post_samples,
        "methods": [item[0] for item in METHODS],
        "ar_bayes_prior": {
            "intercept_precision": prior.intercept_precision,
            "lag_precision": prior.lag_precision,
            "alpha": prior.alpha,
            "beta": prior.beta,
        },
        "regime_shift_prior": {
            "inclusion_probability": 0.0,
            "variance_inclusion_probability": 1.0,
            "log_variance_sd": 0.7,
        },
        "convergence": convergence,
        "summaries": summaries,
    }
    return rows, report


def write_outputs(
    output_root: Path,
    rows: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "replicates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# BSRC-AR variance integration synthetic validation",
        "",
        "Adaptive q11とq15の差が十分小さいかを収束診断とする。",
        "",
        "| True ratio | Median abs log BF diff | Max abs log BF diff | "
        "Median abs ratio diff | Max abs ratio diff |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in report["convergence"]:
        lines.append(
            f"| {item['true_variance_ratio']:.1f} | "
            f"{item['median_abs_log_bf_q11_q15']:.3e} | "
            f"{item['max_abs_log_bf_q11_q15']:.3e} | "
            f"{item['median_abs_ratio_mean_q11_q15']:.3e} | "
            f"{item['max_abs_ratio_mean_q11_q15']:.3e} |"
        )
    (output_root / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--pre-samples", type=int, default=300)
    parser.add_argument("--post-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/analysis/bsrc_variance_integration_synthetic"
        ),
    )
    args = parser.parse_args()
    rows, report = run_validation(
        repetitions=args.repetitions,
        pre_samples=args.pre_samples,
        post_samples=args.post_samples,
        seed=args.seed,
    )
    write_outputs(args.output_root, rows, report)
    print(args.output_root / "summary.md")


if __name__ == "__main__":
    main()

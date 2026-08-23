"""Synthetic validation for the shared-vs-separate AR Bayes Factor."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.ar_bayes_factor import ARBayesFactorPrior, ar_change_bayes_factor


@dataclass(frozen=True)
class Scenario:
    name: str
    post_mean: float = 0.0
    post_phi: float = 0.65
    post_sigma: float = 1.0
    initial_spike: float = 0.0
    expected_change: bool = True


SCENARIOS = (
    Scenario("no_change", expected_change=False),
    Scenario("persistent_mean_shift", post_mean=2.0),
    Scenario("ar_coefficient_change", post_phi=-0.20),
    Scenario("innovation_variance_change", post_sigma=2.0),
    Scenario("single_spike", initial_spike=6.0, expected_change=False),
)


def _simulate_ar1(
    rng: np.random.Generator,
    *,
    pre_samples: int,
    post_samples: int,
    burn_in: int,
    pre_mean: float = 0.0,
    pre_phi: float = 0.65,
    pre_sigma: float = 1.0,
    post_mean: float = 0.0,
    post_phi: float = 0.65,
    post_sigma: float = 1.0,
    initial_spike: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not abs(pre_phi) < 1.0 or not abs(post_phi) < 1.0:
        raise ValueError("Synthetic AR(1) coefficients must be stationary")
    if min(pre_samples, post_samples, burn_in) <= 0:
        raise ValueError("Synthetic segment lengths must be positive")

    value = float(pre_mean)
    pre_all = np.empty(burn_in + pre_samples, dtype=float)
    pre_intercept = (1.0 - pre_phi) * pre_mean
    for index in range(pre_all.size):
        value = pre_intercept + pre_phi * value + rng.normal(0.0, pre_sigma)
        pre_all[index] = value
    pre = pre_all[-pre_samples:]

    post = np.empty(post_samples, dtype=float)
    post_intercept = (1.0 - post_phi) * post_mean
    for index in range(post_samples):
        innovation = rng.normal(0.0, post_sigma)
        if index == 0:
            innovation += initial_spike
        value = post_intercept + post_phi * value + innovation
        post[index] = value
    return pre, post


def run_validation(
    *,
    repetitions: int = 200,
    pre_samples: int = 300,
    post_samples: int = 300,
    burn_in: int = 200,
    seed: int = 20260823,
    ar_order: int = 1,
    prior: ARBayesFactorPrior | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    rng = np.random.default_rng(seed)
    model_prior = prior or ARBayesFactorPrior()
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for repetition in range(1, repetitions + 1):
            pre, post = _simulate_ar1(
                rng,
                pre_samples=pre_samples,
                post_samples=post_samples,
                burn_in=burn_in,
                post_mean=scenario.post_mean,
                post_phi=scenario.post_phi,
                post_sigma=scenario.post_sigma,
                initial_spike=scenario.initial_spike,
            )
            result = ar_change_bayes_factor(
                pre, post, order=ar_order, prior=model_prior
            )
            pre_posterior = result["posterior_pre"]
            post_posterior = result["posterior_post"]
            rows.append({
                "scenario": scenario.name,
                "expected_change": scenario.expected_change,
                "repetition": repetition,
                "log_bayes_factor": result["log_bayes_factor"],
                "pre_intercept_mean": pre_posterior["coefficient_mean"][0],
                "post_intercept_mean": post_posterior["coefficient_mean"][0],
                "pre_phi1_mean": pre_posterior["coefficient_mean"][1],
                "post_phi1_mean": post_posterior["coefficient_mean"][1],
                "pre_innovation_variance_mean": (
                    pre_posterior["innovation_variance_mean"]
                ),
                "post_innovation_variance_mean": (
                    post_posterior["innovation_variance_mean"]
                ),
            })

    changed_strong_fraction_min = 0.80
    unchanged_positive_fraction_max = 0.10
    summaries = []
    for scenario in SCENARIOS:
        values = np.asarray([
            row["log_bayes_factor"]
            for row in rows
            if row["scenario"] == scenario.name
        ], dtype=float)
        positive_fraction = float(np.mean(values > 0.0))
        strong_evidence_fraction = float(np.mean(values > np.log(10.0)))
        passed = (
            strong_evidence_fraction >= changed_strong_fraction_min
            if scenario.expected_change
            else positive_fraction <= unchanged_positive_fraction_max
        )
        summaries.append({
            "scenario": scenario.name,
            "expected_change": scenario.expected_change,
            "repetitions": values.size,
            "mean_log_bayes_factor": float(np.mean(values)),
            "median_log_bayes_factor": float(np.median(values)),
            "q05_log_bayes_factor": float(np.quantile(values, 0.05)),
            "q95_log_bayes_factor": float(np.quantile(values, 0.95)),
            "positive_fraction": positive_fraction,
            "strong_evidence_fraction": strong_evidence_fraction,
            "passed": passed,
        })
    report = {
        "schema_version": 1,
        "hypothesis": "shared_ar_vs_separate_ar",
        "seed": seed,
        "ar_order": ar_order,
        "pre_samples": pre_samples,
        "post_samples": post_samples,
        "burn_in": burn_in,
        "prior": {
            "intercept_mean": model_prior.intercept_mean,
            "lag_mean": model_prior.lag_mean,
            "intercept_precision": model_prior.intercept_precision,
            "lag_precision": model_prior.lag_precision,
            "alpha": model_prior.alpha,
            "beta": model_prior.beta,
        },
        "validation_rule": {
            "changed_strong_evidence_fraction_min": changed_strong_fraction_min,
            "unchanged_positive_fraction_max": unchanged_positive_fraction_max,
        },
        "all_checks_passed": all(item["passed"] for item in summaries),
        "summaries": summaries,
    }
    return rows, report


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# Direct AR Bayes Factor synthetic validation",
        "",
        (
            f"Seed: {report['seed']}; AR order: {report['ar_order']}; "
            f"pre/post samples: {report['pre_samples']}/{report['post_samples']}."
        ),
        "",
        "`log BF > 0` supports separate pre/post AR processes; "
        "`log BF > log(10)` is counted as strong evidence.",
        "",
        f"Overall validation: **{'PASS' if report['all_checks_passed'] else 'FAIL'}**.",
        "",
        "| Scenario | Expected structural change | Median log BF | 5%-95% | Positive | Strong | Check |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["summaries"]:
        lines.append(
            f"| {item['scenario']} | {item['expected_change']} | "
            f"{item['median_log_bayes_factor']:.4f} | "
            f"[{item['q05_log_bayes_factor']:.4f}, "
            f"{item['q95_log_bayes_factor']:.4f}] | "
            f"{item['positive_fraction']:.1%} | "
            f"{item['strong_evidence_fraction']:.1%} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "The single-spike scenario is a transient innovation propagated by the "
        "unchanged AR process, not a persistent parameter change.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    output_root: Path,
    rows: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "replicates.csv").open(
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--pre-samples", type=int, default=300)
    parser.add_argument("--post-samples", type=int, default=300)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--ar-order", type=int, default=1)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/analysis/ar_bayes_factor_synthetic"),
    )
    args = parser.parse_args()
    rows, report = run_validation(
        repetitions=args.repetitions,
        pre_samples=args.pre_samples,
        post_samples=args.post_samples,
        burn_in=args.burn_in,
        seed=args.seed,
        ar_order=args.ar_order,
    )
    write_outputs(args.output_root, rows, report)
    print(f"Scored {len(rows)} synthetic series; wrote {args.output_root}")


if __name__ == "__main__":
    main()

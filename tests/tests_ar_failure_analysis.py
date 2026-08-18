import json

from scripts.analyze_ar_failure import analyze


def _case(score, rank, ac1, residualization):
    return {
        "dataset": "demo",
        "fault_type": "checkout_cpu",
        "run_id": 1,
        "evaluation_granularity": "metric",
        "metrics": {"AC@1": ac1, "Avg@1": ac1},
        "evaluation_ground_truth": ["checkout_cpu"],
        "amber_diagnostics": {
            "residualization": residualization,
            "metrics": [{
                "metric": "checkout_cpu",
                "service": "checkout",
                "score": score,
                "rank": rank,
                "ar_coefficients": [1.0, 0.2, 0.3] if residualization == "ar" else [],
                "raw_normal": [1.0, 2.0],
                "raw_abnormal": [3.0],
                "ar_prediction_normal": [1.0],
                "ar_prediction_abnormal": [2.0],
                "ar_residual_normal": [0.0],
                "ar_residual_abnormal": [1.0],
                "standardized_residual_normal": [0.0],
                "standardized_residual_abnormal": [1.0],
            }],
        },
    }


def test_analyze_joins_cases_and_writes_delta_reports(tmp_path):
    amber_dir = tmp_path / "results/main/amber/metric/demo"
    raw_dir = tmp_path / "results/ablation/no_ar/metric/demo"
    amber_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    with (amber_dir / "checkout_cpu_run1.json").open("w") as handle:
        json.dump(_case(2.0, 3, 0.0, "ar"), handle)
    with (raw_dir / "checkout_cpu_run1.json").open("w") as handle:
        json.dump(_case(5.0, 1, 1.0, "raw"), handle)

    output = tmp_path / "analysis/ar_failure"
    report = analyze("metric", tmp_path / "results/main/amber", tmp_path / "results/ablation/no_ar", output)

    assert report["n_joined_cases"] == 1
    row = report["cases"][0]
    assert row["delta_AC@1"] == -1.0
    assert row["delta_root_score"] == -3.0
    assert row["delta_root_rank"] == 2
    assert row["sum_phi"] == 0.5
    assert (output / "case_deltas_metric.csv").is_file()
    assert (output / "summary_metric.json").is_file()

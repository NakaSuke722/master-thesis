import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.analyze_ar_unit_invariance import (
    AffineTransform,
    _apply_affine,
    _compare_results,
    _summarize,
    analyze,
)


def test_affine_transform_changes_only_numeric_values():
    frame = pd.DataFrame({
        "float_metric": [1.0, 2.0],
        "integer_metric": [10, 20],
        "label": ["a", "b"],
    })

    transformed = _apply_affine(frame, AffineTransform("unit", 1000.0, 5.0))

    assert transformed["float_metric"].tolist() == [1005.0, 2005.0]
    assert transformed["integer_metric"].tolist() == [10005.0, 20005.0]
    assert transformed["label"].tolist() == ["a", "b"]
    assert frame["integer_metric"].tolist() == [10, 20]


def test_result_comparison_detects_rank_and_root_movement():
    baseline = pd.DataFrame({
        "service": ["root", "other"],
        "score": [4.0, 3.0],
    })
    transformed = pd.DataFrame({
        "service": ["other", "root"],
        "score": [5.0, 2.0],
    })
    diagnostics = {
        "root_cpu": {
            "ar_coefficients": [0.0, 0.5],
            "normal_scale": 1.0,
            "forecast_uncertainty_multiplier": [1.0, 1.1],
            "ar_stationarity_constrained": False,
        }
    }

    row = _compare_results(
        baseline,
        transformed,
        "root",
        AffineTransform("scaled", 2.0, 0.0),
        diagnostics,
        diagnostics,
    )

    assert row["service_ranking_identical"] is False
    assert row["top1_same"] is False
    assert row["baseline_root_rank"] == 1
    assert row["transformed_root_rank"] == 2
    assert row["root_rank_delta"] == -1
    assert row["service_scores_allclose"] is False


def test_summary_counts_invariance_per_transformation():
    rows = [
        {
            "transformation": "scaled",
            "service_ranking_identical": True,
            "top1_same": True,
            "root_rank_same": True,
            "service_scores_allclose": True,
            "lag_coefficients_allclose": True,
            "root_rank_delta": 0,
            "mean_service_rank_displacement": 0.0,
            "max_abs_service_score_diff": 0.0,
            "max_abs_lag_coefficient_diff": 0.0,
            "max_relative_normal_scale_error": 0.0,
        },
        {
            "transformation": "scaled",
            "service_ranking_identical": False,
            "top1_same": False,
            "root_rank_same": False,
            "service_scores_allclose": False,
            "lag_coefficients_allclose": False,
            "root_rank_delta": -2,
            "mean_service_rank_displacement": 1.0,
            "max_abs_service_score_diff": 3.0,
            "max_abs_lag_coefficient_diff": 0.2,
            "max_relative_normal_scale_error": 0.1,
        },
    ]

    summary = _summarize(rows, "overall", "overall")

    assert summary["n_cases"] == 2
    assert summary["identical_service_ranking_fraction"] == 0.5
    assert summary["same_root_rank_fraction"] == 0.5
    assert summary["mean_absolute_root_rank_delta"] == 1.0
    assert summary["max_absolute_root_rank_delta"] == 2


def _write_synthetic_case(root: Path) -> None:
    case_dir = root / "default" / "rcaeval_re1" / "re1_ob" / "case-1"
    case_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    normal = pd.DataFrame({
        "root_cpu": np.sin(np.arange(80) / 5.0) + rng.normal(0, 0.05, 80),
        "other_cpu": np.cos(np.arange(80) / 7.0) + rng.normal(0, 0.05, 80),
    })
    abnormal = pd.DataFrame({
        "root_cpu": np.sin(np.arange(30) / 5.0) + 3.0,
        "other_cpu": np.cos(np.arange(30) / 7.0),
    })
    normal.to_csv(case_dir / "normal_data.csv", index=False)
    abnormal.to_csv(case_dir / "abnormal_data.csv", index=False)
    (case_dir / "case_info.json").write_text(json.dumps({
        "case_id": "case-1",
        "dataset": "re1_ob",
        "fault_type": "cpu",
        "root_cause_service": "root",
    }), encoding="utf-8")


def test_analysis_scores_synthetic_case_and_writes_artifacts(tmp_path):
    processed = tmp_path / "processed"
    _write_synthetic_case(processed)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "benchmark": {"name": "rcaeval_re1"},
        "model": {
            "preprocess_strategy": "default",
            "params": {
                "residualization": "counterfactual_ar",
                "scoring": "bayes_factor",
                "ar_input_scaling": "normal_standard",
                "ar_order": 2,
                "ridge": 0.001,
                "ar_stationarity": "root_projection",
                "stationarity_radius": 0.98,
                "counterfactual_bounds": "none",
                "horizon_aware_uncertainty": True,
                "forecast_error_covariance": "diagonal",
                "min_scale": 1e-6,
                "relative_scale_floor": 1e-3,
                "winsor_quantile": None,
                "prior": {"m": 0.0, "kappa": 0.001, "alpha": 2.0, "beta": 1.0},
            },
        },
        "paths": {"processed_data_dir": str(processed)},
        "evaluation": {"service_aggregation": {"method": "mean_top3"}},
        "datasets": ["re1_ob"],
    }), encoding="utf-8")
    output = tmp_path / "out"

    report = analyze(
        config_path,
        output,
        transforms=(AffineTransform("scaled", 1000.0, 0.0),),
        workers=1,
    )

    assert report["n_cases"] == 1
    assert report["summaries"][0]["transformation"] == "scaled"
    assert report["summaries"][0]["identical_service_ranking_fraction"] == 1.0
    assert report["summaries"][0]["score_allclose_fraction"] == 1.0
    assert report["summaries"][0]["lag_coefficients_allclose_fraction"] == 1.0
    assert (output / "case_comparisons.csv").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "summary.md").is_file()
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["invariance_target"]["service_ranking"] == "exact equality"


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf])
def test_affine_transform_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        AffineTransform("invalid", scale, 0.0)

import json

import pandas as pd
import pytest
import yaml

from scripts.analyze_ar_pseudo_fault_calibration import _build_model, analyze


def test_pseudo_fault_calibration_splits_normal_window_and_writes_summary(tmp_path):
    processed = tmp_path / "processed"
    case_dir = processed / "default" / "rcaeval_re1" / "re1_ob" / "case-a"
    case_dir.mkdir(parents=True)
    pd.DataFrame({
        "root_cpu": range(20),
        "other_cpu": [1.0] * 20,
    }).to_csv(case_dir / "normal_data.csv", index=False)
    (case_dir / "case_info.json").write_text(json.dumps({
        "case_id": "case-a",
        "dataset": "re1_ob",
        "fault_type": "cpu",
        "root_cause_service": "root",
    }), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "model": {"params": {
            "ar_order": 1,
            "ridge": 0.001,
            "min_scale": 1e-6,
            "relative_scale_floor": 1e-3,
            "winsor_quantile": None,
            "ar_input_scaling": "normal_standard",
            "prior": {"m": 0, "kappa": 0.001, "alpha": 2, "beta": 1},
        }}
    }), encoding="utf-8")

    report = analyze(
        processed, config, tmp_path / "out",
        modes=("raw", "counterfactual_ar"), fit_fraction=0.5,
    )

    assert report["n_cases"] == 1
    assert report["modes"] == ["raw", "counterfactual_ar"]
    assert (tmp_path / "out" / "case_calibration.csv").is_file()
    assert (tmp_path / "out" / "summary.md").is_file()


def test_pseudo_fault_calibration_rejects_nonpositive_workers(tmp_path):
    with pytest.raises(ValueError, match="workers must be positive"):
        analyze(
            tmp_path,
            tmp_path / "missing.yaml",
            tmp_path / "out",
            workers=0,
        )


def test_configured_mode_preserves_sensitivity_axes():
    model = _build_model({
        "residualization": "counterfactual_ar",
        "ar_order": 5,
        "ar_input_scaling": "normal_standard",
        "ar_stationarity": "root_projection",
        "stationarity_radius": 0.95,
        "counterfactual_bounds": "none",
        "horizon_aware_uncertainty": False,
        "forecast_error_covariance": "diagonal",
    }, "configured")

    assert model.residualization == "counterfactual_ar"
    assert model.ar_order == 5
    assert model.stationarity_radius == 0.95
    assert model.horizon_aware_uncertainty is False


def test_named_modes_do_not_leak_unrelated_configured_axes():
    model = _build_model({
        "residualization": "counterfactual_ar",
        "ar_stationarity": "root_projection",
        "stationarity_radius": 0.95,
        "horizon_aware_uncertainty": True,
    }, "raw")

    assert model.residualization == "raw"
    assert model.ar_stationarity == "none"
    assert model.stationarity_radius == 0.98
    assert model.horizon_aware_uncertainty is False

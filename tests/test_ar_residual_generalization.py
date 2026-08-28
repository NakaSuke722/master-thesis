import json

import numpy as np
import pandas as pd
import yaml

from models.amber import NIG
from scripts.analyze_ar_residual_generalization import (
    analyze,
    residual_generalization_stats,
)


def test_residual_generalization_detects_larger_holdout_scale():
    train = np.tile([-1.0, 1.0], 40)
    holdout = np.tile([-3.0, 3.0], 20)

    result = residual_generalization_stats(
        train,
        holdout,
        np.array([0.0]),
        order=0,
        relative_scale_floor=1e-3,
        min_scale=1e-6,
        prior=NIG(m=0.0, kappa=1e-3, alpha=2.0, beta=1.0),
    )

    assert result["scale_ratio"] == 3.0
    assert result["abs_log_scale_ratio"] == result["log_scale_ratio"]
    assert result["center_shift_z"] == 0.0
    assert result["one_step_log_bayes_factor"] > 0.0


def test_analysis_writes_metric_service_case_and_summary_outputs(tmp_path):
    processed = tmp_path / "processed"
    case_dir = processed / "default/rcaeval_re1/re1_ob/case-a"
    case_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    values = np.cumsum(rng.normal(0.0, 0.2, size=100))
    pd.DataFrame({
        "root_cpu": values,
        "other_cpu": 0.5 * values + rng.normal(0.0, 0.1, size=100),
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
            "ridge": 1e-3,
            "min_scale": 1e-6,
            "relative_scale_floor": 1e-3,
            "winsor_quantile": None,
            "ar_input_scaling": "normal_standard",
            "prior": {"m": 0.0, "kappa": 1e-3, "alpha": 2.0, "beta": 1.0},
        }}
    }), encoding="utf-8")

    report = analyze(
        processed,
        config,
        tmp_path / "out",
        workers=1,
    )

    assert report["n_cases"] == 1
    assert (tmp_path / "out/metric_residuals.csv").is_file()
    assert (tmp_path / "out/service_diagnostics.csv").is_file()
    assert (tmp_path / "out/case_diagnostics.csv").is_file()
    assert (tmp_path / "out/summary.json").is_file()
    assert (tmp_path / "out/summary.md").is_file()

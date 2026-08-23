import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_ar_bayes_factor_pseudo_fault import (
    analyze,
    load_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "configs/sensitivity/direct_ar_bayes_factor_pseudo_fault.yaml"
)


def _ar1_series(seed: int, length: int, phi: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty(length, dtype=float)
    state = 0.0
    for index in range(length):
        state = phi * state + rng.normal(0.0, 0.5)
        values[index] = state
    return values


def test_pseudo_fault_protocol_defines_three_splits_and_proper_priors():
    protocol = load_protocol(CONFIG)

    assert protocol["fit_fractions"] == [0.4, 0.5, 0.6]
    assert protocol["ar_order"] == 3
    assert set(protocol["prior_profiles"]) == {"weak", "reference", "strong"}
    for parameters in protocol["prior_profiles"].values():
        assert parameters["beta"] / (parameters["alpha"] - 1.0) == 1.0


def test_pseudo_fault_analysis_scores_case_and_writes_summary(tmp_path):
    processed = tmp_path / "processed"
    case_dir = (
        processed / "default/rcaeval_re1/re1_ob/case-a"
    )
    case_dir.mkdir(parents=True)
    stable = _ar1_series(3, 160, 0.6)
    changed = np.concatenate([
        _ar1_series(4, 80, 0.6),
        _ar1_series(5, 80, -0.4) + 2.0,
    ])
    pd.DataFrame({
        "stable_cpu": stable,
        "changed_cpu": changed,
    }).to_csv(case_dir / "normal_data.csv", index=False)
    (case_dir / "case_info.json").write_text(json.dumps({
        "case_id": "case-a",
        "dataset": "re1_ob",
    }), encoding="utf-8")
    protocol = load_protocol(CONFIG)

    report = analyze(
        processed,
        tmp_path / "out",
        fit_fractions=(0.5,),
        prior_profiles={
            "reference": protocol["prior_profiles"]["reference"]
        },
        ar_order=1,
        strong_threshold=np.log(10.0),
    )

    assert report["n_cases"] == 1
    case_rows = pd.read_csv(tmp_path / "out/case_calibration.csv")
    assert case_rows.loc[0, "top_service"] == "changed"
    assert case_rows.loc[0, "metric_failures"] == 0
    assert (tmp_path / "out/summary.json").is_file()
    assert (tmp_path / "out/summary.md").is_file()

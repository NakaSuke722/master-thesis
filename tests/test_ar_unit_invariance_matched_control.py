import pandas as pd
import pytest

from scripts.analyze_ar_unit_invariance_matched_control import pair_case_rows


def _row(case_id: str, score: float) -> dict:
    return {
        "case_id": case_id,
        "dataset": "re1_ob",
        "fault_type": "cpu",
        "root_cause_service": "root",
        "residualization": "stationary_counterfactual_ar_uncertainty",
        "max_service_score": score,
        "median_service_score": score / 2.0,
        "p90_service_score": score * 0.9,
        "positive_service_fraction": 0.8,
        "n_metrics": 10,
        "n_services": 4,
    }


def test_pair_case_rows_joins_by_case_and_uses_treatment_minus_control_delta():
    control = pd.DataFrame([_row("case-b", 20.0), _row("case-a", 10.0)])
    treatment = pd.DataFrame([_row("case-a", 7.0), _row("case-b", 25.0)])

    paired = pair_case_rows(control, treatment)

    assert paired["case_id"].tolist() == ["case-a", "case-b"]
    assert paired["max_service_score_delta"].tolist() == [-3.0, 5.0]
    assert paired["median_service_score_delta"].tolist() == [-1.5, 2.5]


def test_pair_case_rows_rejects_unmatched_case_sets():
    control = pd.DataFrame([_row("case-a", 10.0)])
    treatment = pd.DataFrame([_row("case-b", 10.0)])

    with pytest.raises(ValueError, match="case sets differ"):
        pair_case_rows(control, treatment)

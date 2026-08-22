import pytest

from main import select_amber_diagnostics


SERIES_KEYS = {
    "raw_normal",
    "raw_abnormal",
    "ar_prediction_normal",
    "ar_prediction_abnormal",
    "ar_residual_normal",
    "ar_residual_abnormal",
    "standardized_residual_normal",
    "standardized_residual_abnormal",
}


def sample_diagnostics():
    return {
        "schema_version": 1,
        "residualization": "ar",
        "metrics": [
            {
                "metric": "checkout_cpu",
                "score": 4.2,
                "ar_coefficients": [0.5, 0.2],
                **{key: [1.0, 2.0] for key in SERIES_KEYS},
            }
        ],
        "services": [{"service": "checkout", "score": 4.2}],
    }


def test_summary_diagnostics_keep_scalars_but_drop_time_series():
    diagnostics = sample_diagnostics()

    compact = select_amber_diagnostics(diagnostics, "summary")

    assert compact["schema_version"] == 1
    assert compact["services"] == diagnostics["services"]
    assert compact["metrics"][0]["score"] == 4.2
    assert compact["metrics"][0]["ar_coefficients"] == [0.5, 0.2]
    assert SERIES_KEYS.isdisjoint(compact["metrics"][0])
    assert SERIES_KEYS.issubset(diagnostics["metrics"][0])


def test_full_and_none_diagnostics_modes_are_backward_compatible():
    diagnostics = sample_diagnostics()

    assert select_amber_diagnostics(diagnostics, "full") is diagnostics
    assert select_amber_diagnostics(diagnostics, "none") is None


def test_unknown_diagnostics_mode_is_rejected():
    with pytest.raises(ValueError, match="amber_diagnostics"):
        select_amber_diagnostics(None, "verbose")

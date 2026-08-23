import numpy as np
import pandas as pd
import pytest

from models.amber import (
    AMBER,
    _ar_forecast_error_cholesky,
    _ar_forecast_uncertainty_multipliers,
    _ar_residuals,
    _ar_spectral_radius,
    _counterfactual_ar_forecast,
    _project_ar_stationary,
    _whiten_ar_forecast_errors,
)


def test_counterfactual_forecast_never_feeds_abnormal_observations_back_as_lags():
    coefficients = np.array([0.0, 0.9])
    abnormal = np.array([10.0, 10.0, 10.0])

    counterfactual_prediction, clipped = _counterfactual_ar_forecast(
        history=np.array([0.0]),
        coef=coefficients,
        order=1,
        steps=abnormal.size,
    )
    counterfactual_residual = abnormal - counterfactual_prediction
    observed_lag_residual = _ar_residuals(
        np.concatenate([np.array([0.0]), abnormal]),
        coefficients,
        order=1,
    )

    assert counterfactual_prediction.tolist() == [0.0, 0.0, 0.0]
    assert clipped == 0
    assert counterfactual_residual.tolist() == [10.0, 10.0, 10.0]
    assert observed_lag_residual.tolist() == pytest.approx([10.0, 1.0, 1.0])


def test_counterfactual_forecast_recursively_uses_its_own_predictions():
    prediction, clipped = _counterfactual_ar_forecast(
        history=np.array([10.0]),
        coef=np.array([1.0, 0.5]),
        order=1,
        steps=3,
    )

    assert prediction.tolist() == pytest.approx([6.0, 4.0, 3.0])
    assert clipped == 0


def test_counterfactual_forecast_clips_before_recursive_feedback():
    prediction, clipped = _counterfactual_ar_forecast(
        history=np.array([10.0]),
        coef=np.array([0.0, 2.0]),
        order=1,
        steps=3,
        bounds=(0.0, 10.0),
    )

    assert prediction.tolist() == [10.0, 10.0, 10.0]
    assert clipped == 3


def test_counterfactual_ar_is_available_end_to_end_and_recorded_in_diagnostics():
    normal = pd.DataFrame({"service_cpu": np.zeros(20)})
    abnormal = pd.DataFrame({"service_cpu": np.full(10, 10.0)})
    model = AMBER(
        ar_order=1,
        winsor_quantile=None,
        residualization="counterfactual_ar",
    )

    result = model.fit_predict(normal, abnormal)

    assert np.isfinite(result.loc[0, "score"])
    assert model.diagnostics_["residualization"] == "counterfactual_ar"
    metric = model.diagnostics_["metrics"][0]
    assert metric["ar_prediction_abnormal"] == pytest.approx([0.0] * 10)
    assert metric["ar_residual_abnormal"] == pytest.approx([10.0] * 10)
    assert metric["counterfactual_clipped_predictions"] == 0


def test_unknown_residualization_is_rejected():
    with pytest.raises(ValueError, match="Unknown residualization"):
        AMBER(residualization="recursive")


def test_stationarity_projection_moves_companion_roots_inside_radius():
    projected, before, after, constrained = _project_ar_stationary(
        np.array([0.0, 1.2]),
        normal_mean=5.0,
        radius=0.98,
    )

    assert constrained is True
    assert before == pytest.approx(1.2)
    assert after <= 0.98 + 1e-12
    assert _ar_spectral_radius(projected) <= 0.98 + 1e-12
    assert projected[0] / (1.0 - projected[1]) == pytest.approx(5.0)


def test_horizon_uncertainty_matches_ar1_forecast_variance_formula():
    multipliers = _ar_forecast_uncertainty_multipliers(
        np.array([0.0, 0.5]),
        steps=3,
    )

    assert multipliers == pytest.approx([
        1.0,
        np.sqrt(1.0 + 0.5 ** 2),
        np.sqrt(1.0 + 0.5 ** 2 + 0.25 ** 2),
    ])


def test_full_forecast_covariance_contains_variances_and_cross_horizon_covariances():
    factor = _ar_forecast_error_cholesky(
        np.array([0.0, 0.5]),
        steps=3,
    )
    covariance = factor @ factor.T

    assert np.allclose(factor, np.array([
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.25, 0.5, 1.0],
    ]))
    assert np.diag(covariance) == pytest.approx([
        1.0,
        1.0 + 0.5 ** 2,
        1.0 + 0.5 ** 2 + 0.25 ** 2,
    ])
    assert covariance[0, 1] == pytest.approx(0.5)
    assert covariance[1, 2] == pytest.approx(0.5 + 0.5 * 0.25)


def test_fast_full_covariance_whitening_matches_cholesky_solve():
    coefficients = np.array([0.0, 0.4, -0.2])
    errors = np.array([1.2, -0.7, 2.0, 0.5])
    factor = _ar_forecast_error_cholesky(coefficients, errors.size)

    expected = np.linalg.solve(factor, errors)
    actual = _whiten_ar_forecast_errors(errors, coefficients)

    assert actual == pytest.approx(expected)


def test_stationary_counterfactual_is_clip_free_and_uncertainty_aware():
    normal = pd.DataFrame({"service_cpu": np.arange(30, dtype=float)})
    abnormal = pd.DataFrame({"service_cpu": np.arange(30, 45, dtype=float)})
    model = AMBER(
        ar_order=1,
        winsor_quantile=None,
        residualization="counterfactual_ar",
        ar_stationarity="root_projection",
        stationarity_radius=0.98,
        counterfactual_bounds="none",
        horizon_aware_uncertainty=True,
    )

    model.fit_predict(normal, abnormal)
    metric = model.diagnostics_["metrics"][0]

    assert metric["counterfactual_clipped_predictions"] == 0
    assert metric["counterfactual_bounds"] == "none"
    assert metric["ar_spectral_radius_after"] <= 0.98 + 1e-12
    assert metric["forecast_uncertainty_final_multiplier"] >= 1.0
    assert len(metric["forecast_uncertainty_multiplier"]) == len(abnormal)


def test_horizon_uncertainty_rejects_non_counterfactual_mode():
    with pytest.raises(ValueError, match="requires counterfactual_ar"):
        AMBER(residualization="ar", horizon_aware_uncertainty=True)


def test_full_covariance_requires_horizon_uncertainty():
    with pytest.raises(ValueError, match="requires horizon_aware_uncertainty"):
        AMBER(forecast_error_covariance="full")


def test_full_covariance_counterfactual_equals_observed_lag_innovations():
    normal_values = np.sin(np.arange(80, dtype=float) / 4.0)
    abnormal_values = np.concatenate([
        normal_values[-1:] + 4.0,
        np.full(19, normal_values[-1] + 4.0),
    ])
    normal = pd.DataFrame({"service_cpu": normal_values})
    abnormal = pd.DataFrame({"service_cpu": abnormal_values})
    common = dict(
        ar_order=3,
        winsor_quantile=None,
        ar_stationarity="root_projection",
        stationarity_radius=0.98,
        counterfactual_bounds="none",
    )
    observed = AMBER(residualization="ar", **common)
    full = AMBER(
        residualization="counterfactual_ar",
        horizon_aware_uncertainty=True,
        forecast_error_covariance="full",
        **common,
    )

    observed.fit_predict(normal, abnormal)
    full.fit_predict(normal, abnormal)
    observed_metric = observed.diagnostics_["metrics"][0]
    full_metric = full.diagnostics_["metrics"][0]

    assert full_metric["forecast_error_covariance"] == "full"
    assert full_metric["standardized_residual_abnormal"] == pytest.approx(
        observed_metric["standardized_residual_abnormal"], abs=1e-10
    )
    assert full_metric["score"] == pytest.approx(
        observed_metric["score"], abs=1e-10
    )

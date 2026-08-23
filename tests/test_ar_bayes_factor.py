import numpy as np

from models.ar_bayes_factor import (
    ARBayesFactorPrior,
    _ar_design,
    ar_change_bayes_factor,
    ar_intercept_shift_bayes_factor,
)


def _simulate(
    seed: int,
    *,
    post_mean: float = 0.0,
    post_phi: float = 0.6,
    post_sigma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    value = 0.0
    pre = np.empty(400)
    for index in range(pre.size):
        value = 0.6 * value + rng.normal(0.0, 0.5)
        pre[index] = value
    post = np.empty(400)
    intercept = (1.0 - post_phi) * post_mean
    for index in range(post.size):
        value = intercept + post_phi * value + rng.normal(0.0, post_sigma)
        post[index] = value
    return pre, post


def test_post_design_uses_pre_boundary_then_observed_post_lags():
    design, target = _ar_design(
        np.array([10.0, 20.0, 30.0]),
        order=2,
        history=np.array([1.0, 2.0]),
    )

    assert np.allclose(design, np.array([
        [1.0, 2.0, 1.0],
        [1.0, 10.0, 2.0],
        [1.0, 20.0, 10.0],
    ]))
    assert np.allclose(target, [10.0, 20.0, 30.0])


def test_ar_design_drops_invalid_rows_without_collapsing_time_gaps():
    design, target = _ar_design(
        np.array([1.0, np.nan, 3.0, 4.0]),
        order=1,
    )

    assert np.allclose(design, [[1.0, 3.0]])
    assert np.allclose(target, [4.0])


def test_ar_change_bayes_factor_prefers_clear_regime_change():
    pre_same, post_same = _simulate(11)
    pre_changed, post_changed = _simulate(11, post_mean=2.0)

    same = ar_change_bayes_factor(pre_same, post_same, order=1)
    changed = ar_change_bayes_factor(pre_changed, post_changed, order=1)

    assert same["log_bayes_factor"] < 0.0
    assert changed["log_bayes_factor"] > 10.0
    assert changed["log_bayes_factor"] > same["log_bayes_factor"]


def test_ar_change_bayes_factor_detects_dynamics_and_variance_changes():
    pre_phi, post_phi = _simulate(21, post_phi=-0.2)
    pre_var, post_var = _simulate(22, post_sigma=1.2)

    phi_result = ar_change_bayes_factor(pre_phi, post_phi, order=1)
    variance_result = ar_change_bayes_factor(pre_var, post_var, order=1)

    assert phi_result["log_bayes_factor"] > 10.0
    assert variance_result["log_bayes_factor"] > 10.0


def test_intercept_shift_bayes_factor_detects_persistent_mean_change():
    prior = ARBayesFactorPrior(
        intercept_precision=0.1,
        lag_precision=10.0,
        alpha=5.0,
        beta=4.0,
    )
    pre_same, post_same = _simulate(19)
    pre_shift, post_shift = _simulate(19, post_mean=2.0)

    same = ar_intercept_shift_bayes_factor(
        pre_same, post_same, order=1, prior=prior
    )
    shifted = ar_intercept_shift_bayes_factor(
        pre_shift, post_shift, order=1, prior=prior
    )

    assert same["log_bayes_factor"] < 0.0
    assert shifted["log_bayes_factor"] > 10.0
    assert shifted["posterior_h1"]["intercept_shift_mean"] > 1.0
    assert len(shifted["posterior_h1"]["shared_lag_mean"]) == 1


def test_intercept_shift_bayes_factor_does_not_reward_pure_dynamics_or_variance():
    prior = ARBayesFactorPrior(
        intercept_precision=0.1,
        lag_precision=10.0,
        alpha=5.0,
        beta=4.0,
    )
    pre_phi, post_phi = _simulate(19, post_phi=-0.2)
    pre_var, post_var = _simulate(19, post_sigma=1.2)

    phi_result = ar_intercept_shift_bayes_factor(
        pre_phi, post_phi, order=1, prior=prior
    )
    variance_result = ar_intercept_shift_bayes_factor(
        pre_var, post_var, order=1, prior=prior
    )

    assert phi_result["log_bayes_factor"] < 0.0
    assert variance_result["log_bayes_factor"] < 0.0


def test_normal_only_scaling_does_not_change_when_post_is_rescaled():
    pre, post = _simulate(31)
    baseline = ar_change_bayes_factor(pre, post, order=1)
    shifted = ar_change_bayes_factor(pre, post + 100.0, order=1)

    assert baseline["normalization"] == shifted["normalization"]


def test_proper_prior_keeps_constant_metric_score_finite():
    result = ar_change_bayes_factor(
        np.ones(100),
        np.ones(80),
        order=3,
    )

    assert np.isfinite(result["log_bayes_factor"])

    intercept_result = ar_intercept_shift_bayes_factor(
        np.ones(100),
        np.ones(80),
        order=3,
    )
    assert np.isfinite(intercept_result["log_bayes_factor"])


def test_prior_validation_rejects_improper_precision():
    try:
        ARBayesFactorPrior(lag_precision=0.0)
    except ValueError as exc:
        assert "lag_precision" in str(exc)
    else:
        raise AssertionError("Expected improper prior precision to be rejected")

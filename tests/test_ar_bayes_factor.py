import numpy as np

from models.ar_bayes_factor import (
    ARBayesFactorPrior,
    ARRegimeShiftPrior,
    _ar_design,
    _ar_design_with_indices,
    _bayesian_regression_log_marginal,
    _intervention_basis,
    _normal_only_standardize,
    ar_change_bayes_factor,
    ar_intervention_bayes_factor,
    ar_intercept_shift_bayes_factor,
    ar_shrinkage_regime_bayes_factor,
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


def test_intervention_bayes_factor_averages_shapes_and_detects_step():
    prior = ARBayesFactorPrior(
        intercept_precision=0.1,
        lag_precision=10.0,
        alpha=5.0,
        beta=4.0,
    )
    pre_same, post_same = _simulate(29)
    pre_shift, post_shift = _simulate(29, post_mean=2.0)

    same = ar_intervention_bayes_factor(
        pre_same, post_same, order=1, prior=prior,
    )
    shifted = ar_intervention_bayes_factor(
        pre_shift, post_shift, order=1, prior=prior,
        onset_offsets=(0, 5), onset_prior_decay=0.15,
    )

    assert same["log_bayes_factor"] < 0.0
    assert shifted["log_bayes_factor"] > 10.0
    assert shifted["posterior_map"]["shape"] == "step"
    assert shifted["posterior_map"]["onset_offset"] == 0
    probability = sum(
        model["posterior_model_probability"]
        for model in shifted["posterior_models"]
    )
    assert np.isclose(probability, 1.0)


def test_intervention_bayes_factor_preserves_missing_time_alignment():
    pre, post = _simulate(37, post_mean=1.5)
    post[10] = np.nan

    result = ar_intervention_bayes_factor(
        pre, post, order=2, shapes=("step", "ramp"),
    )

    assert np.isfinite(result["log_bayes_factor"])
    assert result["posterior_map"]["post_rows"] == post.size - 3


def test_intervention_sufficient_statistics_match_full_design_calculation():
    prior = ARBayesFactorPrior(
        intercept_precision=0.1,
        lag_precision=10.0,
        alpha=5.0,
        beta=4.0,
    )
    pre, post = _simulate(43, post_mean=1.5)
    post[10] = np.nan
    result = ar_intervention_bayes_factor(
        pre,
        post,
        order=2,
        prior=prior,
        shapes=("step", "step_ramp"),
        onset_offsets=(0, 5),
    )

    scaled_pre, scaled_post, _, _ = _normal_only_standardize(
        pre, post, 1e-6
    )
    pre_design, pre_target = _ar_design(scaled_pre, 2)
    post_design, post_target, retained = _ar_design_with_indices(
        scaled_post, 2, history=scaled_pre
    )
    pooled_design = np.vstack([pre_design, post_design])
    pooled_target = np.concatenate([pre_target, post_target])
    base_prior_mean = np.array([0.0, 0.0, 0.0])
    base_prior_precision = np.array([0.1, 10.0, 10.0])

    for candidate in result["posterior_models"]:
        basis = _intervention_basis(
            post.size,
            shape=candidate["shape"],
            onset_offset=candidate["onset_offset"],
            half_life=10.0,
        )[retained]
        intervention = np.vstack([
            np.zeros((pre_design.shape[0], basis.shape[1])),
            basis,
        ])
        design = np.column_stack([pooled_design, intervention])
        prior_mean = np.concatenate([
            base_prior_mean, np.zeros(basis.shape[1]),
        ])
        prior_precision = np.concatenate([
            base_prior_precision, np.full(basis.shape[1], 0.1),
        ])
        log_marginal, mean, _, _ = _bayesian_regression_log_marginal(
            design,
            pooled_target,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            alpha=prior.alpha,
            beta=prior.beta,
        )

        assert np.isclose(candidate["log_marginal"], log_marginal)
        assert np.allclose(
            candidate["base_coefficient_mean"], mean[:3]
        )
        assert np.allclose(
            candidate["intervention_coefficient_mean"], mean[3:]
        )


def test_intervention_posterior_detail_does_not_change_evidence():
    pre, post = _simulate(47, post_mean=1.5)
    kwargs = {
        "order": 2,
        "shapes": ("step", "ramp", "exp_decay"),
        "onset_offsets": (0, 5),
    }

    full = ar_intervention_bayes_factor(
        pre, post, posterior_detail="full", **kwargs
    )
    map_only = ar_intervention_bayes_factor(
        pre, post, posterior_detail="map", **kwargs
    )
    none = ar_intervention_bayes_factor(
        pre, post, posterior_detail="none", **kwargs
    )

    for key in ("log_bayes_factor", "log_marginal_h0", "log_marginal_h1"):
        assert np.isclose(full[key], map_only[key])
        assert np.isclose(full[key], none[key])
    assert map_only["posterior_map"]["base_coefficient_mean"]
    assert np.isclose(sum(
        candidate["posterior_model_probability"]
        for candidate in map_only["posterior_models"]
    ), 1.0)
    assert none["posterior_h0"] == {
        "n_rows": full["posterior_h0"]["n_rows"],
    }
    assert none["posterior_models"] == []
    assert none["posterior_map"] is None


def test_intervention_basis_is_cached_and_read_only():
    first = _intervention_basis(
        100, shape="ramp", onset_offset=5, half_life=10.0
    )
    second = _intervention_basis(
        100, shape="ramp", onset_offset=5, half_life=10.0
    )

    assert first is second
    assert not first.flags.writeable


def test_bsrc_ar_distinguishes_sparse_regime_change_types():
    prior = ARBayesFactorPrior(
        intercept_precision=0.1,
        lag_precision=10.0,
        alpha=5.0,
        beta=4.0,
    )
    regime_prior = ARRegimeShiftPrior(inclusion_probability=0.5)
    pre_same, post_same = _simulate(13)
    pre_mean, post_mean = _simulate(13, post_mean=2.0)
    pre_phi, post_phi = _simulate(13, post_phi=-0.2)
    pre_var, post_var = _simulate(13, post_sigma=1.2)

    same = ar_shrinkage_regime_bayes_factor(
        pre_same, post_same, order=1, prior=prior,
        regime_prior=regime_prior,
    )
    mean = ar_shrinkage_regime_bayes_factor(
        pre_mean, post_mean, order=1, prior=prior,
        regime_prior=regime_prior,
    )
    dynamics = ar_shrinkage_regime_bayes_factor(
        pre_phi, post_phi, order=1, prior=prior,
        regime_prior=regime_prior,
    )
    variance = ar_shrinkage_regime_bayes_factor(
        pre_var, post_var, order=1, prior=prior,
        regime_prior=regime_prior,
    )

    assert same["log_bayes_factor"] < 0.0
    assert mean["log_bayes_factor"] > 10.0
    assert dynamics["log_bayes_factor"] > 10.0
    assert variance["log_bayes_factor"] > 10.0
    assert mean["parameter_change_inclusion_probability"]["intercept"] > 0.99
    assert dynamics["parameter_change_inclusion_probability"]["lag_1"] > 0.99
    assert variance["posterior_variance_ratio_mean"] > 2.0


def test_bsrc_ar_model_average_is_normalized_and_predictive():
    pre, post = _simulate(53, post_mean=1.0)
    result = ar_shrinkage_regime_bayes_factor(
        pre,
        post,
        order=2,
        regime_prior=ARRegimeShiftPrior(
            inclusion_probability=1.0 / 3.0,
            variance_quadrature_points=4,
        ),
    )

    assert len(result["posterior_models"]) == (2 ** 3) * 4
    assert np.isclose(sum(
        model["posterior_model_probability"]
        for model in result["posterior_models"]
    ), 1.0)
    assert np.isclose(
        result["log_bayes_factor"],
        result["log_marginal_h1"] - result["log_marginal_h0"],
    )
    assert np.isclose(
        result["log_marginal_h0"],
        result["log_joint_h0"] - result["log_marginal_normal"],
    )
    assert result["posterior_map"]["pre_rows"] == pre.size - 2
    assert result["posterior_map"]["post_rows"] == post.size


def test_bsrc_ar_weighted_sufficient_statistics_match_full_design():
    prior = ARBayesFactorPrior(
        intercept_precision=0.1,
        lag_precision=10.0,
        alpha=5.0,
        beta=4.0,
    )
    regime_prior = ARRegimeShiftPrior(
        intercept_precision=0.25,
        lag_precision=1.0,
        inclusion_probability=0.5,
        log_variance_sd=0.7,
        variance_quadrature_points=2,
    )
    pre, post = _simulate(67, post_mean=1.0)
    result = ar_shrinkage_regime_bayes_factor(
        pre, post, order=1, prior=prior, regime_prior=regime_prior,
    )
    candidate = next(
        model for model in result["posterior_models"]
        if model["changed_parameters"] == ["intercept"]
    )

    scaled_pre, scaled_post, _, _ = _normal_only_standardize(
        pre, post, 1e-6
    )
    pre_design, pre_target = _ar_design(scaled_pre, 1)
    post_design, post_target = _ar_design(
        scaled_post, 1, history=scaled_pre
    )
    ratio = candidate["variance_ratio"]
    pre_candidate = np.column_stack([
        pre_design, np.zeros(pre_design.shape[0]),
    ])
    post_candidate = np.column_stack([
        post_design, post_design[:, 0],
    ]) / np.sqrt(ratio)
    design = np.vstack([pre_candidate, post_candidate])
    target = np.concatenate([pre_target, post_target / np.sqrt(ratio)])
    log_marginal, _, _, _ = _bayesian_regression_log_marginal(
        design,
        target,
        prior_mean=np.zeros(3),
        prior_precision=np.array([0.1, 10.0, 0.25]),
        alpha=5.0,
        beta=4.0,
    )
    log_marginal -= 0.5 * post_target.size * np.log(ratio)

    assert np.isclose(candidate["log_marginal"], log_marginal)


def test_bsrc_ar_default_prior_supports_zero_order_model():
    rng = np.random.default_rng(61)
    result = ar_shrinkage_regime_bayes_factor(
        rng.normal(size=100),
        rng.normal(size=80),
        order=0,
    )

    assert np.isfinite(result["log_bayes_factor"])
    assert result["regime_shift_prior"]["inclusion_probability"] == 0.5


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

    intervention_result = ar_intervention_bayes_factor(
        np.ones(100),
        np.ones(80),
        order=3,
    )
    assert np.isfinite(intervention_result["log_bayes_factor"])

    regime_result = ar_shrinkage_regime_bayes_factor(
        np.ones(100),
        np.ones(80),
        order=3,
    )
    assert np.isfinite(regime_result["log_bayes_factor"])


def test_prior_validation_rejects_improper_precision():
    try:
        ARBayesFactorPrior(lag_precision=0.0)
    except ValueError as exc:
        assert "lag_precision" in str(exc)
    else:
        raise AssertionError("Expected improper prior precision to be rejected")

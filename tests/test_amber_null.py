import numpy as np
import pandas as pd
import pytest

from models.amber import AMBER
from models.ar_bayes_factor import ARBayesFactorPrior, ARRegimeShiftPrior


def test_unchanged_metric_has_finite_score():
    normal = pd.DataFrame({
        "service_cpu": np.ones(100),
    })
    abnormal = pd.DataFrame({
        "service_cpu": np.ones(50),
    })

    model = AMBER(
        ar_order=3,
        winsor_quantile=None,
    )

    result = model.fit_predict(normal, abnormal)

    assert np.isfinite(result.loc[0, "score"])


def test_changed_metric_ranks_above_stable_metric():
    normal = pd.DataFrame({
        "stable_cpu": np.ones(100),
        "changed_cpu": np.ones(100),
    })

    abnormal = pd.DataFrame({
        "stable_cpu": np.ones(50),
        "changed_cpu": np.full(50, 10.0),
    })

    model = AMBER(
        ar_order=3,
        winsor_quantile=None,
    )

    result = model.fit_predict(normal, abnormal)

    assert result.iloc[0]["metric"] == "changed_cpu"


def test_direct_ar_bayes_factor_ranks_changed_process_first():
    rng = np.random.default_rng(17)
    normal = {"stable_cpu": [], "changed_cpu": []}
    abnormal = {"stable_cpu": [], "changed_cpu": []}
    state = {"stable_cpu": 0.0, "changed_cpu": 0.0}
    for target in (normal, abnormal):
        for _ in range(300):
            for metric in target:
                phi = -0.4 if target is abnormal and metric == "changed_cpu" else 0.6
                state[metric] = phi * state[metric] + rng.normal(0.0, 0.5)
                target[metric].append(state[metric])

    model = AMBER(
        ar_order=1,
        residualization="ar_model",
        scoring="ar_bayes_factor",
        winsor_quantile=None,
        ar_bayes_prior=ARBayesFactorPrior(
            intercept_precision=0.1,
            lag_precision=10.0,
            alpha=5.0,
            beta=4.0,
        ),
    )
    result = model.fit_predict(pd.DataFrame(normal), pd.DataFrame(abnormal))

    assert result.iloc[0]["metric"] == "changed_cpu"
    assert model.diagnostics_["residualization"] == "ar_model"
    assert model.diagnostics_["scoring"] == "ar_bayes_factor"
    assert model.diagnostics_["ar_bayes_prior"]["lag_precision"] == 10.0


def test_intercept_shift_ar_bayes_factor_ranks_persistent_shift_first():
    rng = np.random.default_rng(23)
    normal = {"stable_cpu": [], "shifted_cpu": []}
    abnormal = {"stable_cpu": [], "shifted_cpu": []}
    state = {"stable_cpu": 0.0, "shifted_cpu": 0.0}
    for _ in range(300):
        for metric in normal:
            state[metric] = 0.6 * state[metric] + rng.normal(0.0, 0.5)
            normal[metric].append(state[metric])
    for _ in range(300):
        for metric in abnormal:
            intercept = 0.8 if metric == "shifted_cpu" else 0.0
            state[metric] = intercept + 0.6 * state[metric] + rng.normal(0.0, 0.5)
            abnormal[metric].append(state[metric])

    model = AMBER(
        ar_order=1,
        residualization="ar_model",
        scoring="ar_intercept_bayes_factor",
        winsor_quantile=None,
        ar_bayes_prior=ARBayesFactorPrior(
            intercept_precision=0.1,
            lag_precision=10.0,
            alpha=5.0,
            beta=4.0,
        ),
    )
    result = model.fit_predict(pd.DataFrame(normal), pd.DataFrame(abnormal))

    assert result.iloc[0]["metric"] == "shifted_cpu"
    shifted = next(
        row for row in model.diagnostics_["metrics"]
        if row["metric"] == "shifted_cpu"
    )
    assert shifted["ar_hypothesis"] == (
        "shared_intercept_vs_pre_post_intercepts_shared_ar"
    )
    assert shifted["ar_intercept_shift"] > 1.0


def test_adaptive_direct_ar_bayes_factor_uses_normal_only_calibration():
    rng = np.random.default_rng(41)
    normal = {"stable_cpu": [], "shifted_cpu": []}
    abnormal = {"stable_cpu": [], "shifted_cpu": []}
    state = {"stable_cpu": 0.0, "shifted_cpu": 0.0}
    for _ in range(300):
        for metric in normal:
            state[metric] = 0.5 * state[metric] + rng.normal(0.0, 0.5)
            normal[metric].append(state[metric])
    for index in range(300):
        for metric in abnormal:
            effect = 0.6 * (index + 1) / 300 if metric == "shifted_cpu" else 0.0
            state[metric] = effect + 0.5 * state[metric] + rng.normal(0.0, 0.5)
            abnormal[metric].append(state[metric])

    model = AMBER(
        ar_order=1,
        residualization="ar_model",
        scoring="ar_intervention_bayes_factor",
        winsor_quantile=None,
        ar_intervention_shapes=("step", "ramp", "exp_rise"),
        ar_intervention_onset_offsets=(0, 5),
        ar_intervention_onset_prior_decay=0.15,
        ar_null_calibration_fractions=(0.4, 0.5, 0.6),
        ar_null_calibration_quantile=0.9,
        ar_null_calibration_mode="per_row_excess",
    )
    result = model.fit_predict(pd.DataFrame(normal), pd.DataFrame(abnormal))

    assert result.iloc[0]["metric"] == "shifted_cpu"
    shifted = next(
        row for row in model.diagnostics_["metrics"]
        if row["metric"] == "shifted_cpu"
    )
    assert len(shifted["null_log_bayes_factors"]) == 3
    assert np.isclose(
        shifted["score"],
        shifted["raw_log_bayes_factor_per_row"]
        - shifted["null_calibration_baseline"],
    )
    assert shifted["ar_intervention_map_shape"] in {
        "step", "ramp", "exp_rise",
    }
    assert np.isclose(
        sum(shifted["ar_intervention_shape_posterior"].values()), 1.0
    )
    assert np.isclose(
        sum(shifted["ar_intervention_onset_posterior"].values()), 1.0
    )


def test_bsrc_ar_bayes_factor_ranks_sparse_regime_change_first():
    rng = np.random.default_rng(59)
    normal = {"stable_cpu": [], "changed_cpu": []}
    abnormal = {"stable_cpu": [], "changed_cpu": []}
    state = {"stable_cpu": 0.0, "changed_cpu": 0.0}
    for _ in range(300):
        for metric in normal:
            state[metric] = 0.6 * state[metric] + rng.normal(0.0, 0.5)
            normal[metric].append(state[metric])
    for _ in range(300):
        for metric in abnormal:
            intercept = 0.8 if metric == "changed_cpu" else 0.0
            state[metric] = (
                intercept + 0.6 * state[metric] + rng.normal(0.0, 0.5)
            )
            abnormal[metric].append(state[metric])

    model = AMBER(
        ar_order=1,
        residualization="ar_model",
        scoring="bsrc_ar_bayes_factor",
        winsor_quantile=None,
        ar_bayes_prior=ARBayesFactorPrior(
            intercept_precision=0.1,
            lag_precision=10.0,
            alpha=5.0,
            beta=4.0,
        ),
        ar_regime_shift_prior=ARRegimeShiftPrior(
            inclusion_probability=0.5,
        ),
    )
    result = model.fit_predict(pd.DataFrame(normal), pd.DataFrame(abnormal))

    assert result.iloc[0]["metric"] == "changed_cpu"
    changed = next(
        row for row in model.diagnostics_["metrics"]
        if row["metric"] == "changed_cpu"
    )
    assert changed["ar_hypothesis"] == (
        "normal_ar_continuation_vs_sparse_regime_change"
    )
    assert "intercept" in changed[
        "ar_regime_parameter_inclusion_probability"
    ]
    assert model.diagnostics_["ar_regime_shift_prior"][
        "variance_quadrature_points"
    ] == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"residualization": "ar_model", "scoring": "bayes_factor"},
        {"residualization": "ar", "scoring": "ar_bayes_factor"},
        {"residualization": "ar", "scoring": "ar_intercept_bayes_factor"},
        {"residualization": "ar", "scoring": "ar_intervention_bayes_factor"},
        {"residualization": "ar", "scoring": "bsrc_ar_bayes_factor"},
        {
            "residualization": "ar_model",
            "scoring": "ar_bayes_factor",
            "winsor_quantile": 0.01,
        },
    ],
)
def test_direct_ar_bayes_factor_rejects_inconsistent_modes(kwargs):
    with pytest.raises(ValueError):
        AMBER(**kwargs)

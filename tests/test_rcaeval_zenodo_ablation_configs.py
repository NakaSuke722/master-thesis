from copy import deepcopy
from pathlib import Path

from experiments.config import load_config
from experiments.paths import experiment_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = PROJECT_ROOT / "configs/main/rcaeval_re1_zenodo_v2.yaml"
ABLATION_DIR = PROJECT_ROOT / "configs/ablation/rcaeval_re1_zenodo_v2"

FORMAL_ABLATION_BASE_PARAMS = {
    "residualization": "ar",
    "scoring": "bayes_factor",
    "ar_order": 3,
    "ridge": 1.0e-3,
    "min_scale": 1.0e-6,
    "relative_scale_floor": 1.0e-3,
    "winsor_quantile": None,
    "prior": {
        "m": 0.0,
        "kappa": 1.0e-3,
        "alpha": 2.0,
        "beta": 1.0,
    },
}

FINAL_MAIN_PARAMS = {
    **FORMAL_ABLATION_BASE_PARAMS,
    "residualization": "counterfactual_ar",
    "ar_input_scaling": "normal_standard",
    "ar_stationarity": "root_projection",
    "stationarity_radius": 0.98,
    "counterfactual_bounds": "none",
    "horizon_aware_uncertainty": True,
    "forecast_error_covariance": "diagonal",
}

ADAPTIVE_DIRECT_OVERRIDES = {
    "residualization": "ar_model",
    "scoring": "ar_intervention_bayes_factor",
    "winsor_quantile": None,
    "ar_bayes_prior": {
        "intercept_mean": 0.0, "lag_mean": 0.0,
        "intercept_precision": 0.1, "lag_precision": 10.0,
        "alpha": 5.0, "beta": 4.0,
    },
    "ar_intervention_shapes": [
        "step", "ramp", "exp_rise", "exp_decay", "step_ramp",
    ],
    "ar_intervention_onset_offsets": [0, 5, 15],
    "ar_intervention_onset_prior_decay": 0.15,
    "ar_intervention_half_life": 10.0,
    "ar_intervention_precision": 0.1,
    "ar_null_calibration_fractions": [0.4, 0.5, 0.6],
    "ar_null_calibration_quantile": 0.9,
    "ar_null_calibration_mode": "per_row_excess",
}
ADAPTIVE_ROLLBACK_AXES = {
    "adaptive_direct_no_null_calibration": {
        "ar_null_calibration_fractions",
    },
    "adaptive_direct_fixed_onset": {
        "ar_intervention_onset_offsets",
    },
    "adaptive_direct_step_only": {
        "ar_intervention_shapes",
    },
    "adaptive_direct_no_step_ramp": {
        "ar_intervention_shapes",
    },
    "adaptive_direct_no_per_row_normalization": {
        "ar_null_calibration_mode",
    },
}

EXPECTED_VARIANTS = {
    "counterfactual_ar": {
        "residualization": "counterfactual_ar", "scoring": "bayes_factor",
    },
    "no_ar": {"residualization": "raw", "scoring": "bayes_factor"},
    "no_bayes": {"residualization": "ar", "scoring": "glrt"},
    "no_ar_no_bayes": {"residualization": "raw", "scoring": "glrt"},
    "stationary_ar": {
        "residualization": "ar", "scoring": "bayes_factor",
        "ar_stationarity": "root_projection", "stationarity_radius": 0.98,
        "counterfactual_bounds": "none", "horizon_aware_uncertainty": False,
    },
    "stationary_counterfactual_ar": {
        "residualization": "counterfactual_ar", "scoring": "bayes_factor",
        "ar_stationarity": "root_projection", "stationarity_radius": 0.98,
        "counterfactual_bounds": "none", "horizon_aware_uncertainty": False,
    },
    "stationary_counterfactual_ar_uncertainty": {
        "residualization": "counterfactual_ar", "scoring": "bayes_factor",
        "ar_stationarity": "root_projection", "stationarity_radius": 0.98,
        "counterfactual_bounds": "none", "horizon_aware_uncertainty": True,
        "forecast_error_covariance": "diagonal",
    },
    "stationary_counterfactual_ar_uncertainty_unit_invariance_control": {
        "residualization": "counterfactual_ar", "scoring": "bayes_factor",
        "ar_input_scaling": "none",
        "ar_stationarity": "root_projection", "stationarity_radius": 0.98,
        "counterfactual_bounds": "none", "horizon_aware_uncertainty": True,
        "forecast_error_covariance": "diagonal",
    },
    "stationary_counterfactual_ar_uncertainty_unit_invariant": {
        "residualization": "counterfactual_ar", "scoring": "bayes_factor",
        "ar_input_scaling": "normal_standard",
        "ar_stationarity": "root_projection", "stationarity_radius": 0.98,
        "counterfactual_bounds": "none", "horizon_aware_uncertainty": True,
        "forecast_error_covariance": "diagonal",
    },
    "stationary_counterfactual_ar_full_covariance": {
        "residualization": "counterfactual_ar", "scoring": "bayes_factor",
        "ar_stationarity": "root_projection", "stationarity_radius": 0.98,
        "counterfactual_bounds": "none", "horizon_aware_uncertainty": True,
        "forecast_error_covariance": "full",
    },
    "direct_ar_bayes_factor": {
        "residualization": "ar_model", "scoring": "ar_bayes_factor",
        "winsor_quantile": None,
        "ar_bayes_prior": {
            "intercept_mean": 0.0, "lag_mean": 0.0,
            "intercept_precision": 0.1, "lag_precision": 10.0,
            "alpha": 5.0, "beta": 4.0,
        },
    },
    "intercept_shift_ar_bayes_factor": {
        "residualization": "ar_model",
        "scoring": "ar_intercept_bayes_factor",
        "winsor_quantile": None,
        "ar_bayes_prior": {
            "intercept_mean": 0.0, "lag_mean": 0.0,
            "intercept_precision": 0.1, "lag_precision": 10.0,
            "alpha": 5.0, "beta": 4.0,
        },
    },
    "adaptive_direct_ar_bayes_factor": ADAPTIVE_DIRECT_OVERRIDES,
    "adaptive_direct_no_null_calibration": {
        **ADAPTIVE_DIRECT_OVERRIDES,
        "ar_null_calibration_fractions": [],
    },
    "adaptive_direct_fixed_onset": {
        **ADAPTIVE_DIRECT_OVERRIDES,
        "ar_intervention_onset_offsets": [0],
    },
    "adaptive_direct_step_only": {
        **ADAPTIVE_DIRECT_OVERRIDES,
        "ar_intervention_shapes": ["step"],
    },
    "adaptive_direct_no_step_ramp": {
        **ADAPTIVE_DIRECT_OVERRIDES,
        "ar_intervention_shapes": [
            "step", "ramp", "exp_rise", "exp_decay",
        ],
    },
    "adaptive_direct_no_per_row_normalization": {
        **ADAPTIVE_DIRECT_OVERRIDES,
        "ar_null_calibration_mode": "subtract",
    },
    "bsrc_ar_bayes_factor": {
        "residualization": "ar_model",
        "scoring": "bsrc_ar_bayes_factor",
        "winsor_quantile": None,
        "ar_bayes_prior": {
            "intercept_mean": 0.0, "lag_mean": 0.0,
            "intercept_precision": 0.1, "lag_precision": 10.0,
            "alpha": 5.0, "beta": 4.0,
        },
        "ar_regime_shift_prior": {
            "intercept_precision": 0.25,
            "lag_precision": 1.0,
            "inclusion_probability": 0.25,
            "log_variance_sd": 0.7,
            "variance_quadrature_points": 4,
        },
    },
    "bsrc_ar_variance_spike_slab": {
        "residualization": "ar_model",
        "scoring": "bsrc_ar_bayes_factor",
        "winsor_quantile": None,
        "ar_bayes_prior": {
            "intercept_mean": 0.0, "lag_mean": 0.0,
            "intercept_precision": 0.1, "lag_precision": 10.0,
            "alpha": 5.0, "beta": 4.0,
        },
        "ar_regime_shift_prior": {
            "intercept_precision": 0.25,
            "lag_precision": 1.0,
            "inclusion_probability": 0.25,
            "variance_inclusion_probability": 0.25,
            "log_variance_sd": 0.7,
            "variance_quadrature_points": 8,
        },
    },
    "bsrc_ar_variance_spike_slab_q4": {
        "residualization": "ar_model",
        "scoring": "bsrc_ar_bayes_factor",
        "winsor_quantile": None,
        "ar_bayes_prior": {
            "intercept_mean": 0.0, "lag_mean": 0.0,
            "intercept_precision": 0.1, "lag_precision": 10.0,
            "alpha": 5.0, "beta": 4.0,
        },
        "ar_regime_shift_prior": {
            "intercept_precision": 0.25,
            "lag_precision": 1.0,
            "inclusion_probability": 0.25,
            "variance_inclusion_probability": 0.25,
            "log_variance_sd": 0.7,
            "variance_quadrature_points": 4,
        },
    },
    "bsrc_ar_variance_slab_q8": {
        "residualization": "ar_model",
        "scoring": "bsrc_ar_bayes_factor",
        "winsor_quantile": None,
        "ar_bayes_prior": {
            "intercept_mean": 0.0, "lag_mean": 0.0,
            "intercept_precision": 0.1, "lag_precision": 10.0,
            "alpha": 5.0, "beta": 4.0,
        },
        "ar_regime_shift_prior": {
            "intercept_precision": 0.25,
            "lag_precision": 1.0,
            "inclusion_probability": 0.25,
            "variance_inclusion_probability": 1.0,
            "log_variance_sd": 0.7,
            "variance_quadrature_points": 8,
        },
    },
    "bsrc_ar_coefficient_only": {
        "residualization": "ar_model",
        "scoring": "bsrc_ar_bayes_factor",
        "winsor_quantile": None,
        "ar_bayes_prior": {
            "intercept_mean": 0.0, "lag_mean": 0.0,
            "intercept_precision": 0.1, "lag_precision": 10.0,
            "alpha": 5.0, "beta": 4.0,
        },
        "ar_regime_shift_prior": {
            "intercept_precision": 0.25,
            "lag_precision": 1.0,
            "inclusion_probability": 0.25,
            "variance_inclusion_probability": 0.0,
            "log_variance_sd": 0.7,
            "variance_quadrature_points": 8,
        },
    },
    "bsrc_ar_variance_only": {
        "residualization": "ar_model",
        "scoring": "bsrc_ar_bayes_factor",
        "winsor_quantile": None,
        "ar_bayes_prior": {
            "intercept_mean": 0.0, "lag_mean": 0.0,
            "intercept_precision": 0.1, "lag_precision": 10.0,
            "alpha": 5.0, "beta": 4.0,
        },
        "ar_regime_shift_prior": {
            "intercept_precision": 0.25,
            "lag_precision": 1.0,
            "inclusion_probability": 0.0,
            "variance_inclusion_probability": 1.0,
            "log_variance_sd": 0.7,
            "variance_quadrature_points": 8,
        },
    },
    "bsrc_ar_adaptive_variance": {
        "residualization": "ar_model",
        "scoring": "bsrc_ar_bayes_factor",
        "winsor_quantile": None,
        "ar_bayes_prior": {
            "intercept_mean": 0.0, "lag_mean": 0.0,
            "intercept_precision": 0.1, "lag_precision": 10.0,
            "alpha": 5.0, "beta": 4.0,
        },
        "ar_regime_shift_prior": {
            "intercept_precision": 0.25,
            "lag_precision": 1.0,
            "inclusion_probability": 0.25,
            "variance_inclusion_probability": 0.25,
            "log_variance_sd": 0.7,
            "variance_integration": "adaptive_gh",
            "variance_quadrature_points": 11,
            "variance_integration_tolerance": 1.0e-6,
        },
    },
}


def test_legacy_hugging_face_rcaeval_main_config_is_removed():
    assert not (PROJECT_ROOT / "configs/main/rcaeval_re1.yaml").exists()


def test_formal_main_uses_final_counterfactual_ar_candidate():
    main = load_config(MAIN_CONFIG)

    assert main["model"]["params"] == FINAL_MAIN_PARAMS


def test_zenodo_ablation_configs_share_formal_conditions_and_declared_model_axes():
    main = load_config(MAIN_CONFIG)

    for name, overrides in EXPECTED_VARIANTS.items():
        ablation = load_config(ABLATION_DIR / f"{name}.yaml")

        expected = deepcopy(main)
        expected["experiment"] = {"category": "ablation", "name": name}
        # The historical ablation family is defined against the original
        # observed-lag AR baseline. Promoting the validated final candidate to
        # main must not silently mutate those archived experiment definitions.
        expected["model"]["params"] = deepcopy(FORMAL_ABLATION_BASE_PARAMS)
        expected["model"]["params"].update(overrides)

        assert ablation == expected


def test_zenodo_ablation_result_paths_are_benchmark_scoped():
    for name in EXPECTED_VARIANTS:
        config = load_config(ABLATION_DIR / f"{name}.yaml")

        assert experiment_dir(config) == Path(
            "results/ablation/rcaeval_re1"
        ) / name


def test_unit_invariance_matched_pair_differs_only_in_ar_input_scaling():
    control = load_config(
        ABLATION_DIR
        / "stationary_counterfactual_ar_uncertainty_unit_invariance_control.yaml"
    )
    treatment = load_config(
        ABLATION_DIR
        / "stationary_counterfactual_ar_uncertainty_unit_invariant.yaml"
    )

    control_comparable = deepcopy(control)
    treatment_comparable = deepcopy(treatment)
    control_comparable["experiment"]["name"] = "matched"
    treatment_comparable["experiment"]["name"] = "matched"
    control_comparable["model"]["params"].pop("ar_input_scaling")
    treatment_comparable["model"]["params"].pop("ar_input_scaling")

    assert control_comparable == treatment_comparable
    assert control["model"]["params"]["ar_input_scaling"] == "none"
    assert (
        treatment["model"]["params"]["ar_input_scaling"]
        == "normal_standard"
    )


def test_adaptive_rollbacks_each_change_exactly_one_model_axis():
    full = load_config(ABLATION_DIR / "adaptive_direct_ar_bayes_factor.yaml")
    full_params = full["model"]["params"]

    for name, expected_axes in ADAPTIVE_ROLLBACK_AXES.items():
        rollback = load_config(ABLATION_DIR / f"{name}.yaml")
        rollback_params = rollback["model"]["params"]
        actual_axes = {
            key for key in full_params.keys() | rollback_params.keys()
            if full_params.get(key) != rollback_params.get(key)
        }

        assert actual_axes == expected_axes


def test_bsrc_variance_spike_and_quadrature_form_a_two_by_two_ablation():
    names = {
        (1.0, 4): "bsrc_ar_bayes_factor",
        (1.0, 8): "bsrc_ar_variance_slab_q8",
        (0.25, 4): "bsrc_ar_variance_spike_slab_q4",
        (0.25, 8): "bsrc_ar_variance_spike_slab",
    }

    for (variance_inclusion, quadrature_points), name in names.items():
        config = load_config(ABLATION_DIR / f"{name}.yaml")
        prior = config["model"]["params"]["ar_regime_shift_prior"]
        assert prior.get("variance_inclusion_probability", 1.0) == (
            variance_inclusion
        )
        assert prior["variance_quadrature_points"] == quadrature_points


def test_bsrc_adaptive_subset_uses_the_same_model_as_full_variant():
    full = load_config(ABLATION_DIR / "bsrc_ar_adaptive_variance.yaml")
    subset = load_config(
        PROJECT_ROOT / "configs/sensitivity/bsrc_ar_adaptive_subset.yaml"
    )

    assert subset["model"] == full["model"]
    assert subset["data"] == full["data"]
    assert subset["evaluation"] == full["evaluation"]
    assert subset["datasets"] == full["datasets"]
    assert subset["subset_protocol"] == {
        "selection_seed": 20260825,
        "cases_per_dataset_fault": 5,
        "fault_types": ["cpu", "mem", "disk", "delay", "loss"],
    }

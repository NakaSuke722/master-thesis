from copy import deepcopy
from pathlib import Path

from experiments.config import load_config
from experiments.paths import experiment_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs/sensitivity/rcaeval_re1_zenodo_v2"
REFERENCE = CONFIG_DIR / "unit_invariant_r0_98_p3.yaml"
EXPECTED_AXES = {
    "unit_invariant_r0_95_p3": {"stationarity_radius": 0.95},
    "unit_invariant_r0_99_p3": {"stationarity_radius": 0.99},
    "unit_invariant_r0_98_p1": {"ar_order": 1},
    "unit_invariant_r0_98_p5": {"ar_order": 5},
    "unit_invariant_no_horizon_uncertainty": {
        "horizon_aware_uncertainty": False,
    },
}


def _without_experiment(config: dict) -> dict:
    comparable = deepcopy(config)
    comparable.pop("experiment")
    return comparable


def test_final_ar_sensitivity_variants_change_exactly_one_model_axis():
    reference = load_config(REFERENCE)
    reference_params = reference["model"]["params"]

    for name, expected in EXPECTED_AXES.items():
        variant = load_config(CONFIG_DIR / f"{name}.yaml")
        actual_axes = {
            key
            for key in reference_params.keys() | variant["model"]["params"].keys()
            if reference_params.get(key) != variant["model"]["params"].get(key)
        }

        assert actual_axes == set(expected)
        assert {
            key: variant["model"]["params"][key] for key in actual_axes
        } == expected
        comparable = deepcopy(variant)
        comparable["model"]["params"].update(reference_params)
        assert _without_experiment(comparable) == _without_experiment(reference)


def test_final_ar_sensitivity_outputs_are_separate_and_benchmark_scoped():
    names = ["unit_invariant_r0_98_p3", *EXPECTED_AXES]
    for name in names:
        config = load_config(CONFIG_DIR / f"{name}.yaml")
        assert config["experiment"] == {
            "category": "sensitivity",
            "name": name,
        }
        assert experiment_dir(config) == Path(
            "results/sensitivity/rcaeval_re1"
        ) / name

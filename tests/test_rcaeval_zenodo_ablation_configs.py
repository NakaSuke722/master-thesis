from copy import deepcopy
from pathlib import Path

from experiments.config import load_config
from experiments.paths import experiment_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = PROJECT_ROOT / "configs/main/rcaeval_re1_zenodo_v2.yaml"
ABLATION_DIR = PROJECT_ROOT / "configs/ablation/rcaeval_re1_zenodo_v2"

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
    },
}


def test_legacy_hugging_face_rcaeval_main_config_is_removed():
    assert not (PROJECT_ROOT / "configs/main/rcaeval_re1.yaml").exists()


def test_zenodo_ablation_configs_match_formal_main_except_variant_axes():
    main = load_config(MAIN_CONFIG)

    for name, overrides in EXPECTED_VARIANTS.items():
        ablation = load_config(ABLATION_DIR / f"{name}.yaml")

        expected = deepcopy(main)
        expected["experiment"] = {"category": "ablation", "name": name}
        expected["model"]["params"].update(overrides)

        assert ablation == expected


def test_zenodo_ablation_result_paths_are_benchmark_scoped():
    for name in EXPECTED_VARIANTS:
        config = load_config(ABLATION_DIR / f"{name}.yaml")

        assert experiment_dir(config) == Path(
            "results/ablation/rcaeval_re1"
        ) / name

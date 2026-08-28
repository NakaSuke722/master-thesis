from pathlib import Path

from experiments.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/baselines/baro.yaml"


def test_baro_model_baseline_is_not_the_baro_pilot_dataset():
    config = load_config(CONFIG_PATH)

    assert config["benchmark"]["name"] == "rcaeval_re1"
    assert config["experiment"] == {
        "category": "baselines",
        "name": "baro_robust_scorer_known_onset",
    }
    assert config["model"]["target"] == "baro_robust_scorer"
    assert config["model"]["params"]["protocol"] == (
        "known_onset_robust_scorer"
    )
    assert config["model"]["params"]["score_mode"] == "max_signed"

    assert config["paths"]["raw_data_dir"] == (
        "data/raw/rcaeval_zenodo_v2"
    )
    assert config["paths"]["processed_data_dir"] == (
        "data/processed/rcaeval_zenodo_v2"
    )
    assert config["paths"]["raw_data_dir"] != "data/raw/baro"


def test_baro_baseline_uses_formal_rcaeval_conditions():
    config = load_config(CONFIG_PATH)

    assert config["data"]["source"] == {
        "provider": "Zenodo",
        "record": "14590730",
        "doi": "10.5281/zenodo.14590730",
        "version": "v2",
    }
    assert config["data"]["normal_window_points"] == 600
    assert config["data"]["abnormal_window_points"] == 600
    assert config["evaluation"]["granularity"] == "service"
    assert config["evaluation"]["service_aggregation"]["method"] == (
        "baro_metric_order_deduplication"
    )
    assert config["datasets"] == ["re1_ob", "re1_ss", "re1_tt"]

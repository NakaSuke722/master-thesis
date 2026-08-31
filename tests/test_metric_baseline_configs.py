from pathlib import Path

from experiments.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "epsilon_diagnosis": "epsilon_diagnosis.yaml",
    "rcd": "rcd.yaml",
    "circa": "circa.yaml",
    "run": "run.yaml",
}


def test_metric_baselines_share_formal_rcaeval_conditions():
    names = set()
    for target, filename in CONFIGS.items():
        config = load_config(PROJECT_ROOT / "configs/baselines" / filename)
        assert config["benchmark"]["name"] == "rcaeval_re1"
        assert config["experiment"]["category"] == "baselines"
        assert config["experiment"]["name"] not in names
        names.add(config["experiment"]["name"])
        assert config["model"]["target"] == target
        assert config["data"]["source"] == {
            "provider": "Zenodo",
            "record": "14590730",
            "doi": "10.5281/zenodo.14590730",
            "version": "v2",
        }
        assert config["data"]["normal_window_points"] == 600
        assert config["data"]["abnormal_window_points"] == 600
        assert config["paths"]["processed_data_dir"] == (
            "data/processed/rcaeval_zenodo_v2"
        )
        assert config["evaluation"]["granularity"] == "service"
        assert config["evaluation"]["service_aggregation"]["method"] == (
            "metric_order_deduplication"
        )
        assert config["datasets"] == ["re1_ob", "re1_ss", "re1_tt"]


def test_run_training_is_known_onset_and_normal_only():
    config = load_config(PROJECT_ROOT / "configs/baselines/run.yaml")
    assert config["model"]["params"]["training_scope"] == "normal_only"
    assert config["model"]["params"]["device"] == "cpu"
    assert config["model"]["params"]["execution_backend"] == "vectorized"
    assert config["model"]["params"]["torch_num_threads"] == 1


def test_circa_declares_tractable_pc_adapter_rules():
    config = load_config(PROJECT_ROOT / "configs/baselines/circa.yaml")
    params = config["model"]["params"]
    assert params["pc_redundancy_threshold"] == 0.999999999999
    assert params["pc_max_conditioning_set"] == 1
    assert params["pc_max_metrics"] == 60

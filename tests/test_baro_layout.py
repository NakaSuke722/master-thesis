from pathlib import Path

from experiments.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BARO_ABLATION_DIR = PROJECT_ROOT / "configs/ablation/baro"

EXPECTED_VARIANTS = {
    "no_ar": ("raw", "bayes_factor"),
    "no_bayes": ("ar", "glrt"),
    "no_ar_no_bayes": ("raw", "glrt"),
}


def test_baro_ablation_configs_have_a_dedicated_directory():
    for name, (residualization, scoring) in EXPECTED_VARIANTS.items():
        assert not (PROJECT_ROOT / f"configs/ablation/{name}.yaml").exists()

        config = load_config(BARO_ABLATION_DIR / f"{name}.yaml")
        assert config["paths"]["raw_data_dir"] == "data/raw/baro"
        assert config["paths"]["processed_data_dir"] == "data/processed/baro"
        assert config["model"]["params"]["residualization"] == residualization
        assert config["model"]["params"]["scoring"] == scoring
        assert config["evaluation"]["granularity"] == "metric"


def test_baro_data_scaffold_is_benchmark_scoped():
    for dataset in ("online_boutique", "sock_shop", "train_ticket"):
        assert (PROJECT_ROOT / "data/raw/baro" / dataset).is_dir()
        assert (
            PROJECT_ROOT / "data/processed/baro/default" / dataset
        ).is_dir()

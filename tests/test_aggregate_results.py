import json

from aggregate_results import aggregate_config


def test_aggregate_config_keeps_only_matching_variant(tmp_path):
    results_root = tmp_path / "results"
    result_dir = results_root / "ablation" / "no_ar" / "metric" / "demo"
    result_dir.mkdir(parents=True)

    matching = {
        "dataset": "demo",
        "metrics": {"AC@1": 1.0, "Avg@1": 1.0},
        "execution_time_sec": 2.5,
        "evaluation_granularity": "metric",
        "model_used": "amber",
        "experiment_category": "ablation",
        "experiment_name": "no_ar",
    }
    unrelated = {**matching, "experiment_name": "no_bayes"}

    for name, data in (("match.json", matching), ("other.json", unrelated)):
        with (result_dir / name).open("w", encoding="utf-8") as f:
            json.dump(data, f)

    config = {
        "datasets": ["demo"],
        "model": {"target": "amber"},
        "paths": {"results_root": str(results_root)},
        "experiment": {"category": "ablation", "name": "no_ar"},
    }

    summary = aggregate_config(config, "metric")

    assert summary["summary"] == {"demo": {"AC@1": 1.0, "Avg@1": 1.0}}
    assert summary["pure_python_execution_time_sec"] == 2.5
    assert summary["details"] == [matching]
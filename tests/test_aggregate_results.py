import json

from aggregate_results import (
    aggregate_config,
    load_recorded_summary,
    load_result_for_aggregation,
)


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
    assert "details" not in summary


def test_result_loader_skips_large_trailing_diagnostics(tmp_path):
    filepath = tmp_path / "case.json"
    result = {
        "dataset": "demo",
        "metrics": {"AC@1": 1.0},
        "execution_time_sec": 1.0,
        "amber_diagnostics": {
            "metrics": [{"raw_normal": list(range(10_000))}],
        },
    }
    filepath.write_text(
        json.dumps(result, indent=4),
        encoding="utf-8",
    )

    loaded = load_result_for_aggregation(filepath)

    assert loaded == {
        "dataset": "demo",
        "metrics": {"AC@1": 1.0},
        "execution_time_sec": 1.0,
    }


def test_recorded_summary_is_reused_only_when_identity_matches(tmp_path):
    results_root = tmp_path / "results"
    summary_file = (
        results_root
        / "ablation"
        / "no_ar"
        / "summary_metric.json"
    )
    summary_file.parent.mkdir(parents=True)
    recorded = {
        "experiment_category": "ablation",
        "experiment_name": "no_ar",
        "model_used": "amber",
        "evaluation_granularity": "metric",
        "number_of_cases": 1,
        "total_execution_time_sec": 3.0,
        "pure_python_execution_time_sec": 2.5,
        "summary": {"demo": {"AC@1": 1.0}},
    }
    summary_file.write_text(
        json.dumps(recorded),
        encoding="utf-8",
    )
    config = {
        "datasets": ["demo"],
        "model": {"target": "amber"},
        "paths": {"results_root": str(results_root)},
        "experiment": {"category": "ablation", "name": "no_ar"},
    }

    assert load_recorded_summary(config, "metric") == recorded

    summary_file.write_text(
        json.dumps({**recorded, "model_used": "other"}),
        encoding="utf-8",
    )
    assert load_recorded_summary(config, "metric") is None

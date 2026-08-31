import json
from pathlib import Path

import pytest

import aggregate_results
from benchmarks.base import BenchmarkCase

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


@pytest.mark.parametrize("model", [
    "amber", "run", "circa", "rcd", "epsilon_diagnosis", "baro_robust_scorer",
])
def test_result_loader_skips_large_trailing_diagnostics(tmp_path, model):
    filepath = tmp_path / "case.json"
    result = {
        "dataset": "demo",
        "metrics": {"AC@1": 1.0},
        "execution_time_sec": 1.0,
        f"{model}_diagnostics": {
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


def test_run_header_read_is_bounded_even_with_large_attention_table(tmp_path, monkeypatch):
    filepath = tmp_path / "run.json"
    header = {"dataset": "demo", "metrics": {"AC@1": 1}, "case_id": "case-0"}
    filepath.write_text(json.dumps({
        **header, "run_diagnostics": {"attention_by_target": {"x": "x" * (8 * 1024 * 1024)}},
    }, indent=4), encoding="utf-8")
    original_open = Path.open
    read_bytes = []

    class CountedFile:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def read(self, size=-1):
            value = self.file.read(size)
            read_bytes.append(len(value))
            return value

        def __getattr__(self, key):
            return getattr(self.file, key)

    monkeypatch.setattr(Path, "open", lambda path, *a, **kw: CountedFile(original_open(path, *a, **kw)))
    assert load_result_for_aggregation(filepath) == header
    assert sum(read_bytes) <= 64 * 1024 + 1024


def test_diagnostics_marker_across_chunk_boundary(tmp_path):
    prefix = '{\n    "padding": "'
    prefix += "x" * (65530 - len(prefix) - 2) + '",'
    filepath = tmp_path / "boundary.json"
    filepath.write_text(prefix + '\n    "run_diagnostics": {\n        "x": 1\n    }\n}', encoding="utf-8")
    assert set(load_result_for_aggregation(filepath)) == {"padding"}


@pytest.mark.parametrize("indent", [None, 2])
def test_result_loader_preserves_alternative_layouts(tmp_path, indent):
    result = {"run_diagnostics": {"x": 1}, "metrics": {"AC@1": 0.5}}
    filepath = tmp_path / "legacy.json"
    filepath.write_text(json.dumps(result, indent=indent), encoding="utf-8")
    assert load_result_for_aggregation(filepath) == result


def test_result_loader_rejects_truncated_diagnostics(tmp_path):
    filepath = tmp_path / "interrupted.json"
    filepath.write_text('{\n    "metrics": {"AC@1": 1},\n    "run_diagnostics": {\n        "x": [1,', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_result_for_aggregation(filepath)


def _complete_benchmark_fixture(tmp_path, monkeypatch):
    cases = [BenchmarkCase("synthetic", "demo", f"case-{i}", "root", "cpu", 10) for i in range(2)]
    monkeypatch.setattr("data_loader.list_benchmark_processed_cases", lambda **kwargs: cases)
    config = {
        "benchmark": {"name": "synthetic"},
        "datasets": ["demo"],
        "model": {"target": "run"},
        "paths": {"results_root": str(tmp_path)},
        "experiment": {"category": "baselines", "name": "run_test"},
        "evaluation": {"k_values": [1]},
    }
    directory = tmp_path / "baselines/synthetic/run_test/service/demo"
    directory.mkdir(parents=True)
    for i, case in enumerate(cases):
        result = {
            "dataset": "demo", "case_id": case.case_id,
            "model_used": "run", "experiment_category": "baselines", "experiment_name": "run_test",
            "evaluation_granularity": "service", "execution_time_sec": 2,
            "metrics": {"AC@1": float(i), "Avg@1": float(i)},
            "run_diagnostics": {"attention_by_target": {"x": 1}},
        }
        (directory / f"{case.case_id}.json").write_text(json.dumps(result, indent=4), encoding="utf-8")
    return config, directory


def test_complete_aggregation_matches_full_read(tmp_path, monkeypatch):
    config, _ = _complete_benchmark_fixture(tmp_path, monkeypatch)
    fast = aggregate_config(config, "service", require_complete=True)
    monkeypatch.setattr(aggregate_results, "load_result_for_aggregation", lambda path: json.loads(path.read_text()))
    reference = aggregate_config(config, "service", require_complete=True)
    assert fast == reference
    assert fast["number_of_cases"] == 2
    assert fast["summary"] == {"demo": {"AC@1": 0.5, "Avg@1": 0.5}}


def test_incomplete_aggregation_does_not_overwrite_summary(tmp_path, monkeypatch):
    config, directory = _complete_benchmark_fixture(tmp_path, monkeypatch)
    (directory / "case-1.json").unlink()
    summary_file = directory.parents[1] / "summary_service.json"
    summary_file.write_text('{"previous": true}', encoding="utf-8")
    monkeypatch.setattr(aggregate_results, "load_config", lambda path: config)
    monkeypatch.setattr("sys.argv", ["aggregate_results.py", "--require-complete"])
    with pytest.raises(ValueError, match="demo: 1/2 saved; missing 1"):
        aggregate_results.main()
    assert summary_file.read_text() == '{"previous": true}'


def test_reaggregation_preserves_recorded_run_duration(tmp_path, monkeypatch):
    config, directory = _complete_benchmark_fixture(tmp_path, monkeypatch)
    summary = aggregate_config(config, "service", total_time=99, require_complete=True)
    summary_file = directory.parents[1] / "summary_service.json"
    summary_file.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(aggregate_results, "load_config", lambda path: config)
    monkeypatch.setattr("sys.argv", ["aggregate_results.py", "--require-complete"])
    aggregate_results.main()
    assert json.loads(summary_file.read_text()) == summary


@pytest.mark.parametrize("field,value,error", [
    ("case_id", "other-case", "case identity"),
    ("dataset", "wrong-dataset", "case identity"),
    ("metrics", {}, "Missing evaluation metrics"),
    ("model_used", "other-model", "Incomplete benchmark results"),
])
def test_complete_aggregation_rejects_wrong_result(tmp_path, monkeypatch, field, value, error):
    config, directory = _complete_benchmark_fixture(tmp_path, monkeypatch)
    path = directory / "case-1.json"
    result = json.loads(path.read_text())
    result[field] = value
    path.write_text(json.dumps(result, indent=4), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        aggregate_config(config, "service", require_complete=True)


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

import json

import pytest

from benchmarks.base import BenchmarkCase
import runner


def _config():
    return {
        "benchmark": {"name": "synthetic"},
        "datasets": ["dataset-a"],
        "experiment": {"name": "experiment"},
        "model": {"target": "model", "preprocess_strategy": "default"},
        "paths": {"processed_data_dir": "processed"},
    }


def _cases():
    return [
        BenchmarkCase(
            benchmark="synthetic",
            dataset="dataset-a",
            case_id=f"case-{index}",
            root_cause_service="root",
            fault_type="cpu",
            inject_time=10,
            repetition=index,
        )
        for index in range(3)
    ]


def test_run_benchmark_sequential_preserves_case_order(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner, "list_benchmark_processed_cases", lambda **kwargs: _cases()
    )

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return {}, f"{kwargs['benchmark_case'].case_id}.json"

    monkeypatch.setattr(runner, "run_experiment", fake_run_experiment)

    generated = runner.run_benchmark(
        _config(), "config.yaml", "service", workers=1
    )

    assert generated == ["case-0.json", "case-1.json", "case-2.json"]
    assert [call["progress"] for call in calls] == [1, 2, 3]
    assert all(call["total_progress"] == 3 for call in calls)


def test_run_benchmark_parallel_uses_requested_worker_count(monkeypatch):
    created = []
    monkeypatch.setattr(
        runner, "list_benchmark_processed_cases", lambda **kwargs: _cases()
    )
    monkeypatch.setattr(
        runner,
        "run_experiment",
        lambda **kwargs: ({}, f"{kwargs['benchmark_case'].case_id}.json"),
    )

    class FakeExecutor:
        def __init__(self, max_workers):
            created.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, tasks):
            return [function(task) for task in tasks]

    monkeypatch.setattr(runner, "ProcessPoolExecutor", FakeExecutor)

    generated = runner.run_benchmark(
        _config(), "config.yaml", "service", workers=4
    )

    assert created == [4]
    assert generated == ["case-0.json", "case-1.json", "case-2.json"]


def test_run_benchmark_rejects_non_positive_workers():
    with pytest.raises(ValueError, match="workers must be positive"):
        runner.run_benchmark(_config(), "config.yaml", "service", workers=0)


def test_run_benchmark_resume_skips_existing_case(monkeypatch, tmp_path):
    calls = []
    cases = _cases()
    monkeypatch.setattr(
        runner, "list_benchmark_processed_cases", lambda **kwargs: cases
    )
    monkeypatch.setattr(
        runner,
        "case_result_dir",
        lambda *args, **kwargs: tmp_path,
    )
    (tmp_path / "case-1.json").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "model_used": "model",
                "experiment_name": "experiment",
                "evaluation_granularity": "service",
                "metrics": {"AC@1": 1},
            }
        ),
        encoding="utf-8",
    )

    def fake_run_experiment(**kwargs):
        calls.append(kwargs["benchmark_case"].case_id)
        return {}, str(tmp_path / f"{kwargs['benchmark_case'].case_id}.json")

    monkeypatch.setattr(runner, "run_experiment", fake_run_experiment)
    generated = runner.run_benchmark(
        _config(),
        "config.yaml",
        "service",
        workers=1,
        resume=True,
    )
    assert calls == ["case-0", "case-2"]
    assert generated == [
        str(tmp_path / "case-1.json"),
        str(tmp_path / "case-0.json"),
        str(tmp_path / "case-2.json"),
    ]

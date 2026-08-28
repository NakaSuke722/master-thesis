from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.base import BenchmarkCase
import main as main_module
from models.baselines.baro import (
    BARORobustScorer,
    to_service_ranking,
)


def test_known_onset_robust_scorer_ranks_metrics_and_services():
    normal = pd.DataFrame(
        {
            "checkout_cpu": [0.0, 1.0, 2.0, 3.0, 4.0],
            "checkout_mem": [10.0, 11.0, 12.0, 13.0, 14.0],
            "payment_cpu": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    abnormal = pd.DataFrame(
        {
            "checkout_cpu": [8.0, 10.0],
            "checkout_mem": [15.0, 16.0],
            "payment_cpu": [3.0, 4.0],
        }
    )

    model = BARORobustScorer(score_mode="max_signed")
    metric_scores = model.predict(normal, abnormal, granularity="metric")

    assert metric_scores["metric"].tolist() == [
        "checkout_cpu",
        "checkout_mem",
        "payment_cpu",
    ]
    assert metric_scores.iloc[0]["normal_median"] == pytest.approx(2.0)
    assert metric_scores.iloc[0]["normal_iqr"] == pytest.approx(2.0)
    assert metric_scores.iloc[0]["score"] == pytest.approx(4.0)
    assert model.predict(normal, abnormal, granularity="service") == [
        "checkout",
        "payment",
    ]


def test_formal_signed_mode_matches_official_upper_deviation_rule():
    normal = pd.DataFrame({"svc_cpu": [0.0, 1.0, 2.0, 3.0, 4.0]})
    abnormal = pd.DataFrame({"svc_cpu": [-10.0, -8.0]})

    signed = BARORobustScorer(score_mode="max_signed").score_metrics(
        normal,
        abnormal,
    )
    absolute = BARORobustScorer(score_mode="max_absolute").score_metrics(
        normal,
        abnormal,
    )

    assert signed.iloc[0]["score"] == pytest.approx(-5.0)
    assert absolute.iloc[0]["score"] == pytest.approx(6.0)


def test_segment_constants_and_time_columns_are_excluded():
    normal = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "constant_normal_cpu": [1.0, 1.0, 1.0],
            "constant_abnormal_cpu": [1.0, 2.0, 3.0],
            "usable_cpu": [1.0, 2.0, 3.0],
        }
    )
    abnormal = pd.DataFrame(
        {
            "time": [3, 4, 5],
            "constant_normal_cpu": [2.0, 3.0, 4.0],
            "constant_abnormal_cpu": [7.0, 7.0, 7.0],
            "usable_cpu": [4.0, 5.0, 6.0],
        }
    )

    model = BARORobustScorer()
    result = model.score_metrics(normal, abnormal)

    assert result["metric"].tolist() == ["usable_cpu"]
    assert model.diagnostics_["excluded_metrics"] == {
        "time": "time_column",
        "constant_normal_cpu": "constant_normal",
        "constant_abnormal_cpu": "constant_abnormal",
    }


def test_service_ranking_preserves_first_metric_occurrence():
    assert to_service_ranking(
        ["cart_cpu", "payment_mem", "cart_latency"]
    ) == ["cart", "payment"]


def test_invalid_score_mode_is_rejected():
    with pytest.raises(ValueError, match="score_mode"):
        BARORobustScorer(score_mode="unknown")


def test_run_experiment_dispatches_baro_baseline(monkeypatch, tmp_path):
    normal = pd.DataFrame(
        {
            "checkout_cpu": [0.0, 1.0, 2.0, 3.0, 4.0],
            "payment_cpu": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    abnormal = pd.DataFrame(
        {
            "checkout_cpu": [8.0, 10.0],
            "payment_cpu": [3.0, 4.0],
        }
    )
    case = BenchmarkCase(
        benchmark="rcaeval_re1",
        dataset="re1_ob",
        case_id="re1_ob__checkout_cpu__1",
        root_cause_service="checkout",
        fault_type="cpu",
        inject_time=100,
        repetition=1,
    )
    config = {
        "experiment": {
            "category": "baselines",
            "name": "baro_robust_scorer_known_onset",
        },
        "model": {
            "target": "baro_robust_scorer",
            "preprocess_strategy": "default",
            "params": {"score_mode": "max_signed"},
        },
        "paths": {
            "processed_data_dir": "unused",
            "results_root": str(tmp_path),
        },
        "evaluation": {
            "k_values": [1, 3, 5],
            "granularity": "service",
        },
    }

    monkeypatch.setattr(
        main_module,
        "load_benchmark_processed_case",
        lambda **kwargs: (normal, abnormal, {"n_metrics": 2}),
    )
    monkeypatch.setattr(
        main_module,
        "case_result_dir",
        lambda *args, **kwargs: tmp_path,
    )

    result, output_file = main_module.run_experiment(
        dataset=case.dataset,
        fault=case.fault_type,
        run=case.repetition,
        config=config,
        config_path="configs/baselines/baro.yaml",
        granularity="service",
        benchmark_case=case,
        batch=True,
        progress=1,
        total_progress=1,
    )

    assert result["model_used"] == "baro_robust_scorer"
    assert result["predicted_ranking"] == ["checkout", "payment"]
    assert result["metrics"]["AC@1"] == 1
    assert result["baro_robust_scorer_diagnostics"]["protocol"] == (
        "known_onset_robust_scorer"
    )
    assert output_file == str(tmp_path / f"{case.case_id}.json")

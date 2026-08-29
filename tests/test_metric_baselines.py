from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from benchmarks.base import BenchmarkCase
import main as main_module
from models.baselines.circa import CIRCAScorer
from models.baselines.common import prepare_paired_metric_data
from models.baselines.epsilon_diagnosis import EpsilonDiagnosisScorer
from models.baselines.rcd import RCDScorer
from models.baselines.run import RUNScorer


def _shift_case(samples: int = 120):
    rng = np.random.default_rng(3)
    normal = pd.DataFrame(
        {f"s{index}_cpu": rng.normal(size=samples) for index in range(4)}
    )
    abnormal = normal.copy()
    abnormal["s2_cpu"] += 5.0
    return normal, abnormal


def test_common_preprocessing_uses_only_common_finite_metrics():
    normal = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "a_cpu": [0.0, 1.0, 2.0],
            "constant_cpu": [1.0, 1.0, 1.0],
        }
    )
    abnormal = pd.DataFrame(
        {
            "time": [3, 4, 5],
            "a_cpu": [3.0, np.nan, 5.0],
            "constant_cpu": [1.0, 2.0, 3.0],
        }
    )
    paired = prepare_paired_metric_data(normal, abnormal)
    assert paired.normal.columns.tolist() == ["a_cpu"]
    assert paired.abnormal["a_cpu"].tolist() == [3.0, 4.0, 5.0]
    assert paired.excluded == {
        "time": "time_column",
        "constant_cpu": "constant_normal",
    }


def test_epsilon_diagnosis_is_seed_reproducible_and_bounded():
    normal, abnormal = _shift_case()
    first = EpsilonDiagnosisScorer(
        bootstrap_time=20, root_cause_top_k=2, seed=7
    )
    second = EpsilonDiagnosisScorer(
        bootstrap_time=20, root_cause_top_k=2, seed=7
    )
    first_result = first.score_metrics(normal, abnormal)
    second_result = second.score_metrics(normal, abnormal)
    pd.testing.assert_frame_equal(first_result, second_result)
    assert len(first_result) <= 2
    assert first.diagnostics_["protocol"] == (
        "pyrca_epsilon_diagnosis_known_onset"
    )


def test_rcd_localized_f_node_finds_large_regime_shift():
    normal, abnormal = _shift_case()
    model = RCDScorer(seed=1)
    result = model.score_metrics(normal, abnormal)
    assert result.iloc[0]["metric"] == "s2_cpu"
    assert model.diagnostics_["conditional_independence_tests"] > 0


def test_circa_rht_scores_each_node_with_explicit_windows(monkeypatch):
    normal, abnormal = _shift_case(80)
    model = CIRCAScorer(
        lookup_window=30,
        detect_window=5,
        score_time_offset=20,
    )
    monkeypatch.setattr(
        model,
        "_learn_graph",
        lambda data: np.zeros((data.shape[1], data.shape[1])),
    )
    result = model.score_metrics(normal, abnormal)
    assert result.iloc[0]["metric"] == "s2_cpu"
    assert model.diagnostics_["actual_score_time_offset"] == 20
    assert model.diagnostics_["graph_learning_scope"] == (
        "normal_and_abnormal"
    )


def test_run_adapter_executes_without_fixed_batch_or_cuda_assumptions():
    normal, abnormal = _shift_case(20)
    model = RUNScorer(
        seq_len=6,
        hidden_size=4,
        moving_average_kernel=5,
        pretrain_epochs=0,
        epochs=0,
        batch_size=7,
        device="cpu",
        seed=2,
    )
    result = model.score_metrics(normal, abnormal)
    assert result["metric"].tolist() == normal.columns.tolist()
    assert model.diagnostics_["training_scope"] == "normal_only"
    assert model.diagnostics_["device"] == "cpu"


def test_run_cycle_pruning_removes_a_cycle():
    graph = nx.DiGraph([("a", "b"), ("b", "a")])
    data = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]})
    assert RUNScorer._prune_cycles(graph, data) == 1
    assert nx.is_directed_acyclic_graph(graph)


def test_run_experiment_dispatches_metric_baseline(monkeypatch, tmp_path):
    normal, abnormal = _shift_case(30)
    case = BenchmarkCase(
        benchmark="rcaeval_re1",
        dataset="re1_ob",
        case_id="re1_ob__s2_cpu__1",
        root_cause_service="s2",
        fault_type="cpu",
        inject_time=30,
        repetition=1,
    )
    config = {
        "experiment": {
            "category": "baselines",
            "name": "epsilon_diagnosis_known_onset",
        },
        "model": {
            "target": "epsilon_diagnosis",
            "preprocess_strategy": "default",
            "params": {
                "alpha": 0.01,
                "bootstrap_time": 10,
                "root_cause_top_k": 5,
                "seed": 2,
            },
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
        lambda **kwargs: (normal, abnormal, {"n_metrics": 4}),
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
        config_path="configs/baselines/epsilon_diagnosis.yaml",
        granularity="service",
        benchmark_case=case,
        batch=True,
        progress=1,
        total_progress=1,
    )
    assert result["model_used"] == "epsilon_diagnosis"
    assert result["epsilon_diagnosis_diagnostics"]["bootstrap_time"] == 10
    assert output_file == str(tmp_path / f"{case.case_id}.json")

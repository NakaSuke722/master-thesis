import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.analyze_bsrc_adaptive_pseudo_fault import (
    _score_case,
    summarize,
    write_outputs as write_pseudo_outputs,
)
from scripts.run_bsrc_adaptive_subset import select_cases
from scripts.validate_bsrc_variance_integration import (
    METHODS,
    VARIANCE_RATIOS,
    run_validation,
    write_outputs,
)


def test_bsrc_synthetic_integration_validation_writes_artifacts(tmp_path):
    rows, report = run_validation(
        repetitions=1,
        pre_samples=80,
        post_samples=80,
        seed=11,
    )

    assert len(rows) == len(METHODS) * len(VARIANCE_RATIOS)
    assert len(report["convergence"]) == len(VARIANCE_RATIOS)
    write_outputs(tmp_path, rows, report)
    assert (tmp_path / "replicates.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.md").is_file()


def test_bsrc_pseudo_fault_scores_a_normal_only_case(tmp_path):
    case_dir = tmp_path / "case-a"
    case_dir.mkdir()
    rng = np.random.default_rng(13)
    pd.DataFrame({"svc_cpu": rng.normal(size=100)}).to_csv(
        case_dir / "normal_data.csv", index=False
    )
    (case_dir / "case_info.json").write_text(json.dumps({
        "case_id": "case-a",
        "dataset": "re1_ob",
        "fault_type": "cpu",
    }), encoding="utf-8")
    protocol = {
        "fit_fractions": [0.5],
        "ar_order": 1,
        "strong_evidence_log_bf": float(np.log(10.0)),
        "ar_bayes_prior": {
            "intercept_precision": 0.1,
            "lag_precision": 10.0,
            "alpha": 5.0,
            "beta": 4.0,
        },
        "ar_regime_shift_prior": {
            "inclusion_probability": 0.25,
            "variance_inclusion_probability": 0.25,
            "variance_integration": "adaptive_gh",
            "variance_quadrature_points": 7,
        },
    }

    rows = _score_case((case_dir, protocol))
    report = summarize(rows)
    write_pseudo_outputs(tmp_path / "out", rows, report)

    assert rows[0]["metric_failures"] == 0
    assert report["n_case_conditions"] == 1
    assert (tmp_path / "out/summary.md").is_file()


def test_subset_selection_is_balanced_and_reproducible(monkeypatch):
    datasets = ["re1_ob", "re1_ss", "re1_tt"]
    faults = ["cpu", "mem", "disk", "delay", "loss"]

    def cases(**kwargs):
        dataset = kwargs["dataset"]
        return [
            SimpleNamespace(
                dataset=dataset,
                fault_type=fault,
                case_id=f"{dataset}-{fault}-{index}",
            )
            for fault in faults
            for index in range(10)
        ]

    monkeypatch.setattr(
        "scripts.run_bsrc_adaptive_subset.list_benchmark_processed_cases",
        cases,
    )
    config = {
        "benchmark": {"name": "rcaeval_re1"},
        "datasets": datasets,
        "model": {"preprocess_strategy": "default"},
        "paths": {"processed_data_dir": "processed"},
        "subset_protocol": {
            "selection_seed": 7,
            "cases_per_dataset_fault": 2,
            "fault_types": faults,
        },
    }

    first = select_cases(config)
    second = select_cases(config)
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert len(first) == 3 * 5 * 2
    for dataset in datasets:
        for fault in faults:
            assert sum(
                case.dataset == dataset and case.fault_type == fault
                for case in first
            ) == 2

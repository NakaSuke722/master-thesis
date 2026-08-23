import json

import pytest

from scripts.analyze_adaptive_direct_rollback import (
    FULL_CANDIDATE,
    ROLLBACKS,
    analyze,
)


def _write(root, case_id, ranking):
    path = root / "service" / "re1_ob" / f"{case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "case_id": case_id,
        "dataset": "re1_ob",
        "fault_type": "cpu",
        "evaluation_granularity": "service",
        "evaluation_ground_truth": "root",
        "predicted_ranking": ranking,
    }), encoding="utf-8")


def test_rollback_analysis_reports_rollback_minus_full(tmp_path):
    roots = {
        method: tmp_path / method
        for method in (FULL_CANDIDATE, *ROLLBACKS)
    }
    _write(roots[FULL_CANDIDATE], "case-a", ["root", "other"])
    for rollback in ROLLBACKS:
        _write(roots[rollback], "case-a", ["other", "root"])

    report = analyze(roots, tmp_path / "out", bootstrap_samples=100, seed=3)

    assert report["difference_definition"] == "rollback_minus_full_adaptive"
    assert report["n_joined_cases"] == 1
    assert len(report["overall"]) == len(ROLLBACKS)
    assert all(
        item["metrics"]["AC@1"]["mean_difference"] == -1.0
        for item in report["overall"]
    )
    assert (tmp_path / "out" / "case_ranks.csv").is_file()
    assert (tmp_path / "out" / "summary.json").is_file()
    assert (tmp_path / "out" / "summary.md").is_file()


def test_rollback_analysis_rejects_different_case_sets(tmp_path):
    roots = {
        method: tmp_path / method
        for method in (FULL_CANDIDATE, *ROLLBACKS)
    }
    for method, root in roots.items():
        _write(root, "case-a", ["root", "other"])
    _write(roots[ROLLBACKS[0]], "case-b", ["root", "other"])

    with pytest.raises(ValueError, match="case_id sets differ"):
        analyze(roots, tmp_path / "out", bootstrap_samples=10, seed=3)

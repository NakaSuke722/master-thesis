import json

import pytest

from scripts.analyze_direct_ar_bayes_factor import analyze


METHODS = (
    "raw", "stationary_counterfactual_ar_uncertainty",
    "direct_ar_bayes_factor",
)


def _write(root, case_id, ranking, *, dataset="re1_ob", scores=False):
    payload = {
        "case_id": case_id,
        "dataset": dataset,
        "fault_type": "cpu",
        "evaluation_granularity": "service",
        "evaluation_ground_truth": "root",
        "predicted_ranking": ranking,
    }
    if scores:
        payload["amber_diagnostics"] = {
            "services": [
                {"rank": rank, "service": service, "score": 10.0 - rank}
                for rank, service in enumerate(ranking, 1)
            ]
        }
    path = root / "service" / dataset / f"{case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_direct_ar_analysis_joins_and_reports_paired_top1_gain(tmp_path):
    roots = {method: tmp_path / method for method in METHODS}
    for method, root in roots.items():
        _write(
            root,
            "case-a",
            ["root", "other"] if method == "direct_ar_bayes_factor"
            else ["other", "root"],
            scores=method == "direct_ar_bayes_factor",
        )
        _write(
            root, "case-b", ["root", "other"],
            scores=method == "direct_ar_bayes_factor",
        )

    report = analyze(roots, tmp_path / "out", bootstrap_samples=100, seed=3)

    assert report["n_joined_cases"] == 2
    raw = report["overall"][0]
    assert raw["metrics"]["AC@1"]["mean_difference"] == 0.5
    assert raw["metrics"]["AC@1"]["mcnemar"]["candidate_only_correct"] == 1
    assert report["rank_movement"][0]["improved"] == 1
    assert (tmp_path / "out" / "case_ranks.csv").is_file()
    assert (tmp_path / "out" / "summary.md").is_file()


def test_direct_ar_analysis_rejects_case_set_mismatch(tmp_path):
    roots = {method: tmp_path / method for method in METHODS}
    for method, root in roots.items():
        _write(
            root, "case-a", ["root", "other"],
            scores=method == "direct_ar_bayes_factor",
        )
    _write(roots["raw"], "case-b", ["root", "other"])

    with pytest.raises(ValueError, match="case_id sets differ"):
        analyze(roots, tmp_path / "out", bootstrap_samples=10)

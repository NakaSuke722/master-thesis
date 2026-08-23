import json

from scripts.analyze_adaptive_direct_ar_bayes_factor import analyze


METHODS = (
    "raw", "stationary_counterfactual_ar_uncertainty",
    "direct_ar_bayes_factor", "intercept_shift_ar_bayes_factor",
    "adaptive_direct_ar_bayes_factor",
)


def _write(root, ranking):
    path = root / "service" / "re1_ob" / "case-a.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "case_id": "case-a",
        "dataset": "re1_ob",
        "fault_type": "cpu",
        "evaluation_granularity": "service",
        "evaluation_ground_truth": "root",
        "predicted_ranking": ranking,
    }), encoding="utf-8")


def test_adaptive_analysis_reports_all_paired_comparisons(tmp_path):
    roots = {method: tmp_path / method for method in METHODS}
    for method, root in roots.items():
        _write(
            root,
            ["root", "other"]
            if method == "adaptive_direct_ar_bayes_factor"
            else ["other", "root"],
        )

    report = analyze(roots, tmp_path / "out", bootstrap_samples=100, seed=3)

    assert report["n_joined_cases"] == 1
    assert len(report["overall"]) == 4
    assert all(
        item["metrics"]["AC@1"]["mean_difference"] == 1.0
        for item in report["overall"]
    )
    assert (tmp_path / "out" / "summary.md").is_file()

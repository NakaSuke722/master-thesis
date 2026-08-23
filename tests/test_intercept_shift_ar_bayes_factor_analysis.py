import json

from scripts.analyze_intercept_shift_ar_bayes_factor import analyze


METHODS = (
    "raw",
    "stationary_counterfactual_ar_uncertainty",
    "direct_ar_bayes_factor",
    "intercept_shift_ar_bayes_factor",
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


def test_intercept_shift_analysis_reports_paired_gain(tmp_path):
    roots = {method: tmp_path / method for method in METHODS}
    for method, root in roots.items():
        ranking = (
            ["root", "other"]
            if method == "intercept_shift_ar_bayes_factor"
            else ["other", "root"]
        )
        _write(root, "case-a", ranking)

    report = analyze(roots, tmp_path / "out", bootstrap_samples=100, seed=3)

    assert report["n_joined_cases"] == 1
    assert report["overall"][0]["metrics"]["AC@1"]["mean_difference"] == 1.0
    assert (tmp_path / "out" / "case_ranks.csv").is_file()
    assert (tmp_path / "out" / "summary.md").is_file()

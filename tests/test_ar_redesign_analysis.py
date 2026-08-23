import json

from scripts.analyze_ar_redesign import analyze


METHODS = (
    "raw", "observed_ar", "counterfactual_ar", "stationary_ar",
    "stationary_counterfactual_ar", "stationary_counterfactual_ar_uncertainty",
    "stationary_counterfactual_ar_full_covariance",
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


def test_ar_redesign_analysis_reports_stagewise_paired_results(tmp_path):
    roots = {method: tmp_path / method for method in METHODS}
    for method, root in roots.items():
        ranking = (
            ["root", "other"]
            if method == "stationary_counterfactual_ar_full_covariance"
            else ["other", "root"]
        )
        _write(root, "case-a", ranking)

    report = analyze(roots, tmp_path / "out", bootstrap_samples=100, seed=3)

    assert report["n_joined_cases"] == 1
    final = report["overall"][-2]
    assert final["metrics"]["AC@1"]["mean_difference"] == 1.0
    assert (tmp_path / "out" / "summary.md").is_file()

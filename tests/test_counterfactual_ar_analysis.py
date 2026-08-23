import json

from scripts.analyze_counterfactual_ar import analyze


def _case(case_id, ranking, *, root="root", clipped=False, dataset="re1_ob"):
    services = []
    metrics = []
    for rank, service in enumerate(ranking, 1):
        services.append({"service": service, "rank": rank, "score": 10 - rank})
        for index in range(3):
            metrics.append({
                "metric": f"{service}_metric_{index}",
                "service": service,
                "rank": rank * 3 + index,
                "score": 10 - rank - index / 10,
                "counterfactual_clipped_predictions": (
                    1 if clipped and service == root and index == 0 else 0
                ),
                "counterfactual_clipped_fraction": (
                    0.1 if clipped and service == root and index == 0 else 0.0
                ),
            })
    return {
        "case_id": case_id,
        "dataset": dataset,
        "fault_type": "cpu",
        "evaluation_granularity": "service",
        "evaluation_ground_truth": root,
        "predicted_ranking": ranking,
        "amber_diagnostics": {"services": services, "metrics": metrics},
    }


def _write(root, payload):
    path = root / "service" / payload["dataset"] / f"{payload['case_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_counterfactual_analysis_joins_cases_and_reports_paired_gain_and_clip(tmp_path):
    raw, observed, counterfactual = (
        tmp_path / "raw", tmp_path / "observed", tmp_path / "counterfactual"
    )
    _write(raw, _case("case-a", ["other", "root"]))
    _write(observed, _case("case-a", ["other", "root"]))
    _write(counterfactual, _case("case-a", ["root", "other"], clipped=True))
    _write(raw, _case("case-b", ["root", "other"]))
    _write(observed, _case("case-b", ["root", "other"]))
    _write(counterfactual, _case("case-b", ["root", "other"]))

    report = analyze(
        raw, observed, counterfactual, tmp_path / "out",
        bootstrap_samples=100, seed=7,
    )

    raw_comparison = report["paired_comparisons"][0]
    assert report["n_joined_cases"] == 2
    assert raw_comparison["metrics"]["AC@1"]["mean_difference"] == 0.5
    assert raw_comparison["metrics"]["AC@1"]["mcnemar"]["candidate_only_correct"] == 1
    gain_group = next(
        row for row in report["clip_groups"]
        if row["group"] == "raw_non_top1_to_counterfactual_top1"
    )
    assert gain_group["n_cases"] == 1
    assert gain_group["root_any_clipped"] == 1
    assert (tmp_path / "out" / "summary.md").is_file()
    assert (tmp_path / "out" / "case_diagnostics.csv").is_file()

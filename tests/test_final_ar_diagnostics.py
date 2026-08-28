import json

import pytest

from scripts.analyze_final_ar_diagnostics import analyze, build_case_rows


def _case(case_id, ranking, *, dataset="re1_ob", fault="cpu", root="root"):
    return {
        "case_id": case_id,
        "dataset": dataset,
        "fault_type": fault,
        "evaluation_ground_truth": root,
        "predicted_ranking": ranking,
        "amber_diagnostics": {"metrics": [{
            "metric": f"{root}_cpu",
            "service": root,
            "score": 3.0,
            "forecast_uncertainty_max_multiplier": 12.0,
            "forecast_uncertainty_final_multiplier": 11.0,
        }]},
    }


def _write(root, payload):
    path = root / payload["dataset"] / f"{payload['case_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_case_rows_classifies_both_comparisons():
    raw = {"a": _case("a", ["other", "root"])}
    no_horizon = {"a": _case("a", ["root", "other"])}
    reference = {"a": _case("a", ["root", "other"])}

    row = build_case_rows(raw, no_horizon, reference)[0]

    assert row["raw_to_reference_delta"] == 1
    assert row["raw_to_reference_movement"] == "improved"
    assert row["horizon_delta"] == 0
    assert row["horizon_movement"] == "same"
    assert row["root_top3_max_uncertainty_multiplier"] == 12.0


def test_build_case_rows_accepts_degenerate_metric_with_null_score():
    payload = _case("a", ["root"])
    payload["amber_diagnostics"]["metrics"].append({
        "metric": "root_constant",
        "service": "root",
        "score": None,
        "forecast_uncertainty_max_multiplier": None,
    })

    row = build_case_rows({"a": payload}, {"a": payload}, {"a": payload})[0]

    assert row["root_top3_max_uncertainty_multiplier"] == 12.0


def test_analyze_writes_artifacts_and_counts_top1_transitions(tmp_path):
    roots = [tmp_path / name for name in ("raw", "no_horizon", "reference")]
    _write(roots[0], _case("a", ["root", "other"]))
    _write(roots[1], _case("a", ["other", "root"]))
    _write(roots[2], _case("a", ["root", "other"]))

    report = analyze(*roots, tmp_path / "output")

    overall = report["scopes"][0]
    assert overall["raw_to_reference"]["same"] == 1
    assert overall["horizon_aware_uncertainty"]["improved"] == 1
    assert overall["horizon_aware_uncertainty"][
        "baseline_non_top1_to_reference_top1"
    ] == 1
    assert report["uncertainty_multipliers"]["ge_10"] == 1
    assert (tmp_path / "output" / "summary.md").is_file()


def test_analyze_rejects_case_set_mismatch(tmp_path):
    roots = [tmp_path / name for name in ("raw", "no_horizon", "reference")]
    _write(roots[0], _case("a", ["root"]))
    _write(roots[1], _case("a", ["root"]))
    _write(roots[2], _case("b", ["root"]))

    with pytest.raises(ValueError, match="case_id sets differ"):
        analyze(*roots, tmp_path / "output")

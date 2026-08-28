import json

import pytest

from scripts.analyze_final_ar_sensitivity import REFERENCE, analyze


def _case(case_id, ranking):
    return {
        "case_id": case_id,
        "dataset": "re1_ob",
        "fault_type": "cpu",
        "evaluation_ground_truth": "root",
        "predicted_ranking": ranking,
    }


def _write_variant(root, name, payload):
    case_path = root / name / "service" / "re1_ob" / "case.json"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(json.dumps(payload), encoding="utf-8")
    summary = {"summary": {"re1_ob": {
        "AC@1": 1.0, "AC@3": 1.0, "AC@5": 1.0, "Avg@5": 1.0,
    }}}
    (root / name / "summary_service.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


def test_analyze_strictly_pairs_and_summarizes(tmp_path):
    results = tmp_path / "results"
    _write_variant(results, REFERENCE, _case("a", ["root", "other"]))
    _write_variant(results, "variant", _case("a", ["other", "root"]))

    report = analyze(
        results, tmp_path / "pseudo", tmp_path / "output",
        variants=(REFERENCE, "variant"),
    )

    comparison = report["variants"][1]["paired_vs_reference"]
    assert comparison["reference_improved"] == 1
    assert comparison["reference_top1_gains"] == 1
    assert report["variants"][1]["pseudo_fault"] is None
    assert (tmp_path / "output" / "case_rank_sensitivity.csv").is_file()


def test_analyze_rejects_case_set_mismatch(tmp_path):
    results = tmp_path / "results"
    _write_variant(results, REFERENCE, _case("a", ["root"]))
    _write_variant(results, "variant", _case("b", ["root"]))

    with pytest.raises(ValueError, match="case_id sets differ"):
        analyze(
            results, tmp_path / "pseudo", tmp_path / "output",
            variants=(REFERENCE, "variant"),
        )

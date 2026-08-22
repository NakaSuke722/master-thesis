import json

import pytest

from scripts.analyze_ar_rank_movement import analyze


def _case(case_id, ranking, dataset="re1_ob", fault_type="cpu", root="cartservice"):
    return {
        "case_id": case_id,
        "dataset": dataset,
        "fault_type": fault_type,
        "evaluation_granularity": "service",
        "evaluation_ground_truth": root,
        "amber_diagnostics": {"services": [
            {"service": service, "rank": rank} for rank, service in enumerate(ranking, 1)
        ]},
    }


def _write(root, name, payload):
    path = root / "service" / payload["dataset"] / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_uses_case_id_ranks_and_writes_grouped_reports(tmp_path):
    raw, ar, output = tmp_path / "raw", tmp_path / "ar", tmp_path / "out"
    _write(raw, "a", _case("case-a", ["other", "cartservice"], fault_type="cpu"))
    _write(ar, "different-file-name", _case("case-a", ["cartservice", "other"], fault_type="cpu"))
    _write(raw, "b", _case("case-b", ["cartservice", "other"], dataset="re1_tt", fault_type="delay"))
    _write(ar, "b", _case("case-b", ["other", "cartservice"], dataset="re1_tt", fault_type="delay"))

    report = analyze(raw, ar, output)

    assert report["overall"]["improved"] == 1
    assert report["overall"]["worsened"] == 1
    assert report["overall"]["raw_non_top1_to_ar_top1"] == 1
    assert report["overall"]["raw_top1_to_ar_non_top1"] == 1
    assert report["by_dataset"][0]["dataset"] == "re1_ob"
    assert (output / "case_rank_movement.csv").is_file()
    assert (output / "by_fault_type.csv").is_file()


def test_analyze_rejects_nonidentical_case_id_sets(tmp_path):
    raw, ar = tmp_path / "raw", tmp_path / "ar"
    _write(raw, "a", _case("case-a", ["cartservice"]))
    _write(ar, "b", _case("case-b", ["cartservice"]))

    with pytest.raises(ValueError, match="case_id sets differ"):
        analyze(raw, ar, tmp_path / "out")


def test_analyze_rejects_top5_only_artifact(tmp_path):
    raw, ar = tmp_path / "raw", tmp_path / "ar"
    payload = _case("case-a", ["cartservice"])
    payload.pop("amber_diagnostics")
    payload["predicted_top_5"] = ["cartservice"]
    _write(raw, "a", payload)
    _write(ar, "a", _case("case-a", ["cartservice"]))

    with pytest.raises(ValueError, match="predicted_top_5 is insufficient"):
        analyze(raw, ar, tmp_path / "out")

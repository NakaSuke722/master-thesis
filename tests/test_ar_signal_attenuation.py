import json

import pytest

from scripts.analyze_ar_signal_attenuation import analyze


def _metric(name, service, score, rank, residualization):
    common = {
        "metric": name,
        "service": service,
        "score": score,
        "rank": rank,
        "raw_normal": [0.0, 0.0, 0.0, 0.0],
        "raw_abnormal": [10.0, 10.0, 10.0, 10.0],
    }
    if residualization == "ar":
        return {
            **common,
            "ar_coefficients": [0.0, 0.6, 0.3],
            "ar_residual_normal": [0.0, 0.0, 0.0],
            "ar_residual_abnormal": [10.0, 4.0, 1.0, 1.0],
        }
    return {
        **common,
        "ar_coefficients": [],
        "ar_residual_normal": common["raw_normal"],
        "ar_residual_abnormal": common["raw_abnormal"],
    }


def _case(case_id, ranking, residualization, fault_type="disk"):
    services = [{"service": service, "rank": rank, "score": 3 - rank} for rank, service in enumerate(ranking, 1)]
    return {
        "case_id": case_id,
        "dataset": "re1_ob",
        "fault_type": fault_type,
        "evaluation_granularity": "service",
        "evaluation_ground_truth": "root",
        "amber_diagnostics": {
            "services": services,
            "metrics": [
                _metric("root_cpu", "root", 2.0 if residualization == "raw" else 1.0, 1, residualization),
                _metric("other_cpu", "other", 0.5, 2, residualization),
            ],
        },
    }


def _write(root, payload):
    path = root / "service" / payload["dataset"] / f"{payload['case_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_diagnoses_shift_retention_decay_and_rank_movement(tmp_path):
    raw_root, ar_root, output = tmp_path / "raw", tmp_path / "ar", tmp_path / "out"
    _write(raw_root, _case("case-1", ["root", "other"], "raw"))
    _write(ar_root, _case("case-1", ["other", "root"], "ar"))

    report = analyze(raw_root, ar_root, output)
    metric_rows = json.loads((output / "root_metric_diagnostics.json").read_text())
    case_rows = json.loads((output / "case_diagnostics.json").read_text())

    assert report["n_joined_cases"] == 1
    assert report["n_root_metrics"] == 1
    assert metric_rows[0]["sum_phi"] == pytest.approx(0.9)
    assert metric_rows[0]["raw_shift"] == 10.0
    assert metric_rows[0]["signal_retention_ratio"] == pytest.approx(0.25)
    assert metric_rows[0]["ar_initial_to_late_retention_ratio"] == pytest.approx(1.0 / 7.0)
    assert metric_rows[0]["ar_initial_to_late_decay_fraction"] == pytest.approx(1.0 - 1.0 / 7.0)
    assert case_rows[0]["rank_delta"] == -1
    assert case_rows[0]["movement"] == "worsened"
    assert (output / "summary.md").is_file()
    assert (output / "case_level_correlations.csv").is_file()


def test_rejects_missing_complete_metric_diagnostics(tmp_path):
    raw_root, ar_root = tmp_path / "raw", tmp_path / "ar"
    raw = _case("case-1", ["root", "other"], "raw")
    ar = _case("case-1", ["root", "other"], "ar")
    ar["amber_diagnostics"]["metrics"] = []
    _write(raw_root, raw)
    _write(ar_root, ar)

    with pytest.raises(ValueError, match="Missing AMBER metric diagnostics"):
        analyze(raw_root, ar_root, tmp_path / "out")


def test_rejects_missing_result_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="No service result artifacts found"):
        analyze(tmp_path / "raw", tmp_path / "ar", tmp_path / "out")


def test_reconstructs_observed_lag_residuals_from_processed_csv(tmp_path):
    raw_root, ar_root = tmp_path / "raw", tmp_path / "ar"
    raw = _case("case-1", ["root", "other"], "raw")
    ar = _case("case-1", ["other", "root"], "ar")
    for payload in (raw, ar):
        for metric in payload["amber_diagnostics"]["metrics"]:
            metric.pop("raw_normal")
            metric.pop("raw_abnormal")
            metric.pop("ar_residual_normal")
            metric.pop("ar_residual_abnormal")
    _write(raw_root, raw)
    _write(ar_root, ar)

    case_dir = tmp_path / "processed/default/rcaeval_re1/re1_ob/case-1"
    case_dir.mkdir(parents=True)
    for filename, values in (("normal_data.csv", [0, 0, 0, 0]), ("abnormal_data.csv", [10, 10, 10, 10])):
        (case_dir / filename).write_text(
            "root_cpu,other_cpu\n" + "".join(f"{value},0\n" for value in values),
            encoding="utf-8",
        )

    analyze(raw_root, ar_root, tmp_path / "out", tmp_path / "processed")
    row = json.loads((tmp_path / "out/root_metric_diagnostics.json").read_text())[0]
    assert row["signal_retention_ratio"] == pytest.approx(0.25)

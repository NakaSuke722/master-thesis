from scripts.validate_ar_bayes_factor_synthetic import (
    run_validation,
    write_outputs,
)


def test_synthetic_validation_is_reproducible_and_writes_artifacts(tmp_path):
    rows_a, report_a = run_validation(
        repetitions=3,
        pre_samples=80,
        post_samples=80,
        burn_in=40,
        seed=7,
    )
    rows_b, report_b = run_validation(
        repetitions=3,
        pre_samples=80,
        post_samples=80,
        burn_in=40,
        seed=7,
    )

    assert rows_a == rows_b
    assert report_a == report_b
    assert len(rows_a) == 15
    assert "all_checks_passed" in report_a
    assert {item["scenario"] for item in report_a["summaries"]} == {
        "no_change",
        "persistent_mean_shift",
        "ar_coefficient_change",
        "innovation_variance_change",
        "single_spike",
    }

    write_outputs(tmp_path, rows_a, report_a)

    assert (tmp_path / "replicates.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.md").is_file()

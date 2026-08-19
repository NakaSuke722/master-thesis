import pandas as pd

from benchmarks.rcaeval_re1 import (
    discover_cases,
    load_case_index,
)


def test_rcaeval_case_index(tmp_path):

    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    case_id = (
        "re1ob_adservice_cpu_1"
    )

    case_dir = (
        raw_root / case_id
    )

    case_dir.mkdir()

    pd.DataFrame(
        {
            "time": [1, 2, 3],
            "adservice_cpu": [
                1.0,
                2.0,
                3.0,
            ],
        }
    ).to_parquet(
        case_dir / "metrics.parquet"
    )

    index = pd.DataFrame(
        {
            "case": [case_id],
            "dataset": ["RE1-OB"],
            "root_cause_service": [
                "adservice"
            ],
            "fault": ["cpu"],
            "repetition": [1],
            "inject_time": [2],
        }
    )

    index.to_parquet(
        raw_root / "cases.parquet"
    )

    df = load_case_index(
        raw_root
    )

    assert len(df) == 1

    assert (
        df.iloc[0][
            "dataset_internal"
        ]
        == "re1_ob"
    )

    cases = discover_cases(
        raw_root
    )

    assert len(cases) == 1

    case = cases[0]

    assert (
        case.root_cause_service
        == "adservice"
    )

    assert case.fault_type == "cpu"
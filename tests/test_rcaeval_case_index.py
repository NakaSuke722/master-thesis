from pathlib import Path

import pandas as pd

from benchmarks.rcaeval_re1 import (
    discover_cases,
    parse_fault_directory,
)


def make_case(
    raw_root: Path,
    *,
    dataset_internal: str,
    dataset_original: str,
    fault_directory: str,
    repetition: int,
    inject_time: int,
) -> Path:
    case_dir = (
        raw_root
        / dataset_internal
        / dataset_original
        / fault_directory
        / str(repetition)
    )

    case_dir.mkdir(
        parents=True
    )

    pd.DataFrame(
        {
            "time": [
                inject_time - 1,
                inject_time,
                inject_time + 1,
            ],
            "adservice_cpu": [
                1.0,
                5.0,
                6.0,
            ],
        }
    ).to_csv(
        case_dir / "data.csv",
        index=False,
    )

    (
        case_dir / "inject_time.txt"
    ).write_text(
        str(inject_time),
        encoding="utf-8",
    )

    return case_dir


def test_parse_fault_directory():

    service, fault = (
        parse_fault_directory(
            "currencyservice_loss"
        )
    )

    assert service == "currencyservice"
    assert fault == "loss"


def test_discover_zenodo_case(
    tmp_path,
):

    raw_root = tmp_path / "raw"

    make_case(
        raw_root,
        dataset_internal="re1_ob",
        dataset_original="RE1-OB",
        fault_directory=(
            "adservice_cpu"
        ),
        repetition=1,
        inject_time=100,
    )

    cases = discover_cases(
        raw_root,
        datasets=["re1_ob"],
    )

    assert len(cases) == 1

    case = cases[0]

    assert (
        case.dataset
        == "re1_ob"
    )

    assert (
        case.case_id
        == "re1_ob__adservice_cpu__1"
    )

    assert (
        case.root_cause_service
        == "adservice"
    )

    assert case.fault_type == "cpu"
    assert case.repetition == 1
    assert case.inject_time == 100

    assert (
        case.source_path
        == (
            raw_root
            / "re1_ob"
            / "RE1-OB"
            / "adservice_cpu"
            / "1"
        )
    )
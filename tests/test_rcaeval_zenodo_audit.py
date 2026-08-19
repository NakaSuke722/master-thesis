from pathlib import Path

import pandas as pd
import pytest

from benchmarks.rcaeval_re1 import (
    discover_cases,
)


RAW_ROOT = Path(
    "data/raw/rcaeval_zenodo_v2"
)


@pytest.mark.skipif(
    not RAW_ROOT.is_dir(),
    reason=(
        "Zenodo RCAEval data "
        "is not downloaded."
    ),
)
def test_zenodo_re1_all_375_cases():

    cases = discover_cases(
        RAW_ROOT,
        datasets=[
            "re1_ob",
            "re1_ss",
            "re1_tt",
        ],
    )

    counts = {
        dataset: sum(
            case.dataset == dataset
            for case in cases
        )
        for dataset in (
            "re1_ob",
            "re1_ss",
            "re1_tt",
        )
    }

    assert counts == {
        "re1_ob": 125,
        "re1_ss": 125,
        "re1_tt": 125,
    }

    assert len(cases) == 375

    assert len(
        {case.case_id for case in cases}
    ) == 375

    for case in cases:
        data_path = (
            case.source_path
            / "data.csv"
        )

        time_df = pd.read_csv(
            data_path,
            usecols=["time"],
        )

        time_series = (
            time_df["time"]
            .sort_values()
            .drop_duplicates()
        )

        time_min = int(
            time_series.min()
        )

        time_max = int(
            time_series.max()
        )

        assert (
            time_min
            < case.inject_time
            <= time_max
        )

        normal_count = int(
            (
                time_series
                < case.inject_time
            ).sum()
        )

        abnormal_count = int(
            (
                time_series
                >= case.inject_time
            ).sum()
        )

        assert normal_count > 0
        assert abnormal_count > 0

        # 実際の前処理後は最大600点。
        assert min(
            normal_count,
            600,
        ) > 0

        assert min(
            abnormal_count,
            600,
        ) > 0
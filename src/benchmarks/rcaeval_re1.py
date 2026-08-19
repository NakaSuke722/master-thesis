from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks.base import BenchmarkCase


BENCHMARK_NAME = "rcaeval_re1"

DATASET_NAME_MAP = {
    "RE1-OB": "re1_ob",
    "RE1-SS": "re1_ss",
    "RE1-TT": "re1_tt",
}

REQUIRED_INDEX_COLUMNS = {
    "case",
    "dataset",
    "root_cause_service",
    "fault",
    "repetition",
    "inject_time",
}


def load_case_index(
    raw_root: str | Path,
) -> pd.DataFrame:
    """RCAEval cases.parquetからRE1のみを読み込む。"""

    raw_root = Path(raw_root)
    index_path = raw_root / "cases.parquet"

    if not index_path.is_file():
        raise FileNotFoundError(
            f"RCAEval case index not found: {index_path}"
        )

    df = pd.read_parquet(index_path)

    missing = (
        REQUIRED_INDEX_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "RCAEval case index is missing columns: "
            f"{sorted(missing)}"
        )

    df = df[
        df["dataset"].isin(
            DATASET_NAME_MAP.keys()
        )
    ].copy()

    df["dataset_internal"] = (
        df["dataset"].map(DATASET_NAME_MAP)
    )

    return df.reset_index(drop=True)


def discover_cases(
    raw_root: str | Path,
    datasets: list[str] | None = None,
) -> list[BenchmarkCase]:
    """RCAEval RE1の全ケースをBenchmarkCaseとして返す。"""

    raw_root = Path(raw_root)

    df = load_case_index(raw_root)

    if datasets is not None:
        df = df[
            df["dataset_internal"].isin(datasets)
        ]

    cases: list[BenchmarkCase] = []

    for row in df.sort_values(
        ["dataset_internal", "case"]
    ).itertuples(index=False):

        source_path = raw_root / row.case

        metrics_path = (
            source_path / "metrics.parquet"
        )

        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"metrics.parquet not found: "
                f"{metrics_path}"
            )

        cases.append(
            BenchmarkCase(
                benchmark=BENCHMARK_NAME,
                dataset=row.dataset_internal,
                case_id=row.case,
                root_cause_service=(
                    row.root_cause_service
                ),
                fault_type=row.fault,
                inject_time=int(row.inject_time),
                repetition=int(row.repetition),
                root_cause_metrics=None,
                source_path=source_path,
            )
        )

    return cases
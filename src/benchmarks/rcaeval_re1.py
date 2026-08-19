from __future__ import annotations

from pathlib import Path

import pandas as pd
import warnings

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

def load_inject_time(
    source_path: Path,
    index_inject_time: int,
) -> int:
    """各ケースのinject_time.txtから障害注入時刻を取得する。

    cases.parquetのinject_timeは索引用metadataとして扱い、
    実際の実験ではRCAEval公式コードと同様に
    case-localなinject_time.txtをsource of truthとする。
    """

    inject_path = (
        source_path / "inject_time.txt"
    )

    if not inject_path.is_file():
        raise FileNotFoundError(
            "inject_time.txt not found: "
            f"{inject_path}"
        )

    try:
        inject_time = int(
            inject_path
            .read_text(encoding="utf-8")
            .strip()
        )
    except ValueError as exc:
        raise ValueError(
            "Invalid inject_time.txt: "
            f"{inject_path}"
        ) from exc

    if inject_time != int(index_inject_time):
        warnings.warn(
            "RCAEval inject-time mismatch: "
            f"{source_path.name}: "
            f"cases.parquet={index_inject_time}, "
            f"inject_time.txt={inject_time}. "
            "Using inject_time.txt.",
            RuntimeWarning,
            stacklevel=2,
        )

    return inject_time


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

        inject_time = load_inject_time(
            source_path=source_path,
            index_inject_time=int(
                row.inject_time
            ),
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
                inject_time=inject_time,
                repetition=int(row.repetition),
                root_cause_metrics=None,
                source_path=source_path,
            )
        )

    return cases
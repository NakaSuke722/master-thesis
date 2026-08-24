from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.rcaeval_re1 import (
    discover_cases,
)
from experiments.config import load_config


def preprocess_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """RCAEval metric seriesへ共通前処理を適用する。"""

    if "time" not in df.columns:
        raise ValueError(
            "RCAEval metrics must contain 'time'"
        )

    work = df.copy()

    work = work.sort_values("time")
    work = work.drop_duplicates(
        subset=["time"],
        keep="last",
    )

    work = work.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    work = work.ffill().fillna(0)

    # pandas renames duplicate CSV headers such as a second ``time`` column
    # to ``time.1``.  These aliases are timestamps, not RCA metrics, and must
    # not reach a ranking model as if they were services.
    time_aliases = [
        column
        for column in work.columns
        if column != "time"
        and re.fullmatch(r"time\.\d+", str(column), flags=re.IGNORECASE)
    ]
    work = work.drop(columns=time_aliases)

    # RCAEval/BARO系評価で利用してきた
    # latency-90へ統一する。
    work = work.loc[
        :,
        ~work.columns.str.endswith(
            "_latency-50"
        ),
    ]

    rename_map = {
        column: column.replace(
            "_latency-90",
            "_latency",
        )
        for column in work.columns
        if column.endswith("_latency-90")
    }

    work = work.rename(
        columns=rename_map
    )

    if work.columns.duplicated().any():
        duplicates = (
            work.columns[
                work.columns.duplicated()
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate metric names after "
            f"canonicalization: {duplicates}"
        )

    # A series that is constant over the complete case cannot distinguish
    # normal from abnormal behavior.  Drop only complete-case constants: a
    # metric that is constant during normal operation but changes after the
    # fault remains available for RCA.
    metric_columns = [
        column for column in work.columns
        if column != "time"
    ]
    no_information_metrics = [
        column
        for column in metric_columns
        if work[column].nunique(dropna=False) <= 1
    ]
    work = work.drop(columns=no_information_metrics)

    return work.reset_index(drop=True)


def split_normal_abnormal(
    df: pd.DataFrame,
    inject_time: int,
    normal_window_points: int | None = None,
    abnormal_window_points: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """障害注入時刻でnormal/abnormalへ分割する。"""

    normal = df[
        df["time"] < inject_time
    ].copy()

    abnormal = df[
        df["time"] >= inject_time
    ].copy()

    if normal.empty:
        raise ValueError(
            "Normal period is empty."
        )

    if abnormal.empty:
        raise ValueError(
            "Abnormal period is empty."
        )

    if normal_window_points is not None:
        normal = normal.tail(
            normal_window_points
        )

    if abnormal_window_points is not None:
        abnormal = abnormal.head(
            abnormal_window_points
        )

    normal = normal.drop(
        columns=["time"]
    ).reset_index(drop=True)

    abnormal = abnormal.drop(
        columns=["time"]
    ).reset_index(drop=True)

    # Judge information content on the exact windows passed to RCA models.
    # A metric may vary outside these windows while being identical throughout
    # both selected segments, in which case it is still uninformative for this
    # case.  Conversely, normal-only constants that change after injection are
    # retained because the concatenated series has multiple values.
    combined = pd.concat(
        [normal, abnormal],
        ignore_index=True,
    )
    no_information_metrics = [
        column
        for column in combined.columns
        if combined[column].nunique(dropna=False) <= 1
    ]
    normal = normal.drop(columns=no_information_metrics)
    abnormal = abnormal.drop(columns=no_information_metrics)

    if normal.shape[1] == 0:
        raise ValueError(
            "No informative RCAEval metrics remain "
            "after preprocessing."
        )

    return normal, abnormal


def prepare_case(
    case,
    raw_root: str | Path,
    processed_root: str | Path,
    strategy: str,
    normal_window_points: int | None,
    abnormal_window_points: int | None,
    data_source: dict,
) -> Path:
    
    if case.source_path is None:
        raise ValueError(
            f"No source path for {case.case_id}"
        )

    metrics_path = (
        case.source_path
        / "data.csv"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"data.csv not found: "
            f"{metrics_path}"
        )

    df = pd.read_csv(metrics_path)
    df = preprocess_metrics(df)

    time_min = int(df["time"].min())
    time_max = int(df["time"].max())

    if not (
        time_min
        < case.inject_time
        <= time_max
    ):
        raise ValueError(
            "Inject time is outside "
            "the valid split range: "
            f"case={case.case_id}, "
            f"inject_time={case.inject_time}, "
            f"time_min={time_min}, "
            f"time_max={time_max}"
        )

    normal, abnormal = (
        split_normal_abnormal(
            df=df,
            inject_time=case.inject_time,
            normal_window_points=(
                normal_window_points
            ),
            abnormal_window_points=(
                abnormal_window_points
            ),
        )
    )

    output_dir = (
        Path(processed_root)
        / strategy
        / case.benchmark
        / case.dataset
        / case.case_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    normal.to_csv(
        output_dir / "normal_data.csv",
        index=False,
    )

    abnormal.to_csv(
        output_dir / "abnormal_data.csv",
        index=False,
    )

    case_info = case.to_dict()

    try:
        source_case_path = (
            case.source_path
            .relative_to(Path(raw_root))
            .as_posix()
        )
    except ValueError:
        source_case_path = (
            case.source_path.as_posix()
        )

    case_info.update(
        {
            "data_source": data_source,
            "source_case_path": (
                source_case_path
            ),
            "normal_samples": len(normal),
            "abnormal_samples": len(abnormal),
            "n_metrics": len(normal.columns),
            "preprocessing": {
                "strategy": strategy,
                "split_rule": {
                    "normal": (
                        "time < inject_time"
                    ),
                    "abnormal": (
                        "time >= inject_time"
                    ),
                },
                "drop_latency_50": True,
                "rename_latency_90": True,
                "drop_time_aliases": True,
                "drop_complete_case_constant_metrics": (
                    True
                ),
                "drop_selected_window_constant_metrics": (
                    True
                ),
                "normal_window_points": (
                    normal_window_points
                ),
                "abnormal_window_points": (
                    abnormal_window_points
                ),
            },
        }
    )

    with (
        output_dir / "case_info.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            case_info,
            f,
            indent=4,
            ensure_ascii=False,
        )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare RCAEval RE1 for AMBER"
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/main/"
            "rcaeval_re1.yaml"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N cases "
            "for smoke testing."
        ),
    )

    args = parser.parse_args()

    config = load_config(args.config)

    benchmark = config["benchmark"]

    if benchmark["name"] != "rcaeval_re1":
        raise ValueError(
            "This script only supports "
            "rcaeval_re1."
        )

    raw_root = config["paths"][
        "raw_data_dir"
    ]

    processed_root = config["paths"][
        "processed_data_dir"
    ]

    strategy = config["model"].get(
        "preprocess_strategy",
        "default",
    )

    if strategy != "default":
        raise ValueError(
            "RCAEval migration currently "
            "supports only strategy='default'."
        )

    data_config = config.get(
        "data",
        {},
    )

    data_source = data_config.get(
    "source",
    {},
    )

    normal_window_points = (
        data_config.get(
            "normal_window_points"
        )
    )

    abnormal_window_points = (
        data_config.get(
            "abnormal_window_points"
        )
    )

    cases = discover_cases(
        raw_root=raw_root,
        datasets=config.get(
            "datasets"
        ),
    )

    if args.limit is not None:
        cases = cases[: args.limit]

    total = len(cases)

    for index, case in enumerate(
        cases,
        start=1,
    ):
        output_dir = prepare_case(
            case=case,
            raw_root=raw_root,
            processed_root=processed_root,
            strategy=strategy,
            normal_window_points=(
                normal_window_points
            ),
            abnormal_window_points=(
                abnormal_window_points
            ),
            data_source=data_source,
        )
        
        print(
            f"[{index}/{total}] "
            f"{case.case_id} -> "
            f"{output_dir}"
        )

    print(
        f"Prepared {total} RCAEval RE1 cases."
    )


if __name__ == "__main__":
    main()

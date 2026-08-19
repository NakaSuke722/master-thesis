from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmarks.base import BenchmarkCase


# ============================================================
# Legacy BARO pilot API
# ============================================================

def get_processed_case_dir(
    dataset: str,
    fault_type: str,
    run_id: int,
    strategy: str = "default",
    processed_root: str = "data/processed",
) -> Path:
    return (
        Path(processed_root)
        / strategy
        / dataset
        / fault_type
        / str(run_id)
    )


def load_processed_case(
    dataset: str,
    fault_type: str,
    run_id: int,
    strategy: str = "default",
    processed_root: str = "data/processed",
    load_graph_info: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict | None,
]:

    target_dir = get_processed_case_dir(
        dataset=dataset,
        fault_type=fault_type,
        run_id=run_id,
        strategy=strategy,
        processed_root=processed_root,
    )

    normal_path = (
        target_dir / "normal_data.csv"
    )

    abnormal_path = (
        target_dir / "abnormal_data.csv"
    )

    if not normal_path.is_file():
        raise FileNotFoundError(
            f"Normal data not found: "
            f"{normal_path}"
        )

    if not abnormal_path.is_file():
        raise FileNotFoundError(
            f"Abnormal data not found: "
            f"{abnormal_path}"
        )

    df_normal = pd.read_csv(normal_path)
    df_abnormal = pd.read_csv(
        abnormal_path
    )

    graph_info = None

    if load_graph_info:
        info_path = (
            target_dir
            / "graph_info.json"
        )

        if not info_path.is_file():
            raise FileNotFoundError(
                f"Graph info not found: "
                f"{info_path}"
            )

        with info_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            graph_info = json.load(f)

    return (
        df_normal,
        df_abnormal,
        graph_info,
    )


def load_timeseries_data(
    dataset: str,
    fault_type: str,
    run_id: int,
    strategy: str = "default",
    processed_root: str = "data/processed",
):
    df_normal, df_abnormal, graph_info = (
        load_processed_case(
            dataset=dataset,
            fault_type=fault_type,
            run_id=run_id,
            strategy=strategy,
            processed_root=processed_root,
            load_graph_info=True,
        )
    )

    df_full = pd.concat(
        [df_normal, df_abnormal],
        ignore_index=True,
    )

    ground_truth = fault_type

    return (
        df_full,
        ground_truth,
        graph_info,
    )


# ============================================================
# Benchmark-independent API
# ============================================================

def get_benchmark_processed_case_dir(
    benchmark: str,
    dataset: str,
    case_id: str,
    strategy: str = "default",
    processed_root: str = "data/processed",
) -> Path:
    return (
        Path(processed_root)
        / strategy
        / benchmark
        / dataset
        / case_id
    )


def load_benchmark_processed_case(
    benchmark: str,
    dataset: str,
    case_id: str,
    strategy: str = "default",
    processed_root: str = "data/processed",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    target_dir = (
        get_benchmark_processed_case_dir(
            benchmark=benchmark,
            dataset=dataset,
            case_id=case_id,
            strategy=strategy,
            processed_root=processed_root,
        )
    )

    normal_path = (
        target_dir / "normal_data.csv"
    )

    abnormal_path = (
        target_dir / "abnormal_data.csv"
    )

    info_path = (
        target_dir / "case_info.json"
    )

    for path in (
        normal_path,
        abnormal_path,
        info_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Processed case file "
                f"not found: {path}"
            )

    df_normal = pd.read_csv(
        normal_path
    )

    df_abnormal = pd.read_csv(
        abnormal_path
    )

    with info_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        case_info = json.load(f)

    return (
        df_normal,
        df_abnormal,
        case_info,
    )


def list_benchmark_processed_cases(
    benchmark: str,
    dataset: str,
    strategy: str = "default",
    processed_root: str = "data/processed",
) -> list[BenchmarkCase]:

    dataset_dir = (
        Path(processed_root)
        / strategy
        / benchmark
        / dataset
    )

    if not dataset_dir.is_dir():
        return []

    cases: list[BenchmarkCase] = []

    for case_dir in sorted(
        dataset_dir.iterdir()
    ):
        if not case_dir.is_dir():
            continue

        info_path = (
            case_dir / "case_info.json"
        )

        if not info_path.is_file():
            continue

        with info_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            info = json.load(f)

        cases.append(
            BenchmarkCase.from_dict(
                info
            )
        )

    return cases
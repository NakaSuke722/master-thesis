# src/data_loader.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def get_processed_case_dir(
    dataset: str,
    fault_type: str,
    run_id: int,
    strategy: str = "default",
    processed_root: str = "data/processed",
) -> Path:
    """Return the directory containing one processed RCA case."""
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    """Load normal and abnormal time-series data for one RCA case."""

    target_dir = get_processed_case_dir(
        dataset=dataset,
        fault_type=fault_type,
        run_id=run_id,
        strategy=strategy,
        processed_root=processed_root,
    )

    normal_path = target_dir / "normal_data.csv"
    abnormal_path = target_dir / "abnormal_data.csv"

    if not normal_path.is_file():
        raise FileNotFoundError(
            f"Normal data not found: {normal_path}"
        )

    if not abnormal_path.is_file():
        raise FileNotFoundError(
            f"Abnormal data not found: {abnormal_path}"
        )

    df_normal = pd.read_csv(normal_path)
    df_abnormal = pd.read_csv(abnormal_path)

    graph_info = None

    if load_graph_info:
        info_path = target_dir / "graph_info.json"

        if not info_path.is_file():
            raise FileNotFoundError(
                f"Graph info not found: {info_path}"
            )

        with info_path.open("r", encoding="utf-8") as f:
            graph_info = json.load(f)

    return df_normal, df_abnormal, graph_info


def load_timeseries_data(
    dataset: str,
    fault_type: str,
    run_id: int,
    strategy: str = "default",
    processed_root: str = "data/processed",
):
    """Backward-compatible loader for models requiring concatenated data."""

    df_normal, df_abnormal, graph_info = load_processed_case(
        dataset=dataset,
        fault_type=fault_type,
        run_id=run_id,
        strategy=strategy,
        processed_root=processed_root,
        load_graph_info=True,
    )

    df_full = pd.concat(
        [df_normal, df_abnormal],
        ignore_index=True,
    )

    ground_truth = fault_type

    return df_full, ground_truth, graph_info
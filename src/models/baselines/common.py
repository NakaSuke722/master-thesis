from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TIME_COLUMNS = {"time", "time.1", "timestamp"}


def service_from_metric(metric: str) -> str:
    """Convert RCAEval's ``service_metric`` name to a service name."""
    return str(metric).split("_", 1)[0]


def to_service_ranking(metric_ranking: list[str]) -> list[str]:
    """Deduplicate services while preserving metric-rank order."""
    ranking: list[str] = []
    seen: set[str] = set()
    for metric in metric_ranking:
        service = service_from_metric(metric)
        if service not in seen:
            ranking.append(service)
            seen.add(service)
    return ranking


def ranking_frame(
    ranking: list[str],
    scores: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Represent a metric ranking in the repository's score-table format."""
    if scores is None:
        scores = {
            metric: float(len(ranking) - index)
            for index, metric in enumerate(ranking)
        }
    return pd.DataFrame(
        {
            "metric": ranking,
            "score": [float(scores[metric]) for metric in ranking],
        }
    )


@dataclass(frozen=True)
class PairedMetricData:
    normal: pd.DataFrame
    abnormal: pd.DataFrame
    excluded: dict[str, str]


def prepare_paired_metric_data(
    normal: pd.DataFrame,
    abnormal: pd.DataFrame,
    *,
    equal_length: bool = False,
    drop_segment_constants: bool = True,
) -> PairedMetricData:
    """Create common numeric metric frames without using ground truth labels.

    ε-Diagnosis and RCD preprocess normal and abnormal segments separately in
    the official RCAEval adapters.  Consequently, a metric constant in either
    segment is excluded by default.  No anomaly-score based feature selection
    is performed here.
    """
    if normal.empty:
        raise ValueError("normal segment is empty")
    if abnormal.empty:
        raise ValueError("abnormal segment is empty")

    usable: list[str] = []
    excluded: dict[str, str] = {}
    for column in normal.columns:
        name = str(column)
        if name.lower() in TIME_COLUMNS:
            excluded[name] = "time_column"
            continue
        if column not in abnormal.columns:
            excluded[name] = "missing_from_abnormal"
            continue
        if not (
            pd.api.types.is_numeric_dtype(normal[column])
            and pd.api.types.is_numeric_dtype(abnormal[column])
        ):
            excluded[name] = "non_numeric"
            continue

        n = pd.to_numeric(normal[column], errors="coerce")
        a = pd.to_numeric(abnormal[column], errors="coerce")
        if n.notna().sum() == 0 or a.notna().sum() == 0:
            excluded[name] = "empty_segment"
            continue
        if drop_segment_constants and n.nunique(dropna=True) <= 1:
            excluded[name] = "constant_normal"
            continue
        if drop_segment_constants and a.nunique(dropna=True) <= 1:
            excluded[name] = "constant_abnormal"
            continue
        usable.append(name)

    if not usable:
        raise ValueError("no usable common numeric metrics remain")

    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.loc[:, usable].apply(pd.to_numeric, errors="coerce")
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.interpolate(limit_direction="both").ffill().bfill()
        return result.astype(float).reset_index(drop=True)

    clean_normal = clean(normal)
    clean_abnormal = clean(abnormal)
    finite = [
        column
        for column in usable
        if np.isfinite(clean_normal[column].to_numpy()).all()
        and np.isfinite(clean_abnormal[column].to_numpy()).all()
    ]
    for column in usable:
        if column not in finite:
            excluded[column] = "non_finite"
    clean_normal = clean_normal.loc[:, finite]
    clean_abnormal = clean_abnormal.loc[:, finite]
    if not finite:
        raise ValueError("no finite metrics remain")

    if equal_length:
        length = min(len(clean_normal), len(clean_abnormal))
        clean_normal = clean_normal.tail(length).reset_index(drop=True)
        clean_abnormal = clean_abnormal.head(length).reset_index(drop=True)

    return PairedMetricData(clean_normal, clean_abnormal, excluded)

from __future__ import annotations

import numpy as np
import pandas as pd


def canonical_metric_name(metric: str) -> str:
    """Normalize names into service_metric_type without percentile suffixes."""
    latency_suffixes = (
        "_latency-50",
        "_latency-90",
        "_latency-95",
        "_latency-99",
        "_latency",
    )
    for suffix in latency_suffixes:
        if metric.endswith(suffix):
            return metric[: -len(suffix)] + "_latency"

    if metric.endswith("_memory"):
        return metric[: -len("_memory")] + "_mem"

    return metric


def aggregate_canonical_metrics(
    metric_result: pd.DataFrame,
    method: str = "max",
) -> list[str]:
    """Aggregate raw metric scores into canonical metric ranking."""
    if "metric" not in metric_result or "score" not in metric_result:
        raise ValueError("metric_result must contain metric and score columns")

    work = metric_result[["metric", "score"]].copy()
    work["canonical_metric"] = work["metric"].map(canonical_metric_name)

    rows: list[tuple[str, float]] = []
    for name, group in work.groupby("canonical_metric", sort=False):
        values = group["score"].dropna().to_numpy(dtype=float)
        if values.size == 0:
            score = float("-inf")
        elif method == "max":
            score = float(np.max(values))
        elif method == "mean":
            score = float(np.mean(values))
        elif method == "logsumexp":
            m = float(np.max(values))
            score = m + float(np.log(np.sum(np.exp(values - m))))
        else:
            raise ValueError(f"Unknown metric aggregation method: {method}")
        rows.append((name, score))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in rows]

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


_TIME_COLUMNS = {"time", "time.1", "timestamp"}


def service_from_metric(metric: str) -> str:
    """Match BARO's published conversion from metric to service name."""
    return str(metric).split("_", 1)[0]


def to_service_ranking(metric_ranking: list[str]) -> list[str]:
    """Deduplicate services in metric-rank order, as BARO does."""
    ranking: list[str] = []
    seen: set[str] = set()
    for metric in metric_ranking:
        service = service_from_metric(metric)
        if service not in seen:
            ranking.append(service)
            seen.add(service)
    return ranking


@dataclass(frozen=True)
class BAROMetricScore:
    metric: str
    score: float
    normal_median: float
    normal_iqr: float


class BARORobustScorer:
    """Known-onset BARO RobustScorer baseline.

    The formal baseline uses ``max_signed``, which reproduces the DataFrame
    implementation used by the official BARO and RCAEval repositories:
    fit a ``RobustScaler`` on the normal segment and take the maximum scaled
    value in the abnormal segment.  ``max_absolute`` is available only to make
    the paper's absolute-deviation equation testable as a separate protocol.
    """

    _VALID_SCORE_MODES = {"max_signed", "max_absolute"}

    def __init__(self, *, score_mode: str = "max_signed") -> None:
        if score_mode not in self._VALID_SCORE_MODES:
            expected = ", ".join(sorted(self._VALID_SCORE_MODES))
            raise ValueError(
                f"score_mode must be one of: {expected}; got {score_mode}"
            )
        self.score_mode = score_mode
        self.metric_scores_: pd.DataFrame | None = None
        self.diagnostics_: dict | None = None

    @staticmethod
    def _usable_columns(
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
    ) -> tuple[list[str], dict[str, str]]:
        """Return official-compatible common, numeric, non-constant metrics."""
        usable: list[str] = []
        excluded: dict[str, str] = {}

        for column in normal.columns:
            name = str(column)
            if name.lower() in _TIME_COLUMNS:
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

            normal_values = normal[column].dropna()
            abnormal_values = abnormal[column].dropna()
            if normal_values.empty or abnormal_values.empty:
                excluded[name] = "empty_segment"
                continue

            # Official BARO preprocesses the two segments separately with
            # drop_constant(), then keeps their column intersection.
            if normal_values.nunique(dropna=True) <= 1:
                excluded[name] = "constant_normal"
                continue
            if abnormal_values.nunique(dropna=True) <= 1:
                excluded[name] = "constant_abnormal"
                continue

            usable.append(name)

        return usable, excluded

    def score_metrics(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
    ) -> pd.DataFrame:
        if normal.empty:
            raise ValueError("normal segment is empty")
        if abnormal.empty:
            raise ValueError("abnormal segment is empty")

        columns, excluded = self._usable_columns(normal, abnormal)
        if not columns:
            raise ValueError("no usable metrics remain for BARO RobustScorer")

        scores: list[BAROMetricScore] = []
        for column in columns:
            normal_values = normal[column].dropna().to_numpy(dtype=float)
            abnormal_values = abnormal[column].dropna().to_numpy(dtype=float)
            if not (
                np.isfinite(normal_values).all()
                and np.isfinite(abnormal_values).all()
            ):
                excluded[column] = "non_finite"
                continue

            scaler = RobustScaler().fit(normal_values.reshape(-1, 1))
            scaled = scaler.transform(abnormal_values.reshape(-1, 1))[:, 0]
            if self.score_mode == "max_signed":
                score = float(np.max(scaled))
            else:
                score = float(np.max(np.abs(scaled)))

            scores.append(
                BAROMetricScore(
                    metric=column,
                    score=score,
                    normal_median=float(scaler.center_[0]),
                    normal_iqr=float(scaler.scale_[0]),
                )
            )

        if not scores:
            raise ValueError("no finite metrics remain for BARO RobustScorer")

        # Python's sort is stable. Equal scores therefore preserve the input
        # column order, matching the official implementation.
        scores.sort(key=lambda item: item.score, reverse=True)
        result = pd.DataFrame(
            [
                {
                    "metric": item.metric,
                    "score": item.score,
                    "normal_median": item.normal_median,
                    "normal_iqr": item.normal_iqr,
                }
                for item in scores
            ]
        )

        self.metric_scores_ = result
        self.diagnostics_ = {
            "protocol": "known_onset_robust_scorer",
            "score_mode": self.score_mode,
            "normal_samples": int(len(normal)),
            "abnormal_samples": int(len(abnormal)),
            "scored_metrics": int(len(result)),
            "excluded_metrics": excluded,
            "metric_scores": result.to_dict(orient="records"),
        }
        return result

    def predict(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
        *,
        granularity: str = "service",
    ) -> list[str] | pd.DataFrame:
        metric_scores = self.score_metrics(normal, abnormal)
        if granularity == "metric":
            return metric_scores
        if granularity != "service":
            raise ValueError(
                "granularity must be 'service' or 'metric'; "
                f"got {granularity}"
            )
        return to_service_ranking(metric_scores["metric"].tolist())

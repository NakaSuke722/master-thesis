from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    prepare_paired_metric_data,
    ranking_frame,
    to_service_ranking,
)


class EpsilonDiagnosisScorer:
    """RCAEval-compatible implementation of PyRCA ε-Diagnosis.

    The statistic and bootstrap procedure follow Salesforce PyRCA.  The
    adapter adds an explicit seed and requests five candidates so AC@5 is a
    meaningful evaluation measure.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.01,
        bootstrap_time: int = 200,
        root_cause_top_k: int = 5,
        seed: int = 42,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if bootstrap_time <= 0:
            raise ValueError("bootstrap_time must be positive")
        if root_cause_top_k <= 0:
            raise ValueError("root_cause_top_k must be positive")
        self.alpha = float(alpha)
        self.bootstrap_time = int(bootstrap_time)
        self.root_cause_top_k = int(root_cause_top_k)
        self.seed = int(seed)
        self.metric_scores_: pd.DataFrame | None = None
        self.diagnostics_: dict | None = None

    def score_metrics(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
    ) -> pd.DataFrame:
        paired = prepare_paired_metric_data(
            normal,
            abnormal,
            equal_length=True,
            drop_segment_constants=True,
        )
        n = paired.normal.to_numpy(dtype=float)
        a = paired.abnormal.to_numpy(dtype=float)
        columns = paired.normal.columns.tolist()
        rng = np.random.default_rng(self.seed)

        # This intentionally mirrors PyRCA's column-wise bootstrap layout.
        sampled = np.empty((n.shape[0], n.shape[1], self.bootstrap_time))
        for column in range(n.shape[1]):
            sampled[:, column, :] = rng.choice(
                n[:, column],
                size=(n.shape[0], self.bootstrap_time),
                replace=True,
            )

        upper = np.triu_indices(n.shape[1], k=1)
        bootstrap_thresholds = []
        for bootstrap in range(self.bootstrap_time):
            correlations = np.atleast_2d(
                np.corrcoef(sampled[:, :, bootstrap], rowvar=False)
            )
            squared_pairs = np.nan_to_num(
                correlations[upper] ** 2,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            bootstrap_thresholds.append(
                float(np.quantile(squared_pairs, 1 - self.alpha))
                if squared_pairs.size
                else 1.0
            )

        # This axis convention intentionally preserves PyRCA's implementation:
        # it produces one threshold per bootstrap replicate and zips those
        # values to metric columns. In normal RCAEval cases p <= 200.
        pair_thresholds = np.asarray(bootstrap_thresholds, dtype=float)
        thresholds = {
            column: float(pair_thresholds[index])
            if index < len(pair_thresholds)
            else float(np.median(pair_thresholds))
            for index, column in enumerate(columns)
        }

        selected: list[tuple[str, float]] = []
        correlations: dict[str, float] = {}
        for index, column in enumerate(columns):
            n_values = n[:, index]
            a_values = a[:, index]
            denominator = float(np.var(n_values) * np.var(a_values))
            score = 0.0
            if denominator > 0:
                score = float(
                    np.cov(n_values, a_values)[0, 1] ** 2 / denominator
                )
            correlations[column] = score
            if score > thresholds[column]:
                selected.append((column, score))

        selected.sort(key=lambda item: item[1], reverse=True)
        selected = selected[: self.root_cause_top_k]
        ranking = [metric for metric, _ in selected]
        result = ranking_frame(ranking, dict(selected))
        self.metric_scores_ = result
        self.diagnostics_ = {
            "protocol": "pyrca_epsilon_diagnosis_known_onset",
            "alpha": self.alpha,
            "bootstrap_time": self.bootstrap_time,
            "root_cause_top_k": self.root_cause_top_k,
            "seed": self.seed,
            "normal_samples": int(len(paired.normal)),
            "abnormal_samples": int(len(paired.abnormal)),
            "scored_metrics": int(len(columns)),
            "selected_metrics": ranking,
            "correlations": correlations,
            "thresholds": thresholds,
            "excluded_metrics": paired.excluded,
        }
        return result

    def predict(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
        *,
        granularity: str = "service",
    ) -> list[str] | pd.DataFrame:
        scores = self.score_metrics(normal, abnormal)
        if granularity == "metric":
            return scores
        if granularity != "service":
            raise ValueError("granularity must be 'service' or 'metric'")
        return to_service_ranking(scores["metric"].tolist())

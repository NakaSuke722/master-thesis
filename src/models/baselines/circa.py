from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from .common import (
    prepare_paired_metric_data,
    ranking_frame,
    to_service_ranking,
)


def _pc_function():
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "CIRCA requires causal-learn. Install requirements-baselines.txt."
        ) from exc
    return pc


def _z_scores(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    center = float(np.mean(train))
    scale = float(np.std(train))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = 1.0
    return (test - center) / scale


class CIRCAScorer:
    """PC graph construction followed by CIRCA's RHT node scoring.

    RCAEval has no supplied call graph, so its published adapter learns a PC
    graph from the complete case and then applies CIRCA's regression-based
    hypothesis test (RHT).  This class makes that adaptation explicit.
    """

    def __init__(
        self,
        *,
        pc_alpha: float = 0.05,
        stable: bool = True,
        lookup_window: int = 120,
        detect_window: int = 10,
        score_time_offset: int = 300,
    ) -> None:
        if not 0 < pc_alpha < 1:
            raise ValueError("pc_alpha must be in (0, 1)")
        if lookup_window < detect_window or detect_window <= 0:
            raise ValueError("lookup_window must be >= detect_window > 0")
        if score_time_offset < 0:
            raise ValueError("score_time_offset must be non-negative")
        self.pc_alpha = float(pc_alpha)
        self.stable = bool(stable)
        self.lookup_window = int(lookup_window)
        self.detect_window = int(detect_window)
        self.score_time_offset = int(score_time_offset)
        self.metric_scores_: pd.DataFrame | None = None
        self.diagnostics_: dict | None = None

    def _learn_graph(self, data: pd.DataFrame) -> np.ndarray:
        pc = _pc_function()
        result = pc(
            data.to_numpy(dtype=float),
            alpha=self.pc_alpha,
            stable=self.stable,
            uc_rule=0,
            uc_priority=-1,
            background_knowledge=None,
            show_progress=False,
            node_names=data.columns.tolist(),
        )
        return np.asarray(result.G.graph)

    @staticmethod
    def _to_digraph(
        adjacency: np.ndarray,
        columns: list[str],
    ) -> nx.DiGraph:
        """Apply the same causal-learn endpoint conversion as RCAEval RHT."""
        graph = nx.DiGraph()
        graph.add_nodes_from(columns)
        for left in range(len(columns)):
            for right in range(left + 1, len(columns)):
                lr = adjacency[left, right]
                rl = adjacency[right, left]
                if lr == rl == 0:
                    continue
                if lr == rl == -1:
                    graph.add_edge(columns[left], columns[right])
                    graph.add_edge(columns[right], columns[left])
                elif lr == 1 and rl == -1:
                    graph.add_edge(columns[left], columns[right])
                elif lr == -1 and rl == 1:
                    graph.add_edge(columns[right], columns[left])
                else:
                    # Circle endpoints can appear for partially oriented PC
                    # output. Treat them as undirected instead of inventing a
                    # direction, consistent with CIRCA's tolerant graph use.
                    graph.add_edge(columns[left], columns[right])
                    graph.add_edge(columns[right], columns[left])
        return graph

    def _score_windows(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        train_window = self.lookup_window - self.detect_window + 1
        normal_context = normal.tail(self.lookup_window + 1)
        train = normal_context.head(train_window)
        if len(train) < 2:
            train = normal.tail(max(2, min(len(normal), train_window)))

        end = min(self.score_time_offset, len(abnormal) - 1)
        start = max(0, end - self.detect_window + 1)
        test = abnormal.iloc[start : end + 1]
        if test.empty:
            test = abnormal.head(self.detect_window)
            end = len(test) - 1
        return train.reset_index(drop=True), test.reset_index(drop=True), end

    def score_metrics(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
    ) -> pd.DataFrame:
        paired = prepare_paired_metric_data(
            normal,
            abnormal,
            drop_segment_constants=False,
        )
        combined = pd.concat(
            [paired.normal, paired.abnormal],
            ignore_index=True,
        )
        adjacency = self._learn_graph(combined)
        columns = combined.columns.tolist()
        if adjacency.shape != (len(columns), len(columns)):
            raise ValueError("PC adjacency shape does not match metric columns")
        graph = self._to_digraph(adjacency, columns)
        train, test, actual_offset = self._score_windows(
            paired.normal,
            paired.abnormal,
        )

        metric_scores: dict[str, float] = {}
        confidences: dict[str, float] = {}
        for metric in columns:
            parents = [parent for parent in graph.predecessors(metric)]
            train_y = train[metric].to_numpy(dtype=float)
            test_y = test[metric].to_numpy(dtype=float)
            if parents:
                train_x = train.loc[:, parents].to_numpy(dtype=float)
                test_x = test.loc[:, parents].to_numpy(dtype=float)
                regressor = LinearRegression().fit(train_x, train_y)
                train_error = train_y - regressor.predict(train_x)
                test_error = test_y - regressor.predict(test_x)
                z = _z_scores(train_error, test_error)
            else:
                z = _z_scores(train_y, test_y)
            score = float(np.max(np.abs(z)))
            metric_scores[metric] = score
            confidences[metric] = float(1 - 2 * norm.cdf(-score))

        ranking = sorted(
            columns,
            key=lambda metric: metric_scores[metric],
            reverse=True,
        )
        result = ranking_frame(ranking, metric_scores)
        self.metric_scores_ = result
        self.diagnostics_ = {
            "protocol": "rcaeval_pc_graph_circa_rht",
            "pc_alpha": self.pc_alpha,
            "pc_stable": self.stable,
            "lookup_window": self.lookup_window,
            "detect_window": self.detect_window,
            "requested_score_time_offset": self.score_time_offset,
            "actual_score_time_offset": actual_offset,
            "graph_learning_scope": "normal_and_abnormal",
            "normal_samples": int(len(paired.normal)),
            "abnormal_samples": int(len(paired.abnormal)),
            "input_metrics": len(columns),
            "graph_edges": [[source, target] for source, target in graph.edges],
            "metric_confidences": confidences,
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

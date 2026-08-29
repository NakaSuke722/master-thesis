from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from .common import (
    prepare_paired_metric_data,
    ranking_frame,
    service_from_metric,
    to_service_ranking,
)


def _pc_orientation_components():
    try:
        from causallearn.graph.GraphClass import CausalGraph
        from causallearn.utils.PCUtils import Meek, UCSepset
        from causallearn.utils.PCUtils.Helper import append_value
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "CIRCA requires causal-learn. Install requirements-baselines.txt."
        ) from exc
    return CausalGraph, Meek, UCSepset, append_value


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
    hypothesis test (RHT). For high-dimensional RCAEval cases, the graph step
    uses an explicitly bounded and service-stratified PC approximation.
    """

    def __init__(
        self,
        *,
        pc_alpha: float = 0.05,
        stable: bool = True,
        lookup_window: int = 120,
        detect_window: int = 10,
        score_time_offset: int = 300,
        pc_redundancy_threshold: float = 1 - 1e-12,
        pc_max_conditioning_set: int = 1,
        pc_max_metrics: int = 60,
    ) -> None:
        if not 0 < pc_alpha < 1:
            raise ValueError("pc_alpha must be in (0, 1)")
        if not stable:
            raise ValueError("CIRCA bounded PC requires stable=True")
        if lookup_window < detect_window or detect_window <= 0:
            raise ValueError("lookup_window must be >= detect_window > 0")
        if score_time_offset < 0:
            raise ValueError("score_time_offset must be non-negative")
        if not 0 < pc_redundancy_threshold <= 1:
            raise ValueError("pc_redundancy_threshold must be in (0, 1]")
        if pc_max_conditioning_set not in {0, 1}:
            raise ValueError("pc_max_conditioning_set must be 0 or 1")
        if pc_max_metrics <= 1:
            raise ValueError("pc_max_metrics must be greater than 1")
        self.pc_alpha = float(pc_alpha)
        self.stable = bool(stable)
        self.lookup_window = int(lookup_window)
        self.detect_window = int(detect_window)
        self.score_time_offset = int(score_time_offset)
        self.pc_redundancy_threshold = float(pc_redundancy_threshold)
        self.pc_max_conditioning_set = int(pc_max_conditioning_set)
        self.pc_max_metrics = int(pc_max_metrics)
        self.metric_scores_: pd.DataFrame | None = None
        self.diagnostics_: dict | None = None

    def _learn_graph(
        self,
        data: pd.DataFrame,
        *,
        bounded: bool = False,
    ) -> np.ndarray:
        """Vectorized bounded-order Fisher-Z PC with standard orientation.

        Full PC is exponential in the conditioning depth and causal-learn's
        Fisher-Z implementation repeatedly inverts tiny correlation matrices.
        RCAEval SS has hundreds of metrics, making that adapter impractical.
        Orders 0 and 1 have closed-form partial correlations, so compute them
        directly and then use causal-learn's standard UC/Meek orientation.
        """
        if not bounded:
            result = _pc_function()(
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

        CausalGraph, Meek, UCSepset, append_value = (
            _pc_orientation_components()
        )
        values = data.to_numpy(dtype=float)
        sample_size, dimension = values.shape
        correlation = np.corrcoef(values, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0)
        correlation = np.clip(correlation, -1 + 1e-12, 1 - 1e-12)

        def fisher_p_value(value: float, order: int) -> float:
            value = float(np.clip(value, -1 + 1e-12, 1 - 1e-12))
            degrees = max(sample_size - order - 3, 1)
            statistic = abs(np.arctanh(value)) * np.sqrt(degrees)
            return float(2 * norm.sf(statistic))

        adjacency = np.ones((dimension, dimension), dtype=bool)
        np.fill_diagonal(adjacency, False)
        separating_sets: dict[tuple[int, int], tuple[int, ...]] = {}

        for left in range(dimension):
            for right in range(left + 1, dimension):
                if fisher_p_value(correlation[left, right], 0) > self.pc_alpha:
                    adjacency[left, right] = adjacency[right, left] = False
                    separating_sets[left, right] = ()
                    separating_sets[right, left] = ()

        if self.pc_max_conditioning_set >= 1:
            depth_zero = adjacency.copy()
            removals: dict[tuple[int, int], tuple[int, ...]] = {}
            for left in range(dimension):
                for right in range(left + 1, dimension):
                    if not depth_zero[left, right]:
                        continue
                    candidates = set(np.flatnonzero(depth_zero[left]))
                    candidates.update(np.flatnonzero(depth_zero[right]))
                    candidates.discard(left)
                    candidates.discard(right)
                    for condition in sorted(candidates):
                        numerator = (
                            correlation[left, right]
                            - correlation[left, condition]
                            * correlation[right, condition]
                        )
                        denominator = np.sqrt(
                            max(1 - correlation[left, condition] ** 2, 1e-12)
                            * max(1 - correlation[right, condition] ** 2, 1e-12)
                        )
                        partial = numerator / denominator
                        if fisher_p_value(partial, 1) > self.pc_alpha:
                            removals[left, right] = (condition,)
                            removals[right, left] = (condition,)
                            break
            for (left, right), separating in removals.items():
                adjacency[left, right] = False
                separating_sets[left, right] = separating

        graph = CausalGraph(dimension, data.columns.tolist())
        for left in range(dimension):
            for right in range(left + 1, dimension):
                if not adjacency[left, right]:
                    edge = graph.G.get_edge(
                        graph.G.nodes[left], graph.G.nodes[right]
                    )
                    if edge is not None:
                        graph.G.remove_edge(edge)
                separating = separating_sets.get((left, right))
                if separating is not None:
                    append_value(graph.sepset, left, right, separating)
                    append_value(graph.sepset, right, left, separating)

        # Priority 2 orients from the separating sets without launching extra
        # Fisher-Z tests (which would reintroduce the singular-inverse issue).
        oriented = UCSepset.uc_sepset(graph, priority=2)
        oriented = Meek.meek(oriented)
        return np.asarray(oriented.G.graph)

    def _select_pc_columns(
        self,
        data: pd.DataFrame,
    ) -> tuple[list[str], dict[str, str], list[str]]:
        """Remove deterministic pairwise redundancy before Fisher-Z PC.

        Fisher-Z in causal-learn inverts correlation submatrices and therefore
        cannot accept perfectly collinear metrics. Selection is independent of
        ground truth and anomaly scores: preserve the first metric and remove
        only later metrics whose absolute correlation with a retained metric
        reaches the configured numerical-equivalence threshold.
        """
        correlation = data.corr().abs()
        selected: list[str] = []
        redundant: dict[str, str] = {}
        for metric in data.columns:
            representative = None
            representative_correlation = float("-inf")
            for retained in selected:
                value = float(correlation.loc[metric, retained])
                if np.isfinite(value) and value > representative_correlation:
                    representative = retained
                    representative_correlation = value
            if (
                representative is not None
                and representative_correlation >= self.pc_redundancy_threshold
            ):
                redundant[metric] = representative
            else:
                selected.append(metric)
        if len(selected) <= self.pc_max_metrics:
            return selected, redundant, []

        # Preserve service coverage instead of selecting metrics by anomaly
        # magnitude. Round-robin over service groups in original column order.
        groups: dict[str, list[str]] = {}
        for metric in selected:
            groups.setdefault(service_from_metric(metric), []).append(metric)
        screened_selection: list[str] = []
        depth = 0
        while len(screened_selection) < self.pc_max_metrics:
            added = False
            for metrics in groups.values():
                if depth < len(metrics):
                    screened_selection.append(metrics[depth])
                    added = True
                    if len(screened_selection) == self.pc_max_metrics:
                        break
            if not added:
                break
            depth += 1
        screened = [
            metric for metric in selected if metric not in screened_selection
        ]
        return screened_selection, redundant, screened

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
        columns = combined.columns.tolist()
        pc_columns, redundant_metrics, screened_metrics = (
            self._select_pc_columns(combined)
        )
        adjacency = self._learn_graph(
            combined.loc[:, pc_columns],
            bounded=bool(screened_metrics),
        )
        if adjacency.shape != (len(pc_columns), len(pc_columns)):
            raise ValueError("PC adjacency shape does not match metric columns")
        graph = self._to_digraph(adjacency, pc_columns)
        # Redundant metrics remain valid RCA candidates. With no PC node they
        # follow CIRCA RHT's standard parentless (univariate z-score) path.
        graph.add_nodes_from(columns)
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
            "protocol": "rcaeval_stratified_adaptive_pc_graph_circa_rht",
            "pc_alpha": self.pc_alpha,
            "pc_stable": self.stable,
            "pc_redundancy_threshold": self.pc_redundancy_threshold,
            "pc_max_conditioning_set": self.pc_max_conditioning_set,
            "pc_max_metrics": self.pc_max_metrics,
            "lookup_window": self.lookup_window,
            "detect_window": self.detect_window,
            "requested_score_time_offset": self.score_time_offset,
            "actual_score_time_offset": actual_offset,
            "graph_learning_scope": "normal_and_abnormal",
            "normal_samples": int(len(paired.normal)),
            "abnormal_samples": int(len(paired.abnormal)),
            "input_metrics": len(columns),
            "pc_input_metrics": len(pc_columns),
            "pc_redundant_metrics": redundant_metrics,
            "pc_screened_metrics": screened_metrics,
            "pc_search": (
                "bounded_order_1"
                if screened_metrics
                else "full_pc"
            ),
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

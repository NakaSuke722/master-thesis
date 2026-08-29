from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from .common import (
    prepare_paired_metric_data,
    ranking_frame,
    to_service_ranking,
)


def _causal_learn_components():
    try:
        from causallearn.utils.cit import CIT
        from causallearn.utils.PCUtils import SkeletonDiscovery
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "RCD requires causal-learn. Install requirements-baselines.txt."
        ) from exc
    return CIT, SkeletonDiscovery


class RCDScorer:
    """Localized multi-phase Root Cause Discovery (RCD).

    This is a small adapter of the official RCD/PyRCA implementation.  It
    keeps the F-node, k-means discretization, randomized gamma-sized chunks,
    localized PC search, and increasing-alpha ordering used by RCAEval.
    """

    def __init__(
        self,
        *,
        gamma: int = 5,
        bins: int = 5,
        localized: bool = True,
        start_alpha: float = 0.001,
        local_alpha: float = 0.01,
        alpha_step: float = 0.1,
        alpha_limit: float = 1.0,
        root_cause_top_k: int = 5,
        seed: int = 42,
    ) -> None:
        if gamma < 2:
            raise ValueError("gamma must be at least 2")
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if root_cause_top_k <= 0:
            raise ValueError("root_cause_top_k must be positive")
        self.gamma = int(gamma)
        self.bins = int(bins)
        self.localized = bool(localized)
        self.start_alpha = float(start_alpha)
        self.local_alpha = float(local_alpha)
        self.alpha_step = float(alpha_step)
        self.alpha_limit = float(alpha_limit)
        self.root_cause_top_k = int(root_cause_top_k)
        self.seed = int(seed)
        self.metric_scores_: pd.DataFrame | None = None
        self.diagnostics_: dict | None = None
        self._ci_tests = 0
        self._rng = np.random.default_rng(self.seed)

    def _discretize(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
    ) -> np.ndarray:
        combined = pd.concat([normal, abnormal], ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed = KBinsDiscretizer(
                n_bins=self.bins,
                encode="ordinal",
                strategy="kmeans",
                random_state=self.seed,
            ).fit_transform(combined)
        f_node = np.concatenate(
            [
                np.zeros(len(normal), dtype=int),
                np.ones(len(abnormal), dtype=int),
            ]
        )
        return np.column_stack([transformed.astype(int), f_node])

    @staticmethod
    def _f_neighbors(graph: np.ndarray, f_index: int) -> list[int]:
        return [
            index
            for index in range(f_index)
            if graph[f_index, index] != 0 or graph[index, f_index] != 0
        ]

    def _run_pc(self, data: np.ndarray, alpha: float):
        CIT, SkeletonDiscovery = _causal_learn_components()
        independence_test = CIT(data, "chisq")
        cg = SkeletonDiscovery.skeleton_discovery(
            data,
            alpha,
            indep_test=independence_test,
            background_knowledge=None,
            stable=False,
            verbose=False,
            show_progress=False,
        )
        self._ci_tests += int(getattr(cg, "no_ci_tests", 0))
        return cg

    def _localized_neighbors(
        self,
        data: np.ndarray,
        alpha: float,
    ) -> tuple[list[int], dict[int, float]]:
        """Localized skeleton search from the official RCD third-party code."""
        CIT, _ = _causal_learn_components()
        independence_test = CIT(data, "chisq")
        f_index = data.shape[1] - 1
        neighbors = list(range(f_index))
        p_values = {index: 1.0 for index in neighbors}
        depth = 0
        while len(neighbors) - 1 >= depth:
            changed = False
            for candidate in self._rng.permutation(neighbors).tolist():
                conditioning_pool = [
                    index for index in neighbors if index != candidate
                ]
                if len(conditioning_pool) < depth:
                    continue
                for conditioning in combinations(conditioning_pool, depth):
                    p_value = float(
                        independence_test(f_index, candidate, conditioning)
                    )
                    self._ci_tests += 1
                    p_values[candidate] = p_value
                    if p_value > alpha:
                        neighbors.remove(candidate)
                        changed = True
                        break
            depth += 1
            if not neighbors:
                break
            # The degree condition above normally terminates the search. This
            # guard also prevents pathological repeated tests after no edge
            # can be removed at the maximum possible conditioning depth.
            if not changed and depth > len(neighbors) - 1:
                break
        return neighbors, p_values

    def _psi_pc(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
        *,
        start_alpha: float,
        min_nodes: int,
    ) -> list[str]:
        if normal.shape[1] == 0:
            return []
        data = self._discretize(normal, abnormal)
        f_index = data.shape[1] - 1
        selected: list[int] = []

        for alpha in np.arange(
            start_alpha,
            self.alpha_limit,
            self.alpha_step,
        ):
            if self.localized:
                neighbors, local_p_values = self._localized_neighbors(
                    data, float(alpha)
                )
                p_values = None
            else:
                cg = self._run_pc(data, float(alpha))
                neighbors = self._f_neighbors(cg.G.graph, f_index)
                p_values = getattr(cg, "p_values", None)
            new_neighbors = [i for i in neighbors if i not in selected]
            if new_neighbors:
                if self.localized:
                    ordered = sorted(
                        new_neighbors,
                        key=lambda i: local_p_values[i],
                    )
                elif p_values is None:
                    ordered = new_neighbors
                else:
                    ordered = sorted(
                        new_neighbors,
                        key=lambda i: float(p_values[f_index, i]),
                    )
                selected.extend(ordered)
            if len(selected) >= min_nodes:
                break
        return [normal.columns[index] for index in selected]

    def _run_multi_phase(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
    ) -> tuple[list[str], list[dict]]:
        rng = np.random.default_rng(self.seed)
        candidates = normal.columns.tolist()
        levels: list[dict] = []
        previous = len(candidates)

        while candidates:
            shuffled = rng.permutation(candidates).tolist()
            chunks = [
                shuffled[index : index + self.gamma]
                for index in range(0, len(shuffled), self.gamma)
            ]
            next_candidates: list[str] = []
            for chunk in chunks:
                next_candidates.extend(
                    self._psi_pc(
                        normal.loc[:, chunk],
                        abnormal.loc[:, chunk],
                        start_alpha=self.local_alpha,
                        min_nodes=1,
                    )
                )
            # Preserve first occurrence if numerical edge cases duplicate one.
            next_candidates = list(dict.fromkeys(next_candidates))
            levels.append(
                {
                    "input_metrics": len(candidates),
                    "output_metrics": len(next_candidates),
                }
            )
            candidates = next_candidates
            if len(candidates) <= self.gamma or len(candidates) == previous:
                break
            previous = len(candidates)

        if not candidates:
            return [], levels
        ranking = self._psi_pc(
            normal.loc[:, candidates],
            abnormal.loc[:, candidates],
            start_alpha=self.start_alpha,
            min_nodes=len(candidates),
        )
        return ranking[: self.root_cause_top_k], levels

    def score_metrics(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
    ) -> pd.DataFrame:
        paired = prepare_paired_metric_data(
            normal,
            abnormal,
            drop_segment_constants=True,
        )
        self._ci_tests = 0
        self._rng = np.random.default_rng(self.seed)
        ranking, levels = self._run_multi_phase(
            paired.normal.copy(),
            paired.abnormal.copy(),
        )
        result = ranking_frame(ranking)
        self.metric_scores_ = result
        self.diagnostics_ = {
            "protocol": "rcd_localized_multi_phase_known_onset",
            "gamma": self.gamma,
            "bins": self.bins,
            "localized": self.localized,
            "start_alpha": self.start_alpha,
            "local_alpha": self.local_alpha,
            "alpha_step": self.alpha_step,
            "alpha_limit": self.alpha_limit,
            "root_cause_top_k": self.root_cause_top_k,
            "seed": self.seed,
            "normal_samples": int(len(paired.normal)),
            "abnormal_samples": int(len(paired.abnormal)),
            "input_metrics": int(paired.normal.shape[1]),
            "selected_metrics": ranking,
            "phase_one_levels": levels,
            "conditional_independence_tests": self._ci_tests,
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

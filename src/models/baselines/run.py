from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pandas as pd

from .common import (
    prepare_paired_metric_data,
    ranking_frame,
    to_service_ranking,
)


def _torch_modules():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "RUN requires PyTorch. Install requirements-baselines.txt."
        ) from exc
    return torch, nn, functional


def _make_dlinear(
    seq_len: int,
    hidden_size: int,
    channels: int,
    kernel_size: int,
):
    torch, nn, functional = _torch_modules()

    class DLinear(nn.Module):
        """Official RUN DLinear with device-safe, dynamic batch shapes."""

        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.Parameter(torch.ones(channels, 1))
            self.seasonal_encoder = nn.ModuleList(
                nn.Linear(seq_len, hidden_size) for _ in range(channels)
            )
            self.trend_encoder = nn.ModuleList(
                nn.Linear(seq_len, hidden_size) for _ in range(channels)
            )
            self.seasonal_decoder = nn.ModuleList(
                nn.Linear(hidden_size, seq_len) for _ in range(channels)
            )
            self.trend_decoder = nn.ModuleList(
                nn.Linear(hidden_size, seq_len) for _ in range(channels)
            )
            self.seasonal_projector = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.PReLU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                )
                for _ in range(channels)
            )
            self.trend_projector = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.PReLU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                )
                for _ in range(channels)
            )
            self.seasonal_output = nn.Linear(seq_len * channels, 1)
            self.trend_output = nn.Linear(seq_len * channels, 1)

        @staticmethod
        def decompose(x):
            # x is batch x time x channel. This is the edge-padded moving
            # average decomposition used by DLinear and official RUN.
            padding = (kernel_size - 1) // 2
            channel_first = x.transpose(1, 2)
            padded = functional.pad(
                channel_first,
                (padding, padding),
                mode="replicate",
            )
            trend = functional.avg_pool1d(
                padded,
                kernel_size=kernel_size,
                stride=1,
            )
            seasonal = channel_first - trend
            return seasonal, trend

        def encode(self, x, *, project: bool):
            seasonal, trend = self.decompose(x)
            seasonal_encoded = []
            trend_encoded = []
            for index in range(channels):
                s = self.seasonal_encoder[index](seasonal[:, index, :])
                t = self.trend_encoder[index](trend[:, index, :])
                if project:
                    s = self.seasonal_projector[index](s)
                    t = self.trend_projector[index](t)
                seasonal_encoded.append(s)
                trend_encoded.append(t)
            return (
                torch.stack(seasonal_encoded, dim=1)
                + torch.stack(trend_encoded, dim=1)
            ).transpose(1, 2)

        def forward(self, x):
            seasonal, trend = self.decompose(x)
            weights = functional.softmax(self.attention, dim=0).T
            seasonal_decoded = []
            trend_decoded = []
            for index in range(channels):
                seasonal_hidden = self.seasonal_encoder[index](
                    seasonal[:, index, :]
                ) * weights[:, index]
                trend_hidden = self.trend_encoder[index](
                    trend[:, index, :]
                ) * weights[:, index]
                seasonal_decoded.append(
                    self.seasonal_decoder[index](seasonal_hidden)
                )
                trend_decoded.append(
                    self.trend_decoder[index](trend_hidden)
                )
            batch = x.shape[0]
            seasonal_flat = torch.stack(seasonal_decoded, dim=1).reshape(
                batch, seq_len * channels
            )
            trend_flat = torch.stack(trend_decoded, dim=1).reshape(
                batch, seq_len * channels
            )
            return self.seasonal_output(seasonal_flat) + self.trend_output(
                trend_flat
            )

    return DLinear()


def _temporal_contrastive_loss(left, right):
    torch, _, functional = _torch_modules()
    length = left.shape[1]
    if length <= 1:
        return left.new_tensor(0.0)
    joined = torch.cat([left, right], dim=1)
    similarities = joined @ joined.transpose(1, 2)
    return -functional.log_softmax(
        similarities[:, :length, length:], dim=-1
    ).mean()


def _hierarchical_contrastive_loss(left, right):
    _, _, functional = _torch_modules()
    loss = left.new_tensor(0.0)
    levels = 0
    while left.shape[1] > 1:
        loss = loss + _temporal_contrastive_loss(left, right)
        left = functional.max_pool1d(
            left.transpose(1, 2), 2
        ).transpose(1, 2)
        right = functional.max_pool1d(
            right.transpose(1, 2), 2
        ).transpose(1, 2)
        levels += 1
    return loss / max(levels, 1)


class RUNScorer:
    """Known-onset adapter of RUN's neural Granger/PageRank pipeline.

    The official DLinear feature-attention graph, correlation cycle pruning,
    sink-personalized PageRank, and one-model-per-target design are retained.
    Fixed CUDA and batch-size assumptions are removed, and model fitting is
    restricted to the normal segment to avoid incident leakage.
    """

    def __init__(
        self,
        *,
        seq_len: int = 32,
        hidden_size: int = 128,
        moving_average_kernel: int = 25,
        pretrain_epochs: int = 1,
        epochs: int = 1,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        device: str = "cpu",
        torch_num_threads: int = 1,
        seed: int = 42,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if moving_average_kernel <= 0 or moving_average_kernel % 2 == 0:
            raise ValueError("moving_average_kernel must be a positive odd integer")
        if moving_average_kernel > seq_len:
            raise ValueError("moving_average_kernel cannot exceed seq_len")
        if min(hidden_size, batch_size) <= 0:
            raise ValueError("hidden_size and batch_size must be positive")
        if torch_num_threads <= 0:
            raise ValueError("torch_num_threads must be positive")
        if min(pretrain_epochs, epochs) < 0:
            raise ValueError("epoch counts cannot be negative")
        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.moving_average_kernel = int(moving_average_kernel)
        self.pretrain_epochs = int(pretrain_epochs)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.device = device
        self.torch_num_threads = int(torch_num_threads)
        self.seed = int(seed)
        self.metric_scores_: pd.DataFrame | None = None
        self.diagnostics_: dict | None = None

    def _resolve_device(self):
        torch, _, _ = _torch_modules()
        torch.set_num_threads(self.torch_num_threads)
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(self.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("RUN device is CUDA but CUDA is unavailable")
        return requested

    def _windows(self, values: np.ndarray):
        torch, _, _ = _torch_modules()
        if len(values) <= self.seq_len:
            raise ValueError(
                f"RUN needs more than {self.seq_len} normal samples"
            )
        x = np.stack(
            [values[i : i + self.seq_len] for i in range(len(values) - self.seq_len)]
        )
        y = values[self.seq_len :]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def _fit_target(self, x, y, target: int, device) -> np.ndarray:
        torch, _, functional = _torch_modules()
        torch.manual_seed(self.seed + target)
        model = _make_dlinear(
            self.seq_len,
            self.hidden_size,
            x.shape[2],
            self.moving_average_kernel,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(self.seed + target)

        for _ in range(self.pretrain_epochs):
            for indices in torch.randperm(len(x), generator=generator).split(
                self.batch_size
            ):
                batch = x[indices].to(device)
                optimizer.zero_grad()
                projected = model.encode(batch, project=True)
                with torch.no_grad():
                    encoded = model.encode(batch, project=False)
                loss = _hierarchical_contrastive_loss(projected, encoded)
                loss.backward()
                optimizer.step()

        for _ in range(self.epochs):
            for indices in torch.randperm(len(x), generator=generator).split(
                self.batch_size
            ):
                batch_x = x[indices].to(device)
                batch_y = y[indices, target].to(device)
                optimizer.zero_grad()
                prediction = model(batch_x).reshape(-1)
                loss = functional.mse_loss(prediction, batch_y)
                loss.backward()
                optimizer.step()
        return model.attention.detach().cpu().numpy().reshape(-1)

    @staticmethod
    def _potential_parents(attention: np.ndarray) -> list[int]:
        order = np.argsort(-attention)
        sorted_scores = attention[order]
        if len(sorted_scores) <= 5:
            return [int(index) for index in order if attention[index] > 1]

        gaps: list[float] = []
        for index in range(len(sorted_scores) - 1):
            if sorted_scores[index] < 1:
                break
            gaps.append(float(sorted_scores[index] - sorted_scores[index + 1]))
        cutoff = 0
        for gap in sorted(gaps, reverse=True):
            index = gaps.index(gap)
            if 0 < index < (len(sorted_scores) - 1) / 2:
                cutoff = index
                break
        return [int(index) for index in order[: cutoff + 1]]

    @staticmethod
    def _prune_cycles(graph: nx.DiGraph, data: pd.DataFrame) -> int:
        removed = 0
        while not nx.is_directed_acyclic_graph(graph):
            edges = list(graph.edges)
            if not edges:
                break
            correlations = []
            for source, target in edges:
                value = data[source].corr(data[target])
                correlations.append(float(value) if np.isfinite(value) else 0.0)
            graph.remove_edge(*edges[int(np.argmin(correlations))])
            removed += 1
        return removed

    def score_metrics(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
    ) -> pd.DataFrame:
        torch, _, _ = _torch_modules()
        paired = prepare_paired_metric_data(
            normal,
            abnormal,
            drop_segment_constants=False,
        )
        columns = paired.normal.columns.tolist()
        mean = paired.normal.mean(axis=0)
        scale = paired.normal.std(axis=0, ddof=0).replace(0, 1.0)
        standardized_normal = (paired.normal - mean) / scale
        x, y = self._windows(standardized_normal.to_numpy(dtype=float))
        device = self._resolve_device()
        random.seed(self.seed)
        np.random.seed(self.seed)

        graph = nx.DiGraph()
        graph.add_nodes_from(columns)
        attention_by_target: dict[str, dict[str, float]] = {}
        for target, target_name in enumerate(columns):
            attention = self._fit_target(x, y, target, device)
            attention_by_target[target_name] = {
                columns[index]: float(score)
                for index, score in enumerate(attention)
            }
            for parent in self._potential_parents(attention):
                graph.add_edge(columns[parent], target_name)

        combined = pd.concat(
            [paired.normal, paired.abnormal], ignore_index=True
        )
        removed_edges = self._prune_cycles(graph, combined)
        sinks = {node for node, degree in graph.out_degree if degree == 0}
        personalization = {
            node: 1.0 if node in sinks else 0.5 for node in graph.nodes
        }
        pagerank = nx.pagerank(graph, personalization=personalization)
        ranking = sorted(columns, key=lambda metric: pagerank[metric], reverse=True)
        result = ranking_frame(ranking, pagerank)
        self.metric_scores_ = result
        self.diagnostics_ = {
            "protocol": "run_neural_granger_known_onset",
            "training_scope": "normal_only",
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "moving_average_kernel": self.moving_average_kernel,
            "pretrain_epochs": self.pretrain_epochs,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "device": str(device),
            "torch_num_threads": self.torch_num_threads,
            "seed": self.seed,
            "normal_samples": int(len(paired.normal)),
            "abnormal_samples": int(len(paired.abnormal)),
            "input_metrics": len(columns),
            "graph_edges": [[source, target] for source, target in graph.edges],
            "cycle_edges_removed": removed_edges,
            "pagerank": {key: float(value) for key, value in pagerank.items()},
            "attention_by_target": attention_by_target,
            "excluded_metrics": paired.excluded,
            "torch_version": torch.__version__,
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

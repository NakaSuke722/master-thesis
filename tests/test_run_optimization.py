"""RUN optimizations must preserve the statistical model and graph rules."""

import copy
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from models.baselines.run import (
    RUNScorer,
    _hierarchical_contrastive_loss,
    _make_dlinear,
    _pack_dlinear,
    _temporal_contrastive_loss,
)


def _paired_models(channels=5):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    reference = _make_dlinear(6, 4, channels, 5).double()
    packed = _pack_dlinear(copy.deepcopy(reference))
    return reference, packed


def _check_banks(reference, packed, *, gradients=False):
    def value(parameter):
        return parameter.grad if gradients else parameter

    for name in ("seasonal_encoder", "trend_encoder", "seasonal_decoder", "trend_decoder"):
        left, right = getattr(reference, name), getattr(packed, name)
        for attribute in ("weight", "bias"):
            tensors = [value(getattr(layer, attribute)) for layer in left]
            actual = value(getattr(right, attribute))
            if tensors[0] is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, torch.stack(tensors), rtol=1e-10, atol=1e-12)
    for name in ("seasonal_projector", "trend_projector"):
        left, right = getattr(reference, name), getattr(packed, name)
        for index, bank in ((0, right.first), (2, right.second)):
            for attribute in ("weight", "bias"):
                tensors = [value(getattr(layer[index], attribute)) for layer in left]
                actual = value(getattr(bank, attribute))
                if tensors[0] is None:
                    assert actual is None
                else:
                    torch.testing.assert_close(actual, torch.stack(tensors), rtol=1e-10, atol=1e-12)
        slopes = [value(layer[1].weight) for layer in left]
        if slopes[0] is None:
            assert value(right.slope) is None
        else:
            torch.testing.assert_close(value(right.slope), torch.stack(slopes), rtol=1e-10, atol=1e-12)
    for name in ("attention", "seasonal_output.weight", "seasonal_output.bias",
                 "trend_output.weight", "trend_output.bias"):
        left = value(reference.get_parameter(name))
        right = value(packed.get_parameter(name))
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(right, left, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("channels,batch", [(1, 1), (3, 7), (9, 4)])
def test_packing_preserves_initialization_rng_predictions_and_gradients(channels, batch):
    reference, packed = _paired_models(channels)
    rng = torch.get_rng_state()
    _pack_dlinear(copy.deepcopy(reference))
    assert torch.equal(rng, torch.get_rng_state())
    _check_banks(reference, packed)
    x = torch.randn(batch, 6, channels, dtype=torch.double)
    torch.testing.assert_close(packed(x), reference(x), rtol=1e-10, atol=1e-12)
    reference(x).square().mean().backward()
    packed(x).square().mean().backward()
    _check_banks(reference, packed, gradients=True)


def test_pretraining_views_gradients_and_adam_steps_preserve_independent_layers():
    reference, packed = _paired_models()
    old_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    new_optimizer = torch.optim.Adam(packed.parameters(), lr=0.001)
    for _ in range(3):
        x = torch.randn(7, 6, 5, dtype=torch.double)
        projected = reference.encode(x, project=True)
        with torch.no_grad():
            encoded = reference.encode(x, project=False)
        new_projected, new_encoded = packed.contrastive_views(x)
        assert not new_encoded.requires_grad
        torch.testing.assert_close(new_projected, projected, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(new_encoded, encoded, rtol=1e-10, atol=1e-12)
        old_optimizer.zero_grad()
        new_optimizer.zero_grad()
        old_loss = _hierarchical_contrastive_loss(projected, encoded)
        new_loss = _hierarchical_contrastive_loss(new_projected, new_encoded)
        torch.testing.assert_close(new_loss, old_loss, rtol=1e-10, atol=1e-12)
        old_loss.backward()
        new_loss.backward()
        _check_banks(reference, packed, gradients=True)
        old_optimizer.step()
        new_optimizer.step()
        _check_banks(reference, packed)


def test_temporal_loss_keeps_original_float32_multiplication_shape():
    torch.manual_seed(4)
    left, right = torch.randn(3, 128, 802), torch.randn(3, 128, 802)
    joined = torch.cat([left, right], dim=1)
    expected = -torch.nn.functional.log_softmax(
        (joined @ joined.transpose(1, 2))[:, :128, 128:], dim=-1,
    ).mean()
    assert torch.equal(_temporal_contrastive_loss(left, right), expected)


@pytest.mark.parametrize("pretrain,epochs", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_trained_attention_matches_reference_including_partial_batches(pretrain, epochs):
    values = np.random.default_rng(7).normal(size=(23, 7))
    outputs = []
    for backend in ("reference", "vectorized"):
        scorer = RUNScorer(seq_len=6, hidden_size=4, moving_average_kernel=5,
                           batch_size=7, pretrain_epochs=pretrain, epochs=epochs,
                           execution_backend=backend)
        device = scorer._resolve_device()
        x, y = scorer._windows(values)
        outputs.append(scorer._fit_target(x, y, 2, device))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=5e-7)
    assert set(RUNScorer._potential_parents(outputs[0])) == set(RUNScorer._potential_parents(outputs[1]))


def _original_prune(graph, data):
    removed = 0
    while not nx.is_directed_acyclic_graph(graph):
        edges = list(graph.edges)
        correlations = []
        for source, target in edges:
            value = data[source].corr(data[target])
            correlations.append(float(value) if np.isfinite(value) else 0.0)
        graph.remove_edge(*edges[int(np.argmin(correlations))])
        removed += 1
    return removed


@pytest.mark.parametrize("seed", range(10))
def test_cycle_pruning_is_exact_with_ties_constants_and_self_loops(seed):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.normal(size=(20, 8)))
    data[1] = data[0]  # positive ties
    data[2] = -data[0]  # negative ties
    data[3] = 1.0  # undefined correlation -> 0, same as reference
    graph = nx.DiGraph()
    graph.add_nodes_from(data.columns)
    edges = [(int(i), int(j)) for i, j in np.argwhere(rng.random((8, 8)) < 0.5)]
    rng.shuffle(edges)
    graph.add_edges_from(edges)
    reference = graph.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        removed = RUNScorer._prune_cycles(graph, data)
        expected = _original_prune(reference, data)
    assert removed == expected
    assert list(graph.nodes) == list(reference.nodes)
    assert list(graph.edges) == list(reference.edges)


def test_cycle_pruning_computes_each_original_edge_correlation_only_once(monkeypatch):
    graph = nx.complete_graph(15, create_using=nx.DiGraph)
    data = pd.DataFrame(np.random.default_rng(3).normal(size=(40, 15)))
    original = pd.Series.corr
    calls = []

    def counted(self, other, *args, **kwargs):
        calls.append((self.name, other.name))
        return original(self, other, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "corr", counted)
    edges = set(graph.edges)
    RUNScorer._prune_cycles(graph, data)
    assert len(calls) == len(edges)
    assert set(calls) == edges


def test_end_to_end_rank_and_graph_match_reference_backend():
    rng = np.random.default_rng(5)
    normal = pd.DataFrame(rng.normal(size=(30, 5)), columns=[f"svc{i}_cpu" for i in range(5)])
    abnormal = normal + 1
    models, rankings = [], []
    for backend in ("reference", "vectorized"):
        model = RUNScorer(seq_len=6, hidden_size=4, moving_average_kernel=5,
                          batch_size=7, execution_backend=backend)
        rankings.append(model.predict(normal, abnormal))
        models.append(model)
    assert rankings[0] == rankings[1]
    assert models[0].diagnostics_["graph_edges"] == models[1].diagnostics_["graph_edges"]
    assert models[1].diagnostics_["execution_backend"] == "vectorized"


def test_execution_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="execution_backend"):
        RUNScorer(execution_backend="unknown")

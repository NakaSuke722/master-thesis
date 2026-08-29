from __future__ import annotations

from collections.abc import Sequence


GroundTruth = str | Sequence[str]


def _as_set(ground_truth: GroundTruth) -> set[str]:
    if isinstance(ground_truth, str):
        return {ground_truth}
    return {str(x) for x in ground_truth}


def calculate_ac_at_k(
    predicted_ranking: list[str],
    ground_truth: GroundTruth,
    k: int,
) -> int:
    """Return 1 if top-k includes any correct answer, else 0."""
    truths = _as_set(ground_truth)
    return int(bool(set(predicted_ranking[:k]) & truths))


def calculate_avg_at_k(
    predicted_ranking: list[str],
    ground_truth: GroundTruth,
    k: int,
) -> float:
    return sum(
        calculate_ac_at_k(predicted_ranking, ground_truth, j)
        for j in range(1, k + 1)
    ) / k


def evaluate_ranking(
    predicted_ranking: list[str],
    ground_truth: GroundTruth,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    if k_values is None:
        k_values = [1, 3, 5]

    truths = _as_set(ground_truth)
    if not truths:
        raise ValueError("ground_truth is empty")

    results: dict[str, float] = {}
    for k in k_values:
        results[f"AC@{k}"] = calculate_ac_at_k(
            predicted_ranking, truths, k
        )
        results[f"Avg@{k}"] = calculate_avg_at_k(
            predicted_ranking, truths, k
        )
    return results

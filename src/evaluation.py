# src/evaluation.py
from typing import List, Dict

def calculate_ac_at_k(predicted_ranking: List[str], ground_truth: str, k: int) -> int:
    """上位K個以内に正解が含まれるかを判定する(1 or 0)"""
    top_k_predictions = predicted_ranking[:k]
    return 1 if ground_truth in top_k_predictions else 0

def calculate_avg_at_k(predicted_ranking: List[str], ground_truth: str, k: int) -> float:
    """1からKまでのAC@jの平均を算出する"""
    ac_sum = sum([calculate_ac_at_k(predicted_ranking, ground_truth, j) for j in range(1, k + 1)])
    return ac_sum / k

def evaluate_ranking(predicted_ranking: List[str], ground_truth: str, k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """指定されたすべてのKについて評価指標を一括計算する"""
    results = {}
    for k in k_values:
        results[f"AC@{k}"] = calculate_ac_at_k(predicted_ranking, ground_truth, k)
        results[f"Avg@{k}"] = calculate_avg_at_k(predicted_ranking, ground_truth, k)
    return results
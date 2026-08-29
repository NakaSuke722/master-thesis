from evaluation import evaluate_ranking


def test_rank_one():
    result = evaluate_ranking(
        ["a", "b", "c", "d", "e"],
        "a",
        [1, 3, 5],
    )

    assert result["AC@1"] == 1
    assert result["AC@3"] == 1
    assert result["AC@5"] == 1
    assert result["Avg@5"] == 1.0


def test_rank_three():
    result = evaluate_ranking(
        ["a", "b", "root", "d", "e"],
        "root",
        [1, 3, 5],
    )

    assert result["AC@1"] == 0
    assert result["AC@3"] == 1
    assert result["AC@5"] == 1
    assert result["Avg@5"] == 0.6


def test_empty_method_output_is_a_valid_all_miss_ranking():
    result = evaluate_ranking([], "root", [1, 3, 5])

    assert result == {
        "AC@1": 0,
        "Avg@1": 0.0,
        "AC@3": 0,
        "Avg@3": 0.0,
        "AC@5": 0,
        "Avg@5": 0.0,
    }

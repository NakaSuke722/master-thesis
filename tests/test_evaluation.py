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
from evaluation import (
    canonical_metric_name,
    make_evaluation_ground_truth,
)


def test_latency_canonicalization():
    assert (
        canonical_metric_name("cart_latency-90")
        == "cart_latency"
    )


def test_memory_canonicalization():
    assert (
        canonical_metric_name("cart_memory")
        == "cart_mem"
    )


def test_service_ground_truth():
    assert (
        make_evaluation_ground_truth(
            "cart_cpu",
            "service",
            {"cpu": ["cpu"]},
        )
        == "cart"
    )


def test_metric_ground_truth():
    assert (
        make_evaluation_ground_truth(
            "cart_cpu",
            "metric",
            {"cpu": ["cpu"]},
        )
        == ["cart_cpu"]
    )
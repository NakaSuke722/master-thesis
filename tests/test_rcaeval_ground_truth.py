from benchmarks.base import (
    BenchmarkCase,
)


def test_rcaeval_service_ground_truth():

    case = BenchmarkCase(
        benchmark="rcaeval_re1",
        dataset="re1_ob",
        case_id=(
            "re1ob_adservice_cpu_1"
        ),
        root_cause_service=(
            "adservice"
        ),
        fault_type="cpu",
        inject_time=123,
        repetition=1,
        root_cause_metrics=None,
    )

    assert (
        case.root_cause_service
        == "adservice"
    )

    # RCAEval RE1では
    # official metric-level GTは使わない。
    assert (
        case.root_cause_metrics
        is None
    )

    restored = (
        BenchmarkCase.from_dict(
            case.to_dict()
        )
    )

    assert restored == case
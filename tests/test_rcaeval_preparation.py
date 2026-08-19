import pandas as pd

from prepare_rcaeval_re1 import (
    preprocess_metrics,
    split_normal_abnormal,
)


def test_rcaeval_preprocessing():

    df = pd.DataFrame(
        {
            "time": [
                1,
                2,
                3,
                4,
            ],
            "svc_cpu": [
                1.0,
                1.0,
                5.0,
                6.0,
            ],
            "svc_latency-50": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "svc_latency-90": [
                0.1,
                0.1,
                0.5,
                0.6,
            ],
        }
    )

    processed = (
        preprocess_metrics(df)
    )

    assert (
        "svc_latency-50"
        not in processed.columns
    )

    assert (
        "svc_latency"
        in processed.columns
    )

    normal, abnormal = (
        split_normal_abnormal(
            processed,
            inject_time=3,
        )
    )

    assert len(normal) == 2
    assert len(abnormal) == 2

    assert (
        "time"
        not in normal.columns
    )

    assert (
        list(normal.columns)
        == list(abnormal.columns)
    )
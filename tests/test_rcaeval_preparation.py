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

def test_rcaeval_window_is_maximum():

    df = pd.DataFrame(
        {
            "time": list(
                range(1400)
            ),
            "svc_cpu": list(
                range(1400)
            ),
        }
    )

    normal, abnormal = (
        split_normal_abnormal(
            df,
            inject_time=700,
            normal_window_points=600,
            abnormal_window_points=600,
        )
    )

    assert len(normal) == 600
    assert len(abnormal) == 600

    # 障害前の末尾600点
    assert (
        normal["svc_cpu"].iloc[0]
        == 100
    )

    assert (
        normal["svc_cpu"].iloc[-1]
        == 699
    )

    # 障害後の先頭600点
    assert (
        abnormal["svc_cpu"].iloc[0]
        == 700
    )

    assert (
        abnormal["svc_cpu"].iloc[-1]
        == 1299
    )


def test_short_case_is_not_padded():

    df = pd.DataFrame(
        {
            "time": list(
                range(721)
            ),
            "svc_cpu": list(
                range(721)
            ),
        }
    )

    normal, abnormal = (
        split_normal_abnormal(
            df,
            inject_time=360,
            normal_window_points=600,
            abnormal_window_points=600,
        )
    )

    assert len(normal) == 360
    assert len(abnormal) == 361
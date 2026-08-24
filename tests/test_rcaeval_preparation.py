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


def test_rcaeval_preprocessing_drops_time_aliases_and_no_information_metrics():
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4],
            "time.1": [1, 2, 3, 4],
            "time.2": [10, 20, 30, 40],
            "always_zero": [0.0, 0.0, 0.0, 0.0],
            "always_five": [5.0, 5.0, 5.0, 5.0],
            "changes_after_fault": [0.0, 0.0, 3.0, 4.0],
        }
    )

    processed = preprocess_metrics(df)

    assert list(processed.columns) == [
        "time",
        "changes_after_fault",
    ]


def test_rcaeval_preprocessing_keeps_normal_constant_metric_that_changes():
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4],
            "svc_error": [0.0, 0.0, 1.0, 2.0],
        }
    )

    processed = preprocess_metrics(df)
    normal, abnormal = split_normal_abnormal(
        processed,
        inject_time=3,
    )

    assert normal["svc_error"].tolist() == [0.0, 0.0]
    assert abnormal["svc_error"].tolist() == [1.0, 2.0]


def test_rcaeval_window_drops_metric_constant_only_in_selected_windows():
    df = pd.DataFrame(
        {
            "time": list(range(8)),
            "outside_window_only": [
                1.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                3.0,
                4.0,
            ],
            "informative": [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                2.0,
                2.0,
                2.0,
            ],
        }
    )

    normal, abnormal = split_normal_abnormal(
        df,
        inject_time=4,
        normal_window_points=2,
        abnormal_window_points=2,
    )

    assert list(normal.columns) == ["informative"]
    assert list(abnormal.columns) == ["informative"]


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

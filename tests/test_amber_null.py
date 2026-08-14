import numpy as np
import pandas as pd

from models.amber import AMBER


def test_unchanged_metric_has_finite_score():
    normal = pd.DataFrame({
        "service_cpu": np.ones(100),
    })
    abnormal = pd.DataFrame({
        "service_cpu": np.ones(50),
    })

    model = AMBER(
        ar_order=3,
        winsor_quantile=None,
    )

    result = model.fit_predict(normal, abnormal)

    assert np.isfinite(result.loc[0, "score"])


def test_changed_metric_ranks_above_stable_metric():
    normal = pd.DataFrame({
        "stable_cpu": np.ones(100),
        "changed_cpu": np.ones(100),
    })

    abnormal = pd.DataFrame({
        "stable_cpu": np.ones(50),
        "changed_cpu": np.full(50, 10.0),
    })

    model = AMBER(
        ar_order=3,
        winsor_quantile=None,
    )

    result = model.fit_predict(normal, abnormal)

    assert result.iloc[0]["metric"] == "changed_cpu"
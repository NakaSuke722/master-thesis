import numpy as np

from models.amber import (
    NIG,
    _nig_log_marginal,
    _nig_update,
)


def test_nig_sequential_marginal_identity():
    """m(D0,D1)=m(D0)m(D1|D0) を確認する。"""

    prior = NIG(
        m=0.0,
        kappa=1e-3,
        alpha=2.0,
        beta=1.0,
    )

    d0 = np.array(
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    )
    d1 = np.array(
        [-0.8, -0.3, 0.2, 0.7]
    )

    pooled = np.concatenate([d0, d1])

    log_joint = _nig_log_marginal(
        pooled,
        prior,
    )

    posterior_d0 = _nig_update(
        prior,
        d0,
    )

    log_sequential = (
        _nig_log_marginal(d0, prior)
        + _nig_log_marginal(
            d1,
            posterior_d0,
        )
    )

    assert np.isclose(
        log_joint,
        log_sequential,
        atol=1e-10,
    )


def test_shift_has_larger_bayes_factor():
    """明確な分布変化で log BF が増えることを確認する。"""

    prior = NIG(
        m=0.0,
        kappa=1e-3,
        alpha=2.0,
        beta=1.0,
    )

    d0 = np.tile(
        [-1.0, 0.0, 1.0],
        50,
    )

    d1_same = d0.copy()
    d1_shift = d0 + 5.0

    def log_bf(d1):
        pooled = np.concatenate([d0, d1])

        return (
            _nig_log_marginal(d0, prior)
            + _nig_log_marginal(d1, prior)
            - _nig_log_marginal(
                pooled,
                prior,
            )
        )

    assert log_bf(d1_shift) > log_bf(d1_same)
"""Bayesian shared-vs-separate autoregressive model comparison.

The null hypothesis uses one AR parameter set for the pre/post segments.  The
alternative gives each segment an independent parameter set drawn from the
same proper Normal-Inverse-Gamma prior.  The conditional AR likelihood is a
Bayesian linear regression, so all marginal likelihoods are analytic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import lgamma, log, pi

import numpy as np


@dataclass(frozen=True)
class ARBayesFactorPrior:
    """Proper conjugate prior for an AR regression.

    ``precision`` values are conditional on the innovation variance:
    ``beta | sigma^2 ~ N(mean, sigma^2 * precision^-1)``.
    """

    intercept_mean: float = 0.0
    lag_mean: float = 0.0
    intercept_precision: float = 0.01
    lag_precision: float = 1.0
    alpha: float = 2.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.intercept_precision <= 0:
            raise ValueError("intercept_precision must be positive")
        if self.lag_precision <= 0:
            raise ValueError("lag_precision must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")


def _normal_only_standardize(
    pre: np.ndarray,
    post: np.ndarray,
    min_scale: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Standardize both segments using location/scale learned from pre only."""
    if min_scale <= 0:
        raise ValueError("min_scale must be positive")
    finite_pre = pre[np.isfinite(pre)]
    if finite_pre.size == 0:
        raise ValueError("pre segment has no finite observations")
    center = float(np.median(finite_pre))
    mad = float(np.median(np.abs(finite_pre - center)))
    robust_scale = 1.4826 * mad
    sd = float(np.std(finite_pre, ddof=1)) if finite_pre.size > 1 else 0.0
    scale = max(robust_scale, 0.1 * sd, min_scale)
    return (pre - center) / scale, (post - center) / scale, center, scale


def _ar_design(
    segment: np.ndarray,
    order: int,
    history: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build conditional-AR rows without collapsing gaps in the time axis."""
    if order < 0:
        raise ValueError("order must be non-negative")
    segment = np.asarray(segment, dtype=float)
    if segment.ndim != 1:
        raise ValueError("segment must be one-dimensional")

    if history is None:
        prefix = np.empty(0, dtype=float)
        first_target = order
    else:
        history = np.asarray(history, dtype=float)
        if history.ndim != 1:
            raise ValueError("history must be one-dimensional")
        if order and history.size < order:
            raise ValueError(f"Need at least ar_order={order} history observations")
        prefix = history[-order:] if order else np.empty(0, dtype=float)
        first_target = prefix.size

    combined = np.concatenate([prefix, segment])
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for index in range(first_target, combined.size):
        if order:
            lags = combined[index - np.arange(1, order + 1)]
        else:
            lags = np.empty(0, dtype=float)
        row = np.concatenate(([1.0], lags))
        target = float(combined[index])
        if np.isfinite(target) and np.all(np.isfinite(row)):
            rows.append(row)
            targets.append(target)

    if not rows:
        return np.empty((0, order + 1), dtype=float), np.empty(0, dtype=float)
    return np.vstack(rows), np.asarray(targets, dtype=float)


def _ar_spectral_radius_from_mean(coef: np.ndarray) -> float:
    order = max(0, coef.size - 1)
    if order == 0:
        return 0.0
    roots = np.roots(np.concatenate(([1.0], -coef[1:])))
    return float(np.max(np.abs(roots))) if roots.size else 0.0


def _posterior_summary(
    mean: np.ndarray,
    alpha: float,
    beta: float,
    n_rows: int,
) -> dict[str, object]:
    denominator = 1.0 - float(np.sum(mean[1:]))
    long_run_mean = (
        float(mean[0] / denominator)
        if abs(denominator) > 1e-10
        else None
    )
    return {
        "n_rows": n_rows,
        "coefficient_mean": mean.astype(float).tolist(),
        "innovation_variance_mean": (
            float(beta / (alpha - 1.0)) if alpha > 1.0 else None
        ),
        "spectral_radius_at_mean": _ar_spectral_radius_from_mean(mean),
        "long_run_mean_at_mean": long_run_mean,
        "alpha": float(alpha),
        "beta": float(beta),
    }


def _bayesian_ar_log_marginal(
    design: np.ndarray,
    target: np.ndarray,
    prior: ARBayesFactorPrior,
) -> tuple[float, dict[str, object]]:
    """Return analytic log marginal likelihood and posterior summary."""
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    if design.ndim != 2 or target.ndim != 1:
        raise ValueError("design must be 2-D and target must be 1-D")
    if design.shape[0] != target.size:
        raise ValueError("design and target row counts differ")
    if target.size == 0:
        raise ValueError("AR marginal likelihood requires at least one row")

    dimension = design.shape[1]
    prior_mean = np.full(dimension, prior.lag_mean, dtype=float)
    prior_mean[0] = prior.intercept_mean
    prior_precision = np.diag(np.concatenate([
        [prior.intercept_precision],
        np.full(dimension - 1, prior.lag_precision, dtype=float),
    ]))
    posterior_precision = prior_precision + design.T @ design
    posterior_rhs = prior_precision @ prior_mean + design.T @ target
    posterior_mean = np.linalg.solve(posterior_precision, posterior_rhs)
    alpha_n = prior.alpha + 0.5 * target.size
    beta_n = prior.beta + 0.5 * (
        float(target @ target)
        + float(prior_mean @ prior_precision @ prior_mean)
        - float(posterior_mean @ posterior_precision @ posterior_mean)
    )
    beta_n = max(float(beta_n), 1e-12)

    prior_sign, prior_logdet = np.linalg.slogdet(prior_precision)
    posterior_sign, posterior_logdet = np.linalg.slogdet(posterior_precision)
    if prior_sign <= 0 or posterior_sign <= 0:
        raise ValueError("AR prior/posterior precision must be positive definite")

    log_marginal = (
        -0.5 * target.size * log(2.0 * pi)
        + 0.5 * (prior_logdet - posterior_logdet)
        + prior.alpha * log(prior.beta)
        - alpha_n * log(beta_n)
        + lgamma(alpha_n)
        - lgamma(prior.alpha)
    )
    return float(log_marginal), _posterior_summary(
        posterior_mean, alpha_n, beta_n, target.size
    )


def ar_change_bayes_factor(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    order: int = 3,
    prior: ARBayesFactorPrior | None = None,
    standardize: bool = True,
    min_scale: float = 1e-6,
) -> dict[str, object]:
    """Compare a shared pre/post AR process against two separate AR processes.

    Post-segment rows use the final pre observations as their initial lags.
    Under H0, pre and post regression rows share ``(c, phi, sigma^2)``.  Under
    H1, the two segments receive independent copies of the same proper prior.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    if pre.ndim != 1 or post.ndim != 1:
        raise ValueError("pre and post must be one-dimensional")
    if pre.size <= order:
        raise ValueError(f"Need more than ar_order={order} pre observations")
    if post.size == 0:
        raise ValueError("post segment must not be empty")

    model_prior = prior or ARBayesFactorPrior()
    if standardize:
        scaled_pre, scaled_post, center, scale = _normal_only_standardize(
            pre, post, min_scale
        )
    else:
        scaled_pre, scaled_post = pre.copy(), post.copy()
        center, scale = 0.0, 1.0

    pre_design, pre_target = _ar_design(scaled_pre, order)
    post_design, post_target = _ar_design(
        scaled_post, order, history=scaled_pre
    )
    if pre_target.size == 0 or post_target.size == 0:
        raise ValueError("No finite conditional-AR rows in one or both segments")

    pooled_design = np.vstack([pre_design, post_design])
    pooled_target = np.concatenate([pre_target, post_target])
    log_pre, posterior_pre = _bayesian_ar_log_marginal(
        pre_design, pre_target, model_prior
    )
    log_post, posterior_post = _bayesian_ar_log_marginal(
        post_design, post_target, model_prior
    )
    log_h0, posterior_shared = _bayesian_ar_log_marginal(
        pooled_design, pooled_target, model_prior
    )
    log_h1 = log_pre + log_post

    return {
        "schema_version": 1,
        "hypothesis": "shared_ar_vs_separate_ar",
        "ar_order": order,
        "log_bayes_factor": float(log_h1 - log_h0),
        "log_marginal_h0": float(log_h0),
        "log_marginal_h1": float(log_h1),
        "normalization": {
            "normal_only": bool(standardize),
            "center": center,
            "scale": scale,
        },
        "prior": asdict(model_prior),
        "posterior_shared": posterior_shared,
        "posterior_pre": posterior_pre,
        "posterior_post": posterior_post,
    }

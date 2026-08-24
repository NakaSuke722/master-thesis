"""Bayesian autoregressive model comparisons for AMBER.

The conditional AR likelihood is Bayesian linear regression under a proper
Normal-Inverse-Gamma prior. This supports analytic shared-vs-separate,
intervention-response, and sparse parameter-regime comparisons.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import lgamma, log, pi, sqrt
from typing import Literal, Sequence

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


@dataclass(frozen=True)
class ARRegimeShiftPrior:
    """Sparse normal-anchored prior for BSRC-AR parameter changes.

    Active coefficient changes use a Gaussian slab conditional on the shared
    innovation variance. Independent Bernoulli indicators provide the spike
    at zero. The innovation variance has its own Bernoulli indicator: its spike
    fixes the post/pre ratio to one, while its slab has a zero-centred normal
    prior on the log scale and is integrated with Gauss--Hermite quadrature.
    ``variance_inclusion_probability=1`` preserves the original BSRC-AR model
    in which every H1 candidate activates the variance-ratio slab.
    """

    intercept_precision: float = 0.25
    lag_precision: float = 1.0
    inclusion_probability: float = 0.25
    variance_inclusion_probability: float = 1.0
    log_variance_sd: float = 0.7
    variance_quadrature_points: int = 4

    def __post_init__(self) -> None:
        if self.intercept_precision <= 0:
            raise ValueError("intercept_precision must be positive")
        if self.lag_precision <= 0:
            raise ValueError("lag_precision must be positive")
        if not 0.0 < self.inclusion_probability < 1.0:
            raise ValueError("inclusion_probability must lie in (0, 1)")
        if not 0.0 <= self.variance_inclusion_probability <= 1.0:
            raise ValueError(
                "variance_inclusion_probability must lie in [0, 1]"
            )
        if self.log_variance_sd < 0:
            raise ValueError("log_variance_sd must be non-negative")
        if self.variance_quadrature_points <= 0:
            raise ValueError("variance_quadrature_points must be positive")


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
    design, target, _ = _ar_design_with_indices(segment, order, history)
    return design, target


def _ar_design_with_indices(
    segment: np.ndarray,
    order: int,
    history: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build AR rows and return their original target indices."""
    if order < 0:
        raise ValueError("order must be non-negative")
    segment = np.asarray(segment, dtype=float)
    if segment.ndim != 1:
        raise ValueError("segment must be one-dimensional")

    if history is None:
        prefix = np.empty(0, dtype=float)
    else:
        history = np.asarray(history, dtype=float)
        if history.ndim != 1:
            raise ValueError("history must be one-dimensional")
        if order and history.size < order:
            raise ValueError(f"Need at least ar_order={order} history observations")
        prefix = history[-order:] if order else np.empty(0, dtype=float)

    if order == 0:
        finite = np.isfinite(segment)
        return (
            np.ones((int(np.sum(finite)), 1), dtype=float),
            segment[finite].astype(float, copy=False),
            np.flatnonzero(finite),
        )

    combined = np.concatenate([prefix, segment])
    if combined.size <= order:
        return (
            np.empty((0, order + 1), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=int),
        )
    windows = np.lib.stride_tricks.sliding_window_view(combined, order + 1)
    # With a history prefix every window ends at one post-segment target;
    # without it the first target is naturally at index ``order``.
    targets = windows[:, -1]
    lags = windows[:, :-1][:, ::-1]
    finite = np.isfinite(targets) & np.all(np.isfinite(lags), axis=1)
    if not np.any(finite):
        return (
            np.empty((0, order + 1), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=int),
        )
    design = np.column_stack([
        np.ones(int(np.sum(finite)), dtype=float),
        lags[finite],
    ])
    target_offset = 0 if history is not None else order
    retained_indices = np.flatnonzero(finite) + target_offset
    return (
        design,
        targets[finite].astype(float, copy=False),
        retained_indices,
    )


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
    dimension = design.shape[1]
    prior_mean = np.full(dimension, prior.lag_mean, dtype=float)
    prior_mean[0] = prior.intercept_mean
    prior_precision = np.concatenate([
        [prior.intercept_precision],
        np.full(dimension - 1, prior.lag_precision, dtype=float),
    ])
    log_marginal, posterior_mean, alpha_n, beta_n = (
        _bayesian_regression_log_marginal(
            design,
            target,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            alpha=prior.alpha,
            beta=prior.beta,
        )
    )
    return log_marginal, _posterior_summary(
        posterior_mean, alpha_n, beta_n, target.size
    )


def _bayesian_regression_log_marginal(
    design: np.ndarray,
    target: np.ndarray,
    *,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[float, np.ndarray, float, float]:
    """Return a Normal-Inverse-Gamma regression marginal likelihood.

    ``prior_precision`` is the diagonal of the coefficient precision matrix,
    conditional on the innovation variance.  Keeping this primitive generic
    allows the intercept-shift alternative to use two intercept columns while
    sharing all lag coefficients.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    if design.ndim != 2 or target.ndim != 1:
        raise ValueError("design must be 2-D and target must be 1-D")
    if design.shape[0] != target.size:
        raise ValueError("design and target row counts differ")
    if target.size == 0:
        raise ValueError("AR marginal likelihood requires at least one row")

    return _bayesian_regression_log_marginal_from_sufficient_statistics(
        design.T @ design,
        design.T @ target,
        float(target @ target),
        target.size,
        prior_mean=prior_mean,
        prior_precision=prior_precision,
        alpha=alpha,
        beta=beta,
    )


def _bayesian_regression_log_marginal_from_sufficient_statistics(
    gram: np.ndarray,
    target_cross: np.ndarray,
    target_square_sum: float,
    n_rows: int,
    *,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[float, np.ndarray, float, float]:
    """Evaluate the same NIG marginal likelihood from reusable statistics."""
    gram = np.asarray(gram, dtype=float)
    target_cross = np.asarray(target_cross, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_precision = np.asarray(prior_precision, dtype=float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a square matrix")
    dimension = gram.shape[0]
    if target_cross.shape != (dimension,):
        raise ValueError("target_cross dimension does not match gram")
    if n_rows <= 0:
        raise ValueError("AR marginal likelihood requires at least one row")
    if prior_mean.shape != (dimension,):
        raise ValueError("prior_mean dimension does not match design")
    if prior_precision.shape != (dimension,):
        raise ValueError("prior_precision dimension does not match design")
    if not np.all(np.isfinite(prior_precision)) or np.any(prior_precision <= 0):
        raise ValueError("prior_precision must be finite and positive")
    if not np.isfinite(target_square_sum) or target_square_sum < 0:
        raise ValueError("target_square_sum must be finite and non-negative")
    if alpha <= 0 or beta <= 0:
        raise ValueError("inverse-gamma parameters must be positive")

    prior_precision_matrix = np.diag(prior_precision)
    posterior_precision = prior_precision_matrix + gram
    posterior_rhs = prior_precision_matrix @ prior_mean + target_cross
    posterior_mean = np.linalg.solve(posterior_precision, posterior_rhs)
    alpha_n = alpha + 0.5 * n_rows
    beta_n = beta + 0.5 * (
        target_square_sum
        + float(prior_mean @ prior_precision_matrix @ prior_mean)
        - float(posterior_mean @ posterior_precision @ posterior_mean)
    )
    beta_n = max(float(beta_n), 1e-12)

    prior_sign, prior_logdet = np.linalg.slogdet(prior_precision_matrix)
    posterior_sign, posterior_logdet = np.linalg.slogdet(posterior_precision)
    if prior_sign <= 0 or posterior_sign <= 0:
        raise ValueError("AR prior/posterior precision must be positive definite")

    log_marginal = (
        -0.5 * n_rows * log(2.0 * pi)
        + 0.5 * (prior_logdet - posterior_logdet)
        + alpha * log(beta)
        - alpha_n * log(beta_n)
        + lgamma(alpha_n)
        - lgamma(alpha)
    )
    return float(log_marginal), posterior_mean, float(alpha_n), float(beta_n)


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


def ar_intercept_shift_bayes_factor(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    order: int = 3,
    prior: ARBayesFactorPrior | None = None,
    standardize: bool = True,
    min_scale: float = 1e-6,
) -> dict[str, object]:
    """Compare one shared AR intercept against pre/post-specific intercepts.

    Both hypotheses share the lag coefficients and innovation variance:

    H0: ``x_t = c + phi' lags_t + epsilon_t``
    H1: ``x_t = c_pre/post + phi' lags_t + epsilon_t``

    Therefore the alternative adds exactly one degree of freedom.  A level
    shift is represented explicitly through ``c_post``.  With highly
    persistent lags its conditional-intercept effect can still be small, so
    this restriction does not by itself eliminate near-unit-root attenuation.
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
    log_h0, posterior_h0 = _bayesian_ar_log_marginal(
        pooled_design, pooled_target, model_prior
    )

    pre_h1 = np.column_stack([
        np.ones(pre_design.shape[0]),
        np.zeros(pre_design.shape[0]),
        pre_design[:, 1:],
    ])
    post_h1 = np.column_stack([
        np.zeros(post_design.shape[0]),
        np.ones(post_design.shape[0]),
        post_design[:, 1:],
    ])
    h1_design = np.vstack([pre_h1, post_h1])
    h1_prior_mean = np.concatenate([
        [model_prior.intercept_mean, model_prior.intercept_mean],
        np.full(order, model_prior.lag_mean, dtype=float),
    ])
    h1_prior_precision = np.concatenate([
        [model_prior.intercept_precision, model_prior.intercept_precision],
        np.full(order, model_prior.lag_precision, dtype=float),
    ])
    log_h1, h1_mean, h1_alpha, h1_beta = _bayesian_regression_log_marginal(
        h1_design,
        pooled_target,
        prior_mean=h1_prior_mean,
        prior_precision=h1_prior_precision,
        alpha=model_prior.alpha,
        beta=model_prior.beta,
    )
    shared_lags = h1_mean[2:]
    pre_coefficients = np.concatenate(([h1_mean[0]], shared_lags))
    post_coefficients = np.concatenate(([h1_mean[1]], shared_lags))
    pre_summary = _posterior_summary(
        pre_coefficients, h1_alpha, h1_beta, pre_target.size
    )
    post_summary = _posterior_summary(
        post_coefficients, h1_alpha, h1_beta, post_target.size
    )
    posterior_h1 = {
        "n_rows": int(pooled_target.size),
        "pre_rows": int(pre_target.size),
        "post_rows": int(post_target.size),
        "pre_intercept_mean": float(h1_mean[0]),
        "post_intercept_mean": float(h1_mean[1]),
        "intercept_shift_mean": float(h1_mean[1] - h1_mean[0]),
        "shared_lag_mean": shared_lags.astype(float).tolist(),
        "pre_coefficient_mean": pre_coefficients.astype(float).tolist(),
        "post_coefficient_mean": post_coefficients.astype(float).tolist(),
        "innovation_variance_mean": (
            float(h1_beta / (h1_alpha - 1.0)) if h1_alpha > 1.0 else None
        ),
        "spectral_radius_at_mean": _ar_spectral_radius_from_mean(
            pre_coefficients
        ),
        "pre_long_run_mean_at_mean": pre_summary["long_run_mean_at_mean"],
        "post_long_run_mean_at_mean": post_summary["long_run_mean_at_mean"],
        "alpha": h1_alpha,
        "beta": h1_beta,
    }

    return {
        "schema_version": 1,
        "hypothesis": "shared_intercept_vs_pre_post_intercepts_shared_ar",
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
        "posterior_h0": posterior_h0,
        "posterior_h1": posterior_h1,
    }


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("logsumexp requires at least one value")
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _variance_ratio_quadrature(
    log_sd: float,
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log-ratios, ratios and normalized log prior weights."""
    if log_sd < 0:
        raise ValueError("log_sd must be non-negative")
    if points <= 0:
        raise ValueError("points must be positive")
    if log_sd == 0:
        return (
            np.zeros(1, dtype=float),
            np.ones(1, dtype=float),
            np.zeros(1, dtype=float),
        )
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    log_ratios = sqrt(2.0) * log_sd * nodes
    ratios = np.exp(log_ratios)
    log_weights = np.log(weights) - 0.5 * log(pi)
    log_weights -= _logsumexp(log_weights)
    return log_ratios, ratios, log_weights


def ar_shrinkage_regime_bayes_factor(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    order: int = 3,
    prior: ARBayesFactorPrior | None = None,
    regime_prior: ARRegimeShiftPrior | None = None,
    standardize: bool = True,
    min_scale: float = 1e-6,
    posterior_detail: Literal["full", "map", "none"] = "full",
) -> dict[str, object]:
    """Compare normal continuation with a sparse AR parameter regime change.

    H0 uses one AR coefficient vector and innovation variance for the normal
    and post periods.  H1 keeps a normal-regime coefficient vector and adds
    post-boundary changes to a sparse subset of the intercept/lag parameters.
    A separate spike-and-slab indicator either fixes the post/pre innovation-
    variance ratio to one or activates a log-normal variance-ratio slab.  H1 is
    conditioned on at least one coefficient or variance change.  The change
    subset and variance ratio are model-averaged, so no response-shape basis or
    post-derived empirical-null correction is used.

    The normal-period marginal likelihood is common to both hypotheses.  The
    returned H0/H1 marginal values are therefore conditional posterior-
    predictive log densities for the post period given the normal period.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if posterior_detail not in {"full", "map", "none"}:
        raise ValueError(f"Unknown posterior_detail={posterior_detail}")
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    if pre.ndim != 1 or post.ndim != 1:
        raise ValueError("pre and post must be one-dimensional")
    if pre.size <= order:
        raise ValueError(f"Need more than ar_order={order} pre observations")
    if post.size == 0:
        raise ValueError("post segment must not be empty")

    model_prior = prior or ARBayesFactorPrior()
    shift_prior = regime_prior or ARRegimeShiftPrior(
        inclusion_probability=min(0.5, 1.0 / (order + 1))
    )
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

    dimension = order + 1
    coefficient_names = ["intercept"] + [
        f"lag_{index}" for index in range(1, order + 1)
    ]
    base_prior_mean = np.concatenate([
        [model_prior.intercept_mean],
        np.full(order, model_prior.lag_mean, dtype=float),
    ])
    base_prior_precision = np.concatenate([
        [model_prior.intercept_precision],
        np.full(order, model_prior.lag_precision, dtype=float),
    ])
    shift_precisions = np.concatenate([
        [shift_prior.intercept_precision],
        np.full(order, shift_prior.lag_precision, dtype=float),
    ])

    log_pre, posterior_normal = _bayesian_ar_log_marginal(
        pre_design, pre_target, model_prior
    )
    pooled_design = np.vstack([pre_design, post_design])
    pooled_target = np.concatenate([pre_target, post_target])
    log_joint_h0, posterior_h0 = _bayesian_ar_log_marginal(
        pooled_design, pooled_target, model_prior
    )
    log_predictive_h0 = float(log_joint_h0 - log_pre)

    pre_gram = pre_design.T @ pre_design
    post_gram = post_design.T @ post_design
    pre_cross = pre_design.T @ pre_target
    post_cross = post_design.T @ post_target
    pre_square = float(pre_target @ pre_target)
    post_square = float(post_target @ post_target)
    log_ratios, variance_ratios, log_variance_weights = (
        _variance_ratio_quadrature(
            shift_prior.log_variance_sd,
            shift_prior.variance_quadrature_points,
        )
    )

    variance_inclusion = shift_prior.variance_inclusion_probability
    variance_states: list[tuple[bool, float, float, float]] = []
    if variance_inclusion < 1.0:
        variance_states.append((
            False,
            0.0,
            1.0,
            log(1.0 - variance_inclusion),
        ))
    if variance_inclusion > 0.0:
        variance_states.extend(
            (
                True,
                float(log_ratio),
                float(variance_ratio),
                float(log(variance_inclusion) + variance_log_weight),
            )
            for log_ratio, variance_ratio, variance_log_weight in zip(
                log_ratios,
                variance_ratios,
                log_variance_weights,
                strict=True,
            )
        )

    inclusion = shift_prior.inclusion_probability
    log_inclusion = log(inclusion)
    log_exclusion = log(1.0 - inclusion)
    candidate_log_marginals: list[float] = []
    candidate_log_weights: list[float] = []
    candidate_states: list[tuple[
        tuple[int, ...], bool, float, float, np.ndarray, float, float,
    ]] = []
    for mask_value in range(1 << dimension):
        selected = tuple(
            index for index in range(dimension)
            if mask_value & (1 << index)
        )
        mask_log_weight = (
            len(selected) * log_inclusion
            + (dimension - len(selected)) * log_exclusion
        )
        selected_array = np.asarray(selected, dtype=int)
        for (
            variance_change_active,
            log_ratio,
            variance_ratio,
            variance_log_weight,
        ) in variance_states:
            # The all-spike state is H0 itself.  Excluding it and normalizing
            # the remaining prior weights defines H1 conditional on at least
            # one genuine regime change.
            if not selected and not variance_change_active:
                continue
            inverse_ratio = 1.0 / float(variance_ratio)
            base_gram = pre_gram + inverse_ratio * post_gram
            base_cross = pre_cross + inverse_ratio * post_cross
            target_square = pre_square + inverse_ratio * post_square
            if selected:
                post_shift = post_design[:, selected_array]
                base_shift_cross = (
                    inverse_ratio * post_design.T @ post_shift
                )
                shift_gram = inverse_ratio * post_shift.T @ post_shift
                gram = np.block([
                    [base_gram, base_shift_cross],
                    [base_shift_cross.T, shift_gram],
                ])
                target_cross = np.concatenate([
                    base_cross,
                    inverse_ratio * post_shift.T @ post_target,
                ])
                candidate_prior_mean = np.concatenate([
                    base_prior_mean, np.zeros(len(selected), dtype=float),
                ])
                candidate_prior_precision = np.concatenate([
                    base_prior_precision, shift_precisions[selected_array],
                ])
            else:
                gram = base_gram
                target_cross = base_cross
                candidate_prior_mean = base_prior_mean
                candidate_prior_precision = base_prior_precision

            log_marginal, mean, alpha_n, beta_n = (
                _bayesian_regression_log_marginal_from_sufficient_statistics(
                    gram,
                    target_cross,
                    target_square,
                    pre_target.size + post_target.size,
                    prior_mean=candidate_prior_mean,
                    prior_precision=candidate_prior_precision,
                    alpha=model_prior.alpha,
                    beta=model_prior.beta,
                )
            )
            # Whitening post rows by sqrt(variance_ratio) changes the density
            # normalization by this Jacobian term.
            log_marginal -= 0.5 * post_target.size * float(log_ratio)
            candidate_log_marginals.append(float(log_marginal))
            candidate_log_weights.append(
                float(mask_log_weight + variance_log_weight)
            )
            if posterior_detail != "none":
                candidate_states.append((
                    selected,
                    variance_change_active,
                    float(log_ratio),
                    float(variance_ratio),
                    mean,
                    alpha_n,
                    beta_n,
                ))

    log_marginals = np.asarray(candidate_log_marginals, dtype=float)
    log_weights = np.asarray(candidate_log_weights, dtype=float)
    log_joint_h1 = (
        _logsumexp(log_marginals + log_weights)
        - _logsumexp(log_weights)
    )
    log_predictive_h1 = float(log_joint_h1 - log_pre)

    candidates: list[dict[str, object]] = []
    posterior_map: dict[str, object] | None = None
    inclusion_posterior = {
        name: 0.0 for name in coefficient_names
    }
    variance_ratio_mean: float | None = None
    variance_change_probability: float | None = None
    if posterior_detail != "none":
        normalizer = _logsumexp(log_marginals + log_weights)
        probabilities = np.exp(log_marginals + log_weights - normalizer)
        map_index = int(np.argmax(probabilities))
        variance_ratio_mean = float(
            np.sum(probabilities * np.asarray([
                state[3] for state in candidate_states
            ], dtype=float))
        )
        variance_change_probability = float(
            np.sum(probabilities * np.asarray([
                state[1] for state in candidate_states
            ], dtype=float))
        )
        for index, (
            selected,
            variance_change_active,
            log_ratio,
            variance_ratio,
            mean,
            alpha_n,
            beta_n,
        ) in enumerate(candidate_states):
            probability = float(probabilities[index])
            for selected_index in selected:
                inclusion_posterior[coefficient_names[selected_index]] += (
                    probability
                )
            candidate: dict[str, object] = {
                "changed_parameters": [
                    coefficient_names[selected_index]
                    for selected_index in selected
                ],
                "variance_change_active": variance_change_active,
                "log_variance_ratio": log_ratio,
                "variance_ratio": variance_ratio,
                "log_prior_weight": float(log_weights[index]),
                "log_marginal": float(log_marginals[index]),
                "posterior_model_probability": probability,
            }
            if posterior_detail == "full" or index == map_index:
                base_mean = mean[:dimension]
                change_mean = np.zeros(dimension, dtype=float)
                if selected:
                    change_mean[np.asarray(selected, dtype=int)] = mean[
                        dimension:
                    ]
                post_mean = base_mean + change_mean
                pre_variance = (
                    float(beta_n / (alpha_n - 1.0))
                    if alpha_n > 1.0 else None
                )
                post_variance = (
                    float(variance_ratio * pre_variance)
                    if pre_variance is not None else None
                )
                pre_summary = _posterior_summary(
                    base_mean, alpha_n, beta_n, pre_target.size
                )
                post_summary = _posterior_summary(
                    post_mean, alpha_n,
                    beta_n * variance_ratio, post_target.size,
                )
                candidate.update({
                    "pre_rows": int(pre_target.size),
                    "post_rows": int(post_target.size),
                    "base_coefficient_mean": base_mean.astype(float).tolist(),
                    "coefficient_change_mean": (
                        change_mean.astype(float).tolist()
                    ),
                    "post_coefficient_mean": post_mean.astype(float).tolist(),
                    "pre_innovation_variance_mean": pre_variance,
                    "post_innovation_variance_mean": post_variance,
                    "pre_spectral_radius_at_mean": (
                        pre_summary["spectral_radius_at_mean"]
                    ),
                    "post_spectral_radius_at_mean": (
                        post_summary["spectral_radius_at_mean"]
                    ),
                    "pre_long_run_mean_at_mean": (
                        pre_summary["long_run_mean_at_mean"]
                    ),
                    "post_long_run_mean_at_mean": (
                        post_summary["long_run_mean_at_mean"]
                    ),
                    "alpha": alpha_n,
                    "beta": beta_n,
                })
            candidates.append(candidate)
        candidates.sort(
            key=lambda item: float(item["posterior_model_probability"]),
            reverse=True,
        )
        posterior_map = candidates[0]

    return {
        "schema_version": 1,
        "hypothesis": "normal_ar_continuation_vs_sparse_regime_change",
        "ar_order": order,
        "known_change_boundary": True,
        "h1_excludes_all_spike_state": True,
        "predictive_rows": int(post_target.size),
        "log_bayes_factor": float(log_predictive_h1 - log_predictive_h0),
        "log_marginal_h0": log_predictive_h0,
        "log_marginal_h1": log_predictive_h1,
        "log_joint_h0": float(log_joint_h0),
        "log_joint_h1": float(log_joint_h1),
        "log_marginal_normal": float(log_pre),
        "normalization": {
            "normal_only": bool(standardize),
            "center": center,
            "scale": scale,
        },
        "prior": asdict(model_prior),
        "regime_shift_prior": asdict(shift_prior),
        "posterior_normal": posterior_normal,
        "posterior_h0": posterior_h0,
        "posterior_models": candidates,
        "posterior_map": posterior_map,
        "parameter_change_inclusion_probability": inclusion_posterior,
        "posterior_variance_change_probability": (
            variance_change_probability
        ),
        "posterior_variance_ratio_mean": variance_ratio_mean,
    }


@lru_cache(maxsize=256)
def _intervention_basis(
    length: int,
    *,
    shape: str,
    onset_offset: int,
    half_life: float,
) -> np.ndarray:
    """Create post-period intervention columns for a candidate response."""
    if length <= 0:
        raise ValueError("intervention length must be positive")
    if onset_offset < 0 or onset_offset >= length:
        raise ValueError("onset_offset must lie within the post period")
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    allowed = {"step", "ramp", "exp_rise", "exp_decay", "step_ramp"}
    if shape not in allowed:
        raise ValueError(f"Unknown intervention shape={shape}")

    horizon = np.arange(length, dtype=float) - float(onset_offset)
    active = horizon >= 0
    step = active.astype(float)
    active_horizon = np.maximum(horizon, 0.0)
    remaining = max(1, length - onset_offset)
    ramp = np.where(active, (active_horizon + 1.0) / remaining, 0.0)
    rise = np.where(
        active,
        1.0 - np.exp(-np.log(2.0) * (active_horizon + 1.0) / half_life),
        0.0,
    )
    decay = np.where(
        active,
        np.exp(-np.log(2.0) * active_horizon / half_life),
        0.0,
    )
    if shape == "step":
        basis = step[:, None]
    elif shape == "ramp":
        basis = ramp[:, None]
    elif shape == "exp_rise":
        basis = rise[:, None]
    elif shape == "exp_decay":
        basis = decay[:, None]
    else:
        basis = np.column_stack([step, ramp])
    basis.setflags(write=False)
    return basis


def ar_intervention_bayes_factor(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    order: int = 3,
    prior: ARBayesFactorPrior | None = None,
    shapes: Sequence[str] = (
        "step", "ramp", "exp_rise", "exp_decay", "step_ramp",
    ),
    onset_offsets: Sequence[int] = (0,),
    half_life: float = 10.0,
    intervention_precision: float = 0.1,
    onset_prior_decay: float = 0.0,
    standardize: bool = True,
    min_scale: float = 1e-6,
    posterior_detail: Literal["full", "map", "none"] = "full",
) -> dict[str, object]:
    """Model-average structured intervention responses on a shared AR process.

    H0 has one baseline AR process.  Each H1 candidate keeps the baseline AR
    coefficients and innovation variance shared, then adds one of several
    deterministic response bases beginning at a candidate onset.  Conditional
    on shape/onset, the model is conjugate Bayesian regression.  The H1
    marginal likelihood averages all candidates rather than selecting the
    largest one, retaining the Bayesian multiplicity penalty.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if intervention_precision <= 0:
        raise ValueError("intervention_precision must be positive")
    if onset_prior_decay < 0:
        raise ValueError("onset_prior_decay must be non-negative")
    if not shapes:
        raise ValueError("At least one intervention shape is required")
    if not onset_offsets:
        raise ValueError("At least one onset offset is required")
    if posterior_detail not in {"full", "map", "none"}:
        raise ValueError(f"Unknown posterior_detail={posterior_detail}")
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
    post_design, post_target, retained_post_indices = _ar_design_with_indices(
        scaled_post, order, history=scaled_pre
    )
    if pre_target.size == 0 or post_target.size == 0:
        raise ValueError("No finite conditional-AR rows in one or both segments")
    if len(retained_post_indices) != post_design.shape[0]:
        raise ValueError("Intervention/design row alignment failed")

    pooled_design = np.vstack([pre_design, post_design])
    pooled_target = np.concatenate([pre_target, post_target])
    base_dimension = pooled_design.shape[1]
    base_prior_mean = np.concatenate([
        [model_prior.intercept_mean],
        np.full(order, model_prior.lag_mean, dtype=float),
    ])
    base_prior_precision = np.concatenate([
        [model_prior.intercept_precision],
        np.full(order, model_prior.lag_precision, dtype=float),
    ])
    base_gram = pooled_design.T @ pooled_design
    base_target_cross = pooled_design.T @ pooled_target
    target_square_sum = float(pooled_target @ pooled_target)
    log_h0, h0_mean, h0_alpha, h0_beta = (
        _bayesian_regression_log_marginal_from_sufficient_statistics(
            base_gram,
            base_target_cross,
            target_square_sum,
            pooled_target.size,
            prior_mean=base_prior_mean,
            prior_precision=base_prior_precision,
            alpha=model_prior.alpha,
            beta=model_prior.beta,
        )
    )
    posterior_h0 = (
        {"n_rows": int(pooled_target.size)}
        if posterior_detail == "none"
        else _posterior_summary(
            h0_mean, h0_alpha, h0_beta, pooled_target.size
        )
    )

    candidate_log_marginals: list[float] = []
    candidate_log_weights: list[float] = []
    candidate_states: list[tuple[
        str, int, np.ndarray, np.ndarray, float, float,
    ]] = []
    for onset_offset in onset_offsets:
        for shape in shapes:
            full_post_basis = _intervention_basis(
                post.size,
                shape=str(shape),
                onset_offset=int(onset_offset),
                half_life=half_life,
            )
            post_basis = full_post_basis[retained_post_indices]
            intervention_dimension = post_basis.shape[1]
            candidate_prior_mean = np.concatenate([
                base_prior_mean, np.zeros(intervention_dimension),
            ])
            candidate_prior_precision = np.concatenate([
                base_prior_precision,
                np.full(intervention_dimension, intervention_precision),
            ])
            base_intervention_cross = post_design.T @ post_basis
            candidate_gram = np.empty(
                (
                    base_dimension + intervention_dimension,
                    base_dimension + intervention_dimension,
                ),
                dtype=float,
            )
            candidate_gram[:base_dimension, :base_dimension] = base_gram
            candidate_gram[:base_dimension, base_dimension:] = (
                base_intervention_cross
            )
            candidate_gram[base_dimension:, :base_dimension] = (
                base_intervention_cross.T
            )
            candidate_gram[base_dimension:, base_dimension:] = (
                post_basis.T @ post_basis
            )
            candidate_target_cross = np.concatenate([
                base_target_cross,
                post_basis.T @ post_target,
            ])
            log_marginal, mean, alpha_n, beta_n = (
                _bayesian_regression_log_marginal_from_sufficient_statistics(
                    candidate_gram,
                    candidate_target_cross,
                    target_square_sum,
                    pooled_target.size,
                    prior_mean=candidate_prior_mean,
                    prior_precision=candidate_prior_precision,
                    alpha=model_prior.alpha,
                    beta=model_prior.beta,
                )
            )
            candidate_log_marginals.append(float(log_marginal))
            candidate_log_weights.append(
                float(-onset_prior_decay * onset_offset)
            )
            if posterior_detail != "none":
                candidate_states.append((
                    str(shape), int(onset_offset), full_post_basis,
                    mean, alpha_n, beta_n,
                ))

    log_marginals = np.asarray(candidate_log_marginals, dtype=float)
    log_weights = np.asarray(candidate_log_weights, dtype=float)
    log_h1 = (
        _logsumexp(log_marginals + log_weights)
        - _logsumexp(log_weights)
    )
    candidates: list[dict[str, object]] = []
    posterior_map: dict[str, object] | None = None
    if posterior_detail != "none":
        posterior_log_normalizer = _logsumexp(log_marginals + log_weights)
        probabilities = np.exp(
            log_marginals + log_weights - posterior_log_normalizer
        )
        map_index = int(np.argmax(probabilities))
        for index, (
            shape, onset_offset, full_post_basis, mean, alpha_n, beta_n,
        ) in enumerate(candidate_states):
            candidate: dict[str, object] = {
                "shape": shape,
                "onset_offset": onset_offset,
                "log_prior_weight": float(log_weights[index]),
                "log_marginal": float(log_marginals[index]),
                "posterior_model_probability": float(probabilities[index]),
            }
            if posterior_detail == "full" or index == map_index:
                base_mean = mean[:base_dimension]
                effect_mean = mean[base_dimension:]
                final_effect = float(full_post_basis[-1] @ effect_mean)
                initial_effect = float(
                    full_post_basis[onset_offset] @ effect_mean
                )
                denominator = 1.0 - float(np.sum(base_mean[1:]))
                post_long_run_mean = (
                    float((base_mean[0] + final_effect) / denominator)
                    if abs(denominator) > 1e-10 else None
                )
                candidate.update({
                    "pre_rows": int(pre_target.size),
                    "post_rows": int(post_target.size),
                    "base_coefficient_mean": (
                        base_mean.astype(float).tolist()
                    ),
                    "intervention_coefficient_mean": (
                        effect_mean.astype(float).tolist()
                    ),
                    "initial_effect_mean": initial_effect,
                    "final_effect_mean": final_effect,
                    "innovation_variance_mean": (
                        float(beta_n / (alpha_n - 1.0))
                        if alpha_n > 1.0 else None
                    ),
                    "spectral_radius_at_mean": (
                        _ar_spectral_radius_from_mean(base_mean)
                    ),
                    "post_long_run_mean_at_end": post_long_run_mean,
                    "alpha": alpha_n,
                    "beta": beta_n,
                })
            candidates.append(candidate)
        candidates.sort(
            key=lambda item: float(item["posterior_model_probability"]),
            reverse=True,
        )
        posterior_map = candidates[0]

    return {
        "schema_version": 1,
        "hypothesis": "shared_ar_vs_model_averaged_intervention_response",
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
        "intervention_precision": float(intervention_precision),
        "onset_prior_decay": float(onset_prior_decay),
        "half_life": float(half_life),
        "posterior_h0": posterior_h0,
        "posterior_models": candidates,
        "posterior_map": posterior_map,
    }

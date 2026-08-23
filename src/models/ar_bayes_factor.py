"""Bayesian shared-vs-separate autoregressive model comparison.

The null hypothesis uses one AR parameter set for the pre/post segments.  The
alternative gives each segment an independent parameter set drawn from the
same proper Normal-Inverse-Gamma prior.  The conditional AR likelihood is a
Bayesian linear regression, so all marginal likelihoods are analytic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import lgamma, log, pi
from typing import Sequence

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
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_precision = np.asarray(prior_precision, dtype=float)
    if design.ndim != 2 or target.ndim != 1:
        raise ValueError("design must be 2-D and target must be 1-D")
    if design.shape[0] != target.size:
        raise ValueError("design and target row counts differ")
    if target.size == 0:
        raise ValueError("AR marginal likelihood requires at least one row")
    if prior_mean.shape != (design.shape[1],):
        raise ValueError("prior_mean dimension does not match design")
    if prior_precision.shape != (design.shape[1],):
        raise ValueError("prior_precision dimension does not match design")
    if not np.all(np.isfinite(prior_precision)) or np.any(prior_precision <= 0):
        raise ValueError("prior_precision must be finite and positive")
    if alpha <= 0 or beta <= 0:
        raise ValueError("inverse-gamma parameters must be positive")

    prior_precision_matrix = np.diag(prior_precision)
    posterior_precision = prior_precision_matrix + design.T @ design
    posterior_rhs = prior_precision_matrix @ prior_mean + design.T @ target
    posterior_mean = np.linalg.solve(posterior_precision, posterior_rhs)
    alpha_n = alpha + 0.5 * target.size
    beta_n = beta + 0.5 * (
        float(target @ target)
        + float(prior_mean @ prior_precision_matrix @ prior_mean)
        - float(posterior_mean @ posterior_precision @ posterior_mean)
    )
    beta_n = max(float(beta_n), 1e-12)

    prior_sign, prior_logdet = np.linalg.slogdet(prior_precision_matrix)
    posterior_sign, posterior_logdet = np.linalg.slogdet(posterior_precision)
    if prior_sign <= 0 or posterior_sign <= 0:
        raise ValueError("AR prior/posterior precision must be positive definite")

    log_marginal = (
        -0.5 * target.size * log(2.0 * pi)
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
        return step[:, None]
    if shape == "ramp":
        return ramp[:, None]
    if shape == "exp_rise":
        return rise[:, None]
    if shape == "exp_decay":
        return decay[:, None]
    return np.column_stack([step, ramp])


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
    # Build the exact retained-target indices without compressing time gaps.
    combined = np.concatenate([
        scaled_pre[-order:] if order else np.empty(0), scaled_post,
    ])
    retained_post_indices: list[int] = []
    for index in range(order, combined.size):
        lags = (
            combined[index - np.arange(1, order + 1)]
            if order else np.empty(0, dtype=float)
        )
        if np.isfinite(combined[index]) and np.all(np.isfinite(lags)):
            retained_post_indices.append(index - order)
    if len(retained_post_indices) != post_design.shape[0]:
        raise ValueError("Intervention/design row alignment failed")

    pooled_design = np.vstack([pre_design, post_design])
    pooled_target = np.concatenate([pre_target, post_target])
    log_h0, posterior_h0 = _bayesian_ar_log_marginal(
        pooled_design, pooled_target, model_prior
    )
    base_dimension = pooled_design.shape[1]
    base_prior_mean = np.concatenate([
        [model_prior.intercept_mean],
        np.full(order, model_prior.lag_mean, dtype=float),
    ])
    base_prior_precision = np.concatenate([
        [model_prior.intercept_precision],
        np.full(order, model_prior.lag_precision, dtype=float),
    ])

    candidates: list[dict[str, object]] = []
    for onset_offset in onset_offsets:
        for shape in shapes:
            full_post_basis = _intervention_basis(
                post.size,
                shape=str(shape),
                onset_offset=int(onset_offset),
                half_life=half_life,
            )
            post_basis = full_post_basis[retained_post_indices]
            intervention = np.vstack([
                np.zeros((pre_design.shape[0], post_basis.shape[1])),
                post_basis,
            ])
            candidate_design = np.column_stack([pooled_design, intervention])
            candidate_prior_mean = np.concatenate([
                base_prior_mean, np.zeros(post_basis.shape[1]),
            ])
            candidate_prior_precision = np.concatenate([
                base_prior_precision,
                np.full(post_basis.shape[1], intervention_precision),
            ])
            log_marginal, mean, alpha_n, beta_n = (
                _bayesian_regression_log_marginal(
                    candidate_design,
                    pooled_target,
                    prior_mean=candidate_prior_mean,
                    prior_precision=candidate_prior_precision,
                    alpha=model_prior.alpha,
                    beta=model_prior.beta,
                )
            )
            base_mean = mean[:base_dimension]
            effect_mean = mean[base_dimension:]
            final_effect = float(full_post_basis[-1] @ effect_mean)
            initial_effect = float(
                full_post_basis[int(onset_offset)] @ effect_mean
            )
            denominator = 1.0 - float(np.sum(base_mean[1:]))
            post_long_run_mean = (
                float((base_mean[0] + final_effect) / denominator)
                if abs(denominator) > 1e-10 else None
            )
            candidates.append({
                "shape": str(shape),
                "onset_offset": int(onset_offset),
                "log_prior_weight": float(-onset_prior_decay * onset_offset),
                "pre_rows": int(pre_target.size),
                "post_rows": int(post_target.size),
                "log_marginal": float(log_marginal),
                "base_coefficient_mean": base_mean.astype(float).tolist(),
                "intervention_coefficient_mean": effect_mean.astype(float).tolist(),
                "initial_effect_mean": initial_effect,
                "final_effect_mean": final_effect,
                "innovation_variance_mean": (
                    float(beta_n / (alpha_n - 1.0)) if alpha_n > 1.0 else None
                ),
                "spectral_radius_at_mean": _ar_spectral_radius_from_mean(
                    base_mean
                ),
                "post_long_run_mean_at_end": post_long_run_mean,
                "alpha": alpha_n,
                "beta": beta_n,
            })

    candidate_log_marginals = np.asarray([
        float(candidate["log_marginal"]) for candidate in candidates
    ])
    candidate_log_weights = np.asarray([
        float(candidate["log_prior_weight"]) for candidate in candidates
    ])
    log_h1 = (
        _logsumexp(candidate_log_marginals + candidate_log_weights)
        - _logsumexp(candidate_log_weights)
    )
    posterior_log_normalizer = _logsumexp(
        candidate_log_marginals + candidate_log_weights
    )
    probabilities = np.exp(
        candidate_log_marginals + candidate_log_weights
        - posterior_log_normalizer
    )
    for candidate, probability in zip(candidates, probabilities, strict=True):
        candidate["posterior_model_probability"] = float(probability)
    candidates.sort(
        key=lambda item: float(item["posterior_model_probability"]),
        reverse=True,
    )

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
        "posterior_map": candidates[0],
    }

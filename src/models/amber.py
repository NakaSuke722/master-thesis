"""Bayesian residual-shift root cause analysis.

This module ranks metrics (and optionally services) by how strongly their
abnormal-period prediction residuals support a distributional change from the
normal-period residual distribution.

Only NumPy and pandas are required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from models.ar_bayes_factor import (
    ARBayesFactorPrior,
    ARRegimeShiftPrior,
    ar_change_bayes_factor,
    ar_intervention_bayes_factor,
    ar_intercept_shift_bayes_factor,
    ar_shrinkage_regime_bayes_factor,
)


@dataclass(frozen=True)
class NIG:
    """Normal-Inverse-Gamma parameters.

    mu | sigma^2 ~ Normal(m, sigma^2 / kappa)
    sigma^2      ~ InvGamma(alpha, beta)
    """

    m: float
    kappa: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")


def _nig_update(prior: NIG, x: np.ndarray) -> NIG:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return prior

    mean = float(np.mean(x))
    ss = float(np.sum((x - mean) ** 2))
    kappa_n = prior.kappa + n
    m_n = (prior.kappa * prior.m + n * mean) / kappa_n
    alpha_n = prior.alpha + 0.5 * n
    beta_n = (
        prior.beta
        + 0.5 * ss
        + 0.5 * (prior.kappa * n / kappa_n) * (mean - prior.m) ** 2
    )
    return NIG(m=m_n, kappa=kappa_n, alpha=alpha_n, beta=max(beta_n, 1e-12))


def _nig_log_marginal(x: np.ndarray, prior: NIG) -> float:
    """Log p(x | prior), integrating out mean and variance."""
    from math import lgamma, log, pi

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return 0.0

    post = _nig_update(prior, x)
    return (
        lgamma(post.alpha)
        - lgamma(prior.alpha)
        + prior.alpha * log(prior.beta)
        - post.alpha * log(post.beta)
        + 0.5 * (log(prior.kappa) - log(post.kappa))
        - 0.5 * n * log(pi)
    )


def _build_ar(y: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if y.size <= order:
        return np.empty((0, order + 1)), np.empty(0)
    rows = y.size - order
    X = np.ones((rows, order + 1), dtype=float)
    for r, t in enumerate(range(order, y.size)):
        X[r, 1:] = y[t - np.arange(1, order + 1)]
    return X, y[order:]


def _ridge_ar_fit(y: np.ndarray, order: int, ridge: float) -> np.ndarray:
    X, target = _build_ar(y, order)
    if target.size == 0:
        raise ValueError(f"Need more than ar_order={order} normal observations.")
    # Solve ridge regression through an augmented least-squares system.
    # This is more stable than solving the normal equations, especially for
    # memory metrics whose raw values can be very large.
    penalty = np.eye(X.shape[1], dtype=float)
    penalty[0, 0] = 1e-3  # barely penalize intercept
    X_aug = np.vstack([X, np.sqrt(ridge) * penalty])
    y_aug = np.concatenate([target, np.zeros(X.shape[1], dtype=float)])
    coef, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    return coef


def _ar_residuals(y: np.ndarray, coef: np.ndarray, order: int) -> np.ndarray:
    X, target = _build_ar(y, order)
    if target.size == 0:
        return np.empty(0)
    return target - X @ coef


def _counterfactual_ar_forecast(
    history: np.ndarray,
    coef: np.ndarray,
    order: int,
    steps: int,
    bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, int]:
    """Recursively forecast a no-fault baseline without abnormal lag inputs.

    The initial lags come from the end of the normal period.  Every later lag
    is a previous model prediction, never an observed abnormal value.  Bounds
    learned only from the normal period may be supplied to keep an unstable
    one-step AR fit from exploding during a long recursive forecast.
    """
    history = np.asarray(history, dtype=float)
    coef = np.asarray(coef, dtype=float)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if coef.size != order + 1:
        raise ValueError(f"Expected {order + 1} AR coefficients, got {coef.size}")
    if order > 0 and history.size < order:
        raise ValueError(f"Need at least ar_order={order} history observations")
    if bounds is not None and bounds[0] > bounds[1]:
        raise ValueError("counterfactual lower bound must not exceed upper bound")

    state = history[-order:].astype(float).tolist() if order else []
    predictions = np.empty(steps, dtype=float)
    clipped_count = 0
    for index in range(steps):
        if order:
            lags = np.asarray(state[-order:][::-1], dtype=float)
            prediction = float(coef[0] + np.dot(coef[1:], lags))
        else:
            prediction = float(coef[0])
        if bounds is not None:
            bounded_prediction = float(np.clip(prediction, bounds[0], bounds[1]))
            clipped_count += int(bounded_prediction != prediction)
            prediction = bounded_prediction
        predictions[index] = prediction
        if order:
            state.append(prediction)
            if len(state) > order:
                del state[:-order]
    return predictions, clipped_count


def _ar_spectral_radius(coef: np.ndarray) -> float:
    """Return the largest companion-root magnitude for an AR coefficient vector."""
    coef = np.asarray(coef, dtype=float)
    order = max(0, coef.size - 1)
    if order == 0:
        return 0.0
    roots = np.roots(np.concatenate(([1.0], -coef[1:])))
    return float(np.max(np.abs(roots))) if roots.size else 0.0


def _project_ar_stationary(
    coef: np.ndarray,
    normal_mean: float,
    radius: float,
) -> tuple[np.ndarray, float, float, bool]:
    """Project companion roots inside ``radius`` and preserve normal mean.

    Ridge fitting remains normal-only and unconstrained.  If its companion
    roots are outside the requested disk, their angles are retained and only
    their magnitudes are projected.  The intercept is then adjusted so the
    projected AR has the observed normal-window mean as its fixed point.
    """
    coef = np.asarray(coef, dtype=float)
    if not 0.0 < radius < 1.0:
        raise ValueError("stationarity_radius must be between zero and one")
    order = max(0, coef.size - 1)
    before = _ar_spectral_radius(coef)
    if order == 0 or before <= radius:
        return coef.copy(), before, before, False

    roots = np.roots(np.concatenate(([1.0], -coef[1:])))
    projected = np.asarray([
        root if abs(root) <= radius else root * (radius / abs(root))
        for root in roots
    ])
    polynomial = np.real_if_close(np.poly(projected), tol=1000)
    if np.iscomplexobj(polynomial):
        raise ValueError("Stationarity projection produced complex AR coefficients")
    ar_coef = -np.asarray(polynomial[1:], dtype=float)
    constrained = np.empty_like(coef)
    constrained[1:] = ar_coef
    constrained[0] = float(normal_mean) * (1.0 - float(np.sum(ar_coef)))
    after = _ar_spectral_radius(constrained)
    return constrained, before, after, True


def _ar_forecast_uncertainty_multipliers(
    coef: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Standard-deviation multipliers for h-step AR forecast errors.

    If ``psi_j`` denotes the MA(infinity) impulse response, then the forecast
    error variance at horizon h is ``sigma^2 * sum(psi_j^2, j=0..h-1)``.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    psi = _ar_impulse_response(coef, steps)
    return np.sqrt(np.cumsum(psi ** 2))


def _ar_impulse_response(coef: np.ndarray, steps: int) -> np.ndarray:
    """Return ``psi_0, ..., psi_(steps-1)`` for an AR model."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    coef = np.asarray(coef, dtype=float)
    phi = coef[1:]
    order = phi.size
    psi = np.zeros(steps, dtype=float)
    if steps:
        psi[0] = 1.0
    for index in range(1, steps):
        psi[index] = sum(
            phi[lag - 1] * psi[index - lag]
            for lag in range(1, min(order, index) + 1)
        )
    return psi


def _ar_forecast_error_cholesky(
    coef: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Return the unit-innovation Cholesky factor of AR forecast errors.

    For innovation variance ``sigma**2``, the complete forecast-error
    covariance is ``Sigma_H = sigma**2 * L @ L.T``.  ``L`` is lower-triangular
    Toeplitz with the AR impulse responses on its diagonals.  This helper is
    primarily useful for verification; scoring uses an algebraically exact
    O(Hp) inverse filter instead of materializing an H-by-H matrix per metric.
    """
    psi = _ar_impulse_response(coef, steps)
    factor = np.zeros((steps, steps), dtype=float)
    for row in range(steps):
        factor[row, :row + 1] = psi[row::-1]
    return factor


def _whiten_ar_forecast_errors(
    errors: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    """Apply the exact inverse-Cholesky transform to AR forecast errors.

    If ``e = L @ innovation`` under the fitted AR model, this returns
    ``L**-1 @ e``.  The AR inverse-filter form avoids an O(H^3) dense solve and
    is mathematically identical because ``L`` is the impulse-response factor.
    """
    errors = np.asarray(errors, dtype=float)
    coef = np.asarray(coef, dtype=float)
    phi = coef[1:]
    order = phi.size
    whitened = errors.copy()
    for index in range(errors.size):
        whitened[index] -= sum(
            phi[lag - 1] * errors[index - lag]
            for lag in range(1, min(order, index) + 1)
        )
    return whitened


def _service_name(metric: str) -> str:
    """Infer service from names such as catalogue_cpu or catalogue_latency-90."""
    for suffix in ("_latency-50", "_latency-90", "_latency-95", "_latency-99",
                   "_latency", "_cpu", "_mem", "_memory"):
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return metric.rsplit("_", 1)[0] if "_" in metric else metric

def _gaussian_mle_loglik(
    x: np.ndarray,
    variance_floor: float = 1e-12,
) -> float:
    """Gaussian maximized log-likelihood."""

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = x.size
    if n == 0:
        return 0.0

    mean = float(np.mean(x))

    variance = float(
        np.mean((x - mean) ** 2)
    )

    variance = max(
        variance,
        variance_floor,
    )

    return float(
        -0.5
        * n
        * (
            np.log(2.0 * np.pi * variance)
            + 1.0
        )
    )


def _gaussian_glrt_score(
    z_normal: np.ndarray,
    z_abnormal: np.ndarray,
) -> float:
    """GLRT statistic for pooled vs separate Gaussian distributions."""

    pooled = np.concatenate([
        z_normal,
        z_abnormal,
    ])

    log_h0 = _gaussian_mle_loglik(
        pooled
    )

    log_h1 = (
        _gaussian_mle_loglik(z_normal)
        + _gaussian_mle_loglik(z_abnormal)
    )

    return float(
        2.0 * (log_h1 - log_h0)
    )


class AMBER:
    """RCA using Bayesian model selection on time-series changes.

    The standard modes compare Gaussian distributions after raw/AR
    residualization.  ``ar_model`` scoring modes instead make AR part of the
    hypothesis: every AR parameter, only the intercept, a structured response,
    or a sparse subset of AR regime parameters can differ.

    The RCA score is
        log BF= log m(z_normal) + log m(z_abnormal) - log m(z_normal and z_abnormal),

    where m(.) is the NIG-integrated marginal likelihood.
    """

    def __init__(
        self,
        ar_order: int = 3,
        ridge: float = 1e-3,
        min_scale: float = 1e-6,
        relative_scale_floor: float = 1e-3,
        winsor_quantile: float | None = 0.01,
        aggregate: Literal["metric", "service"] = "metric",
        service_aggregation: Literal["max", "mean_top3", "logsumexp"] = "max",
        prior: NIG | None = None,
        residualization: Literal[
            "ar", "counterfactual_ar", "raw", "ar_model",
        ] = "ar",
        scoring: Literal[
            "bayes_factor", "glrt", "ar_bayes_factor",
            "ar_intercept_bayes_factor", "ar_intervention_bayes_factor",
            "bsrc_ar_bayes_factor",
        ] = "bayes_factor",
        ar_stationarity: Literal["none", "root_projection"] = "none",
        stationarity_radius: float = 0.98,
        counterfactual_bounds: Literal["normal_range", "none"] = "normal_range",
        horizon_aware_uncertainty: bool = False,
        forecast_error_covariance: Literal["diagonal", "full"] = "diagonal",
        ar_bayes_prior: ARBayesFactorPrior | None = None,
        ar_regime_shift_prior: ARRegimeShiftPrior | None = None,
        ar_intervention_shapes: Sequence[str] = (
            "step", "ramp", "exp_rise", "exp_decay", "step_ramp",
        ),
        ar_intervention_onset_offsets: Sequence[int] = (0,),
        ar_intervention_half_life: float = 10.0,
        ar_intervention_precision: float = 0.1,
        ar_intervention_onset_prior_decay: float = 0.0,
        ar_null_calibration_fractions: Sequence[float] = (),
        ar_null_calibration_quantile: float = 0.9,
        ar_null_calibration_mode: Literal[
            "none", "subtract", "per_row_excess",
        ] = "subtract",
    ) -> None:
        if ar_order < 0:
            raise ValueError("ar_order must be non-negative")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")

        if min_scale <= 0:
            raise ValueError("min_scale must be positive")

        if relative_scale_floor < 0:
            raise ValueError("relative_scale_floor must be non-negative")

        if aggregate not in {"metric", "service"}:
            raise ValueError(f"Unknown aggregate={aggregate}")

        if service_aggregation not in {"max","mean_top3","logsumexp",}:
            raise ValueError("Unknown service_aggregation=" f"{service_aggregation}")

        if residualization not in {"ar", "counterfactual_ar", "raw", "ar_model"}:
            raise ValueError(f"Unknown residualization={residualization}")

        ar_model_scores = {
            "ar_bayes_factor", "ar_intercept_bayes_factor",
            "ar_intervention_bayes_factor", "bsrc_ar_bayes_factor",
        }
        if scoring not in {"bayes_factor", "glrt", *ar_model_scores}:
            raise ValueError(f"Unknown scoring={scoring}")

        if (scoring in ar_model_scores) != (residualization == "ar_model"):
            raise ValueError(
                "AR-model Bayes-factor scoring and ar_model residualization "
                "must be selected together"
            )

        if scoring in ar_model_scores and winsor_quantile is not None:
            raise ValueError(
                "AR-model Bayes factors require winsor_quantile=None so the AR "
                "likelihood is evaluated without clipping"
            )

        if ar_stationarity not in {"none", "root_projection"}:
            raise ValueError(f"Unknown ar_stationarity={ar_stationarity}")

        if not 0.0 < stationarity_radius < 1.0:
            raise ValueError("stationarity_radius must be between zero and one")

        if counterfactual_bounds not in {"normal_range", "none"}:
            raise ValueError(f"Unknown counterfactual_bounds={counterfactual_bounds}")

        if horizon_aware_uncertainty and residualization != "counterfactual_ar":
            raise ValueError(
                "horizon_aware_uncertainty requires counterfactual_ar residualization"
            )

        if forecast_error_covariance not in {"diagonal", "full"}:
            raise ValueError(
                "Unknown forecast_error_covariance="
                f"{forecast_error_covariance}"
            )

        if forecast_error_covariance == "full" and not horizon_aware_uncertainty:
            raise ValueError(
                "full forecast_error_covariance requires "
                "horizon_aware_uncertainty"
            )

        intervention_shapes = tuple(str(value) for value in ar_intervention_shapes)
        intervention_onsets = tuple(
            int(value) for value in ar_intervention_onset_offsets
        )
        allowed_intervention_shapes = {
            "step", "ramp", "exp_rise", "exp_decay", "step_ramp",
        }
        if not intervention_shapes:
            raise ValueError("ar_intervention_shapes must not be empty")
        unknown_shapes = set(intervention_shapes) - allowed_intervention_shapes
        if unknown_shapes:
            raise ValueError(
                "Unknown ar_intervention_shapes="
                + ",".join(sorted(unknown_shapes))
            )
        if not intervention_onsets or any(
            value < 0 for value in intervention_onsets
        ):
            raise ValueError(
                "ar_intervention_onset_offsets must contain non-negative values"
            )
        if ar_intervention_half_life <= 0:
            raise ValueError("ar_intervention_half_life must be positive")
        if ar_intervention_precision <= 0:
            raise ValueError("ar_intervention_precision must be positive")
        if ar_intervention_onset_prior_decay < 0:
            raise ValueError(
                "ar_intervention_onset_prior_decay must be non-negative"
            )
        if not 0.0 <= ar_null_calibration_quantile <= 1.0:
            raise ValueError("ar_null_calibration_quantile must be in [0, 1]")
        if any(not 0.0 < value < 1.0 for value in ar_null_calibration_fractions):
            raise ValueError("ar_null_calibration_fractions must lie in (0, 1)")
        if ar_null_calibration_mode not in {
            "none", "subtract", "per_row_excess",
        }:
            raise ValueError(
                f"Unknown ar_null_calibration_mode={ar_null_calibration_mode}"
            )
        
        self.ar_order = ar_order
        self.ridge = ridge
        self.min_scale = min_scale
        self.relative_scale_floor = relative_scale_floor
        self.winsor_quantile = winsor_quantile
        self.aggregate = aggregate
        self.prior = prior or NIG(
            m=0.0,
            kappa=1e-3,
            alpha=2.0,
            beta=1.0,
        )
        self.service_aggregation = service_aggregation
        self.metric_result_: pd.DataFrame | None = None
        self.residualization = residualization
        self.scoring = scoring
        self.ar_stationarity = ar_stationarity
        self.stationarity_radius = stationarity_radius
        self.counterfactual_bounds = counterfactual_bounds
        self.horizon_aware_uncertainty = horizon_aware_uncertainty
        self.forecast_error_covariance = forecast_error_covariance
        self.ar_bayes_prior = ar_bayes_prior or ARBayesFactorPrior()
        self.ar_regime_shift_prior = (
            ar_regime_shift_prior or ARRegimeShiftPrior(
                inclusion_probability=min(0.5, 1.0 / (ar_order + 1))
            )
        )
        self.ar_intervention_shapes = intervention_shapes
        self.ar_intervention_onset_offsets = intervention_onsets
        self.ar_intervention_half_life = float(ar_intervention_half_life)
        self.ar_intervention_precision = float(ar_intervention_precision)
        self.ar_intervention_onset_prior_decay = float(
            ar_intervention_onset_prior_decay
        )
        self.ar_null_calibration_fractions = tuple(
            float(value) for value in ar_null_calibration_fractions
        )
        self.ar_null_calibration_quantile = float(ar_null_calibration_quantile)
        self.ar_null_calibration_mode = ar_null_calibration_mode
        self.result_: pd.DataFrame | None = None
        # JSON-serializable, per-metric observations for post-hoc analysis.
        # Populated by fit_predict; deliberately separate from metric_result_
        # so the ranking algorithm is not affected by diagnostic collection.
        self.diagnostics_: dict[str, object] | None = None

    @staticmethod
    def _numeric_common_columns(normal: pd.DataFrame, abnormal: pd.DataFrame) -> list[str]:
        common = [c for c in normal.columns if c in abnormal.columns]
        return [
            c for c in common
            if pd.api.types.is_numeric_dtype(normal[c])
            and pd.api.types.is_numeric_dtype(abnormal[c])
        ]

    @staticmethod
    def _finite(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]


    def _prepare_series(
        self,
        normal_y: np.ndarray,
        abnormal_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        normal_y = self._finite(normal_y)
        abnormal_y = self._finite(abnormal_y)

        if (
            normal_y.size
            and self.winsor_quantile is not None
        ):
            q = self.winsor_quantile

            if not 0 <= q < 0.5:
                raise ValueError(
                    "winsor_quantile must be in [0, 0.5)"
                )

            # Learn clipping thresholds from the normal period only.
            lo, hi = np.quantile(
                normal_y,
                [q, 1 - q],
            )

            normal_y = np.clip(
                normal_y,
                lo,
                hi,
            )

            abnormal_y = np.clip(
                abnormal_y,
                lo,
                hi,
            )

        return normal_y, abnormal_y
    
    def _score_metric(self, normal_y: np.ndarray, abnormal_y: np.ndarray) -> dict[str, object]:
        if self.scoring in {
            "ar_bayes_factor", "ar_intercept_bayes_factor",
            "ar_intervention_bayes_factor", "bsrc_ar_bayes_factor",
        }:
            normal_y = np.asarray(normal_y, dtype=float)
            abnormal_y = np.asarray(abnormal_y, dtype=float)
            if (
                np.isfinite(normal_y).sum() <= self.ar_order + 3
                or np.isfinite(abnormal_y).sum() <= self.ar_order
            ):
                return {
                    "score": -np.inf,
                    "evidence_weight": 0.0,
                    "normal_scale": np.nan,
                    "abnormal_mean_z": np.nan,
                    "abnormal_sd_z": np.nan,
                    "ar_coefficients": [],
                    "raw_normal": normal_y.tolist(),
                    "raw_abnormal": abnormal_y.tolist(),
                    "ar_prediction_normal": [],
                    "ar_prediction_abnormal": [],
                    "ar_residual_normal": [],
                    "ar_residual_abnormal": [],
                    "standardized_residual_normal": [],
                    "standardized_residual_abnormal": [],
                }
            try:
                if self.scoring == "ar_bayes_factor":
                    comparison = ar_change_bayes_factor(
                        normal_y, abnormal_y, order=self.ar_order,
                        prior=self.ar_bayes_prior, min_scale=self.min_scale,
                    )
                elif self.scoring == "ar_intercept_bayes_factor":
                    comparison = ar_intercept_shift_bayes_factor(
                        normal_y, abnormal_y, order=self.ar_order,
                        prior=self.ar_bayes_prior, min_scale=self.min_scale,
                    )
                elif self.scoring == "bsrc_ar_bayes_factor":
                    comparison = ar_shrinkage_regime_bayes_factor(
                        normal_y,
                        abnormal_y,
                        order=self.ar_order,
                        prior=self.ar_bayes_prior,
                        regime_prior=self.ar_regime_shift_prior,
                        min_scale=self.min_scale,
                        posterior_detail="map",
                    )
                else:
                    comparison = ar_intervention_bayes_factor(
                        normal_y,
                        abnormal_y,
                        order=self.ar_order,
                        prior=self.ar_bayes_prior,
                        shapes=self.ar_intervention_shapes,
                        onset_offsets=self.ar_intervention_onset_offsets,
                        half_life=self.ar_intervention_half_life,
                        intervention_precision=self.ar_intervention_precision,
                        onset_prior_decay=self.ar_intervention_onset_prior_decay,
                        min_scale=self.min_scale,
                        posterior_detail="map",
                    )
                null_scores: list[float] = []
                null_score_rates: list[float] = []
                if self.scoring == "ar_intervention_bayes_factor":
                    for fraction in self.ar_null_calibration_fractions:
                        split = int(round(normal_y.size * fraction))
                        if (
                            split <= self.ar_order + 3
                            or normal_y.size - split <= self.ar_order
                        ):
                            continue
                        null_comparison = ar_intervention_bayes_factor(
                            normal_y[:split],
                            normal_y[split:],
                            order=self.ar_order,
                            prior=self.ar_bayes_prior,
                            shapes=self.ar_intervention_shapes,
                            onset_offsets=self.ar_intervention_onset_offsets,
                            half_life=self.ar_intervention_half_life,
                            intervention_precision=self.ar_intervention_precision,
                            onset_prior_decay=self.ar_intervention_onset_prior_decay,
                            min_scale=self.min_scale,
                            posterior_detail="none",
                        )
                        null_scores.append(
                            float(null_comparison["log_bayes_factor"])
                        )
                        null_rows = int(
                            null_comparison["posterior_h0"]["n_rows"]
                        )
                        null_score_rates.append(
                            float(null_comparison["log_bayes_factor"])
                            / null_rows
                        )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                return {
                    "score": -np.inf,
                    "evidence_weight": 0.0,
                    "normal_scale": np.nan,
                    "abnormal_mean_z": np.nan,
                    "abnormal_sd_z": np.nan,
                    "ar_coefficients": [],
                    "raw_normal": normal_y.tolist(),
                    "raw_abnormal": abnormal_y.tolist(),
                    "ar_prediction_normal": [],
                    "ar_prediction_abnormal": [],
                    "ar_residual_normal": [],
                    "ar_residual_abnormal": [],
                    "standardized_residual_normal": [],
                    "standardized_residual_abnormal": [],
                }
            raw_score = float(comparison["log_bayes_factor"])
            observed_rows = int(comparison.get(
                "predictive_rows",
                comparison.get("posterior_h0", {}).get("n_rows", 1),
            ))
            raw_score_rate = raw_score / observed_rows
            if self.ar_null_calibration_mode == "per_row_excess":
                null_baseline = (
                    float(np.quantile(
                        null_score_rates, self.ar_null_calibration_quantile
                    ))
                    if null_score_rates else 0.0
                )
                score = raw_score_rate - null_baseline
            elif self.ar_null_calibration_mode == "subtract":
                null_baseline = (
                    float(np.quantile(
                        null_scores, self.ar_null_calibration_quantile
                    ))
                    if null_scores else 0.0
                )
                score = raw_score - null_baseline
            else:
                null_baseline = 0.0
                score = raw_score

            intervention_shape_posterior = None
            intervention_onset_posterior = None
            regime_changed_parameters = None
            regime_variance_ratio = None
            regime_variance_change_active = None
            regime_variance_change_probability = None
            regime_parameter_inclusion = None
            regime_map_probability = None
            if self.scoring == "ar_bayes_factor":
                pre = comparison["posterior_pre"]
                post = comparison["posterior_post"]
                shared = comparison["posterior_shared"]
                pre_coefficients = pre["coefficient_mean"]
                post_coefficients = post["coefficient_mean"]
                pre_variance = pre["innovation_variance_mean"]
                post_variance = post["innovation_variance_mean"]
                pre_radius = pre["spectral_radius_at_mean"]
                post_radius = post["spectral_radius_at_mean"]
                pre_long_run_mean = pre["long_run_mean_at_mean"]
                post_long_run_mean = post["long_run_mean_at_mean"]
                pre_rows = pre["n_rows"]
                post_rows = post["n_rows"]
                intercept_shift = (
                    float(post_coefficients[0] - pre_coefficients[0])
                )
                intervention_shape = None
                intervention_onset = None
                intervention_probability = None
            elif self.scoring == "ar_intercept_bayes_factor":
                shared = comparison["posterior_h0"]
                alternative = comparison["posterior_h1"]
                pre_coefficients = alternative["pre_coefficient_mean"]
                post_coefficients = alternative["post_coefficient_mean"]
                pre_variance = alternative["innovation_variance_mean"]
                post_variance = alternative["innovation_variance_mean"]
                pre_radius = alternative["spectral_radius_at_mean"]
                post_radius = alternative["spectral_radius_at_mean"]
                pre_long_run_mean = alternative["pre_long_run_mean_at_mean"]
                post_long_run_mean = alternative["post_long_run_mean_at_mean"]
                pre_rows = alternative["pre_rows"]
                post_rows = alternative["post_rows"]
                intercept_shift = alternative["intercept_shift_mean"]
                intervention_shape = None
                intervention_onset = None
                intervention_probability = None
            elif self.scoring == "bsrc_ar_bayes_factor":
                shared = comparison["posterior_h0"]
                alternative = comparison["posterior_map"]
                pre_coefficients = alternative["base_coefficient_mean"]
                post_coefficients = alternative["post_coefficient_mean"]
                pre_variance = alternative[
                    "pre_innovation_variance_mean"
                ]
                post_variance = alternative[
                    "post_innovation_variance_mean"
                ]
                pre_radius = alternative[
                    "pre_spectral_radius_at_mean"
                ]
                post_radius = alternative[
                    "post_spectral_radius_at_mean"
                ]
                pre_long_run_mean = alternative[
                    "pre_long_run_mean_at_mean"
                ]
                post_long_run_mean = alternative[
                    "post_long_run_mean_at_mean"
                ]
                pre_rows = alternative["pre_rows"]
                post_rows = alternative["post_rows"]
                intercept_shift = alternative[
                    "coefficient_change_mean"
                ][0]
                intervention_shape = None
                intervention_onset = None
                intervention_probability = None
                regime_changed_parameters = alternative[
                    "changed_parameters"
                ]
                regime_variance_ratio = alternative["variance_ratio"]
                regime_variance_change_active = alternative[
                    "variance_change_active"
                ]
                regime_variance_change_probability = comparison[
                    "posterior_variance_change_probability"
                ]
                regime_parameter_inclusion = comparison[
                    "parameter_change_inclusion_probability"
                ]
                regime_map_probability = alternative[
                    "posterior_model_probability"
                ]
            else:
                shared = comparison["posterior_h0"]
                alternative = comparison["posterior_map"]
                pre_coefficients = alternative["base_coefficient_mean"]
                post_coefficients = list(pre_coefficients)
                post_coefficients[0] += alternative["final_effect_mean"]
                pre_variance = alternative["innovation_variance_mean"]
                post_variance = alternative["innovation_variance_mean"]
                pre_radius = alternative["spectral_radius_at_mean"]
                post_radius = alternative["spectral_radius_at_mean"]
                denominator = 1.0 - float(np.sum(pre_coefficients[1:]))
                pre_long_run_mean = (
                    float(pre_coefficients[0] / denominator)
                    if abs(denominator) > 1e-10 else None
                )
                post_long_run_mean = alternative["post_long_run_mean_at_end"]
                pre_rows = alternative["pre_rows"]
                post_rows = alternative["post_rows"]
                intercept_shift = alternative["final_effect_mean"]
                intervention_shape = alternative["shape"]
                intervention_onset = alternative["onset_offset"]
                intervention_probability = alternative[
                    "posterior_model_probability"
                ]
                intervention_shape_posterior = {}
                intervention_onset_posterior = {}
                for candidate in comparison["posterior_models"]:
                    probability = float(
                        candidate["posterior_model_probability"]
                    )
                    shape = str(candidate["shape"])
                    onset = str(candidate["onset_offset"])
                    intervention_shape_posterior[shape] = (
                        intervention_shape_posterior.get(shape, 0.0)
                        + probability
                    )
                    intervention_onset_posterior[onset] = (
                        intervention_onset_posterior.get(onset, 0.0)
                        + probability
                    )
            return {
                "score": score,
                "raw_log_bayes_factor": raw_score,
                "raw_log_bayes_factor_per_row": raw_score_rate,
                "null_log_bayes_factors": null_scores,
                "null_log_bayes_factors_per_row": null_score_rates,
                "null_calibration_baseline": null_baseline,
                "null_calibration_mode": self.ar_null_calibration_mode,
                "ar_hypothesis": comparison["hypothesis"],
                "ar_intervention_map_shape": intervention_shape,
                "ar_intervention_map_onset_offset": intervention_onset,
                "ar_intervention_map_probability": intervention_probability,
                "ar_intervention_shape_posterior": (
                    intervention_shape_posterior
                ),
                "ar_intervention_onset_posterior": (
                    intervention_onset_posterior
                ),
                "ar_regime_map_changed_parameters": (
                    regime_changed_parameters
                ),
                "ar_regime_map_variance_ratio": regime_variance_ratio,
                "ar_regime_map_variance_change_active": (
                    regime_variance_change_active
                ),
                "ar_regime_variance_change_probability": (
                    regime_variance_change_probability
                ),
                "ar_regime_map_probability": regime_map_probability,
                "ar_regime_parameter_inclusion_probability": (
                    regime_parameter_inclusion
                ),
                "normal_scale": float(comparison["normalization"]["scale"]),
                "abnormal_mean_z": np.nan,
                "abnormal_sd_z": np.nan,
                "log_marginal_h0": float(comparison["log_marginal_h0"]),
                "log_marginal_h1": float(comparison["log_marginal_h1"]),
                "ar_coefficients": pre_coefficients,
                "ar_post_coefficients": post_coefficients,
                "ar_shared_coefficients": shared["coefficient_mean"],
                "ar_intercept_shift": intercept_shift,
                "ar_pre_innovation_variance": pre_variance,
                "ar_post_innovation_variance": post_variance,
                "ar_shared_innovation_variance": shared["innovation_variance_mean"],
                "ar_pre_spectral_radius": pre_radius,
                "ar_post_spectral_radius": post_radius,
                "ar_shared_spectral_radius": shared["spectral_radius_at_mean"],
                "ar_pre_long_run_mean": pre_long_run_mean,
                "ar_post_long_run_mean": post_long_run_mean,
                "ar_shared_long_run_mean": shared["long_run_mean_at_mean"],
                "ar_pre_rows": pre_rows,
                "ar_post_rows": post_rows,
                "raw_normal": normal_y.astype(float).tolist(),
                "raw_abnormal": abnormal_y.astype(float).tolist(),
                "ar_prediction_normal": [],
                "ar_prediction_abnormal": [],
                "ar_residual_normal": [],
                "ar_residual_abnormal": [],
                "standardized_residual_normal": [],
                "standardized_residual_abnormal": [],
            }

        normal_y, abnormal_y = self._prepare_series(
            normal_y,
            abnormal_y,
        )

        if (normal_y.size <= self.ar_order + 3
            or abnormal_y.size <= self.ar_order
        ):
            return {
                "score": -np.inf,
                "evidence_weight": 0.0,
                "normal_scale": np.nan,
                "abnormal_mean_z": np.nan,
                "abnormal_sd_z": np.nan,
                "ar_coefficients": [],
                "raw_normal": normal_y.tolist(),
                "raw_abnormal": abnormal_y.tolist(),
                "ar_prediction_normal": [],
                "ar_prediction_abnormal": [],
                "ar_residual_normal": [],
                "ar_residual_abnormal": [],
                "standardized_residual_normal": [],
                "standardized_residual_abnormal": [],
            }
        
        counterfactual_clipped_count: int | None = None
        spectral_radius_before: float | None = None
        spectral_radius_after: float | None = None
        stationarity_constrained: bool | None = None
        forecast_uncertainty_multiplier = np.empty(0, dtype=float)

        if self.residualization in {"ar", "counterfactual_ar"}:
            coef = _ridge_ar_fit(
                normal_y,
                self.ar_order,
                self.ridge,
            )

            spectral_radius_before = _ar_spectral_radius(coef)
            spectral_radius_after = spectral_radius_before
            stationarity_constrained = False
            if self.ar_stationarity == "root_projection":
                (
                    coef,
                    spectral_radius_before,
                    spectral_radius_after,
                    stationarity_constrained,
                ) = _project_ar_stationary(
                    coef,
                    normal_mean=float(np.mean(normal_y)),
                    radius=self.stationarity_radius,
                )

            r_n = _ar_residuals(
                normal_y,
                coef,
                self.ar_order,
            )

            if self.ar_order > 0:
                history = normal_y[-self.ar_order:]
            else:
                history = np.empty(
                    0,
                    dtype=float,
                )

            _, normal_target = _build_ar(normal_y, self.ar_order)
            normal_prediction = normal_target - r_n

            if self.residualization == "ar":
                history_and_abnormal = np.concatenate([
                    history,
                    abnormal_y,
                ])
                r_a = _ar_residuals(
                    history_and_abnormal,
                    coef,
                    self.ar_order,
                )
                _, abnormal_target = _build_ar(
                    history_and_abnormal,
                    self.ar_order,
                )
                abnormal_prediction = abnormal_target - r_a
            else:
                abnormal_prediction, counterfactual_clipped_count = (
                    _counterfactual_ar_forecast(
                        history,
                        coef,
                        self.ar_order,
                        abnormal_y.size,
                        bounds=(
                            (
                                float(np.min(normal_y)),
                                float(np.max(normal_y)),
                            )
                            if self.counterfactual_bounds == "normal_range"
                            else None
                        ),
                    )
                )
                r_a = abnormal_y - abnormal_prediction

        elif self.residualization == "raw":
            r_n = normal_y.copy()
            r_a = abnormal_y.copy()
            coef = np.empty(0, dtype=float)
            normal_prediction = np.empty(0, dtype=float)
            abnormal_prediction = np.empty(0, dtype=float)

        center = float(np.median(r_n))
        mad = float(np.median(np.abs(r_n - center)))
        robust_scale = 1.4826 * mad
        sd = float(np.std(r_n, ddof=1)) if r_n.size > 1 else 0.0
        level_scale = float(np.median(np.abs(normal_y)))
        relative_floor = self.relative_scale_floor * max(level_scale, self.min_scale)
        scale = max(robust_scale, 0.1 * sd, relative_floor, self.min_scale)

        z_n = (r_n - center) / scale
        if self.horizon_aware_uncertainty:
            forecast_uncertainty_multiplier = (
                _ar_forecast_uncertainty_multipliers(coef, r_a.size)
            )
            if self.forecast_error_covariance == "full":
                # The multi-horizon counterfactual errors satisfy e = L eps.
                # Whitening first recovers innovation-like errors; their
                # normal-only center and scale are then directly comparable
                # with the one-step normal residuals.
                whitened_abnormal = _whiten_ar_forecast_errors(r_a, coef)
                z_a = (whitened_abnormal - center) / scale
            else:
                z_a = (
                    (r_a - center)
                    / (scale * forecast_uncertainty_multiplier)
                )
        else:
            z_a = (r_a - center) / scale

        log_h0 = np.nan
        log_h1 = np.nan

        if self.scoring == "bayes_factor":
            pooled = np.concatenate([
                z_n,
                z_a,
            ])

            log_h0 = _nig_log_marginal(
                pooled,
                self.prior,
            )

            log_h1 = (
                _nig_log_marginal(
                    z_n,
                    self.prior,
                )
                + _nig_log_marginal(
                    z_a,
                    self.prior,
                )
            )

            score = float(
                log_h1 - log_h0
            )

        elif self.scoring == "glrt":
            score = _gaussian_glrt_score(
                z_n,
                z_a,
            )

        return {
            "score": score,
            "normal_scale": scale,
            "abnormal_mean_z": float(np.mean(z_a)),
            "abnormal_sd_z": float(np.std(z_a, ddof=1)) if z_a.size > 1 else 0.0,
            "log_marginal_h0": float(log_h0),
            "log_marginal_h1": float(log_h1),
            "counterfactual_clipped_predictions": counterfactual_clipped_count,
            "counterfactual_clipped_fraction": (
                counterfactual_clipped_count / abnormal_y.size
                if counterfactual_clipped_count is not None and abnormal_y.size
                else None
            ),
            "ar_spectral_radius_before": spectral_radius_before,
            "ar_spectral_radius_after": spectral_radius_after,
            "ar_stationarity_constrained": stationarity_constrained,
            "counterfactual_bounds": (
                self.counterfactual_bounds
                if self.residualization == "counterfactual_ar"
                else None
            ),
            "horizon_aware_uncertainty": self.horizon_aware_uncertainty,
            "forecast_error_covariance": self.forecast_error_covariance,
            "forecast_uncertainty_final_multiplier": (
                float(forecast_uncertainty_multiplier[-1])
                if forecast_uncertainty_multiplier.size
                else None
            ),
            "forecast_uncertainty_max_multiplier": (
                float(np.max(forecast_uncertainty_multiplier))
                if forecast_uncertainty_multiplier.size
                else None
            ),
            "forecast_uncertainty_multiplier": (
                forecast_uncertainty_multiplier.astype(float).tolist()
            ),
            # Keep the observations required to recreate an individual case
            # plot without reloading preprocessed input data.  For AR, the
            # prediction/residual series begin at ar_order; for raw they align
            # one-to-one with the raw series and predictions are unavailable.
            "ar_coefficients": coef.astype(float).tolist(),
            "raw_normal": normal_y.astype(float).tolist(),
            "raw_abnormal": abnormal_y.astype(float).tolist(),
            "ar_prediction_normal": normal_prediction.astype(float).tolist(),
            "ar_prediction_abnormal": abnormal_prediction.astype(float).tolist(),
            "ar_residual_normal": r_n.astype(float).tolist(),
            "ar_residual_abnormal": r_a.astype(float).tolist(),
            "standardized_residual_normal": z_n.astype(float).tolist(),
            "standardized_residual_abnormal": z_a.astype(float).tolist(),
        }

    def fit_predict(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        cols = list(columns) if columns is not None else self._numeric_common_columns(normal, abnormal)
        records: list[dict[str, object]] = []
        diagnostic_records: list[dict[str, object]] = []
        for col in cols:
            out = self._score_metric(normal[col].to_numpy(), abnormal[col].to_numpy())
            ranking_fields = {
                key: value for key, value in out.items()
                if not isinstance(value, (list, dict))
            }
            records.append({"metric": col, "service": _service_name(col), **ranking_fields})
            diagnostic_records.append({
                "metric": col,
                "service": _service_name(col),
                **out,
            })

        metric_df = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan)

        metric_df["evidence_weight"] = 0.0

        finite = metric_df["score"].notna()

        if finite.any():
            values = metric_df.loc[
                finite,
                "score",
            ].to_numpy()

            shifted = values - np.max(values)
            weights = np.exp(shifted)

            metric_df.loc[
                finite,
                "evidence_weight",
            ] = weights / weights.sum()

        metric_df = metric_df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
        metric_df.insert(0, "rank", np.arange(1, len(metric_df) + 1))
        self.metric_result_ = metric_df

        metric_ranks = metric_df.set_index("metric")["rank"].to_dict()
        metric_scores = metric_df.set_index("metric")["score"].to_dict()
        for record in diagnostic_records:
            record["rank"] = int(metric_ranks[record["metric"]])
            score = metric_scores[record["metric"]]
            record["score"] = float(score) if pd.notna(score) else None

        if self.aggregate == "metric":
            self.diagnostics_ = {
                "schema_version": 1,
                "residualization": self.residualization,
                "scoring": self.scoring,
                "ar_order": self.ar_order,
                "ar_stationarity": self.ar_stationarity,
                "stationarity_radius": self.stationarity_radius,
                "counterfactual_bounds": self.counterfactual_bounds,
                "horizon_aware_uncertainty": self.horizon_aware_uncertainty,
                "forecast_error_covariance": self.forecast_error_covariance,
                "ar_intervention_shapes": list(self.ar_intervention_shapes),
                "ar_intervention_onset_offsets": list(
                    self.ar_intervention_onset_offsets
                ),
                "ar_intervention_half_life": self.ar_intervention_half_life,
                "ar_intervention_precision": self.ar_intervention_precision,
                "ar_intervention_onset_prior_decay": (
                    self.ar_intervention_onset_prior_decay
                ),
                "ar_null_calibration_fractions": list(
                    self.ar_null_calibration_fractions
                ),
                "ar_null_calibration_quantile": self.ar_null_calibration_quantile,
                "ar_null_calibration_mode": self.ar_null_calibration_mode,
                "ar_bayes_prior": (
                    asdict(self.ar_bayes_prior)
                    if self.scoring in {
                        "ar_bayes_factor", "ar_intercept_bayes_factor",
                        "ar_intervention_bayes_factor",
                        "bsrc_ar_bayes_factor",
                    } else None
                ),
                "ar_regime_shift_prior": (
                    asdict(self.ar_regime_shift_prior)
                    if self.scoring == "bsrc_ar_bayes_factor" else None
                ),
                "metrics": diagnostic_records,
            }
            self.result_ = metric_df
            return metric_df.copy()

        grouped = []
        for service, g in metric_df.groupby("service", sort=False):
            s = g["score"].dropna().to_numpy()
            if s.size == 0:
                agg = np.nan
            elif self.service_aggregation == "max":
                agg = float(np.max(s))
            elif self.service_aggregation == "mean_top3":
                agg = float(np.mean(np.sort(s)[-min(3, s.size):]))
            elif self.service_aggregation == "logsumexp":
                m = float(np.max(s))
                agg = m + float(np.log(np.sum(np.exp(s - m))))
            else:
                raise ValueError(f"Unknown service_aggregation={self.service_aggregation}")
            grouped.append({
                "service": service,
                "score": agg,
                "top_metric": g.iloc[0]["metric"],
                "n_metrics": len(g),
            })

        service_df = pd.DataFrame(grouped).sort_values("score", ascending=False).reset_index(drop=True)
        service_df.insert(0, "rank", np.arange(1, len(service_df) + 1))
        service_df["evidence_weight"] = 0.0

        finite = service_df["score"].notna()

        if finite.any():
            values = service_df.loc[
                finite,
                "score",
            ].to_numpy()

            shifted = values - np.max(values)
            weights = np.exp(shifted)

            service_df.loc[
                finite,
                "evidence_weight",
            ] = weights / weights.sum()
        self.result_ = service_df
        self.diagnostics_ = {
            "schema_version": 1,
            "residualization": self.residualization,
            "scoring": self.scoring,
            "ar_order": self.ar_order,
            "ar_stationarity": self.ar_stationarity,
            "stationarity_radius": self.stationarity_radius,
            "counterfactual_bounds": self.counterfactual_bounds,
            "horizon_aware_uncertainty": self.horizon_aware_uncertainty,
            "forecast_error_covariance": self.forecast_error_covariance,
            "ar_intervention_shapes": list(self.ar_intervention_shapes),
            "ar_intervention_onset_offsets": list(
                self.ar_intervention_onset_offsets
            ),
            "ar_intervention_half_life": self.ar_intervention_half_life,
            "ar_intervention_precision": self.ar_intervention_precision,
            "ar_intervention_onset_prior_decay": (
                self.ar_intervention_onset_prior_decay
            ),
            "ar_null_calibration_fractions": list(
                self.ar_null_calibration_fractions
            ),
            "ar_null_calibration_quantile": self.ar_null_calibration_quantile,
            "ar_null_calibration_mode": self.ar_null_calibration_mode,
            "ar_bayes_prior": (
                asdict(self.ar_bayes_prior)
                if self.scoring in {
                    "ar_bayes_factor", "ar_intercept_bayes_factor",
                    "ar_intervention_bayes_factor",
                    "bsrc_ar_bayes_factor",
                } else None
            ),
            "ar_regime_shift_prior": (
                asdict(self.ar_regime_shift_prior)
                if self.scoring == "bsrc_ar_bayes_factor" else None
            ),
            "metrics": diagnostic_records,
            "services": service_df.to_dict(orient="records"),
        }
        
        return service_df.copy()

    def predict(self, normal: pd.DataFrame, abnormal: pd.DataFrame) -> list[str]:
        """Convenience API returning ranked metric/service names."""
        result = self.fit_predict(normal, abnormal)
        key = "metric" if self.aggregate == "metric" else "service"
        return result[key].tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("normal_csv")
    parser.add_argument("abnormal_csv")
    parser.add_argument("--ar-order", type=int, default=3)
    parser.add_argument("--aggregate", choices=["metric", "service"], default="service")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    normal_df = pd.read_csv(args.normal_csv)
    abnormal_df = pd.read_csv(args.abnormal_csv)
    model = AMBER(ar_order=args.ar_order, aggregate=args.aggregate)
    print(model.fit_predict(normal_df, abnormal_df).head(args.top).to_string(index=False))

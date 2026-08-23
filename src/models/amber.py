"""Bayesian residual-shift root cause analysis.

This module ranks metrics (and optionally services) by how strongly their
abnormal-period prediction residuals support a distributional change from the
normal-period residual distribution.

Only NumPy and pandas are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


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
    return np.sqrt(np.cumsum(psi ** 2))


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
    """RCA using Bayesian model selection on standardized AR residuals.

    H0: Normal and abnormal residuals share the same Gaussian parameters.
    H1: Normal and abnormal residuals have independent Gaussian parameters.

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
        residualization: Literal["ar", "counterfactual_ar", "raw",] = "ar",
        scoring: Literal["bayes_factor", "glrt",] = "bayes_factor",
        ar_stationarity: Literal["none", "root_projection"] = "none",
        stationarity_radius: float = 0.98,
        counterfactual_bounds: Literal["normal_range", "none"] = "normal_range",
        horizon_aware_uncertainty: bool = False,
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

        if residualization not in {"ar", "counterfactual_ar", "raw"}:
            raise ValueError(f"Unknown residualization={residualization}")

        if scoring not in {"bayes_factor", "glrt",}:
            raise ValueError(f"Unknown scoring={scoring}")

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
                if not isinstance(value, list)
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

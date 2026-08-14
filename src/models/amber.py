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


def _service_name(metric: str) -> str:
    """Infer service from names such as catalogue_cpu or catalogue_latency-90."""
    for suffix in ("_latency-50", "_latency-90", "_latency-95", "_latency-99",
                   "_latency", "_cpu", "_mem", "_memory"):
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return metric.rsplit("_", 1)[0] if "_" in metric else metric


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
        self.result_: pd.DataFrame | None = None

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
    
    def _score_metric(self, normal_y: np.ndarray, abnormal_y: np.ndarray) -> dict[str, float]:
        normal_y, abnormal_y = self._prepare_series(
            normal_y,
            abnormal_y,
        )

        if (
            normal_y.size <= self.ar_order + 3
            or abnormal_y.size <= self.ar_order
        ):
            return {
                "score": -np.inf,
                "evidence_weight": 0.0,
                "normal_scale": np.nan,
                "abnormal_mean_z": np.nan,
                "abnormal_sd_z": np.nan,
            }
        
        coef = _ridge_ar_fit(normal_y, self.ar_order, self.ridge)
        r_n = _ar_residuals(normal_y, coef, self.ar_order)

        if self.ar_order > 0:
            history = normal_y[-self.ar_order:]
        else:
            history = np.empty(
                0,
                dtype=float,
            )

        history_and_abnormal = np.concatenate([
            history,
            abnormal_y,
        ])

        r_a = _ar_residuals(
            history_and_abnormal,
            coef,
            self.ar_order,
        )

        center = float(np.median(r_n))
        mad = float(np.median(np.abs(r_n - center)))
        robust_scale = 1.4826 * mad
        sd = float(np.std(r_n, ddof=1)) if r_n.size > 1 else 0.0
        level_scale = float(np.median(np.abs(normal_y)))
        relative_floor = self.relative_scale_floor * max(level_scale, self.min_scale)
        scale = max(robust_scale, 0.1 * sd, relative_floor, self.min_scale)

        z_n = (r_n - center) / scale
        z_a = (r_a - center) / scale

        pooled = np.concatenate([z_n, z_a])

        # H0: normal and abnormal residuals share one Gaussian distribution.
        log_h0 = _nig_log_marginal(
            pooled,
            self.prior,
        )

        # H1: normal and abnormal residuals have independent Gaussian parameters,
        # both governed by the same weak-information NIG prior.
        log_h1 = (
            _nig_log_marginal(z_n, self.prior)
            + _nig_log_marginal(z_a, self.prior)
        )

        score = float(log_h1 - log_h0)

        return {
            "score": score,
            "normal_scale": scale,
            "abnormal_mean_z": float(np.mean(z_a)),
            "abnormal_sd_z": float(np.std(z_a, ddof=1)) if z_a.size > 1 else 0.0,
            "log_marginal_h0": float(log_h0),
            "log_marginal_h1": float(log_h1),
        }

    def fit_predict(
        self,
        normal: pd.DataFrame,
        abnormal: pd.DataFrame,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        cols = list(columns) if columns is not None else self._numeric_common_columns(normal, abnormal)
        records: list[dict[str, float | str]] = []
        for col in cols:
            out = self._score_metric(normal[col].to_numpy(), abnormal[col].to_numpy())
            records.append({"metric": col, "service": _service_name(col), **out})

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

        if self.aggregate == "metric":
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
        vals = service_df["score"].to_numpy()
        vals = vals - np.nanmax(vals)
        p = np.exp(vals)
        service_df["evidence_weight"] = p / np.nansum(p)
        self.result_ = service_df
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

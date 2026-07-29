import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from scipy.special import gammaln, logsumexp


@dataclass
class NIGPosterior:
    """Normal-Inverse-Gamma distribution parameters for Bayesian AR regression."""

    mean: np.ndarray
    covariance_scale: np.ndarray
    shape: float
    scale: float


class BayesianRCA:
    """
    Bayesian root-cause analysis by model selection between direct and delayed changes.

    For each metric X_k, the model compares:

    H_k^(D): the AR data-generating mechanism changes directly at the known fault time t_F;
    H_k^(P): the mechanism remains normal initially and either changes later or never changes.

    The root-cause score is the log Bayes factor

        S_k = log p(D_abn,k | H_k^(D)) - log p(D_abn,k | H_k^(P)).

    Each AR regime is represented by a Bayesian linear regression with a
    Normal-Inverse-Gamma prior. Regression coefficients and noise variance are
    integrated out analytically, so every score is a true marginal-likelihood
    comparison rather than a heuristic residual score.
    """

    def __init__(
        self,
        ar_lags: int = 3,
        init_window: int = 5,
        eta: float = 5.0,
        max_steps: Optional[int] = None,
        kappa: float = 10.0,
        hazard: float = 0.1,
        prior_variance: float = 100.0,
        prior_shape: float = 2.0,
        change_prior_df: float = 4.0,
        variance_floor: float = 1e-8,
        constant_threshold: float = 1e-6,
    ):
        """
        Parameters
        ----------
        ar_lags:
            AR order p.
        init_window, eta:
            Kept only for backward compatibility with existing configuration.
            They are not used by the marginal-likelihood model.
        max_steps:
            Maximum number of abnormal observations used. None uses all data.
        kappa:
            Inflation factor for the abnormal-regime coefficient prior.
            Larger values allow a larger departure from the normal AR dynamics.
        hazard:
            Constant discrete hazard for a delayed propagated change.
        prior_variance:
            Weak-prior variance scale for the initial normal AR coefficients.
        prior_shape:
            Shape parameter of the weak Inverse-Gamma prior used for normal fitting.
        change_prior_df:
            Effective prior degrees of freedom for abnormal noise variance.
            This controls how strongly the abnormal variance is shrunk toward the
            normal-regime posterior mean variance.
        variance_floor:
            Numerical lower bound for variance-related quantities.
        constant_threshold:
            Metrics with normal-period standard deviation below this value are
            excluded and appended to the end of the ranking.
        """
        if ar_lags < 1:
            raise ValueError("ar_lags must be at least 1.")
        if kappa <= 1.0:
            raise ValueError("kappa must be greater than 1.")
        if not 0.0 < hazard < 1.0:
            raise ValueError("hazard must lie strictly between 0 and 1.")
        if prior_variance <= 0.0:
            raise ValueError("prior_variance must be positive.")
        if prior_shape <= 1.0:
            raise ValueError("prior_shape must exceed 1 so the prior variance mean exists.")
        if change_prior_df <= 2.0:
            raise ValueError("change_prior_df must exceed 2.")

        self.ar_lags = ar_lags
        self.init_window = init_window
        self.eta = eta
        self.max_steps = max_steps
        self.kappa = kappa
        self.hazard = hazard
        self.prior_variance = prior_variance
        self.prior_shape = prior_shape
        self.change_prior_df = change_prior_df
        self.variance_floor = variance_floor
        self.constant_threshold = constant_threshold

        self.normal_posteriors: Dict[str, NIGPosterior] = {}
        self.log_scores_: Dict[str, float] = {}
        self.posterior_probabilities_: Dict[str, float] = {}

    @staticmethod
    def _as_finite_array(series: pd.Series) -> np.ndarray:
        """Convert a series to a finite float array, interpolating isolated missing values."""
        values = pd.to_numeric(series, errors="coerce").astype(float)
        values = values.replace([np.inf, -np.inf], np.nan)
        values = values.interpolate(limit_direction="both")
        if values.isna().any():
            values = values.fillna(values.mean())
        if values.isna().any():
            values = values.fillna(0.0)
        return values.to_numpy(dtype=float)

    def _build_ar_design(self, history: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct the AR(p) design matrix for `target`, using the final p values of
        `history` as the initial lag context.
        """
        p = self.ar_lags
        if target.size == 0:
            return np.empty((0, p + 1)), np.empty(0)
        if history.size < p:
            raise ValueError("The history sequence must contain at least ar_lags observations.")

        context = np.concatenate([history[-p:], target])
        design = np.ones((target.size, p + 1), dtype=float)
        for row in range(target.size):
            current = p + row
            lag_indices = current - np.arange(1, p + 1)
            design[row, 1:] = context[lag_indices]
        return design, target.astype(float, copy=False)

    def _initial_normal_prior(self, dimension: int, empirical_variance: float) -> NIGPosterior:
        """Create a weak Normal-Inverse-Gamma prior for the normal AR model."""
        variance = max(float(empirical_variance), self.variance_floor)
        shape = self.prior_shape
        scale = variance * (shape - 1.0)
        return NIGPosterior(
            mean=np.zeros(dimension, dtype=float),
            covariance_scale=np.eye(dimension, dtype=float) * self.prior_variance,
            shape=shape,
            scale=max(scale, self.variance_floor),
        )

    def _change_prior(self, normal_posterior: NIGPosterior) -> NIGPosterior:
        """
        Construct the abnormal-regime prior.

        Its coefficient mean is the normal posterior mean, but its covariance is
        inflated by kappa. Its noise-variance prior has the same mean as the normal
        posterior variance but substantially lower concentration.
        """
        normal_variance_mean = normal_posterior.scale / max(
            normal_posterior.shape - 1.0,
            self.variance_floor,
        )
        shape = self.change_prior_df / 2.0
        scale = normal_variance_mean * (shape - 1.0)
        return NIGPosterior(
            mean=normal_posterior.mean.copy(),
            covariance_scale=self.kappa * normal_posterior.covariance_scale,
            shape=shape,
            scale=max(scale, self.variance_floor),
        )
    
    def _stabilize_spd(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return a finite symmetric positive-definite matrix and its Cholesky factor."""
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("SPD stabilization requires a square matrix.")

        dimension = array.shape[0]
        identity = np.eye(dimension, dtype=float)

        finite_values = array[np.isfinite(array)]
        finite_scale = (
            float(np.max(np.abs(finite_values)))
            if finite_values.size > 0
            else 1.0
        )
        finite_scale = max(finite_scale, 1.0)

        # Remove NaN/Inf values before symmetrization. Such values can arise from
        # round-off in extremely ill-conditioned AR design matrices.
        bounded = np.nan_to_num(
            array,
            nan=0.0,
            posinf=finite_scale,
            neginf=-finite_scale,
        )
        symmetric = 0.5 * (bounded + bounded.T)

        # Use a scale-aware eigenvalue floor rather than an absolute-only jitter.
        eigen_floor = max(
            self.variance_floor,
            np.finfo(float).eps * finite_scale * max(dimension, 1) * 100.0,
        )

        # First try the inexpensive diagonal-jitter path.
        jitter = eigen_floor
        for _ in range(16):
            candidate = symmetric + jitter * identity
            try:
                chol = np.linalg.cholesky(candidate)
                return candidate, chol
            except np.linalg.LinAlgError:
                jitter *= 10.0

        # Project explicitly onto the SPD cone.
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.nan_to_num(
            eigenvalues,
            nan=eigen_floor,
            posinf=finite_scale,
            neginf=eigen_floor,
        )
        eigenvalues = np.maximum(eigenvalues, eigen_floor)
        candidate = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        candidate = 0.5 * (candidate + candidate.T)

        # Reconstruction itself can introduce a tiny negative eigenvalue, so add
        # a final scale-aware jitter and retry deterministically.
        jitter = eigen_floor
        for _ in range(16):
            stabilized = candidate + jitter * identity
            try:
                chol = np.linalg.cholesky(stabilized)
                return stabilized, chol
            except np.linalg.LinAlgError:
                jitter *= 10.0

        raise np.linalg.LinAlgError(
            "Failed to stabilize covariance matrix as positive definite."
        )

    def _posterior_update(
        self,
        prior: NIGPosterior,
        design: np.ndarray,
        response: np.ndarray,
    ) -> NIGPosterior:
        """Update a Normal-Inverse-Gamma prior with Bayesian linear-regression data."""
        if response.size == 0:
            return NIGPosterior(
                mean=prior.mean.copy(),
                covariance_scale=prior.covariance_scale.copy(),
                shape=prior.shape,
                scale=prior.scale,
            )

        v0, v0_chol = self._stabilize_spd(prior.covariance_scale)
        identity = np.eye(v0.shape[0], dtype=float)
        v0_inv = cho_solve((v0_chol, True), identity, check_finite=False)
        v0_inv = 0.5 * (v0_inv + v0_inv.T)

        gram = design.T @ design
        gram = np.nan_to_num(
            gram,
            nan=0.0,
            posinf=np.finfo(float).max / 100.0,
            neginf=-np.finfo(float).max / 100.0,
        )
        vn_inv_raw = v0_inv + gram
        vn_inv, vn_inv_chol = self._stabilize_spd(vn_inv_raw)

        rhs = v0_inv @ prior.mean + design.T @ response
        rhs = np.nan_to_num(
            rhs,
            nan=0.0,
            posinf=np.finfo(float).max / 100.0,
            neginf=-np.finfo(float).max / 100.0,
        )
        mn = cho_solve((vn_inv_chol, True), rhs, check_finite=False)
        vn = cho_solve((vn_inv_chol, True), identity, check_finite=False)
        vn = 0.5 * (vn + vn.T)
        vn, _ = self._stabilize_spd(vn)

        an = prior.shape + response.size / 2.0

        quadratic = (
            response @ response
            + prior.mean @ v0_inv @ prior.mean
            - mn @ vn_inv @ mn
        )
        if not np.isfinite(quadratic):
            residual = response - design @ mn
            centered_prior = mn - prior.mean
            quadratic = (
                residual @ residual
                + centered_prior @ v0_inv @ centered_prior
            )

        bn = prior.scale + 0.5 * max(float(quadratic), 0.0)

        return NIGPosterior(
            mean=mn,
            covariance_scale=vn,
            shape=an,
            scale=max(float(bn), self.variance_floor),
        )

    def _log_marginal_likelihood(
        self,
        prior: NIGPosterior,
        design: np.ndarray,
        response: np.ndarray,
    ) -> Tuple[float, NIGPosterior]:
        """Compute the exact log marginal likelihood under a stabilized NIG model."""
        if response.size == 0:
            return 0.0, self._posterior_update(prior, design, response)

        stable_prior_covariance, prior_chol = self._stabilize_spd(
            prior.covariance_scale
        )
        stable_prior = NIGPosterior(
            mean=prior.mean,
            covariance_scale=stable_prior_covariance,
            shape=prior.shape,
            scale=max(float(prior.scale), self.variance_floor),
        )
        posterior = self._posterior_update(stable_prior, design, response)
        stable_posterior_covariance, posterior_chol = self._stabilize_spd(
            posterior.covariance_scale
        )
        posterior = NIGPosterior(
            mean=posterior.mean,
            covariance_scale=stable_posterior_covariance,
            shape=posterior.shape,
            scale=max(float(posterior.scale), self.variance_floor),
        )

        logdet0 = 2.0 * np.sum(np.log(np.diag(prior_chol)))
        logdetn = 2.0 * np.sum(np.log(np.diag(posterior_chol)))

        n = response.size
        log_ml = (
            -0.5 * n * np.log(2.0 * np.pi)
            + 0.5 * (logdetn - logdet0)
            + stable_prior.shape * np.log(stable_prior.scale)
            - posterior.shape * np.log(posterior.scale)
            + gammaln(posterior.shape)
            - gammaln(stable_prior.shape)
        )
        return float(log_ml), posterior

    def _fit_normal_model(self, values: np.ndarray) -> NIGPosterior:
        """Fit the normal-period Bayesian AR model and return its posterior."""
        p = self.ar_lags
        if values.size <= p:
            raise ValueError("Normal data must contain more observations than ar_lags.")

        design = np.ones((values.size - p, p + 1), dtype=float)
        for row, t in enumerate(range(p, values.size)):
            lag_indices = t - np.arange(1, p + 1)
            design[row, 1:] = values[lag_indices]
        response = values[p:]

        empirical_variance = np.var(response, ddof=1) if response.size > 1 else np.var(values)
        prior = self._initial_normal_prior(p + 1, empirical_variance)
        _, posterior = self._log_marginal_likelihood(prior, design, response)
        return posterior

    def _metric_log_score(self, normal: np.ndarray, abnormal: np.ndarray, column: str) -> float:
        """Compute log BF(direct change at t_F versus delayed/no propagated change)."""
        normal_posterior = self.normal_posteriors[column]
        change_prior = self._change_prior(normal_posterior)
        design_abn, response_abn = self._build_ar_design(normal, abnormal)

        # Direct-change model: the abnormal regime starts immediately at t_F.
        log_direct, _ = self._log_marginal_likelihood(
            change_prior,
            design_abn,
            response_abn,
        )

        delayed_components: List[float] = []
        total_steps = response_abn.size

        # Delayed propagation: normal regime for d observations, then a changed regime.
        # d ranges from 1 to total_steps - 1, so the candidate cannot change directly at t_F.
        for d in range(1, total_steps):
            log_pre, _ = self._log_marginal_likelihood(
                normal_posterior,
                design_abn[:d],
                response_abn[:d],
            )
            log_post, _ = self._log_marginal_likelihood(
                change_prior,
                design_abn[d:],
                response_abn[d:],
            )
            log_weight = np.log(self.hazard) + (d - 1) * np.log1p(-self.hazard)
            delayed_components.append(log_weight + log_pre + log_post)

        # No propagated change during the observed abnormal interval.
        log_no_change, _ = self._log_marginal_likelihood(
            normal_posterior,
            design_abn,
            response_abn,
        )
        log_no_change_weight = max(total_steps - 1, 0) * np.log1p(-self.hazard)
        delayed_components.append(log_no_change_weight + log_no_change)

        log_delayed = float(logsumexp(np.asarray(delayed_components, dtype=float)))
        return float(log_direct - log_delayed)

    def fit_predict(
        self,
        df_normal: pd.DataFrame,
        df_abnormal: pd.DataFrame,
        dataset_name: Optional[str] = None,
    ) -> List[str]:
        """Fit normal AR models, compare direct/delayed changes, and return a ranking."""
        del dataset_name
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        if df_normal.empty:
            raise ValueError("df_normal must contain at least one observation.")
        if df_abnormal.empty:
            raise ValueError("df_abnormal must contain at least one observation.")

        common_columns = [column for column in df_normal.columns if column in df_abnormal.columns]
        if not common_columns:
            raise ValueError("Normal and abnormal data have no common metric columns.")

        normal_numeric = pd.DataFrame(
            {column: self._as_finite_array(df_normal[column]) for column in common_columns}
        )
        abnormal_numeric = pd.DataFrame(
            {column: self._as_finite_array(df_abnormal[column]) for column in common_columns}
        )

        standard_deviations = normal_numeric.std(axis=0, ddof=1)
        valid_columns = standard_deviations[
            standard_deviations > self.constant_threshold
        ].index.tolist()
        excluded_columns = [column for column in common_columns if column not in valid_columns]

        if not valid_columns:
            return excluded_columns

        abnormal_steps = len(abnormal_numeric)
        if self.max_steps is not None:
            abnormal_steps = min(abnormal_steps, int(self.max_steps))
        if abnormal_steps < 1:
            raise ValueError("max_steps leaves no abnormal observations.")

        self.normal_posteriors.clear()
        self.log_scores_.clear()
        self.posterior_probabilities_.clear()

        for column in valid_columns:
            normal_values = normal_numeric[column].to_numpy(dtype=float)
            abnormal_values = abnormal_numeric[column].to_numpy(dtype=float)[:abnormal_steps]
            self.normal_posteriors[column] = self._fit_normal_model(normal_values)
            self.log_scores_[column] = self._metric_log_score(
                normal_values,
                abnormal_values,
                column,
            )

        score_array = np.asarray(
            [self.log_scores_[column] for column in valid_columns],
            dtype=float,
        )
        log_prior = -np.log(len(valid_columns))
        log_posterior = log_prior + score_array
        log_posterior -= logsumexp(log_posterior)
        posterior = np.exp(log_posterior)

        self.posterior_probabilities_ = {
            column: float(probability)
            for column, probability in zip(valid_columns, posterior)
        }

        ranking = sorted(
            valid_columns,
            key=lambda column: self.posterior_probabilities_[column],
            reverse=True,
        )
        ranking.extend(excluded_columns)
        return ranking
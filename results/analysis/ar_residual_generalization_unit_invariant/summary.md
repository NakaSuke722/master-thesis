# In-sample vs time-ordered holdout AR residuals

The first normal block fits a unit-invariant stationary AR model. The held-out normal block is predicted one step ahead with frozen coefficients and observed lags.

`scale_ratio = OOS residual scale / in-sample residual scale`; values above one mean the fitted-window residual reference is narrower than future normal residuals.

| Scope | Cases | Median case scale ratio (95% CI) | Ratio > 1 | Ratio > 1.1 | Median absolute center shift / in-scale | Median max CF BF | Median max 1-step BF | Positive CF services | Positive 1-step services |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 375 | 0.981 (0.973, 0.989) | 35.2% | 4.5% | 0.259 | 582.757 | 690.731 | 89.6% | 71.9% |
| re1_ob | 125 | 0.996 (0.980, 1.014) | 48.0% | 12.0% | 0.205 | 245.050 | 256.656 | 84.8% | 58.5% |
| re1_ss | 125 | 0.979 (0.957, 0.991) | 34.4% | 1.6% | 0.305 | 534.599 | 694.681 | 86.6% | 70.0% |
| re1_tt | 125 | 0.975 (0.967, 0.987) | 23.2% | 0.0% | 0.258 | 958.058 | 961.742 | 97.6% | 87.1% |
| cpu | 75 | 0.988 (0.979, 0.996) | 32.0% | 1.3% | 0.242 | 542.558 | 694.781 | 90.1% | 70.6% |
| mem | 75 | 0.958 (0.952, 0.973) | 28.0% | 5.3% | 0.264 | 620.923 | 694.691 | 90.3% | 72.0% |
| disk | 75 | 0.999 (0.982, 1.017) | 46.7% | 8.0% | 0.285 | 528.729 | 583.604 | 90.8% | 76.6% |
| delay | 75 | 0.977 (0.961, 0.992) | 36.0% | 5.3% | 0.247 | 586.729 | 695.092 | 88.4% | 70.8% |
| loss | 75 | 0.980 (0.959, 0.994) | 33.3% | 2.7% | 0.254 | 513.418 | 633.819 | 88.6% | 69.4% |

## Service-level associations

Spearman correlations use all case-service rows in each scope.

| Scope | CF BF vs absolute log-scale ratio | CF BF vs absolute center shift | CF BF vs 1-step BF | 1-step BF vs absolute log-scale ratio | 1-step BF vs absolute center shift |
|---|---:|---:|---:|---:|---:|
| overall | 0.637 (p=0) | 0.474 (p=0) | 0.788 (p=0) | 0.841 (p=0) | 0.217 (p=1.7e-140) |
| re1_ob | 0.698 (p=1.18e-237) | 0.401 (p=8.21e-64) | 0.819 (p=0) | 0.739 (p=8.12e-281) | 0.267 (p=6.27e-28) |
| re1_ss | 0.522 (p=2.34e-197) | 0.497 (p=6.44e-176) | 0.799 (p=0) | 0.773 (p=0) | 0.263 (p=6.99e-46) |
| re1_tt | 0.618 (p=0) | 0.394 (p=0) | 0.727 (p=0) | 0.855 (p=0) | 0.086 (p=8.36e-16) |
| cpu | 0.620 (p=8.13e-281) | 0.520 (p=1.13e-183) | 0.768 (p=0) | 0.846 (p=0) | 0.263 (p=2.62e-43) |
| mem | 0.645 (p=1.24e-311) | 0.446 (p=1.06e-129) | 0.778 (p=0) | 0.860 (p=0) | 0.172 (p=4.79e-19) |
| disk | 0.643 (p=2.26e-309) | 0.405 (p=2.54e-105) | 0.833 (p=0) | 0.788 (p=0) | 0.185 (p=6.02e-22) |
| delay | 0.628 (p=1.08e-290) | 0.498 (p=6.27e-166) | 0.765 (p=0) | 0.839 (p=0) | 0.207 (p=5.17e-27) |
| loss | 0.649 (p=1.04e-316) | 0.468 (p=4.07e-144) | 0.791 (p=0) | 0.864 (p=0) | 0.223 (p=3.83e-31) |

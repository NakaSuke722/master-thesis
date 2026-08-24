# Intercept-shift AR Bayes Factor RCAEval paired validation

Joined cases: 375

Differences are Intercept-shift AR-BF minus baseline with paired case-level 95% bootstrap intervals.

| Comparison | Candidate AC@1 | AC@3 | AC@5 | Avg@5 | AC@1 difference (95% CI) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Raw+BF → Intercept-shift AR BF (shared lags and variance) | 0.6053 | 0.8773 | 0.9413 | 0.8272 | -0.0347 [-0.0987, +0.0293] | 0.319 |
| Stationary counterfactual AR+BF + horizon uncertainty → Intercept-shift AR BF (shared lags and variance) | 0.6053 | 0.8773 | 0.9413 | 0.8272 | -0.0587 [-0.1200, +0.0027] | 0.07555 |
| Direct shared-vs-separate AR BF → Intercept-shift AR BF (shared lags and variance) | 0.6053 | 0.8773 | 0.9413 | 0.8272 | +0.1200 [+0.0507, +0.1893] | 0.0008903 |

## Intercept-shift AR-BF by dataset

| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|
| re1_ob | 0.6240 | 0.8800 | 0.9200 | 0.8288 |
| re1_ss | 0.7360 | 0.9280 | 0.9840 | 0.8992 |
| re1_tt | 0.4560 | 0.8240 | 0.9200 | 0.7536 |

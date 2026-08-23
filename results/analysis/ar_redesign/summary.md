# Stationarity and forecast-error covariance AR redesign

Joined cases: 375

Candidate-minus-baseline differences use paired case-level bootstrap intervals.

| Comparison | AC@1 | AC@3 | AC@5 | Avg@5 | AC@1 difference (95% CI) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Observed-lag AR+BF → Stationary observed-lag AR+BF | 0.6107 | 0.8587 | 0.9227 | 0.8181 | -0.0027 [-0.0080, +0.0000] | 1 |
| Stationary observed-lag AR+BF → Stationary counterfactual AR+BF | 0.6640 | 0.8747 | 0.9387 | 0.8379 | +0.0533 [+0.0160, +0.0907] | 0.0066 |
| Stationary counterfactual AR+BF → Stationary counterfactual AR+BF + horizon uncertainty | 0.6640 | 0.8853 | 0.9440 | 0.8496 | +0.0000 [-0.0213, +0.0213] | 1 |
| Stationary counterfactual AR+BF + horizon uncertainty → Stationary counterfactual AR+BF + full forecast covariance | 0.6107 | 0.8587 | 0.9227 | 0.8181 | -0.0533 [-0.0853, -0.0240] | 0.001193 |
| Stationary observed-lag AR+BF → Stationary counterfactual AR+BF + full forecast covariance | 0.6107 | 0.8587 | 0.9227 | 0.8181 | +0.0000 [+0.0000, +0.0000] | 1 |
| Raw+BF → Stationary counterfactual AR+BF + full forecast covariance | 0.6107 | 0.8587 | 0.9227 | 0.8181 | -0.0293 [-0.0587, +0.0000] | 0.08014 |
| Bounded counterfactual AR+BF → Stationary counterfactual AR+BF + full forecast covariance | 0.6107 | 0.8587 | 0.9227 | 0.8181 | -0.0560 [-0.0933, -0.0187] | 0.004601 |

## Full-covariance variant by dataset

| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|
| re1_ob | 0.4480 | 0.8560 | 0.9440 | 0.7808 |
| re1_ss | 0.8480 | 0.9360 | 0.9920 | 0.9312 |
| re1_tt | 0.5360 | 0.7840 | 0.8320 | 0.7424 |

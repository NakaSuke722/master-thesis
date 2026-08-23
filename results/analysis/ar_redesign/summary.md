# Stationarity and horizon-uncertainty AR redesign

Joined cases: 375

Candidate-minus-baseline differences use paired case-level bootstrap intervals.

| Comparison | AC@1 | AC@3 | AC@5 | Avg@5 | AC@1 difference (95% CI) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Observed-lag AR+BF → Stationary observed-lag AR+BF | 0.6107 | 0.8587 | 0.9227 | 0.8181 | -0.0027 [-0.0080, +0.0000] | 1 |
| Stationary observed-lag AR+BF → Stationary counterfactual AR+BF | 0.6640 | 0.8747 | 0.9387 | 0.8379 | +0.0533 [+0.0160, +0.0907] | 0.0066 |
| Stationary counterfactual AR+BF → Stationary counterfactual AR+BF + horizon uncertainty | 0.6640 | 0.8853 | 0.9440 | 0.8496 | +0.0000 [-0.0213, +0.0213] | 1 |
| Raw+BF → Stationary counterfactual AR+BF + horizon uncertainty | 0.6640 | 0.8853 | 0.9440 | 0.8496 | +0.0240 [+0.0080, +0.0427] | 0.01172 |
| Bounded counterfactual AR+BF → Stationary counterfactual AR+BF + horizon uncertainty | 0.6640 | 0.8853 | 0.9440 | 0.8496 | -0.0027 [-0.0240, +0.0187] | 1 |

## Final candidate by dataset

| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|
| re1_ob | 0.5360 | 0.8960 | 0.9760 | 0.8304 |
| re1_ss | 0.8480 | 0.9680 | 0.9920 | 0.9408 |
| re1_tt | 0.6080 | 0.7920 | 0.8640 | 0.7776 |

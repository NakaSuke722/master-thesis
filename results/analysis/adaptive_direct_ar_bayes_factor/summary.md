# Adaptive intervention-response Direct AR-BF paired validation

Joined cases: 375

Differences are adaptive candidate minus baseline with paired case-level 95% bootstrap intervals.

| Comparison | Candidate AC@1 | AC@3 | AC@5 | Avg@5 | AC@1 difference (95% CI) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Raw+BF → Adaptive intervention-response Direct AR BF | 0.6880 | 0.8800 | 0.9520 | 0.8576 | +0.0480 [-0.0133, +0.1093] | 0.1326 |
| Stationary counterfactual AR+BF + horizon uncertainty → Adaptive intervention-response Direct AR BF | 0.6880 | 0.8800 | 0.9520 | 0.8576 | +0.0240 [-0.0347, +0.0827] | 0.4744 |
| Direct shared-vs-separate AR BF → Adaptive intervention-response Direct AR BF | 0.6880 | 0.8800 | 0.9520 | 0.8576 | +0.2027 [+0.1387, +0.2667] | 1.16e-09 |
| Intercept-shift AR BF (shared lags and variance) → Adaptive intervention-response Direct AR BF | 0.6880 | 0.8800 | 0.9520 | 0.8576 | +0.0827 [+0.0373, +0.1280] | 0.0003713 |

## Adaptive Direct AR-BF by dataset

| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|
| re1_ob | 0.7440 | 0.9120 | 0.9600 | 0.8848 |
| re1_ss | 0.8240 | 0.9520 | 0.9840 | 0.9312 |
| re1_tt | 0.4960 | 0.7760 | 0.9120 | 0.7568 |

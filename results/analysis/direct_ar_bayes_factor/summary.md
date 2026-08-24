# Direct AR Bayes Factor RCAEval paired validation

Joined cases: 375

Differences are Direct AR-BF minus baseline with paired case-level 95% bootstrap intervals.

| Comparison | Direct AC@1 | Direct AC@3 | Direct AC@5 | Direct Avg@5 | AC@1 difference (95% CI) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Raw+BF → Direct shared-vs-separate AR BF | 0.4853 | 0.7387 | 0.8160 | 0.6976 | -0.1547 [-0.2000, -0.1093] | 7.864e-11 |
| Stationary counterfactual AR+BF + horizon uncertainty → Direct shared-vs-separate AR BF | 0.4853 | 0.7387 | 0.8160 | 0.6976 | -0.1787 [-0.2240, -0.1333] | 1.367e-13 |

## Root-service rank movement

delta = baseline rank - Direct AR-BF rank; positive means Direct improved.

| Baseline | Improved | Same | Worsened | Mean delta | Median delta | Top-1 lost | Top-1 gained |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 21 | 201 | 153 | -1.9040 | +0 | 71 | 13 |
| stationary_counterfactual_ar_uncertainty | 17 | 199 | 159 | -2.3227 | +0 | 78 | 11 |

## Direct AR-BF by dataset

| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 |
|---|---:|---:|---:|---:|
| re1_ob | 0.2320 | 0.6880 | 0.8240 | 0.6112 |
| re1_ss | 0.7200 | 0.9120 | 0.9920 | 0.8896 |
| re1_tt | 0.5040 | 0.6160 | 0.6320 | 0.5920 |

# Counterfactual AR paired validation

Joined cases: 375

## Paired performance

Differences are candidate minus baseline. CIs are paired 95% bootstrap intervals.

| Comparison | Metric | Baseline | Candidate | Difference | 95% CI | McNemar p |
|---|---|---:|---:|---:|---:|---:|
| Raw+BF → Bounded counterfactual AR+BF | AC@1 | 0.6400 | 0.6667 | +0.0267 | [+0.0027, +0.0507] | 0.05248 |
| Raw+BF → Bounded counterfactual AR+BF | AC@3 | 0.8480 | 0.8800 | +0.0320 | [+0.0080, +0.0560] | — |
| Raw+BF → Bounded counterfactual AR+BF | AC@5 | 0.9280 | 0.9440 | +0.0160 | [-0.0027, +0.0347] | — |
| Raw+BF → Bounded counterfactual AR+BF | Avg@5 | 0.8187 | 0.8416 | +0.0229 | [+0.0085, +0.0379] | — |
| Observed-lag AR+BF → Bounded counterfactual AR+BF | AC@1 | 0.6133 | 0.6667 | +0.0533 | [+0.0160, +0.0907] | 0.007787 |
| Observed-lag AR+BF → Bounded counterfactual AR+BF | AC@3 | 0.8587 | 0.8800 | +0.0213 | [-0.0107, +0.0507] | — |
| Observed-lag AR+BF → Bounded counterfactual AR+BF | AC@5 | 0.9227 | 0.9440 | +0.0213 | [+0.0000, +0.0427] | — |
| Observed-lag AR+BF → Bounded counterfactual AR+BF | Avg@5 | 0.8187 | 0.8416 | +0.0229 | [+0.0027, +0.0432] | — |

## Clip attribution

The competitor is the highest-ranked non-root service in the counterfactual result.

| Group | Cases | Root any clip | Competitor any clip | Either any clip | Root mean fraction | Competitor mean fraction |
|---|---:|---:|---:|---:|---:|---:|
| all | 375 | 102 | 87 | 167 | 0.1098 | 0.1152 |
| raw_non_top1_to_counterfactual_top1 | 16 | 2 | 4 | 5 | 0.0417 | 0.0698 |
| raw_top1_to_counterfactual_non_top1 | 6 | 2 | 2 | 3 | 0.0559 | 0.2222 |
| counterfactual_rank_better_than_raw | 59 | 15 | 13 | 26 | 0.0877 | 0.1319 |
| counterfactual_rank_worse_than_raw | 23 | 5 | 6 | 9 | 0.0724 | 0.1436 |
| observed_ar_non_top1_to_counterfactual_top1 | 36 | 15 | 5 | 17 | 0.1944 | 0.0219 |
| observed_ar_top1_to_counterfactual_non_top1 | 16 | 2 | 4 | 5 | 0.0209 | 0.1667 |
| counterfactual_rank_better_than_observed_ar | 81 | 28 | 13 | 37 | 0.1474 | 0.0714 |
| counterfactual_rank_worse_than_observed_ar | 36 | 4 | 7 | 9 | 0.0370 | 0.1195 |

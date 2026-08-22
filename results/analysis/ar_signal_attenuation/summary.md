# Observed-lag AR persistent-shift attenuation diagnostic

`rank_delta = raw_rank - ar_rank`; positive values mean AR improved the root-service rank.
`signal_retention_ratio = |AR residual median shift| / |Raw median shift|`; values below 1 indicate attenuation.
Positive `ar_early_to_late_decay_fraction` indicates that the AR residual shift became weaker from the first to the second half of the abnormal period.

Joined cases: 375; root metrics: 7464 (Raw/AR top-3 union: 1277)

## By movement

| Movement | Cases | Root metrics | Median sum(phi) | Median retention | Median AR decay |
|---|---:|---:|---:|---:|---:|
| improved | 44 | 734 | 0.4683 | 0.5387 | -0.0164 |
| same | 274 | 5801 | 0.1328 | 0.8058 | 0.0000 |
| worsened | 57 | 929 | 0.9400 | 0.0607 | -0.0047 |

## Case-level correlation with rank movement

| Diagnostic | N | Pearson r | Spearman rho |
|---|---:|---:|---:|
| sum_phi | 375 | -0.1895 | -0.2491 |
| raw_shift_abs | 375 | 0.0079 | -0.1012 |
| raw_late_to_early_shift_ratio | 374 | 0.1582 | 0.0346 |
| raw_shift_sign_consistency | 375 | -0.0572 | -0.0577 |
| signal_retention_ratio | 374 | 0.1928 | 0.2568 |
| ar_late_to_early_shift_ratio | 371 | 0.0245 | 0.0316 |
| ar_early_to_late_decay_fraction | 371 | -0.0245 | -0.0316 |
| ar_initial_to_late_retention_ratio | 154 | -0.0388 | 0.1423 |
| ar_initial_to_late_decay_fraction | 154 | 0.0388 | -0.1423 |
| ar_score_minus_raw_score | 375 | 0.2801 | 0.4335 |

## Interpretation guardrails

- This report diagnoses association; it does not by itself prove that attenuation caused a rank change.
- Case summaries use the union of the Raw and AR root-service top-3 metrics because service scoring uses `mean_top3`.
- Ratios with a practically zero denominator are recorded as null rather than guessed.

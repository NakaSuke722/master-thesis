# AMBER affine-unit invariance diagnosis

Every metric in both windows is transformed as `y = scale * x + offset`.
The underlying case is unchanged, so exact score and ranking preservation is the target.

Cases: 375

| Scope | Transform | Cases | Full ranking same | Top-1 same | Root rank same | Scores close | Lag coefficients close | Mean rank displacement | Max root-rank shift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | scale_down_0_001 | 375 | 0.0% | 85.3% | 79.2% | 0.0% | 0.0% | 1.4657 | 11 |
| re1_ob | scale_down_0_001 | 125 | 0.0% | 86.4% | 81.6% | 0.0% | 0.0% | 0.9120 | 4 |
| re1_ss | scale_down_0_001 | 125 | 0.0% | 90.4% | 88.0% | 0.0% | 0.0% | 0.8097 | 2 |
| re1_tt | scale_down_0_001 | 125 | 0.0% | 79.2% | 68.0% | 0.0% | 0.0% | 2.6756 | 11 |
| overall | scale_down_0_001_offset_100 | 375 | 0.0% | 59.7% | 57.9% | 0.0% | 0.0% | 5.0409 | 47 |
| re1_ob | scale_down_0_001_offset_100 | 125 | 0.0% | 44.0% | 40.8% | 0.0% | 0.0% | 2.9853 | 9 |
| re1_ss | scale_down_0_001_offset_100 | 125 | 0.0% | 74.4% | 73.6% | 0.0% | 0.0% | 2.9158 | 12 |
| re1_tt | scale_down_0_001_offset_100 | 125 | 0.0% | 60.8% | 59.2% | 0.0% | 0.0% | 9.2215 | 47 |
| overall | scale_up_1000 | 375 | 4.8% | 89.9% | 81.1% | 0.0% | 0.0% | 0.7273 | 4 |
| re1_ob | scale_up_1000 | 125 | 6.4% | 94.4% | 88.8% | 0.0% | 0.0% | 0.5251 | 1 |
| re1_ss | scale_up_1000 | 125 | 8.0% | 96.0% | 93.6% | 0.0% | 0.0% | 0.3241 | 2 |
| re1_tt | scale_up_1000 | 125 | 0.0% | 79.2% | 60.8% | 0.0% | 0.0% | 1.3326 | 4 |

`root_rank_delta = baseline_root_rank - transformed_root_rank`; a non-zero value is unit sensitivity, not a performance gain or loss.
`Scores close` uses `rtol=1e-6, atol=1e-8`; ranking equality is checked exactly.

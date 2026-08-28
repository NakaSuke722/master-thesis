# AMBER affine-unit invariance diagnosis

Every metric in both windows is transformed as `y = scale * x + offset`.
The underlying case is unchanged, so exact score and ranking preservation is the target.

Cases: 375

| Scope | Transform | Cases | Full ranking same | Top-1 same | Root rank same | Scores close | Lag coefficients close | Mean rank displacement | Max root-rank shift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | scale_down_0_001 | 375 | 100.0% | 100.0% | 100.0% | 99.7% | 99.7% | 0.0000 | 0 |
| re1_ob | scale_down_0_001 | 125 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0000 | 0 |
| re1_ss | scale_down_0_001 | 125 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0000 | 0 |
| re1_tt | scale_down_0_001 | 125 | 100.0% | 100.0% | 100.0% | 99.2% | 99.2% | 0.0000 | 0 |
| overall | scale_down_0_001_offset_100 | 375 | 99.7% | 99.7% | 99.7% | 99.2% | 97.6% | 0.0007 | 1 |
| re1_ob | scale_down_0_001_offset_100 | 125 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0000 | 0 |
| re1_ss | scale_down_0_001_offset_100 | 125 | 100.0% | 100.0% | 100.0% | 98.4% | 92.8% | 0.0000 | 0 |
| re1_tt | scale_down_0_001_offset_100 | 125 | 99.2% | 99.2% | 99.2% | 99.2% | 100.0% | 0.0021 | 1 |
| overall | scale_up_1000 | 375 | 100.0% | 100.0% | 100.0% | 99.7% | 99.7% | 0.0000 | 0 |
| re1_ob | scale_up_1000 | 125 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0000 | 0 |
| re1_ss | scale_up_1000 | 125 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0000 | 0 |
| re1_tt | scale_up_1000 | 125 | 100.0% | 100.0% | 100.0% | 99.2% | 99.2% | 0.0000 | 0 |

`root_rank_delta = baseline_root_rank - transformed_root_rank`; a non-zero value is unit sensitivity, not a performance gain or loss.
`Scores close` uses `rtol=1e-5, atol=1e-7` to allow float64 cancellation after affine offsets; ranking equality is checked exactly.

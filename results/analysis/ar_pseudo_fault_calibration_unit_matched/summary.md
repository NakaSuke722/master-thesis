# AR unit-invariance matched control: normal-only pseudo-fault

Both variants use the same current code, processed data, cases, and split.
The only model difference is `ar_input_scaling`: `none` (control) versus `normal_standard` (unit-invariant).

Paired delta = unit-invariant - control. For BF calibration statistics, a negative delta means less no-fault evidence after unit standardization.

Cases: 375

| Scope | Variant | Cases | Median max BF | P90 max BF | Median service BF | Positive service fraction |
|---|---|---:|---:|---:|---:|---:|
| overall | control | 375 | 1360.4878 | 3247.6000 | 134.8673 | 89.7% |
| overall | unit_invariant | 375 | 582.7567 | 1089.2729 | 156.0765 | 89.6% |
| re1_ob | control | 125 | 254.1720 | 1188.2955 | 40.4723 | 85.0% |
| re1_ob | unit_invariant | 125 | 245.0502 | 567.8624 | 35.7025 | 84.8% |
| re1_ss | control | 125 | 2285.2104 | 3260.4212 | 171.5848 | 86.5% |
| re1_ss | unit_invariant | 125 | 534.5991 | 807.6069 | 161.4181 | 86.6% |
| re1_tt | control | 125 | 1794.2283 | 2604.1415 | 173.7405 | 97.7% |
| re1_tt | unit_invariant | 125 | 958.0582 | 1379.4050 | 256.0254 | 97.6% |

## Paired deltas

| Scope | Statistic | Median delta | Mean delta | Unit-invariant lower | Same | Unit-invariant higher |
|---|---|---:|---:|---:|---:|---:|
| overall | max_service_score | -500.3609 | -837.8232 | 281 | 0 | 94 |
| overall | median_service_score | 7.0004 | 14.4081 | 153 | 0 | 222 |
| overall | positive_service_fraction | 0.0000 | -0.0010 | 85 | 204 | 86 |
| re1_ob | max_service_score | 0.0101 | -147.1380 | 57 | 0 | 68 |
| re1_ob | median_service_score | -3.0435 | -2.4598 | 77 | 0 | 48 |
| re1_ob | positive_service_fraction | 0.0000 | -0.0024 | 32 | 60 | 33 |
| re1_ss | max_service_score | -1722.8927 | -1587.9875 | 109 | 0 | 16 |
| re1_ss | median_service_score | -2.1702 | -9.4351 | 72 | 0 | 53 |
| re1_ss | positive_service_fraction | 0.0000 | 0.0004 | 13 | 98 | 14 |
| re1_tt | max_service_score | -731.9938 | -778.3440 | 115 | 0 | 10 |
| re1_tt | median_service_score | 49.6562 | 55.1191 | 4 | 0 | 121 |
| re1_tt | positive_service_fraction | 0.0000 | -0.0009 | 40 | 46 | 39 |

# Final Counterfactual AR diagnostics

Positive rank delta means the final reference moved the ground-truth service upward.
The reference is unit-invariant stationary Counterfactual AR+BF with horizon-aware uncertainty.

| Scope | Comparison | Improved | Same | Worsened | Mean delta | Top-1 lost | Top-1 gained | Exact p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | Raw+BF -> reference | 107 | 230 | 38 | 0.6987 | 31 | 70 | 0.0001 |
| overall | No horizon -> reference | 33 | 314 | 28 | -0.0080 | 4 | 12 | 0.0768 |
| re1_ob | Raw+BF -> reference | 46 | 75 | 4 | 0.4640 | 1 | 33 | 0.0000 |
| re1_ob | No horizon -> reference | 14 | 101 | 10 | 0.0720 | 2 | 5 | 0.4531 |
| re1_ss | Raw+BF -> reference | 15 | 97 | 13 | 0.0720 | 11 | 12 | 1.0000 |
| re1_ss | No horizon -> reference | 6 | 113 | 6 | 0.0400 | 2 | 2 | 1.0000 |
| re1_tt | Raw+BF -> reference | 46 | 58 | 21 | 1.5600 | 19 | 25 | 0.4514 |
| re1_tt | No horizon -> reference | 13 | 100 | 12 | -0.1360 | 0 | 5 | 0.0625 |
| cpu | Raw+BF -> reference | 13 | 46 | 16 | -0.1333 | 16 | 13 | 0.7111 |
| cpu | No horizon -> reference | 5 | 69 | 1 | 0.0933 | 0 | 1 | 1.0000 |
| delay | Raw+BF -> reference | 39 | 34 | 2 | 2.1067 | 1 | 21 | 0.0000 |
| delay | No horizon -> reference | 14 | 56 | 5 | 0.2533 | 1 | 5 | 0.2188 |
| disk | Raw+BF -> reference | 12 | 55 | 8 | 0.0400 | 8 | 12 | 0.5034 |
| disk | No horizon -> reference | 1 | 71 | 3 | -0.0267 | 1 | 0 | 1.0000 |
| loss | Raw+BF -> reference | 33 | 33 | 9 | 1.3867 | 3 | 14 | 0.0127 |
| loss | No horizon -> reference | 12 | 44 | 19 | -0.3733 | 2 | 6 | 0.2891 |
| mem | Raw+BF -> reference | 10 | 62 | 3 | 0.0933 | 3 | 10 | 0.0923 |
| mem | No horizon -> reference | 1 | 74 | 0 | 0.0133 | 0 | 0 | 1.0000 |

## Forecast-uncertainty multiplier audit

- Metrics: 137845
- Maximum: 92.0680
- P99: 6.7638
- >=10: 186 (0.13%)
- >=50: 12 (0.01%)
- Cases whose root top-3 includes >=10: 0
- Cases whose root top-3 includes >=50: 0

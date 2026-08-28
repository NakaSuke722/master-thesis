# Final Counterfactual AR one-axis sensitivity

Reference: `unit_invariant_r0_98_p3`. Rank delta is variant - reference, so positive favors the reference.

Pseudo-fault BF is a relative calibration diagnostic; lower is better, but the absolute BF is not calibrated as a detector threshold.

| Variant | Macro AC@1 | AC@3 | AC@5 | Avg@5 | Ref improved | Same | Ref worsened | Ref Top-1 gain/loss | Top-1 exact p | Constrained metrics | Runtime sec | Pseudo median max BF | Pseudo P90 max BF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unit_invariant_r0_98_p3 | 0.7440 | 0.9280 | 0.9680 | 0.8891 | 0 | 375 | 0 | 0/0 | 1.0000 | 25.2% | 94.0 | 582.7567 | 1083.8826 |
| unit_invariant_r0_95_p3 | 0.7387 | 0.9147 | 0.9627 | 0.8805 | 21 | 346 | 8 | 2/0 | 0.5000 | 51.2% | 104.0 | 602.4623 | 1126.5611 |
| unit_invariant_r0_99_p3 | 0.7493 | 0.9227 | 0.9653 | 0.8901 | 7 | 354 | 14 | 1/3 | 0.6250 | 14.7% | 117.0 | 601.1103 | 1060.7517 |
| unit_invariant_r0_98_p1 | 0.7227 | 0.9280 | 0.9680 | 0.8880 | 13 | 344 | 18 | 11/3 | 0.0574 | 25.3% | 108.0 | 578.2783 | 1011.3185 |
| unit_invariant_r0_98_p5 | 0.7493 | 0.9333 | 0.9653 | 0.8928 | 13 | 348 | 14 | 4/6 | 0.7539 | 26.6% | 126.0 | 585.6639 | 1149.7488 |
| unit_invariant_no_horizon_uncertainty | 0.7227 | 0.9093 | 0.9627 | 0.8784 | 33 | 314 | 28 | 12/4 | 0.0768 | 25.2% | 104.0 | 715.5428 | 1344.3515 |

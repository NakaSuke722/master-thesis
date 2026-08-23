# Direct AR Bayes Factor synthetic validation

Seed: 20260823; AR order: 1; pre/post samples: 300/300.

`log BF > 0` supports separate pre/post AR processes; `log BF > log(10)` is counted as strong evidence.

Overall validation: **PASS**.

| Scenario | Expected structural change | Median log BF | 5%-95% | Positive | Strong | Check |
|---|---:|---:|---:|---:|---:|---:|
| no_change | False | -8.2485 | [-9.3889, -6.0529] | 0.0% | 0.0% | PASS |
| persistent_mean_shift | True | 15.1197 | [7.5127, 24.9465] | 100.0% | 100.0% | PASS |
| ar_coefficient_change | True | 55.0641 | [37.1900, 76.7170] | 100.0% | 100.0% | PASS |
| innovation_variance_change | True | 57.4011 | [39.3381, 76.0101] | 100.0% | 100.0% | PASS |
| single_spike | False | -7.9076 | [-9.2303, -4.6901] | 0.0% | 0.0% | PASS |

The single-spike scenario is a transient innovation propagated by the unchanged AR process, not a persistent parameter change.

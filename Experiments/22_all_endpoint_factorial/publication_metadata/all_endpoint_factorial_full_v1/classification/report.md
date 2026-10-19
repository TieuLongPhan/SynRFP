# Full-row classification factor analysis

5040 completed canonical cells across 14 panels/regimes.
All original declared rows are retained; no subsampling. No baseline fits are included.
Observed ranks are exploratory diagnostics and do not designate a configuration for final benchmarking.

| Configuration | Mean panel rank | Worst panel rank | Equal-panel mean score |
|---|---:|---:|---:|
| synrfp-morgan-r2-b2048-parity-delta_union | 7.36 | 21 | 0.94569 |
| synrfp-wl-r2-b2048-parity-delta_union | 7.79 | 19 | 0.94597 |
| synrfp-morgan-r2-b2048-parity-delta | 10.71 | 34 | 0.94178 |
| synrfp-wl-r2-b2048-parity-delta | 11.21 | 35 | 0.94131 |
| synrfp-path-r3-b2048-parity-delta_union | 12.29 | 34 | 0.94145 |
| synrfp-path-r2-b2048-parity-delta_union | 12.71 | 36 | 0.93674 |
| synrfp-wl-r2-b2048-parity-roles | 14.14 | 23 | 0.93907 |
| synrfp-morgan-r2-b2048-parity-roles | 15.07 | 27 | 0.93858 |
| synrfp-path-r2-b2048-parity-roles | 16.79 | 43 | 0.93412 |
| synrfp-path-r3-b2048-parity-delta | 18.71 | 50 | 0.93508 |
| synrfp-morgan-r3-b2048-parity-delta_union | 21.57 | 39 | 0.93339 |
| synrfp-path-r3-b2048-parity-roles | 21.64 | 61 | 0.93499 |
| synrfp-morgan-r3-b2048-parity-delta | 21.86 | 38 | 0.93360 |
| synrfp-wl-r3-b2048-parity-delta | 21.93 | 39 | 0.93358 |
| synrfp-wl-r3-b2048-parity-delta_union | 22.50 | 43 | 0.93214 |

Main effects and all higher-order interactions are in the paired/panel CSVs. All effects remain task-separated.
Classification/assay repeats are five held-out grouped folds; clustering repeats are five initialization seeds.
Yield uses all original64 saved partitions, with BH random and published OOD reported separately.
Clustering K derives only from sample size. AMI/ARI are external diagnostics; internal silhouette and seed stability support label-free assessment.
These populations were previously inspected; untouched evaluation is required for a confirmatory selected-configuration claim.

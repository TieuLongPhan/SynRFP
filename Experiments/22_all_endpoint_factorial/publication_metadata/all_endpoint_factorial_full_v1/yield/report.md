# Full-row yield factor analysis

4608 completed canonical cells across 4 panels/regimes.
All original declared rows are retained; no subsampling. No baseline fits are included.
Observed ranks are exploratory diagnostics and do not designate a configuration for final benchmarking.

| Configuration | Mean panel rank | Worst panel rank | Equal-panel mean score |
|---|---:|---:|---:|
| synrfp-wl-r3-b2048-parity-roles | 3.00 | 5 | 0.71473 |
| synrfp-morgan-r3-b2048-parity-roles | 5.25 | 9 | 0.71002 |
| synrfp-morgan-r2-b2048-parity-delta_union | 7.75 | 15 | 0.70685 |
| synrfp-wl-r3-b2048-parity-delta_union | 8.75 | 14 | 0.70087 |
| synrfp-wl-r3-b2048-parity-union | 10.00 | 16 | 0.70324 |
| synrfp-morgan-r3-b2048-parity-delta | 10.25 | 27 | 0.69159 |
| synrfp-morgan-r3-b2048-parity-delta_union | 10.25 | 24 | 0.69822 |
| synrfp-wl-r3-b2048-parity-delta | 10.50 | 16 | 0.69859 |
| synrfp-morgan-r2-b2048-parity-roles | 11.75 | 25 | 0.71771 |
| synrfp-wl-r2-b2048-parity-delta_union | 13.50 | 20 | 0.70168 |
| synrfp-morgan-r3-b2048-parity-union | 14.00 | 29 | 0.70288 |
| synrfp-wl-r2-b2048-parity-roles | 15.00 | 27 | 0.71198 |
| synrfp-wl-r2-b2048-parity-union | 16.00 | 29 | 0.70365 |
| synrfp-morgan-r2-b2048-parity-union | 17.75 | 32 | 0.70398 |
| synrfp-morgan-r3-b2048-cw-union | 21.75 | 38 | 0.67575 |

Main effects and all higher-order interactions are in the paired/panel CSVs. All effects remain task-separated.
Classification/assay repeats are five held-out grouped folds; clustering repeats are five initialization seeds.
Yield uses all original64 saved partitions, with BH random and published OOD reported separately.
Clustering K derives only from sample size. AMI/ARI are external diagnostics; internal silhouette and seed stability support label-free assessment.
These populations were previously inspected; untouched evaluation is required for a confirmatory selected-configuration claim.

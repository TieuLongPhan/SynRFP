# Full-row clustering factor analysis

4320 completed canonical cells across 12 panels/regimes.
All original declared rows are retained; no subsampling. No baseline fits are included.
Observed ranks are exploratory diagnostics and do not designate a configuration for final benchmarking.

| Configuration | Mean panel rank | Worst panel rank | Equal-panel mean score |
|---|---:|---:|---:|
| synrfp-path-r2-b2048-parity-delta | 4.50 | 17 | 0.57001 |
| synrfp-morgan-r2-b2048-parity-delta | 7.58 | 19 | 0.55897 |
| synrfp-path-r2-b2048-minhash-delta | 8.08 | 23 | 0.55815 |
| synrfp-wl-r2-b2048-parity-delta | 8.58 | 23 | 0.55692 |
| synrfp-path-r2-b2048-minhash-delta_union | 9.67 | 24 | 0.55860 |
| synrfp-path-r2-b2048-cw-delta | 10.33 | 29 | 0.55152 |
| synrfp-morgan-r2-b2048-minhash-delta | 11.83 | 30 | 0.54043 |
| synrfp-wl-r2-b2048-minhash-delta | 12.08 | 28 | 0.54201 |
| synrfp-path-r2-b2048-cw-delta_union | 12.92 | 32 | 0.54879 |
| synrfp-path-r3-b2048-parity-delta | 14.00 | 35 | 0.54431 |
| synrfp-morgan-r2-b2048-cw-delta | 14.33 | 28 | 0.53681 |
| synrfp-wl-r2-b2048-cw-delta | 15.25 | 26 | 0.53533 |
| synrfp-path-r3-b2048-minhash-delta | 17.58 | 38 | 0.53271 |
| synrfp-path-r2-b2048-parity-delta_union | 18.08 | 39 | 0.54013 |
| synrfp-path-r3-b2048-cw-delta | 19.75 | 43 | 0.52799 |

Main effects and all higher-order interactions are in the paired/panel CSVs. All effects remain task-separated.
Classification/assay repeats are five held-out grouped folds; clustering repeats are five initialization seeds.
Yield uses all original64 saved partitions, with BH random and published OOD reported separately.
Clustering K derives only from sample size. AMI/ARI are external diagnostics; internal silhouette and seed stability support label-free assessment.
These populations were previously inspected; untouched evaluation is required for a confirmatory selected-configuration claim.

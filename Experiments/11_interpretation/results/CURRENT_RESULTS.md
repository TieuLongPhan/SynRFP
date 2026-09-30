# Current label-free classification status

This snapshot is generated from completed benchmark records. XGBoost TPL coverage is incomplete, so its ranking is provisional.

## KNN
Label-free endpoints represented: **14**; SynRFP leaders: **11/14**.

| Fingerprint | Mean MCC across represented endpoints |
|---|---:|
| synrfp-wl-r2-b1024-parity-delta | 0.900 |
| drfp | 0.865 |
| rdkit_diff | 0.769 |
| rxnfp_mlm | 0.724 |
| rdkit_struct | 0.553 |

## MLP
Label-free endpoints represented: **14**; SynRFP leaders: **11/14**.

| Fingerprint | Mean MCC across represented endpoints |
|---|---:|
| synrfp-wl-r2-b1024-parity-delta | 0.939 |
| drfp | 0.917 |
| rdkit_diff | 0.912 |
| rdkit_struct | 0.870 |
| rxnfp_mlm | 0.843 |

## XGB
Label-free endpoints represented: **14**; SynRFP leaders: **4/14**.

| Fingerprint | Mean MCC across represented endpoints |
|---|---:|
| rdkit_diff | 0.936 |
| synrfp-wl-r2-b1024-parity-delta | 0.932 |
| drfp | 0.915 |
| rdkit_struct | 0.886 |
| rxnfp_mlm | 0.748 |

## Coverage

| Model | Completed label-free records | All completed benchmark records |
|---|---:|---:|
| kNN | 1,820 | 2,188 |
| MLP | 1,960 | benchmark extras included; see inventory |
| XGBoost | 1,870 | 2,249 |

The authoritative machine-readable inventory is `classification_progress.json`.

## Family sensitivity

The leave-one-family-out analysis remains positive for SynRFP versus DRFP:

| Model | All matched | Range after leaving out one family |
|---|---:|---:|
| kNN | +0.0314 | +0.0256 to +0.0411 |
| MLP | +0.0215 | +0.0160 to +0.0276 |
| XGBoost (provisional) | +0.0181 | +0.0149 to +0.0215 |

XGBoost versus RDKit-difference is inconclusive in the same analysis
(-0.0056 to +0.0021 after leaving out one family).

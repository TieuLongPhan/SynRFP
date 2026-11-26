# Within-dataset barrier and reaction-energy targets

Each endpoint uses the five archived group folds, the same row membership for both targets and arms, the published 512-unit MLP recipe, and training-only target standardization. The primary barrier fits are the original CUDA fits. The second targets use the equivalent native CPU worker with the same hyperparameters and seed. CPU and CUDA can follow different optimization paths.

| Dataset | Training folds passing frozen rule | Primary normalized MSE gain | Second normalized MSE gain | Second minus primary, 95% CI |
|---|---:|---:|---:|---:|
| b97xd3 | 0/5 | +0.0001 | -0.0051 | -0.0052 [-0.0198, +0.0100] |
| cyclo | 0/5 | +0.0100 | -0.0075 | -0.0175 [-0.0284, -0.0066] |
| gdb7 | 0/5 | +0.0017 | -0.0018 | -0.0035 [-0.0197, +0.0119] |

A paired interaction excludes zero for cyclo. QuantumPioneer's held-out interaction was inconclusive. The frozen eligibility thresholds were 5% of training rows, 30 query groups and ten exact-D classes. These benchmark-cohort comparisons are exploratory. All 10,000 bootstrap draws resample the same held-out groups for both targets and arms.

The within-exact-class variance resolved by U, as a fraction of total variance, was b97xd3 0.0097% and 0.0016%, cyclo 0.0000% and 0.0000%, gdb7 0.0006% and 0.0001% for the primary and second targets, respectively. The variance measure uses all rows. The paired predictive gains use held-out rows and therefore answer a different question.

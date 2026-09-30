# SynRFP applicability and failure-mode analysis

This descriptive analysis uses completed CV classification rows and the
existing ten-start clustering summaries. It is not a new benchmark or
a replacement for the completeness-gated final leaderboard.

## Classification (CV rows)

| Model | Baseline | SynRFP wins | Ties | Losses | Mean MCC margin | Loss datasets |
|---|---|---:|---:|---:|---:|---|
| knn | drfp | 12 | 0 | 2 | +0.0344 | tpl_b, tpl_ub |
| knn | rdkit_diff | 12 | 0 | 2 | +0.1317 | ecreact_1, ecreact_3 |
| knn | rdkit_struct | 14 | 0 | 0 | +0.3479 | none |
| knn | rxnfp_mlm | 14 | 0 | 0 | +0.1766 | none |
| mlp | drfp | 11 | 0 | 3 | +0.0229 | syntemp_r2, tpl_b, tpl_ub |
| mlp | rdkit_diff | 13 | 0 | 1 | +0.0277 | tpl_b |
| mlp | rdkit_struct | 14 | 0 | 0 | +0.0702 | none |
| mlp | rxnfp_mlm | 14 | 0 | 0 | +0.0976 | none |
| xgb | drfp | 11 | 0 | 3 | +0.0172 | syntemp_r2, tpl_b, tpl_ub |
| xgb | rdkit_diff | 5 | 0 | 9 | -0.0038 | ecreact_1, ecreact_3, mech_1, mech_2, schneider_b, schneider_ub, tpl_b, tpl_ub, uspto50k_ub |
| xgb | rdkit_struct | 14 | 0 | 0 | +0.0467 | none |
| xgb | rxnfp_mlm | 14 | 0 | 0 | +0.1856 | none |

The completed rows show a consistent pattern: SynRFP is strongest
against structural and RXNFP-MLM references, while RDKit difference is
the closest XGBoost comparator. TPL and fine-resolution SynTemp rows
are recurring stress cases and should be discussed explicitly.

## Clustering

The clustering leader is recorded per phase/dataset for both AMI and ARI
in `results/applicability_analysis.json`. Large-data exceptions should
be reported instead of averaged away.

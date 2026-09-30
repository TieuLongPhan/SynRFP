# Partial TPL XGBoost analysis

This is a live, descriptive summary of completed log cells. It must not
be used as the final TPL ranking until both workers finish all 25 splits
and the completeness audit passes.

| Dataset | Complete splits | Baseline | SynRFP wins | Losses | Mean MCC margin |
|---|---:|---|---:|---:|---:|
| tpl_ub | 21 | drfp | 0 | 21 | -0.0193 |
| tpl_ub | 20 | rdkit_diff | 0 | 20 | -0.0365 |
| tpl_ub | 20 | rdkit_struct | 20 | 0 | +0.0395 |
| tpl_ub | 20 | rxnfp | 20 | 0 | +0.1162 |
| tpl_ub | 20 | rxnfp_mlm | 20 | 0 | +0.2807 |
| tpl_b | 18 | drfp | 0 | 18 | -0.0094 |
| tpl_b | 18 | rdkit_diff | 0 | 18 | -0.0347 |
| tpl_b | 18 | rdkit_struct | 18 | 0 | +0.0454 |
| tpl_b | 18 | rxnfp | 18 | 0 | +0.2774 |
| tpl_b | 18 | rxnfp_mlm | 18 | 0 | +0.3531 |

The completed cells already show RDKit difference ahead of SynRFP on
both TPL views, with DRFP also ahead on the available matched splits.
This is a useful stress-case explanation, but the final table must be
regenerated after the remaining TPL cells complete.

# Cyclo and GDB7 grouped activation-barrier results

These are the two D-MPNN entries for Table S23. Both use the study 14
`chemprop_v2_default` model on the exact five grouped folds used by the
fingerprint comparisons.

| Endpoint | Target | Population | D-MPNN RMSE (kcal/mol) |
| --- | --- | ---: | ---: |
| Cyclo | Activation free energy, `G_act` | 5,269 | 4.194 ± 0.319 |
| GDB7 | Electronic activation barrier, `dE0` | 11,926 | 8.683 ± 0.289 |

The table reports the arithmetic mean and sample standard deviation across
five held-out fold RMSEs. Each source row is tested exactly once. The grouped
comparison now has D-MPNN entries for all nine endpoints in Table S23.

[cells.csv](cells.csv) provides the ten per-fold MAE, RMSE and R² scores
with counts and split hashes. [summary.csv](summary.csv) gives the aggregate
values at full precision. [AUDIT.json](AUDIT.json) records the selector and
prediction checks. The export uses completed archived fits.

# Raw scalar results for the activation-barrier comparison

- `per_fit_scores.csv`: 1,800 scores, covering nine endpoints × eight methods ×
  five repeats × five folds. Includes RMSE, MAE, R², configuration, sample
  counts, repeat/fold identifiers, and source hashes/paths.
- `per_repeat_mean_scores.csv`: 360 rows, averaging the five fold metrics
  within each dataset/method/repeat.
- `paired_repeat_scores_wide.csv`: 45 dataset/repeat rows, with every method's
  scores in separate columns for paired analysis. Keys are `dataset, repeat`.
- `AUDIT.json`: completeness checks and evidence boundaries. All 225 D-MPNN
  receipts and 1,525 fingerprint receipts were individually hash/metric checked.
  The other 50 fingerprint scores retain the published per-fit table as their
  hash-bound source, rather than inventing an individual receipt pointer.

Cyclo and GDB7 use their original study14 matched references; the other seven
endpoints use study40, with the repaired full-row RGD1 overlay from study66.
Do not replace Cyclo with the separately mapped study40 `cycloadd` cohort.

Repeat and fold indices are zero based. `cv_r0_f0` is repeat 1, fold 1.
The supplied repeat means match Table S13's stated aggregation: calculate each
metric per fold, average five fold metrics per repeat, then report the mean
and sample standard deviation across five repeat means. This differs from
calculating RMSE on pooled out-of-fold predictions.

The 25 folds are not independent replicates; repeated CV also reuses the same
underlying reactions. Choose a statistical method that accounts for paired
evaluation and overlapping training/test populations. These files do not
establish external confirmation of the exploratory task-selected SynRFP preset.
Do not pool different endpoint units into a single error mean. No statistical
test or new model training was run during this export.

These are raw **per-fit metric results**, not per-reaction predictions. The full
prediction arrays and source row ledgers remain local at the original artifact
paths; weights, raw chemical records and heavy arrays are not included here.

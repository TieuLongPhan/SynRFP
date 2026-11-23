# Completed reference results, 23 November 2026

`publication_metadata/20261123/` exports completed scalar results obtained after
the seven-task main/factorial publication (`9ff0611a`). It contains no model
weights, raw datasets, features, or per-row predictions. `MANIFEST.json` records
the local source paths and SHA-256 values of copied artifacts.

- Yield: all 55 method/evaluation-panel summaries (11 methods, five panels),
  R² and MAE tables, exact split counts, and the comparison audit. Only USPTO
  G-NAN uses the explicitly adapted FP64/log-sum-exp/variance-floor profile.
  BH random and Suzuki summarize ten saved splits, BH OOD four holdouts,
  NCS ten iteration means (four holdouts each), and USPTO five grouped folds.
  The selected SynRFP configuration remains exploratory.
- D-MPNN: all 396 matched scalar results. The 31 original RGD1 results are
  replaced in this export by the completed 353,984-row correction; all other
  results are retained. Random 5×5 summaries average folds within repeat
  before reporting variation across five repeat means. Grouped folds and
  published partitions remain separate. Target errors retain native units.
- SynCat classification: 70 full-cohort models across 14 endpoints, five folds
  once, with original selectors and no row cap. The summary reports sample SD
  across folds; this is not five repeated cross-validation runs.
- SynCat feasibility: five optimizer seeds on the same fixed 751 expert test
  rows, with calibration performed separately. Variation across seeds is not
  variation across independent test folds.

All 396 D-MPNN scalar receipt hashes and all 70 classification receipt hashes
were checked before publication. Historical sources, failed attempts and local
weights remain untouched. Parrot (2/30) and T5Chem (2/10) are incomplete and are
not represented as completed references in this snapshot. Existing seven-task
fingerprint main and factorial results remain in their previously published
locations, indexed by `docs/SEVEN_TASK_RESULTS.md` and
`docs/SEVEN_TASK_REFRESH_2026-10-28.md`.

## User-requested stop, 24 November 2026

All remaining benchmark jobs were stopped and automatic startup was disabled
so the workstation can be used for other tasks. Parrot has 2/30 completed fits
and T5Chem has 3/10; their unfinished comparisons are excluded from the completed
benchmark tables. Partial-run means are not presented as full references.
All local checkpoints and completed results remain available, including the
interrupted Parrot fit's saved optimizer/RNG checkpoint. Work must not resume
without a new user instruction. The stop receipt is
`publication_metadata/20261124_USER_STOPPED.json`.

## Primary barrier reference export correction

The earlier export omitted study14's already-completed Cyclo and GDB7 random
5×5 D-MPNN results. These are now exported separately as
`dmpnn/primary14_random_5x5_{cells,summary}.csv`, with 25 fits per endpoint.
All 50 native scalar receipts and exact shared DRFP split/row hashes were
checked. RMSE is 4.0288658443 ± 0.0478038997 for Cyclo and
8.5630382284 ± 0.0765523778 for GDB7 (kcal/mol). These use mean fold metrics
within repeat, then sample SD across the five repeat means, matching Table S13's
stated aggregation. Do not substitute pooled out-of-fold statistics or the
separately mapped study40 cycloadd cohort. No new training was performed.

# Request 93: yield with a second learner (random forest, optional XGBoost)

Written 27 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Report the branch and commit.

## Why

Every yield result in the paper uses one learner, the 512-unit MLP. Two claims depend on it:
1. SI S7.3: the yield configuration (WL radius 3, parity, D+S) ranks first among the seven
   fingerprints on the five yield panels.
2. SI S4: adding context to the signed delta raises yield R2 (D against D+S and D+U).
Neves et al. 2026 (Nat. Comput. Sci.) found that the representation mattered more than the model on
out-of-distribution Buchwald-Hartwig prediction, comparing RF, XGBoost, LR and MLP. We want to know
whether our ranking and our context effect survive a change of learner.

## Panels and splits

The five yield panels of the benchmark release (classic BH random, classic BH published holdouts,
Suzuki random, expanded BH 40 cluster holdouts, USPTO grouped), with exactly the saved splits and
fit-train rows used for the released MLP results. Same input fields and preprocessing.

## Representations

1. The seven benchmark fingerprints: SynRFP standard, SynRFP yield configuration, DRFP, cRXNFP,
   RDKit difference, RDKit structural, RXNFP-MLM (cached vectors, no re-encoding).
2. SynRFP context variants at the yield settings (WL radius 3, parity, 2,048 bits, seed 1): D only,
   D+S and D+U. D+S is the yield configuration above; reuse it.

## Learners

1. Random forest regressor (scikit-learn): 500 trees, max_features = sqrt, min_samples_leaf = 1,
   n_jobs as available, seed 42. No tuning on test data. If you tune at all, use only the fit-train
   rows (e.g. an inner split) and report the grid.
2. Optional: XGBoost regressor with default settings plus early stopping on an inner validation split
   of the fit-train rows. Skip and say so if it is not installed or too slow on USPTO.

## Output

On a new branch `exp/93-yield-learner`:
- `cells.csv`: panel, split, representation, learner, R2, MAE, RMSE, n_test, runtime.
- `summary.csv`: mean and SD per panel, representation and learner, with the released MLP values
  joined for comparison (same splits).
- `ranking.csv`: per panel and learner, the rank of each fingerprint and the Kendall correlation with
  the MLP ranking.
- `context.csv`: per panel and learner, paired differences D+S minus D and D+U minus D over splits
  (mean, SD, wins).
- `note.md`: what ran, skips, runtime, hardware, package versions.

## Constraints

Same as requests 74 to 92: pinned commits and hashes, package versions, no weights or full prediction
dumps. Do not change existing results. Report negative results as they are.

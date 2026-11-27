# Request 92: out-of-distribution yield analysis (distance bins, condition ranking, examples)

Written 27 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Report the branch and commit.

## Why

SI S7.3 shows, split by split, that the yield configuration (WL radius 3, parity, D+S) gains most
over the other fingerprints on held-out reactions, while the standard configuration (signed delta
only) falls behind there. We only have split-level R2. To show where the gain comes from, in the
style of Neves et al. 2026 (Nat. Comput. Sci., Figs. 4 and 5), we need reaction-level predictions.

Panels: the 40 released cluster holdouts of the expanded BH set (`ncs_bh`, splits i0_f0 to i9_f3)
and the 5 grouped USPTO folds. Same yield protocol, learner and seeds as
`Experiments/13_yield/publication_metadata/continuous_allfps_v1` and the benchmark release.

## A. Reaction-level test predictions

For SynRFP standard, SynRFP yield configuration, DRFP and RDKit structural (the strongest other
fingerprints on these panels): one row per test reaction and split with reaction id, observed yield,
prediction. Use saved predictions if they exist; otherwise refit with the exact saved protocol and
confirm that split-level R2 reproduces the released values (report any difference). If cheap, also
fit five learner seeds per split so a prediction spread is available for B.2.

## B. Performance against distance from training (Fig. 4 analogue)

1. For each test reaction, its maximum Tanimoto similarity to the training reactions of that split,
   computed in one fixed label-free space for all methods: DRFP of the full reaction (the space in
   which the clusters were built). Also report it in each method's own space.
2. Five quantile bins of this similarity (pooled over splits). Per method and bin: MAE, R2,
   number and share of test reactions, mean and SD over the 40 splits.
3. If five seeds were fitted: bin by the SD of the seed predictions instead (a confidence proxy) and
   report the same quantities.

## C. Ranking conditions for held-out substrate pairs (Fig. 5a analogue)

On the expanded BH cluster holdouts, for every held-out aryl halide and amine pair tested under at
least 8 conditions: Spearman correlation between predicted and observed yields across its
conditions, whether the top-1 and best of the top-4 predicted conditions reach yield above 10%, and
the same for a training-mean baseline. Per method: mean over pairs, and the number of pairs where
nothing above 10% was observed. Report how many pairs qualify.

## D. Examples

Same selection rules as requests 88 and 89 (deterministic sort key, one per Bemis-Murcko scaffold of
the largest component, prefer fewer heavy atoms, report the eligible pool):
up to 4 held-out reactions where |yield-config error| is much smaller than the best of |DRFP error|
and |RDKit structural error|, and up to 2 of the reverse. For each: reaction with catalyst, base and
solvent, observed yield, the four predictions, split, and the training similarity from B.1.
Also up to 2 held-out substrate pairs from C where the yield configuration ranks a successful
condition first and DRFP does not.

## Output

On a new branch `exp/92-yield-ood-analysis`: `predictions_summary.csv` (split-level reproduction),
`distance_bins.csv`, `confidence_bins.csv` (if B.3 ran), `condition_ranking.csv`,
`examples.csv`, `note.md`. No full per-row prediction dumps beyond what B to D need.

## Constraints

Same as requests 74 to 91: pinned commits and hashes, package versions, no weights. Do not change the
benchmark results. Report negative results as they are.

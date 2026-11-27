# Request 89: reaction examples for the feasibility and barrier comparisons with neural models

Written 27 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Report the branch and commit.

## Why

SI S7 compares SynRFP with task-trained neural models. Two results need reactions to show what lies
behind the numbers: the large feasibility win over SynCat on the expert test, and the large barrier
gap to D-MPNN on the small-molecule DFT sets. Use saved predictions only. No new training.

Same selection rules as request 88: deterministic (state the sort key), at most one reaction per
Bemis-Murcko scaffold of the largest component, prefer fewer heavy atoms, report the eligible pool.
Use the column conventions of `data/chemistry_cases` in the paper repo and atom-mapped SMILES where a
mapping exists.

## A. Feasibility: SynRFP against SynCat on the expert test (751 reactions, 341 feasible)

Using the saved predictions of the SynRFP standard preset (seed 1 or the first saved seed) and the
adapted SynCat reference on the fixed expert test:

1. Counts of the four outcome cells (both right, only SynRFP right, only SynCat right, both wrong),
   separately for expert-feasible and expert-infeasible reactions.
2. Up to 4 reactions labelled infeasible by the experts that SynRFP classifies correctly and SynCat
   does not, and up to 2 of the reverse (SynCat right, SynRFP wrong). Prefer reactions whose expert
   annotation gives a reason, if the dataset records one.
3. For each: reaction, expert label (and reason if recorded), both predicted probabilities and the
   decision thresholds used.

## B. Barriers: largest fingerprint errors relative to D-MPNN

On GDB7 and RGD1, grouped folds, using the saved test predictions of the SynRFP task-selected preset
and D-MPNN on the same folds:

1. For each dataset, the 3 reactions with the largest value of
   |SynRFP error| - |D-MPNN error|, and for comparison the 1 reaction with the largest value of
   |D-MPNN error| - |SynRFP error|.
2. For each: reaction, reference barrier, both predictions, fold, and the number of training
   reactions with the same exact signed counts (WL radius 2).
3. A short descriptive note per dataset: do the worst SynRFP cases share a feature (ring strain,
   multiple bond changes, small molecule size, unusual elements)? Describe only what the examples
   show; do not claim a mechanism.

## Output

On a new branch `exp/89-...`: `feasibility_outcomes.csv`, `feasibility_examples.csv`,
`barrier_examples.csv`, `note.md`.

## Constraints

Same as requests 74 to 88: pinned commits and hashes, package versions, no weights. Example rows only,
no full per-row prediction dumps. Do not change the benchmark results.

# Request 85: T5Chem reagent export and Parrot condition audit

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Report the branch and commit.

## Why

The SI compares the fingerprints with task-trained neural models on classification, yield and
barriers. Reagent and condition prediction have no neural reference yet. The workstation reported:

- T5Chem reagents, native split, seeds 42, 43, 44 complete: 14,215 test rows, seed 42 exact-set
  16.71 %, micro-F1 0.5050, Jaccard 0.4059.
- Parrot conditions, native split, seeds 42 and 43: 65,946 test rows, seed 42 exact-set 2.50 %,
  micro-F1 0.0917, Jaccard 0.0827.

T5Chem looks usable. Parrot does not. The published Parrot-LM-E overall top-1 on USPTO-Condition is
about 27.4 % (all condition slots correct, order ignored, 68,075 random-split test reactions), and
even the 2018 fingerprint baseline FPRCR reports 16.9 %. Our fingerprints reach 5 to 7 % exact-set
on the native condition split. A Parrot score of 2.5 % points to a scoring or test-set mismatch.

## Task A (no training): T5Chem export

Export per-seed metrics for seeds 42, 43, 44 on the native reagent test set, computed with the same
scorer as the fingerprint reagent benchmark (exact-set accuracy, micro-F1, Jaccard), plus mean and
sample SD. Confirm the 14,215 test rows are the same rows as the fingerprint native test set
(row identifiers or a hash of the sorted test IDs).

## Task B (no training first): Parrot audit and rescoring

1. Test rows: which rows were scored? Compare with the fingerprint native condition test set
   (65,946 rows) and with Parrot's original 68,075-row test set. Report overlap counts.
2. Rescore saved predictions of seeds 42 and 43 with:
   - empty slots (no catalyst, no reagent 2, ...) dropped from both prediction and reference, not
     scored as a component;
   - components compared as canonical SMILES (RDKit, same canonicalization as the fingerprint
     benchmark), so name/SMILES or tautomer-free format differences do not count as errors;
   - slot order ignored.
   Report exact-set, micro-F1 and Jaccard under our scorer, and also Parrot's own top-1 overall
   accuracy under its original evaluation script, on the same rows.
3. Decision rule: only if the rescored seed 42 and 43 exact-set or Parrot top-1 lands in a plausible
   range (roughly 15 % or more, near the published value) train seed 44. If it stays near 2.5 %,
   do not train more seeds. Describe what is wrong instead (vocabulary, label format, checkpoint,
   test rows).

## Output

On a new branch `exp/85-...`:
`t5chem_reagent_seeds.csv`, `parrot_condition_audit.csv` (one row per seed and scoring variant),
`test_row_overlap.csv`, `neural_reagent_condition_note.md` (one paragraph: are both references
usable, on which rows, and how they were scored).

## Constraints

Same as requests 74 to 84: pinned commits and hashes, package versions, no weights, no per-row
predictions, state anything that cannot be reproduced. Do not change the fingerprint results.

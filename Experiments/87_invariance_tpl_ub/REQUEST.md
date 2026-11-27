# Request 87: invariance audit on TPL UB instead of TPL B

Written 27 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Report the branch and commit.

## Why

Request 86 (`exp/86-invariance-all-fps`, 3fdd4446) audited Schneider UB, TPL B, ECReact L1 and Mech L2.
The paper wants the unbalanced TPL endpoint, matching Schneider UB, so the table uses TPL UB in place
of TPL B. Everything else stays as in request 86.

## Task

Run exactly the request 86 protocol and code (`Experiments/86_invariance_all_fps/`) on TPL UB:

- 2,000 reactions sampled without replacement with `np.random.default_rng(0)` from the benchmark
  release's n1 inputs;
- original, randomized (component and SMILES order) and reversed forms;
- all seven benchmark arms with their release settings, plus the DRFP canonical-input control;
- the same columns as `perturbation_all_fps.csv`, the encoding failure ledger and the warnings table.

Keep the three other endpoints of request 86 unchanged; do not rerun them unless the code requires it.

## Output

On a new branch `exp/87-...`: `perturbation_tpl_ub.csv` (same columns as
`perturbation_all_fps.csv`), `encoding_failures_tpl_ub.csv`, `encoding_warnings_tpl_ub.csv`, and a
short `note.md` (one paragraph; state any encoding failures and their cause).

## Constraints

Same as requests 74 to 86: pinned commits and hashes, package versions, no weights, no per-row
fingerprints. Do not change the benchmark results.

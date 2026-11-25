# Request 76: do stereo-aware signed counts resolve the ECReact stereo-only reactions?

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Export the files named here and
report the branch and commit.

## Why

Request 75 (commit `9e2c2de9`) rescued 60 classification reactions where D+U is right and D is
wrong. In 15 of them the reaction is a stereo-only transformation (reactant and product are
identical once stereochemistry is removed): 7 of 12 at ECReact L2, 8 of 12 at L3 (exports
`case_a.csv`, the rows whose `rxn`, flattened with `Chem.RemoveStereochemistry`, has equal sides).
For these the signed counts are empty, so D carries no information. Request 74 showed that a
stereo-aware token separates the two alcohol stereoisomer pairs. The question now is whether the
same stereo-aware token resolves the ECReact cases at scale, which would turn a stated limitation
into a result.

## Experiment A (required): ceilings with a stereo-aware token

Same endpoints as request 74 for ECReact only (ecreact_1, ecreact_2, ecreact_3), WL radius 2,
exact counts, no sketching, same cleaned rows and labels as study 65.

1. Compute the ordinary signed count vector D and a stereo-aware D_stereo, using the stereo-aware
   token of request 74 (exact rooted configured stereograph digest at the tetrahedral centres;
   state how double-bond E/Z is handled, and if it is not, add it as a second profile
   `stereo_ez` that appends the canonical bond stereo to the bond label).
2. For D, D_stereo, U, U_stereo and the pairs (D,U) and (D_stereo,U_stereo), report per endpoint:
   n reactions, n exact classes, collision share (reactions in classes with more than one label),
   and the oracle ceiling, as in `endpoint_summary.csv` of request 74.
3. Report the share of reactions whose D is empty (stereo-only or no signed change) before and
   after adding stereo, and their label purity.
4. For the 15 stereo-only rescued reactions of request 75 (list their `case_id` from `case_a.csv`):
   whether D_stereo is non-empty, and whether the stereo-aware class now differs from the class of
   the reaction's collision partner(s).

Outputs in `Experiments/76_stereo_ecreact/publication_metadata/stereo_ecreact_v1/`:
`endpoint_summary.csv`, `stereo_only_rescues.csv` (the 15 rows with the quantities above, no
SMILES beyond `rxn`), `empty_delta_summary.csv`, `AUDIT.json`, `COMPLETE.json`, README with the
table, and a note stating which token profiles were used.

## Experiment B (optional, only if A shows the ceiling gap closes)

Fit the seven-arm MLP of study 65 on ECReact L2 and L3 with the stereo-aware token in D and D+U on
5 of the 25 saved grouped partitions (seeds 1 to 5), same optimizer settings, and report MCC per
partition and arm. Do not run the full 25 without asking the user. State the compute cost first.

## Constraints

Same as requests 74 and 75: pinned commit and file hashes, package versions, no weights or per-row
predictions beyond the 15 exported reactions, state anything that cannot be reproduced. Push on a
new branch and report the commit.

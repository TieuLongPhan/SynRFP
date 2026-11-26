# Request 82: target pairs inside the barrier datasets

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Export the files named here and report the
branch and commit.

## Why

The paper tests whether the value of pooled context depends on the prediction target by holding
the reactions fixed and changing only the target: QuantumPioneer (barrier against reaction
enthalpy) and EnzymeMap (EC class against bond-edit class). Three barrier datasets in the benchmark
also carry a second target that the benchmark has not used:

- B97XD3: `ea` and `dh`;
- GDB7 (wB97X-D3): `ea` and `dh`;
- Cyclo: `G_act` and `G_r`.

These are further same-reaction target pairs. The paper also needs to state why QuantumPioneer was
chosen for the prespecified test.

## Experiment A (required): coverage of the prespecified rule

For each of the three datasets, apply the training-only prediction rule of the QuantumPioneer and
EnzymeMap test (coverage of at least 5% of training rows, 30 query groups and ten exact-delta classes)
on the same group folds as the barrier benchmark. Report whether the rule produces a predicted
ordering, and the coverage numbers. If it does not, say so; do not relax the rule.

## Experiment B (required): paired gains for both targets

With the barrier benchmark protocol (same folds, 512-unit MLP, standardized targets, seeds), fit
2,048-bit D and D+U (1,024 + 1,024) for the second target of each dataset, and reuse the published
barrier fits for the first target. Report for each dataset and target the error reduction from D to
D+U (normalized MSE and percent RMSE) with paired group-bootstrap 95% intervals, and the difference
between the two targets' gains with its interval.

## Experiment C (required): same-signed-count variance for the second target

Repeat the request 78 barrier analysis (variance resolved by pooled counts within exact WL radius 2
signed-count classes) for `dh` and `G_r`, so both targets of each dataset are comparable.

Outputs: `rule_coverage.csv`, `target_pair_gains.csv`, `target_pair_variance.csv`,
`target_pairs_note.md` (one paragraph: does the gain differ between the targets, and is this
consistent with QuantumPioneer and with the variance analysis).

## Constraints

Same as requests 74 to 81: pinned commits and hashes, package versions, no weights, no per-row
predictions, state anything that cannot be reproduced. Push on a new branch named `exp/82-...` and
report the commit.

# EnzymeMap target-dependence replication

The training-only prediction ranked EC level 2 above the bond-edit multiset.
The held-out comparison used 9,631 reactions across 420
scaffold groups. The 30 fits form the full grid of two targets, five fingerprint
arms and three optimizer seeds.

| Group-weighted error reduction, percentage points | Estimate and 95% interval |
| --- | ---: |
| EC level 2, pooled versus standard delta | +2.11 [+0.45, +3.94] |
| Bond edits, pooled versus standard delta | -0.61 [-2.32, +1.11] |
| Difference between the two gains | +2.71 [+0.45, +5.10] |

The interaction interval supports the frozen ordering. The intervals use 10,000 paired scaffold-group bootstrap draws,
conditional on the fitted models. Positive values indicate lower error with
pooled context.

RXNMapper v2 and source-side validation retained 13,469/13,976
training reactions, 1,730/1,774 validation reactions and
9,631/14,744 test reactions. The comparison uses the exact same
retained rows for both targets and all arms. Of the 5,113
test refusals, 5,041
have different explicit atom counts by element or isotope across the source
sides. The estimated gains apply to the admitted mapping domain. See
[input_coverage.json](input_coverage.json) for the split-specific admission
audit. Training sets the target vocabularies. Test labels unseen in training
count as errors.

The frozen prediction and its verification are in
[prediction.json](prediction.json) and [VERIFIED.json](VERIFIED.json). See
[scores.json](scores.json), [fit_scores.csv](fit_scores.csv), and
[plan.json](plan.json) for the held-out metrics and provenance. Reaction maps,
split identities, row-level predictions, and model weights are retained locally.

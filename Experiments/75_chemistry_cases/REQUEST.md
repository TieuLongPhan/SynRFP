# Request 75: chemistry case studies for the Supplementary Information

Written 25 November 2026 by the paper agent, after request 74 (delta-only ceilings,
branch `exp/74-delta-ceiling`, commit `c2f046a9`) supported the hypothesis that context
helps where signed counts leave labels mixed. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`. Do not edit it. Export the files named here and
report the branch and commit.

## Why chemistry examples

The paper argues from numbers. A chemist reading it needs to see, in structures, where
context rescues a prediction, where it costs one, and why SynRFP separates reactions that
set-based fingerprints cannot. Every example must be chosen by a stated deterministic rule
and logged in the export, so that nothing is cherry-picked. Include failures as well as
successes. Prefer small molecules, and avoid near-duplicates (at most one example per
Murcko scaffold of the largest component).

## Case A: successes of pooled context (classification)

Endpoints: ecreact_1, ecreact_2, ecreact_3, tpl_b, schneider_b.
Model: the standard preset's MLP on D (2,048 bits, WL r2 parity) and on D+U (1,024+1,024),
same partitions and seeds as study 65 (use the saved grouped partitions, seed 1 first).

For each endpoint export up to 12 test reactions that D+U classifies correctly and D
classifies wrongly in the majority of the 25 grouped partitions where the reaction is in
the test set. Rule: rank by the number of partitions where D is wrong and D+U is right,
break ties by fewer heavy atoms. For each reaction also export the closest training
reaction that has identical signed counts (exact D equality) and a different label, which is
the collision that D could not resolve.

Columns: `endpoint, case_id, r_id, rxn, true_label, pred_D, pred_DU, n_partitions,
n_D_wrong_DU_right, collision_r_id, collision_rxn, collision_label, n_heavy_atoms`.

## Case B: failures of pooled context (classification)

Endpoints: mech_1, mech_2, syntemp_r1, syntemp_r2, uspto50k_b, uspto50k_ub.
Same rule reversed: reactions that D classifies correctly and D+U classifies wrongly in the
majority of partitions. For each, report whether the error is a sketch collision: the
number of distinct exact-D classes that share bits with this reaction after folding to
1,024 bits (parity, seed 1) but not after folding to 2,048 bits, and one example of the
colliding reaction from another class.

Columns: as Case A, plus `n_classes_merged_1024`, `n_classes_merged_2048`,
`merged_r_id`, `merged_rxn`, `merged_label`.

Also export, per endpoint, the share of exact-D classes that collide after folding at
1,024 bits and at 2,048 bits (`fold_collision_summary.csv`), for D, U and D+S. This
quantifies the resolution that context costs.

## Case C: why SynRFP separates what DRFP cannot

Dataset: ECReact and the USPTO50K reaction set.
Find pairs of reactions whose DRFP fingerprints (2,048 bits, radius 3, the published
settings) are identical but whose exact signed count vectors D differ, and pairs where the
SynRFP D vectors are identical but the DRFP fingerprints differ. Typical causes to look
for: a molecule that appears on both sides and removes a shared token from the symmetric
difference, and repeated groups whose counts differ but whose token sets do not.

Export up to 20 pairs per dataset and direction, smallest first, with the shared and
differing tokens (decoded to atom environments if available).

Columns: `dataset, direction, pair_id, side, r_id, rxn, label, drfp_equal, D_equal,
n_heavy_atoms, differing_tokens`.

## Case D: yield screens (chemistry of the matched holdouts)

Buchwald-Hartwig, the three signed-count classes (aryl chlorides, iodides, bromides; each
has the 4-CF3, 4-OMe and 4-Et forms).
For each class and each of the three joint holdouts (study 73, e3), export per assay:
`class, holdout, aryl_halide, ligand, additive, base, measured_yield, pred_zero_gap_arm,
pred_pooled_arm`. Then the paper agent can draw measured and predicted yield gaps between
forms for each recipe, including the class where pooled context loses (aryl chlorides).
If saved predictions do not exist, say so and do not refit.

## Case E: alcohol screen

The paper agent already has `data/condition_ranking/predictions.csv`. Nothing is needed
except one item: for the three alcohols where pooled context improved the best of three
and the three where it worsened it, export the full 20-condition predicted scores for both
arms so that the rankings can be drawn. If they are already in `predictions.csv`, say so.

## Case F: EnzymeMap and QuantumPioneer

If saved held-out predictions exist for the fitted models, export per test reaction for up
to 10 reactions per dataset: the encoding arm with the largest improvement from pooled
context over delta, and the one with the largest loss, using the same rule as Case A.
EnzymeMap: `reaction, EC label, predicted label under D and D+U, bond-edit label,
predicted bond-edit under D and D+U`. QuantumPioneer: `reaction, true barrier, true
enthalpy, predictions under D and D+U for each target`. If predictions were not saved,
say so and do not refit.

## Output layout

`Experiments/75_chemistry_cases/publication_metadata/chemistry_cases_v1/` with one CSV per
case, `selection_rules.md` that restates every rule used, `AUDIT.json` (code sha256, input
sha256, package versions, counts kept and dropped), `COMPLETE.json`, and a README that
lists for each case how many candidates met the rule before the export was capped. No model
weights and no per-row predictions beyond the exported examples.

## How the paper will use it (so you can check the columns)

- SI figure with Case A and B side by side: structures of rescued and lost reactions with
  true and predicted labels, and the colliding reaction beneath each.
- SI figure for Case C: pairs with identical DRFP but different D, and the reverse.
- SI figure for Case D: three rows, one per signed-count class, showing measured and
  predicted yield differences between the held-out forms.
- Cases E and F as one figure each if the data exist.

Optional from request 74 still open: SynCat on SynRFP's partitions was deferred as too
costly, which is accepted. The stereo-aware probe result is welcome in a short markdown note.

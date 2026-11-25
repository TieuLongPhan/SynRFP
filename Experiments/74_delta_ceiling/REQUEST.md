# Request 74: delta-only ceilings for the classification endpoints

Written 25 November 2026 for the workstation agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`. Do not edit it. Return results here and
tell the user when they are ready.

## Why

In the paper, moving half of the 2,048 delta bits to pooled context (D+U) changes
the MLP score of the 14 classification endpoints as follows (mean over the 18
tokenizer x radius x sketch constructions, from
`data/benchmark_refresh/source_data/context_score_changes.csv`):

| gains | MCC change | losses | MCC change |
| --- | ---: | --- | ---: |
| TPL B / TPL UB | +0.033 / +0.026 | USPTO50K B / UB | -0.010 / -0.013 |
| Schneider B | +0.021 | Mech L1 / L2 | -0.010 / -0.007 |
| ECReact L1 / L2 / L3 | +0.013 / +0.013 / +0.019 | SynTemp R0 / R1 / R2 | -0.005 / -0.011 / -0.011 |
|  |  | Schneider UB | -0.004 |

Hypothesis to test: labels set by the transformation itself (Mech, SynTemp,
USPTO50K) are determined by the signed counts, so delta alone already identifies
them and spending bits on context only costs resolution. Labels that depend on
the surroundings (ECReact enzyme classes, TPL, Schneider B) are not determined by
the signed counts, so context can help.

## Experiment A (required): label purity of exact signed-count classes

No model fitting. Use the unsketched count vectors only.

For each endpoint, on the same cleaned reaction sets and label definitions as the
published classification study (study 65, `default_optimal_v1`):

1. Encode each reaction with the WL tokenizer at radius 2 and compute the exact
   signed count vector D = C_H - C_G (the same code path as the paper, no
   sketching, no folding). Repeat for the WL radius 3, Morgan radius 2 and path
   radius 2 tokenizers if time allows, because the ceiling depends on the
   tokenizer.
2. Group reactions by the exact D vector (a hash of the sorted non-zero
   token:count pairs is enough).
3. For every group report: size, number of distinct labels, majority label share.
4. Per endpoint report:
   - n reactions, n groups, n singleton groups,
   - collision share: fraction of reactions that sit in a group with more than one
     label,
   - delta-only ceiling: sum over groups of the majority-label count, divided by n
     (the best accuracy any delta-only predictor can reach),
   - the same ceiling computed only on reactions whose group contains reactions
     from both the fitting and test partitions of the published grouped folds,
     because a group seen only in test cannot be learned.
5. Repeat steps 2 to 4 for U = C_G + C_H, and for the pair (D, S) with
   S = min(C_G, C_H), to give the context ceilings. The gap between the D ceiling
   and the (D, U) ceiling is the most any amount of pooled context could add.

Endpoints: ecreact_1, ecreact_2, ecreact_3, mech_1, mech_2, schneider_b,
schneider_ub, syntemp_r0, syntemp_r1, syntemp_r2, tpl_b, tpl_ub, uspto50k_b,
uspto50k_ub. Data locations: ECReact
`Experiments/work/synrxn_cache/classification__ecreact.csv.gz` (185,734 rows;
columns r_id, rxn, ec1, ec2, ec3, split, orig_index), Mech31k
`Experiments/data/mech31k.csv.gz`, the others through the same loader used by
study 65.

### Outputs (write to `Experiments/74_delta_ceiling/publication_metadata/delta_ceiling_v1/`)

- `endpoint_summary.csv`: one row per endpoint x tokenizer x channel set (D, U,
  D+U, D+S) with the quantities in step 4.
- `group_table.csv.gz`: group id, endpoint, size, n_labels, majority share (no
  reaction SMILES).
- `examples.csv`: for ECReact L1, L2 and L3, up to 20 pairs of reactions with
  identical D (WL r2) and different labels, chosen to be small (fewest heavy
  atoms), with r_id, rxn, ec1, ec2, ec3. These are for drawing in the
  Supplementary Information.
- `AUDIT.json` with the code sha256, input file sha256, package versions and the
  counts of reactions kept and dropped per endpoint, and `COMPLETE.json`.
- A `README.md` giving the table of ceilings per endpoint (D and D+U) next to the
  D+U minus D MCC change above.

## Experiment B (optional, cheap): fingerprints on SynCat's original folds

SynCat was run once on its original five folds of each full classification
dataset (70 models), which is too expensive to repeat on the 25 saved grouped
partitions. For a same-fold comparison, run the seven fingerprint arms (MLP, the
published settings) on SynCat's original fold indices for the 14 endpoints. Report
MCC per endpoint and fold, plus mean and sample SD across the five folds, in the
same layout as `syncat_classification/cells.csv`. Only do this if the fold indices
are available in the SynCat export; otherwise say what is missing.

## Experiment C (optional): stereochemistry feasibility on the alcohol screen

The alcohol deoxyfluorination screen has two stereoisomer pairs (the Boc
hydroxyproline methyl esters and the 4-phthalimidocyclohexanols). SynKit's
`rsmi_to_graph(..., drop_non_aam=False)` stores stereo descriptors in
`graph.graph["stereo_descriptors"]` but the tokenizers read only node and edge
attributes. Check whether a stereo-aware token label (for example the canonical
descriptor from `synkit.Graph.Stereo.canonical` appended to the atom or bond label)
separates the two members of each pair. Report only whether the four fingerprints
differ. Do not claim improved prediction from four compounds.

## Constraints

- No model weights, per-row predictions or raw datasets in the exports.
- Pin the SynRFP commit, record package versions and file hashes as in studies 65
  and 72.
- State any endpoint that cannot be reproduced and why.
- When done, commit on a new branch, push, and give the user the branch and commit
  hash so that `code/generate_refresh.py`-style readers can pin it.

# Request 88: reaction examples for four SI results that have no chemistry yet

Written 27 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Report the branch and commit.

## Why

The SI now reports four results as numbers only. A reviewer will ask what the reactions look like.
Each part below asks for a few concrete reactions, selected by a fixed rule, to draw in the SI
(later possibly the main text). Use existing encodings and predictions where they exist. No new
training.

Selection rules for every part: deterministic (state the sort key), at most one reaction per
Bemis-Murcko scaffold of the largest component across all exported examples, prefer smaller
reactions (fewer heavy atoms) so they can be drawn, and report the size of the eligible pool.

## A. Count magnitude matters (Table S6, sign-only stage)

On ECReact L1 the sign-only stage added 13 forced errors for D and 55 for D+U (`d21452cc`,
`error_floor_ec_mech_v1`). TPL B also lost 0.035 points at this stage (paper-side computation).
Export up to 4 pairs from ECReact L1 and up to 4 from TPL B: two reactions with different exact
signed counts but identical sign-only keys, and different labels. For each pair give both reactions,
both labels (EC number or TPL template class), and the token(s) whose count magnitude differs
(e.g. +1 against +2), with the WL environment they describe if available.

## B. Compression collisions

The fold-collision summary (Experiments of request 78/80, `fold_collision_summary.csv`) found 2
exact D classes on the six Mech/SynTemp/USPTO50K endpoints, and ECReact L1 merged 1 D class at the
parity stage (`d21452cc`). Export every such case (expected 3): the two reactions with different
exact signed counts and identical parity bits, their labels, the bit width, and the keys that land on
the shared bits.

## C. Reactions with the same SynRFP vector as their reverse

In request 86, 0.15 to 0.20% of sampled reactions had the same SynRFP standard vector as their
reverse (TPL B, ECReact L1). Export all of them from the request 86 sample (expected about 7) with
the reaction, the signed delta (empty or not) and a short cause (e.g. empty delta because only
stereochemistry or an unlabelled attribute changes, or reactant and product sides identical).

## D. Where the fingerprint loses to a generative model (reagents)

On the native reagent test set (14,215 reactions) export 4 reactions where T5Chem (seed 42) predicts
the exact reagent set and the SynRFP standard preset (seed 42) does not, and 2 where SynRFP is exact
and T5Chem is not. For each give the reaction, the true reagent set and both predicted sets. Also give
the counts of the four outcome cells (both exact, only T5Chem, only SynRFP, neither) on all 14,215
rows.

## Output

On a new branch `exp/88-...`: `count_magnitude_pairs.csv`, `fold_collision_pairs.csv`,
`reverse_equal_reactions.csv`, `reagent_examples.csv`, `reagent_outcomes.csv`, and `note.md` (one
short paragraph per part). Use the column conventions of `data/chemistry_cases` in the paper repo
(`endpoint, pair_id, side, r_id, rxn, label, n_heavy_atoms, scaffold, ...`) and give atom-mapped
SMILES where a mapping already exists, so the paper can mark the reaction center.

## Constraints

Same as requests 74 to 87: pinned commits and hashes, package versions, no weights. Example rows are
allowed; no full per-row prediction dumps. Do not change the benchmark results.

# Chemistry-case selection rules

The examples describe saved held-out results and representation collisions.
A scaffold is the achiral Murcko scaffold of the largest component by
heavy-atom count across the reaction SMILES. Canonical achiral component SMILES
break component ties and identify scaffold-free components. A and B each use
distinct query scaffolds across endpoints. In C the same scaffold may appear
on both sides of a single pair and is used only once across pairs.

## A and B. Classification

Use the study 65 paired WL/r2/parity/seed 1 MLPs. D assigns all 2,048 bits to
signed delta. D+U assigns 1,024 bits to each channel. Every row has five held-out
appearances among the 25 saved partitions. Require at least three correctness
changes in the requested direction. Rank by most votes, fewest total heavy
atoms, source ID and source position. Process endpoints in their request order.
Take up to 12 cases per endpoint. Prefer partition seed 1 for display whenever
it supports the chosen case and witness.

**A.** Require a different-label reaction with the same exact D in the actual
optimizer fitting rows of a rescuing fold. Maximize Tanimoto similarity in
the pooled 1,024 U bits, then prefer fewer heavy atoms and the smaller ID and
position. Display the first supporting fold with this witness.

**B.** Reverse the vote to require D correct and D+U wrong. Count other exact
D classes sharing each D bit vector. The difference between the 1,024- and
2,048-bit counts gives the width-specific collision count. Prefer a witness
fitted in a supporting fold, then a different true label, fewer heavy atoms,
ID and position. Keep failures with zero width-specific collisions. Empty
merged-witness fields indicate no width-specific witness. The exact-D training
witness fields from A are retained in B for schema alignment.

The collision summary reports classes sharing a bit vector with a different
exact key of the same channel set. The additional cross-D columns count
distinct D keys coalescing under each representation. D and U use standalone
1,024- or 2,048-bit blocks. D+S uses two 512- or 1,024-bit blocks. Smaller
widths use parity XOR nesting. Lower-width D and U bytes match the archived
D+U blocks.

## C. DRFP and signed counts

Use ECReact with level-3 labels and original unbalanced USPTO50K. In each
exact equality bucket, find the smallest pair differing under the other
representation, using size, source ID and position. Rank pairs by total size
and IDs and positions. Take up to 20 per dataset and direction after scaffold
deduplication. Count eligible unordered pairs and equality buckets separately.

The two rows share a pair ID. The differing signed tokens hold
[token, current count, partner count]. Shared signed tokens hold equal
nonzero counts. The rooted WL examples provide an emitting reaction side,
radius, graph vertex and local neighborhood. Atom fields are [graph ID,
element, charge, aromatic, H count]. Bond fields are [graph ID A, graph ID B,
order]. DRFP fragment columns contain decoded pre-fold environments. The
runner independently replays each selected DRFP bit vector and exact D key.

## D. Matched yield assays

Export every recipe-matched pair from the nine saved E3 joint holdouts.
Require the same ligand, additive and base on both aryl-halide forms.
Use the recipe ID to join the two rows. The zero-gap predictions use the
independent delta rehash arm. Standard delta predictions are also included.
Recomputed pooled matched-gap MSE agrees with the original E3 fit summary.
Yield values are percentages.

## E. Alcohol rankings

Use the paper's saved predicted scores and best-of-three regrets. Rank by
the change in observed yield achieved in the top three, pooled minus
standard delta. Retain up to three gains and losses each, ordered by effect
magnitude, substrate size and canonical string. Export the 20 conditions per
substrate. Prediction ties use base and fluoride strings in ascending order.

## F. External tasks

Use saved paired optimizer seeds 17, 29 and 43. Retain up to five gains
and five losses per dataset, with distinct largest-component scaffolds.

**EnzymeMap.** Require at least two of three paired EC level-2 predictions
to change correctness in the required direction. Rank by vote count, size
and row ID. Display the lowest supporting seed and its EC and bond-edit
predictions from both arms.

**QuantumPioneer.** Select on reaction enthalpy, the frozen higher-ranked
target. Rank by the magnitude of mean squared-error reduction normalized by
training variance. Require the chosen direction in at least two paired seeds
and in the mean. Break ties by vote count, size and ID. Provide measured values
and both arms' individual predictions and their means for both targets.

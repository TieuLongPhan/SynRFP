# Request 80: why D+U misclassifies the lost cases

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Export the files named here and report the
branch and commit. No new model fits; use the request 75 predictions and partitions.

## Why

Request 75 Case B selected reactions that the D model classifies correctly and the D+U model does
not (2,308 eligible failures on the Mech, SynTemp and USPTO50K endpoints, 72 shown). The paper shows
that none of them comes from a fold-only sketch collision, but it does not say why D+U goes wrong.
The working explanation, parallel to the clustering result of request 78, is that the pooled block
pulls the prediction toward training reactions with similar substrates and a different
transformation. This needs a test before it goes in the paper.

## Experiment (required)

For every eligible Case B failure (all 2,308, not only the 72 shown), in its own held-out partition
and training set:

1. Find the nearest training reaction by Tanimoto similarity of the pooled block (U, 1,024 bits) and,
   separately, by the delta block (D, 1,024 bits) and by D+U (2,048 bits).
2. Record whether the D+U prediction equals the class of the nearest-by-U training reaction, and
   whether the D prediction equals the class of the nearest-by-D training reaction.
3. Record the reaction-center (D) Tanimoto between the failure and its nearest-by-U neighbour, and the
   U Tanimoto between the failure and its nearest-by-D neighbour.
4. As a reference, repeat 1 and 2 for an equal number of reactions that both models classify
   correctly on the same endpoints and partitions (fixed seed).

Report per endpoint and pooled: the share of failures whose D+U prediction matches the nearest-by-U
class, the same share in the reference set, the median D and U similarities, and a one-line verdict on
whether the substrate-neighbour explanation is supported. For the 72 shown cases, add the nearest-by-U
training reaction SMILES and class so the paper can draw one or two of them next to the failure.

Outputs: `lost_neighbours.csv` (one row per failure), `lost_neighbours_summary.csv`,
`lost_neighbours_shown.csv` (72 rows with neighbour SMILES), `lost_neighbours_note.md`.

## Constraints

Same as requests 74 to 79: pinned commits and hashes, package versions, no weights, state anything
that cannot be reproduced. Push on a new branch named `exp/80-...` and report the commit.

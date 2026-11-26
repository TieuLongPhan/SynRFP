# Request 79: clustering with the two channels, k-means and BitBIRCH

Written 26 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Export the files named here and
report the branch and commit.

## Why

Request 78 showed that pooled context (U) lowers k-means AMI on 10 of 12 clustering endpoints,
including endpoints whose classes depend on context (TPL B, Schneider B, ECReact), because the U
block adds distance without separating classes. Down-weighting U recovered AMI. The paper now says
that clustering follows the same label rule as classification once U is down-weighted, and that
k-means cannot learn the weighting. Two open points:

1. Is this specific to Euclidean k-means, or does a Tanimoto method on bit vectors behave the same?
   The repository already supports BitBIRCH (`synrfp_exp/clustering.py`, `bitbirch`).
2. The paper groups endpoints by how their classes are defined: reaction-center labels (Mech L1, L2,
   SynTemp R0, USPTO50K B, UB, Schneider UB) and context-dependent labels (ECReact L1 to L3,
   Schneider B, TPL B, UB). A per-group summary of both methods is needed.

## Experiment A (required): BitBIRCH with D, U and D+U

WL radius 2, parity, fingerprint seed 1, the 12 clustering endpoints, all rows (placeholders as in
the published clustering). Constructions:
- D at 2,048 bits (published reference);
- U at 2,048 bits;
- D+U at 1,024 + 1,024 bits (published);
- D+U with unequal allocation: D 1,536 + U 512, and D 1,792 + U 256. This is the bit-vector analogue
  of the k-means weight scan, since BitBIRCH cannot weight bits.

For BitBIRCH use the threshold rule already used in the repository for the benchmark, and report it.
If that rule does not exist, use Tanimoto thresholds 0.5, 0.65 and 0.8 and report all three. Report
AMI, number of clusters, singleton share and runtime per endpoint and construction. BitBIRCH is
deterministic for a given input order; if `bitbirch_order_stability.py` shows order dependence, use
five shuffled orders with fixed seeds and report mean and SD.

## Experiment B (required): per-group summary for both methods

Using request 78 k-means results (reference and factorial K, weights 0 to 1) and Experiment A,
report for each method the mean AMI change from D to each D+U construction, separately for the
reaction-center group and the context group above, and per endpoint. State in one note whether
reducing the U share helps the context group more than the reaction-center group under each
method.

Outputs: `bitbirch_scan.csv`, `group_summary.csv`, `clustering_channels_note.md`.

## Constraints

Same as requests 74 to 78: pinned commits and hashes, package versions, no weights, no per-row
assignments, state anything that cannot be reproduced. Push on a new branch named `exp/79-...` and
report the commit.

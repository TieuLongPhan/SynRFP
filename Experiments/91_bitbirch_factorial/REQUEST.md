# Request 91: BitBIRCH on the 72-configuration factorial and the fingerprint benchmark

Written 27 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Report the branch and commit.

## Why

Clustering in the paper uses k-means everywhere (factorial, configuration selection, benchmark).
BitBIRCH appears only in two places that do not match each other or the current configurations:
the channel ablation (`clustering_channels`, SynRFP standard and four D/U splits, threshold 0.35,
five input orders) and an older benchmark run (SynRFP 1,024-bit delta, DRFP, RDKit structural,
one order, thresholds tuned by AMI on USPTO50K B). We want one BitBIRCH study on the same
representations as the k-means factorial and benchmark, with no threshold chosen from labels.

## Representations

Use the cached fingerprints of the benchmark release; do not re-encode.

1. All 72 configurations of the factorial (tokenizer, radius, sketch, sides; 2,048 bits, seed 1),
   which include the standard and every task-selected configuration.
2. The other benchmark fingerprints: DRFP and RDKit structural (bit vectors, native).
3. cRXNFP, RDKit difference and RXNFP-MLM are not bit vectors. Run them only after binarization
   (nonzero to 1 for count vectors; for RXNFP-MLM, value > 0 to 1), mark them `binarized=True`,
   and report them separately. Skip and say so if binarization is not meaningful for one of them.

## Endpoints and runs

The 12 clustering endpoints of the benchmark, all records, refusal placeholders kept as in k-means.
The three ECReact levels share one clustering; score it against each level. Five input-order seeds
(20260924 to 20260928, as in the channel ablation). BitBIRCH with Tanimoto, branching factor 50 and
diameter merging, as in the ablation. No class count is used.

## Thresholds (label-free)

1. Sweep 0.20 to 0.80 in steps of 0.05 for every representation and endpoint.
2. From the sweep, report two label-free operating points per representation and endpoint:
   a. the fixed threshold 0.35 (continuity with the ablation);
   b. the threshold whose mean cluster count is closest to the k-means row-count rule
      k = min(512, max(2, round(sqrt(n/2)))), so BitBIRCH and count-blind k-means are compared at
      the same resolution.
   Do not select any threshold by AMI or another label-based score.

Estimated size: about 3,850 fits per threshold (77 representations x 10 distinct datasets x 5
orders), about 50,000 over the sweep. Parallelize as the workstation allows.

## Output

On a new branch `exp/91-bitbirch-factorial`:
- `cells.csv`: representation, binarized flag, endpoint, threshold, order seed, AMI, ARI, NMI,
  number of clusters, singleton share, largest-cluster share, runtime.
- `summary.csv`: mean and SD over orders per representation, endpoint and threshold.
- `operating_points.csv`: rows for operating points 2a and 2b.
- `factorial_variance.csv`: share of AMI variance explained by tokenizer, radius, sketch and sides at
  operating point 2b, same decomposition as the k-means factorial.
- `note.md`: what ran, skips, runtime, hardware.

## Constraints

Same as requests 74 to 90: pinned commits and hashes, package versions, no weights. Do not change
existing results. Report negative results as they are.

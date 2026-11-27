# Request 90: matched comparison with SynCluster on USPTO50K UB

Written 27 November 2026 by the paper agent. The manuscript is in
`~/Documents/TACsy/paper/synrfp_paper`; do not edit it. Report the branch and commit.

## Why

SI S7.3 compares SynRFP with SynCluster on the 43,381 USPTO50K UB reactions shared with the
SynCluster release (`Experiments/35_specialist_references/syncluster/publication_metadata/full_reference_v1`).
SynCluster (native, raw) reaches AMI 0.476 with 378 clusters; SynRFP standard
(`synrfp-wl-r2-b2048-parity-delta`) with MiniBatchKMeans at k = 147 (row-count rule) reaches 0.422.
The comparison changes three things at once: representation, algorithm and number of clusters.
We need runs that change one thing at a time, so the paper can say how much of the 0.054 gap is the
fingerprint.

Use exactly the 43,381 matched records and the reference labels of the existing comparison
(`comparison_cells.csv`, `protocol.json`). Same MiniBatchKMeans parameters as `protocol.json`
(batch 8,192, n_init 3, max_iter 100, seeds 0 to 9).

## A. Same number of clusters

SynRFP standard with MiniBatchKMeans at k = 378 (SynCluster raw count) and at the cluster count of
`SynCluster_native_residual_merged`. Ten seeds each. Report AMI, ARI, NMI, occupied clusters,
singleton fraction.

## B. Same algorithm (Butina)

Butina clustering on the SynRFP standard bit vectors with Tanimoto distance, using the same Butina
implementation and ordering as the SynCluster run.
1. At the SynCluster similarity threshold.
2. A threshold sweep (step 0.05) and the threshold whose cluster count is closest to 378.
Deterministic, so one run per threshold. Same metrics as A.

## C. Same algorithm and count for the other fingerprints

For context, repeat A (k = 378, ten seeds) for DRFP, cRXNFP, RDKit difference, RDKit structural and
RXNFP-MLM as encoded in the benchmark release, and B.1 for DRFP if Butina runs in reasonable time.

## D. SynCluster representation with k-means (optional)

If the SynCluster template fingerprints can be exported as vectors for the 43,381 records, run
MiniBatchKMeans on them at k = 147 and k = 378, ten seeds. This isolates the representation from the
algorithm. Skip and say so if the vectors are not accessible.

## Output

On a new branch `exp/90-syncluster-matched`: `matched_cells.csv` (one row per method, algorithm,
k or threshold, seed), `matched_summary.csv` (mean and SD per setting), `note.md` (what was run,
anything skipped, runtime).

## Constraints

Same as requests 74 to 89: pinned commits and hashes, package versions, no weights. Do not change the
benchmark or the existing SynCluster release. Report negative results as they are.

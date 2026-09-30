# Clustering follow-up: final results

Final refresh: 2026-09-25. AMI is the primary metric. Values below are mean ±
sample SD. K-means uses ten internal restarts for every external seed.

## Full-data K-means on the seven smaller endpoints

Ten external seeds per cell.

| Dataset | SynRFP | DRFP | RDKit structural | RDKit difference | RXNFP-MLM |
|---|---:|---:|---:|---:|---:|
| Schneider UB | **0.575 ± 0.011** | 0.571 ± 0.005 | 0.187 ± 0.004 | 0.438 ± 0.021 | 0.538 ± 0.004 |
| Schneider B | **0.719 ± 0.008** | 0.649 ± 0.009 | 0.218 ± 0.007 | 0.518 ± 0.021 | 0.490 ± 0.003 |
| USPTO50K UB | **0.402 ± 0.011** | 0.314 ± 0.016 | 0.099 ± 0.000 | 0.297 ± 0.013 | 0.262 ± 0.000 |
| USPTO50K B | **0.435 ± 0.019** | 0.334 ± 0.013 | 0.106 ± 0.001 | 0.297 ± 0.006 | 0.320 ± 0.008 |
| SynTemp R0 | **0.702 ± 0.007** | 0.594 ± 0.007 | 0.146 ± 0.002 | 0.323 ± 0.006 | 0.586 ± 0.002 |
| Mechanistic level 1 | **0.363 ± 0.018** | 0.263 ± 0.018 | 0.087 ± 0.000 | 0.261 ± 0.009 | 0.080 ± 0.000 |
| Mechanistic level 2 | **0.613 ± 0.009** | 0.475 ± 0.013 | 0.129 ± 0.001 | 0.301 ± 0.009 | 0.158 ± 0.003 |

SynRFP has the highest mean AMI on all seven endpoints. Schneider UB is the
closest comparison: 0.575 for SynRFP versus 0.571 for DRFP.

## Large-data MiniBatchKMeans

Ten external seeds per cell; batch size 8,192, three internal initializations,
and at most 100 iterations.

| Dataset | SynRFP | DRFP | RDKit structural | RDKit difference | RXNFP-MLM |
|---|---:|---:|---:|---:|---:|
| TPL UB | 0.630 ± 0.007 | **0.653 ± 0.030** | 0.312 ± 0.002 | 0.540 ± 0.003 | 0.499 ± 0.001 |
| TPL B | **0.742 ± 0.002** | 0.719 ± 0.005 | 0.325 ± 0.001 | 0.561 ± 0.003 | 0.461 ± 0.001 |
| ECReact level 1 | **0.341 ± 0.023** | 0.332 ± 0.022 | 0.260 ± 0.010 | 0.314 ± 0.034 | 0.246 ± 0.005 |
| ECReact level 2 | **0.524 ± 0.010** | 0.475 ± 0.007 | 0.445 ± 0.005 | 0.472 ± 0.007 | 0.373 ± 0.004 |
| ECReact level 3 | **0.587 ± 0.004** | 0.539 ± 0.002 | 0.529 ± 0.003 | 0.534 ± 0.003 | 0.458 ± 0.003 |

SynRFP leads four of the five large endpoints. DRFP leads TPL UB.

## Paired 80% scaffold-group subsamples

Twenty paired subsamples per cell. Every fingerprint sees exactly the same
rows within a sample.

| Dataset | SynRFP | DRFP | RDKit structural | RDKit difference | RXNFP-MLM |
|---|---:|---:|---:|---:|---:|
| Schneider B | **0.725 ± 0.008** | 0.642 ± 0.009 | 0.220 ± 0.008 | 0.522 ± 0.015 | 0.492 ± 0.004 |
| USPTO50K B | **0.437 ± 0.019** | 0.335 ± 0.023 | 0.104 ± 0.005 | 0.296 ± 0.011 | 0.327 ± 0.008 |
| SynTemp R0 | **0.701 ± 0.005** | 0.594 ± 0.008 | 0.149 ± 0.004 | 0.322 ± 0.007 | 0.583 ± 0.004 |
| Mechanistic level 2 | **0.614 ± 0.011** | 0.469 ± 0.011 | 0.129 ± 0.002 | 0.300 ± 0.011 | 0.159 ± 0.004 |

SynRFP leads every representative endpoint under sampling variation.

## Resolution and input-order sensitivity

SynRFP remains the highest-AMI fingerprint on all four representative
endpoints at both `k/classes = 0.5` and `k/classes = 2.0`. BitBIRCH was repeated
under ten shuffled input orders on the three compatible binary fingerprints.
SynRFP beat both DRFP and RDKit structural in all 80 paired AMI comparisons.

## SynRFP ablation

The one-factor-at-a-time study contains four datasets, ten configurations, and
three seeds. Its main findings are:

- Radius 1 is best on Schneider B, SynTemp R0, and Mechanistic level 2.
- Morgan is best on USPTO50K B and improves over the default on all four
  representative endpoints.
- Increasing the fingerprint from 1,024 to 2,048 bits gives a small improvement
  on all four endpoints.
- Union mode is the worst configuration on every endpoint. Delta encoding
  contributes to the clustering advantage; this comparison does not isolate
  the effects of cancelling unchanged motifs and encoding reaction direction.
- MinHash and consistent weighted sampling (CWS, configured as `cw`) do not
  improve over parity in this local ablation.

The ablation is a local sensitivity analysis around
`wl/r2/1024/parity/delta`; it is not the full 162-configuration factorial
design used by the classification study.

# SynRFP explanation package

This page is derived from completed classification and clustering outputs. It
is intended to guide the main text and supplement; the ongoing XGBoost queues
do not change these completed kNN and clustering contrasts.

## Core mechanism

The strongest controlled result is the signed reaction difference. In the
complete 8,100-cell kNN factorial, the mean MCC is **0.9262** for `delta` and
**0.7475** for `union`. The paired delta-minus-union contrast is positive for
all ten evaluated endpoints. The contrast is small on ECReact L1/L3
(0.0108/0.0124) and much larger on Mechanistic L1/L2 (0.2777/0.2871) and
USPTO50K UB (0.3121). This supports a chemistry-dependent representation
effect rather than a claim that one setting is uniformly optimal.

The clustering ablation gives the same directional result at the default
WL/radius-2/1,024-bit/parity context:

| Endpoint | Delta AMI | Union AMI |
|---|---:|---:|
| Schneider B | 0.7161 | 0.6439 |
| USPTO50K B | 0.4415 | 0.2889 |
| SynTemp R0 | 0.7003 | 0.5633 |
| Mechanistic L2 | 0.6090 | 0.3033 |

The current ablation does not by itself separate cancellation from direction
information, because delta and union also change token-set size. The next
mechanistic control should therefore add an unsigned-delta representation with
the same hashing and density diagnostics.

The direct representation controls in `results/representation_controls.json`
support the intended distinction: reversing `CCO>>CC=O` changes SynRFP (30-bit
Hamming distance) but leaves DRFP identical; adding an unchanged reactant or
product spectator leaves SynRFP identical but changes DRFP (4 and 3 bits).
The unsigned-delta control stays identical under reversal and under both
spectator additions, isolating context cancellation from directional encoding
in this toy example.

## Hierarchy behavior

SynRFP remains ahead of DRFP across the existing hierarchy profiles, although
the margin narrows at finer SynTemp resolution:

| Dataset / label | SynRFP AMI | DRFP AMI |
|---|---:|---:|
| SynTemp R0 | 0.702 | 0.594 |
| SynTemp R1 | 0.749 | 0.661 |
| SynTemp R2 | 0.555 | 0.539 |
| Mechanistic L1 | 0.427 | 0.330 |
| Mechanistic L2 | 0.613 | 0.475 |

These are related label views on shared reactions, so they should be presented
as a hierarchy profile rather than five independent datasets.

In the matched 80%-subsample stability runs, SynRFP exceeded DRFP on all
**20/20** samples for both Schneider B and USPTO50K B. Mean paired AMI gains
were **+0.0828** and **+0.1016**, respectively. These are paired sensitivity
runs sharing sampled records, not 20 independent datasets.

## Claims supported by clustering

- SynRFP has the highest mean AMI and ARI on all seven small K-means endpoints.
- Large-data results are mixed: DRFP leads SynRFP on TPL UB, and RDKit
  difference leads SynRFP on ECReact L1 ARI.
- AMI must be accompanied by ARI, cluster count, singleton fraction,
  homogeneity, and completeness. The frozen-threshold Mech L1 BitBIRCH result
  has high homogeneity but substantial fragmentation, so it is not evidence of
  uniformly useful clusters.

## Required wording and controls

Use “consistent weighted sampling (CWS)” for the `cw` sketch. Parity keeps one
copy of each signed nonzero delta token after subtraction and therefore loses
count magnitude; only the pre-sketch count difference is antisymmetric. The
default graph attributes omit stereochemistry. These are representation limits
to quantify, not reasons to discard the main benchmark.

Before publication, add spectator, reversal, serialization, valid-row, hash
seed, and stereo sensitivity checks. Keep RXNFP-MLM as the label-free neural
reference and identify RXNFP-FT/TPL-1K as supervised references. Treat
non-significant paired comparisons as inconclusive rather than equivalent.

The cached-feature audit found SynRFP encoding failures in five of the seven
clustering datasets (0.005%--0.352% of rows; the largest rate is USPTO50K B),
while the four baseline representations had no failed rows. These rows are
retained as zero vectors by the current pipeline, so validity rates should be
reported with the clustering results and a shared-valid-row sensitivity is a
useful follow-up.

The first reaction-level assignment replay is now available for Mechanistic
L2 (31,667 rows, seed 0, 10 internal K-means starts). All five label-free
fingerprints encoded every row. At the fine mechanistic label, SynRFP reached
AMI/ARI **0.595/0.151**, compared with DRFP **0.452/0.066**, RDKit difference
**0.300/0.050**, RXNFP-MLM **0.156/0.031**, and RDKit structural
**0.133/0.023**. The assignments and validity flags are in
`results/assignments/`; `results/cluster_examples_mech2.csv` contains the
fixed-rule representative rows with cluster size and modal-label purity. This
is explanatory evidence, not an independent replication of the clustering
benchmark.

A second replay on Schneider B gives the same qualitative pattern at a
different chemistry scale: SynRFP AMI/ARI **0.731/0.462** versus DRFP
**0.660/0.190**. SynRFP had 49,975 valid rows (25 failed encodings) and DRFP
had 50,000; the validity difference is retained explicitly in the replay
metadata. Representative rows are in
`results/cluster_examples_schneider_b.csv`, selected by the same fixed rule.

The shared-valid-row and distance-geometry sensitivity is also complete for
Schneider B. After jointly retaining 49,975 valid rows, SynRFP scored
AMI/ARI **0.731/0.462** with native vectors and **0.728/0.464** after L2
normalization. DRFP scored **0.645/0.227** and **0.682/0.330** under the same
two choices. SynRFP remains ahead when invalid rows and vector scale are
controlled explicitly.

The hash-seed sensitivity control on 2,000 Schneider B reactions found a
mean pairwise change of about **58/1,024 bits (5.7%)** across four fixed hash
seeds, with all rows valid. This bounds collision sensitivity while preserving
the primary fixed seed; it should be reported as a representation limitation,
not folded into the primary performance uncertainty.

While the TPL XGBoost cells are running, the interim completed-cell summary is
available at `results/benchmark_completed_summary_partial.csv`. It is not a
final ranking: the non-TPL cells show SynRFP leading ECReact L2 and SynTemp
R0/R1, while RDKit difference leads several mechanistic and ECReact endpoints.
The file records per-fingerprint counts so incomplete TPL coverage cannot be
mistaken for a finished benchmark.

The companion `results/benchmark_label_free_leaderboard_partial.csv` reports
coverage-aware endpoint leaders. On the currently represented endpoints, SynRFP
leads **11/14 kNN**, **11/14 MLP**, and **4/14 XGBoost** label-free endpoint
comparisons. The XGBoost TPL endpoints remain incomplete, so this is a
provisional model-specific summary rather than a universal SynRFP claim.

Family sensitivity is available in
`results/classification_family_leave_one_out.csv`. Across the currently
matched endpoints, the SynRFP-minus-DRFP MCC margin remains positive after
leaving out any one family: **+0.0256 to +0.0411** for kNN, **+0.0160 to
0.0276** for MLP, and **+0.0149 to +0.0215** for the provisional XGBoost
records. The XGBoost SynRFP-minus-RDKit-difference contrast ranges from
**-0.0056 to +0.0021**, so that comparison should remain inconclusive until
the TPL cells are complete.

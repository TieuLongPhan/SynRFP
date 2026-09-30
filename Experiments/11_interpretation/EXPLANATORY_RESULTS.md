# SynRFP explanatory results

This note summarizes analyses that explain the representation. They are
separate from the frozen classification benchmark, whose XGBoost and MLP
matrices are still running.

## Reaction-level clustering replay

Each replay uses K-means with the reference number of clusters, external seed
0, and ten internal starts. Assignments and deterministic representative
reactions are saved under `results/assignments/` and in the two
`cluster_examples_*.csv` files.

| Endpoint | Fingerprint | Valid rows | AMI | ARI |
|---|---|---:|---:|---:|
| Mech L2 | SynRFP | 31,667 | 0.595 | 0.151 |
| Mech L2 | DRFP | 31,667 | 0.452 | 0.066 |
| Mech L2 | RDKit difference | 31,667 | 0.300 | 0.050 |
| Mech L2 | RXNFP-MLM | 31,667 | 0.156 | 0.031 |
| Mech L2 | RDKit structural | 31,667 | 0.133 | 0.023 |
| Schneider B | SynRFP | 49,975 | 0.731 | 0.462 |
| Schneider B | DRFP | 50,000 | 0.660 | 0.190 |

The Schneider B SynRFP replay has 25 failed encodings. They are retained in
the metadata and excluded from the score calculation rather than silently
treated as a cluster.

## Preprocessing sensitivity

On Schneider B, both fingerprints were evaluated on the same 49,975 valid
rows:

| Fingerprint | Native AMI/ARI | L2-normalized AMI/ARI |
|---|---:|---:|
| SynRFP | 0.731 / 0.462 | 0.728 / 0.464 |
| DRFP | 0.645 / 0.227 | 0.682 / 0.330 |

SynRFP remains ahead under both preprocessing choices. This is a targeted
sensitivity check, not a replacement for the primary clustering protocol.

## Hash-seed sensitivity

On 2,000 Schneider B reactions, all four tested hash seeds produced valid
encodings. Across six seed pairs, the mean Hamming distance was approximately
58 of 1,024 bits (5.7%). The primary benchmark keeps hash seed 1 fixed; this
control bounds collision sensitivity and is not additional performance
uncertainty.

## Reaction-centre protocol audit

The historical E8 script used the mechanistic `updated_reaction` column and
radius 3, while the primary SynRFP representation uses the original reaction
and radius 2. On a fixed 200-reaction sample, the primary convention gave a
mean delta-token centre fraction of **0.908**, compared with **0.828** for the
updated-reaction variant at radius 2. The stricter occurrence-level fractions
were **0.881** and **0.806**, respectively. The occurrence-level metric is now
implemented in the E8 script, and the full corrected run is complete for all
**31,673** mechanistic reactions. It gives a mean occurrence-level
delta-token centre fraction of **0.875** (median **0.946**) versus **0.213** for
the union baseline, with mean centre-atom coverage **0.99998**. This is
pre-sketch localization evidence, not proof that every hashed bit uniquely
identifies a reaction-centre atom. The pilot remains available at
`results/reaction_center_protocol_audit.csv`; full outputs are under
`08_reaction_center/results/`.

The corresponding distribution and class-summary figure is
`results/reaction_center_localization.png` (with a PDF copy for manuscript
use). It shows the separation from the union baseline while retaining
class-level variation rather than reporting only a pooled mean.

The cached-feature cost summary shows the practical trade-off before model
training: across seven datasets, SynRFP's estimated dense payload is about
**70.6 MB**, versus **141.2 MB** for DRFP, **282.4 MB** for RDKit structural,
**564.7 MB** for RXNFP-MLM, and **4.52 GB** for RDKit difference. SynRFP's
mean valid-row fraction is **0.9994**; the small failure rate is reported
explicitly rather than hidden. These are payload estimates from cached matrix
shapes, not hardware throughput measurements.

## Interpretation and limits

The controls support three restricted claims: signed reaction differences
retain useful directional information, unchanged context is suppressed, and
the clustering advantage survives common-row and normalization controls. They
do not establish universal superiority. Large-data clustering has documented
exceptions, XGBoost/MLP coverage remains incomplete, and the current graph
parser omits stereochemistry. RXNFP-MLM is the label-free neural reference;
fine-tuned RXNFP variants must remain visibly separate supervised references.

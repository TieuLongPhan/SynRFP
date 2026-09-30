# Clustering follow-up run — 2026-09-24

The follow-up is organized so that the standard benchmark stays immutable and
all exploratory sensitivity outputs are clearly separated.

| Stage | Scope | Fits | Status/output |
|---|---|---:|---|
| Extended BitBIRCH thresholds | 7 endpoints × 3 binary fingerprints × 5 new thresholds | 105 | Complete in `results/`; original frozen thresholds retained for the primary report |
| K-means initialization | 7 endpoints × 5 fingerprints × 10 seeds | 350 total | Complete in `results/` |
| Paired 80% group subsamples | 4 representative endpoints × 5 fingerprints × 20 samples | 400 | Complete in `results_followup/stability80/` |
| BitBIRCH input order | 4 representative endpoints × 3 fingerprints × 10 orders | 120 | Complete in `results_followup/bitbirch_order/` |
| Large MiniBatchKMeans | 5 endpoints × 5 fingerprints × 10 seeds | 250 | Complete in `results_followup/large_minibatch/` |
| RDKit-difference HDBSCAN | 7 endpoints × 4 minimum-cluster sizes | 28 | Complete in `results/` |
| K-means K/C sensitivity | 4 representative endpoints × 5 fingerprints × 2 new K ratios × 3 seeds | 120 | Complete in `results_followup/k_resolution/`; K/C=1 reuses the standard benchmark |
| SynRFP design ablation | 4 representative endpoints × 10 one-factor configurations × 3 seeds | 120 | Complete in `results_followup/ablation/` |

Representative endpoints were fixed before these runs as Schneider B,
USPTO50k B, SynTemp R0, and Mech level 2. The paired subsamples use product
scaffold groups, seed 20260924, and the identical row sets for every
fingerprint. Full-data K-means uses 10 internal restarts per external seed.
The complete K-means/HDBSCAN comparison contains SynRFP, DRFP, RDKit
structural, RDKit difference, and RXNFP-MLM. BitBIRCH contains only SynRFP,
DRFP, and RDKit structural because it requires packed binary fingerprints.

Primary reporting remains AMI, with ARI, homogeneity, completeness, number of
clusters, singleton fraction, and largest-cluster fraction presented alongside
it. Seed variation is initialization sensitivity. Overlapping group subsamples
are sampling sensitivity and will not be presented as independent datasets or
as a formal bootstrap confidence interval.

The SynRFP ablation changes one axis from the default at a time: tokenizer
(Morgan/path), radius (1/3), size (512/2048), sketch (MinHash/CW), and reaction
mode (union). The default WL/r2/1024/parity/delta configuration is included as
the reference. It uses the same fixed factorial-study rows for every variant.

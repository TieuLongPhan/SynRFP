# SynRFP explanatory-analysis audit

Inspected 2026-09-25. Recommendation: extract explanatory evidence from the
completed experiments, then add a compact mechanism/robustness package. A full
162-configuration clustering grid and more initialization seeds are lower
priority. No new training or clustering jobs were launched in this audit.

## New analysis produced now

`inspect_existing.py` validates the complete 8,100-cell kNN factorial and
exports descriptive contrasts, interactions, clustering AMI/ARI summaries, and
hierarchy profiles (`clustering_hierarchy_ami.csv`).
`results/delta_union_evidence.pdf` and `.png` compare paired delta-versus-union
effects across classification and clustering. Error bars are sample SD of
split/initialization contrasts, not confidence intervals for new datasets.

Across the complete random-holdout kNN factorial, averaged equally over all
ten endpoints and other factor settings, mean MCC is **0.9262 for delta** and
**0.7475 for union** (difference **+0.1787**). Every endpoint's average contrast
is positive, but magnitude differs substantially:

| Endpoint | Mean delta-minus-union MCC | SD across five split-level contrasts |
|---|---:|---:|
| ECReact L1 | 0.0108 | 0.0024 |
| ECReact L3 | 0.0124 | 0.0017 |
| Mechanistic L1 | 0.2777 | 0.0026 |
| Mechanistic L2 | 0.2871 | 0.0091 |
| Schneider B | 0.1217 | 0.0031 |
| Schneider UB | 0.1540 | 0.0017 |
| SynTemp R0 | 0.2520 | 0.0027 |
| TPL B | 0.1263 | 0.0034 |
| USPTO50K B | 0.2331 | 0.0010 |
| USPTO50K UB | 0.3121 | 0.0017 |

Each split-level contrast averages 81 otherwise matched factor combinations.
These are descriptive effects in this factorial, not main-benchmark scaffold
CV improvements or ten independent corpus replications.

The full crossing reveals an interaction worth presenting: CWS mean MCC is
0.9222 for delta but 0.6604 for union; parity is 0.9300 versus 0.7975. CWS's
poor overall marginal is driven much more by union mode than by delta mode.
Calling CWS universally weak would miss that distinction.

Radius is task-dependent. With WL/1024/parity/delta fixed, kNN TPL B is
0.7267 at radius 1 versus 0.8224 at radius 2. Mechanistic L2 favours radius 1
(0.9776 versus 0.9658). This helps explain why one default need not win every
endpoint. It does not authorize retuning the main benchmark on its test scores.

The clustering ablation corroborates delta's benefit. On the same three seeds,
delta versus union AMI is 0.7161/0.6439 (Schneider B), 0.4415/0.2889 (USPTO B),
0.7003/0.5633 (SynTemp R0), and 0.6090/0.3033 (Mechanistic L2).

## Recommended order

| Priority | Analysis | What it would establish | Inputs / compute |
|---|---|---|---|
| 1 | Factorial matched contrasts and interactions | Which design choices matter, and how the effect varies by chemistry and model | kNN complete now; other models after completion; no new fits |
| 1 | AMI/ARI, completeness, fragmentation, and hierarchy profiles | Whether higher agreement corresponds to useful cluster structure and at what chemical resolution | Existing JSON scores; no new fits |
| 1 | Signed delta vs unsigned delta support vs union | Separates background cancellation from direction information | One added unsigned variant, same tokenizer/sketch/bit budget/agent policy; start with the four existing ablation endpoints and three seeds |
| 1 | Spectator, reversal, and serialization perturbations | Tests concrete representation properties rather than label agreement alone | Small fixed sample, all five label-free fingerprints; fingerprint computation only |
| 2 | Reaction-centre localization and representative retrieval examples | Connects changed tokens/neighbours to chemically meaningful transformations across different scaffolds | Corrected full E8 localization is complete; remaining work is only to render representative chemical panels |
| 2 | Valid-row, density, and normalization controls | Checks whether geometry conclusions depend on failed encodings or vector preprocessing | Existing feature audit first; targeted common-valid/L2-normalized reruns only |
| 2 | Storage and encoding throughput | Quantifies the practical compactness/cost tradeoff | Fresh fixed-hardware timing pilot, separate from training |
| 3 | Fingerprint hash-seed and stereo sensitivity | Bounds collision sensitivity and chemistry omitted by the default representation | Small fixed subset; current fp_seed is always 1 |

For the unsigned control, use surviving delta-token support and a fixed token
namespace/hash protocol that differs only in retaining the sign. Match density
diagnostics and repeat fingerprint hash seeds where feasible. Delta-to-union
also changes token set size, so the current ablation alone is not proof that
direction encoding caused the gain.

For robustness, add identical disconnected spectators to both reaction sides,
including spectators whose motifs overlap the transformed molecules. Count
delta cancels the shared contribution algebraically. Set symmetric difference
can lose changed motifs when those motifs are introduced on both sides. Use
this as a controlled computational representation test, not as a claim that
an arbitrary added reagent would leave an experimental reaction unchanged.
Normalize middle-field agent treatment across methods for this control.

For interpretability, replay one fixed clustering seed on two representative
endpoints and save row IDs plus assignments for all five fingerprints (ten
fits). Choose exemplars by a documented rule and include failures. Current
clustering runners save scores but no per-reaction assignments, so examples,
within-cluster scaffold diversity, and assignment agreement cannot be recovered
from their JSON alone. Existing hierarchy scores do not require a replay.

## Evidence that should accompany the strengths

- SynRFP has the highest mean **ARI as well as AMI on all seven small KMeans
  endpoints**. Schneider B ARI is 0.423 versus DRFP 0.211; Mechanistic L2 is
  0.175 versus 0.110.
- Large ECReact L1 differs by metric: SynRFP has AMI 0.341 but ARI 0.192;
  RDKit difference has AMI 0.314 and ARI 0.239. TPL UB also favours DRFP in
  both MiniBatchKMeans metrics. Avoid representation-wide dominance claims.
- Schneider UB SynRFP-minus-DRFP AMI is only +0.0042 and positive for 6/10
  paired initialization seeds. ECReact L1 is +0.0093 and positive for 5/10.
  The 20 scaffold subsamples favour SynRFP over DRFP in 20/20 samples for each
  of the four representative endpoints, but those overlapping samples are
  sensitivity checks, not independent population replications.
- The same SynTemp R0 KMeans partitions have SynRFP AMI 0.702/0.750/0.555
  against R0/R1/R2, versus DRFP 0.594/0.661/0.539. The advantage narrows at
  fine chemical resolution. Keep identical partitions/rows for this profile.
- Frozen-threshold BitBIRCH Mech L1 has SynRFP AMI 0.159 but ARI 0.00204:
  14,433 clusters for 31,673 reactions, 32.4% singleton reactions,
  homogeneity 0.978, completeness 0.176. Plot resolution and fragmentation
  rather than presenting AMI alone.

## Reporting and execution issues to resolve

1. **Fixed in the report code:** `09_report/report.py` now uses paired
   per-seed t intervals and Holm-adjusted tests for Track-A contrasts. Its 25
   codes are five repeats times five dependent folds, so they are treated as
   paired observations rather than independent groups. The `=` legend remains
   “inconclusive” unless an equivalence criterion is actually tested. See the official
   [scikit-learn correlated-CV comparison example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html)
   and [statsmodels paired equivalence test](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttost_paired.html).
   The standard corrected-CV approximation also needs justification for the
   present grouped, rotating-validation design; it is not an automatic fix.
2. Fourteen classification endpoints include repeated views of six named
   data families. Add family-weighted summaries and leave-one-family-out
   sensitivity; shared patent provenance may introduce further dependence.
3. Keep RXNFP-MLM in label-free comparisons and visibly label RXNFP-FT/TPL-1K
   as supervised references. The current report includes all saved
   fingerprints, including TPL-1K Track A extras, in its statistics.
4. Factorial ANOVA averages seeds before fitting and its marginal SD combines
   datasets/configurations. It is not five-split uncertainty. Retain matched
   dataset/split contrasts for interpretation and clearly label random-split
   factorial results versus scaffold-CV benchmark results.
5. The scheduled shell's failed inventory check did not stop its following
   ANOVA command. **Fixed in this audit:** `04_factorial/anova.py` now validates
   the expected dataset/model/fingerprint/seed matrix itself, rejects
   duplicates and nonfinite metrics, and exits before writing final outputs
   if incomplete. Real incomplete data and synthetic valid/invalid cases
   were checked.
6. **Still pending:** deferred GPU non-TPL XGBoost sweeps overlap the independent
   CPU XGBoost queues. The workers have an existence check but no shared
   per-cell lock. Assign disjoint ownership before the deferred GPU stage
   begins, and join the CPU stage before final analysis. No live jobs were
   rescheduled in this inspection.

## Existing analysis scripts needing corrections before execution

- Retrieval E6, robustness E7, and reaction-centre E8 have no completed result
  artifacts. Their presence is not evidence that the studies were run.
- E6 omits RXNFP-MLM and includes supervised RXNFP in its configured panel.
  `metrics.retrieval` normalizes AP by observed top-k hits, not by available
  relevant database records. Correct its definition or use P@k/Recall@k.
- E7 retains failed-encoding zero vectors; these can inflate equality-based
  invariance. Audit `Features.ok` and report valid-row coverage.
- E8 now defaults to radius 2 and `original_reactions`, matching the main
  benchmark, and reports occurrence-level localization in addition to the
  historical token-level metric. Describe it as pre-sketch evidence, not proof
  that every hashed bit uniquely identifies a reaction-centre atom. The full
  corrected run remains pending.
- Historical parallel-run elapsed times are unsuitable for fair throughput
  rankings: thread budgets and concurrent machine load differ.

## Representation semantics

CW is **consistent weighted sampling (CWS)**, not CountSketch. The clustering
summary's name was corrected in this audit. Default parity consumes one copy
of each signed nonzero delta token; it discards magnitude after subtraction.
Only the pre-sketch count delta is antisymmetric. The final binary vector is
direction-sensitive, with zero-delta and finite-hash collision limitations.
Default graph attributes omit stereochemistry. These are useful limits to
measure and explain alongside the strengths.

## Suggested paper additions

1. A design-effects figure: delta/union, tokenizer-radius, and sketch-mode,
   with effects broken down by dataset family.
2. A chemical explanation figure: localized changed motifs, spectator/reversal
   controls, and cross-scaffold retrieved reactions plus counterexamples.
3. A reliability/utility supplement: AMI and ARI, fragmentation versus cluster
   count, valid-feature coverage, matched preprocessing, and measured cost.

Keep the existing default fingerprint frozen. Analyses and controls selected
after viewing these results are exploratory; do not replace the main table
with the best new configuration.

# SynRFP analysis expansion decision

This decision is based on the completed clustering studies, the complete kNN
factorial, the current partial MLP/XGBoost inventories, and the representation
audits. It is an analysis plan, not a replacement for the frozen primary
benchmark.

## What is already strong enough to present

- **Representation mechanism:** signed delta, unchanged-spectator, and
  reversal controls are complete. The delta-versus-union effect is positive
  across all ten kNN endpoints and all four existing clustering ablation
  endpoints. The unsigned control shows that context cancellation and
  direction sensitivity can be discussed separately for the toy controls.
- **Clustering performance:** ten K-means seeds on seven small endpoints and
  ten MiniBatch seeds on five large endpoints are complete. AMI and ARI are
  both reported, with the large-data exceptions retained. Twenty paired
  scaffold-group subsamples provide stability evidence.
- **Cross-task consistency:** the saved comparison shows SynRFP beats DRFP on
  every matched small clustering endpoint and every matched classification
  model comparison currently available. This is descriptive evidence, not a
  pooled significance test.
- **Protocol and data quality:** split/seed metadata, feature-validity rates,
  family sensitivity, and artifact integrity audits pass.

## Highest-value additions

1. **Save cluster assignments and show chemical examples.** This is now
   complete for the mechanistic L2 endpoint and Schneider B at seed 0. The
   replay saved reaction IDs, fingerprint assignments, and validity flags
   under `results/assignments/`; the deterministic selector wrote
   `results/cluster_examples_mech2.csv` and
   `results/cluster_examples_schneider_b.csv`. The remaining publication step
   is to render a small panel from these rows and add within-cluster scaffold
   diversity and representative reaction changes.
2. **Add cluster-quality diagnostics to the main figure.** Report AMI and ARI
   together with completeness, homogeneity, cluster count, and singleton
   fraction. The Mech L1 BitBIRCH result demonstrates why AMI alone can hide
   fragmentation; the large ECReact ARI reversal is a useful limitation.
3. **Run a shared-valid-row and normalized-distance sensitivity check.** This
   is now complete for Schneider B. On the same 49,975 valid rows, SynRFP
   scored AMI/ARI 0.731/0.462 without normalization and 0.728/0.464 after
   L2 normalization; DRFP scored 0.645/0.227 and 0.682/0.330, respectively.
   SynRFP remains ahead under both choices, so the result is not explained by
   its 25 failed encodings or vector scale.
4. **Add hash-seed and stereo sensitivity controls.** The default hash seed is
   fixed at one and the graph attributes omit stereochemistry. The hash-seed
   portion is now complete on 2,000 Schneider B reactions: pairwise seeds
   changed about 58 of 1,024 bits on average (5.7%; all rows remained valid).
   A stereo-only perturbation remains the next optional control.

## What should wait or remain out of scope

- Do not launch another full clustering factorial or more optimizer seeds
  before the pending XGBoost and MLP matrices finish; existing ten-seed
  summaries already quantify initialization variability.
- Do not pool Track-A scaffold CV, Track-B fixed splits, TPL-1K, and
  RXNFP-MLM into one significance claim. Keep supervised/TPL-1K references
  visibly separate from label-free comparisons.
- Do not call a non-significant comparison “equivalent.” Use “inconclusive”
  unless an equivalence margin has been tested.
- Do not claim universal superiority: SynRFP loses selected large-data AMI or
  ARI comparisons, and its XGBoost-versus-RDKit-difference margin is currently
  task-dependent.

## Execution order

1. Let the active TPL XGBoost workers and the deferred GPU ownership handoff
   complete; then finish factorial XGBoost and MLP.
2. Run the completeness-gated factorial contrasts and family-weighted summary.
3. Run the compact assignment replay and shared-valid-row/normalization
   sensitivity on idle resources.
4. Regenerate `run_audit.sh`, `artifact_integrity.py`, and the final report,
   then freeze the manuscript tables.

The current queue is therefore sufficient for the headline benchmark. The
additional work should explain *why* SynRFP helps and where it fails, rather
than search for another best configuration.

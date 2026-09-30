# Clustering follow-up and presentation plan

Prepared 2026-09-24 from the completed standard benchmark. This is a proposed
protocol; no new clustering jobs were launched while preparing it.

## Recommendation

Extend K-means to 10 external seeds (0–9), retaining `n_init=10`, all current
fingerprints, and the existing primary SynRFP configuration. Prioritize resolution
sensitivity, sampling stability, and chemical interpretation alongside this
extension. Additional seeds assess optimizer variability on the same data; they
do not create independent chemical datasets.

| Total external seeds | Total K-means fits, 7 endpoints × 5 fingerprints | Additional fits | Approximate additional serial time |
|---|---:|---:|---:|
| 5 | 175 | 91 | approximately 2–3 hours |
| 10 | 350 | 266 | approximately 7–9 hours |

Time estimates use the intervals between the existing consecutive seed outputs,
summed across dataset/fingerprint cells. They include the previous compute load
and are rough scheduling estimates, not guaranteed ETAs. Each fit uses 10 internal
initializations and returns the lowest-inertia solution. Do not count these
internal initializations as separate benchmark observations.

## What the present evidence supports

- SynRFP has the highest mean K-means AMI in all seven standard endpoints.
  The difference from DRFP ranges from about 0.009 on Schneider UB to 0.134
  on Mech level 2. The Schneider UB result is close and needs uncertainty.
- SynRFP also has the highest frozen-threshold BitBIRCH AMI in these endpoints.
  This is not a statement that it wins every metric: for SynTemp R0, DRFP ARI
  is 0.01684 versus SynRFP 0.01091; for Mech level 1, RDKit structural ARI is
  0.01229 versus SynRFP 0.00204.
- At the frozen threshold, SynRFP produces 14,433 clusters from 31,673 Mech
  level-1 reactions (nine reference classes), with 32.4% of reactions in
  singleton clusters. This is substantial fragmentation despite leading AMI.
- Balanced/unbalanced pairs and hierarchy endpoints are dependent views.
  Seven endpoints are not seven independent corpora; shared patent provenance
  can create additional dependence across named sources.
- The BitBIRCH thresholds were chosen using USPTO50k B labels. That row is
  calibration performance, and USPTO50k UB is a related view rather than a
  fully independent validation corpus. The frozen comparison remains useful
  but should be called label-calibrated.

## Prioritized experiments

### 1. Initialization and sampling stability

Run the 196 missing K-means fits for seeds 3–9. Retain the existing scores for
seeds 0–2. Save reaction IDs, assignments, inertia, iterations, elapsed fit and
scoring time, preprocessing, software versions, and feature/config hashes for
new runs. Existing JSON files contain scores but not per-reaction assignments;
any analysis requiring old assignments will need a separate replay.

Report AMI as the existing primary metric, ARI as a prominent secondary metric,
and mean/SD across seeds as initialization sensitivity. A seed-only confidence
interval, if shown, concerns expected performance over initialization on this
fixed dataset, not population generalization.

For sampling robustness, predeclare 20 shared 80% subsamples of reaction groups,
recluster every fingerprint on identical selected records, and report the
distribution of paired AMI/ARI differences plus assignment agreement on shared
records. Group by source/patent when reliable IDs exist, or product scaffold as
an explicitly documented proxy. Fix K to the full endpoint's class count for
this experiment. Treat the overlapping subsets as sensitivity experiments, not
20 independent datasets. Start with one endpoint per named source family
(Schneider B, USPTO50k B, SynTemp R0, Mech level 2), chosen in advance.

If confidence intervals on evaluation scores are needed, separately bootstrap
the reaction groups with replacement using shared draws across fingerprints
(target 1,000 draws after a timing pilot). For an economical analysis, use saved
partitions from a predeclared seed and explicitly label the intervals as
conditional on those fitted partitions. Full pipeline intervals would require
refitting during resampling and a much larger compute budget. Never label five
80% subsamples a bootstrap CI, as the old exp.md shorthand suggests.

Use per-endpoint paired effect sizes, not pooled seed-level significance tests
across all endpoints. Any formal multiple-comparison testing needs a documented
resampling model and correction (e.g. Holm); exploratory follow-ups selected
after viewing these results must remain identified as exploratory.

### 2. Clustering resolution and BitBIRCH order sensitivity

Plot full AMI and ARI curves against both threshold and number of clusters, with
singleton fraction and homogeneity/completeness. Compare overlapping cluster-count
ranges across fingerprints without choosing operating points from label scores.
Different fingerprints need not have comparable Tanimoto distributions at the
same numerical threshold.

Extend the shared BitBIRCH sweep downward to 0.05, 0.10, 0.15, 0.20, and 0.25:
several current optimum scores are at the lowest tested threshold, 0.30. Keep the
old frozen-threshold results as the original benchmark. Treat the extended sweep
as a disclosed sensitivity analysis; do not silently replace the original
threshold with a new best-per-dataset setting.

At a small, predeclared set of operating points, test 10 shared random input
orders per dataset for all binary fingerprints and undo the permutation before
scoring. Assess whether the geometry comparison survives ordering changes.

For K-means, add K/C = 0.5 and 2 around the existing K/C = 1, with three seeds
and all five fingerprints on the four representative endpoints above. This
adds 120 fits. Present full curves rather than selecting the best K per
fingerprint. Round to valid distinct integer K values. K/C is a label-informed
resolution diagnostic, not automatic unsupervised selection of K.

### 3. Hierarchical chemistry and interpretable examples

Use existing partitions scored against SynTemp R0/R1/R2 and Mech levels 1/2 to
show whether useful agreement persists at multiple chemical resolutions. Use
the same reaction rows when comparing hierarchy levels; current endpoint filters
may differ. Begin with the existing threshold profiles, requiring no refits.

For chemical examples, predeclare a sampling rule covering large/medium clusters
and a fixed random sample, then show central reactions and diverse scaffolds.
Report whether reactions of the same mechanistic class remain grouped across
different product scaffolds. Include failure examples and report the aggregate
statistic before selecting illustrations. Do not infer that varying BitBIRCH
thresholds yields a nested tree unless assignment nesting is verified.

### 4. Explain the representation advantage with limited controls

Keep the main default fingerprint frozen. In a separate, predeclared ablation
on the representative endpoints, test delta versus union, WL versus Morgan,
and SynRFP at 1,024 versus 2,048 bits (DRFP currently has 2,048 bits). Change
one factor at a time. Use existing feature caches where available and apply the
same clustering protocol; avoid replacing the main default with the best
observed ablation.

Audit valid-feature masks, all-zero vectors, bit density, duplicates, and shared
row ordering. K-means currently uses raw vectors; supplement this with a fixed
L2-normalization sensitivity check for all fingerprints if conclusions are
presented as representation-wide. Include RDKit difference as a label-free
K-means/HDBSCAN baseline; it is not binary and does not belong in BitBIRCH.
Label-trained RXNFP checkpoints may be displayed
as separate supervised references, not included in label-free rankings.

### 5. Large-data utility and efficiency

Use the existing TPL and ECReact BitBIRCH extension to test generality outside
the small standard panel. Show mixed outcomes as well as wins. Collect elapsed
clustering time, peak memory, sample size, and fingerprint storage with identical
hardware/thread budgets; report feature-generation cost separately. Test a few
fixed sample sizes before any expensive rerun. EC hierarchy levels are multiple
scores of related partitions, not independent scalability experiments.

## Suggested paper figures

1. K-means AMI/ARI comparisons and SynRFP-minus-baseline differences; distinguish
   initialization variation from conditional group-bootstrap uncertainty.
2. BitBIRCH AMI/ARI versus cluster count, with fragmentation/coverage panels.
3. Hierarchy profiles and objectively selected chemical cluster examples.
4. A compact TPL/ECReact quality-versus-time/memory panel if measurements support
   a useful efficiency claim.

Keep HDBSCAN's per-dataset best settings in the supplement as label-tuned
diagnostics. Any main-table HDBSCAN setting needs an explicit frozen selection
rule and full-data scores, noise coverage, and a check of noise treatment.
The current implementation converts noise points to singletons; coverage must
therefore accompany its homogeneity and cluster counts.

The defensible current statement is that SynRFP gives stronger label agreement
under the tested K-means and calibrated BitBIRCH protocols. The follow-up tests
whether that advantage is stable, persists across resolution, and corresponds
to useful chemical organization.

## Methodological references

- [scikit-learn KMeans: n_init, inertia selection, random_state](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Vinh, Epps and Bailey (2010): clustering measures and correction for chance](https://jmlr.org/papers/v11/vinh10a.html)
- [Levine and Domany (2000): cluster validation under resampling](https://arxiv.org/abs/physics/0005046)

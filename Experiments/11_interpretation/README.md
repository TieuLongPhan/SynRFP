# SynRFP interpretation artifacts

Run the completed-result audit from the repository root with:

```bash
python Experiments/11_interpretation/inspect_existing.py
```

The script performs no model fits. It validates the complete 8,100-cell kNN
factorial, reads the completed clustering outputs, and regenerates the tables
and figures under `results/`.

For the classification-only audit, run:

```bash
python Experiments/11_interpretation/classification_audit.py
```

This writes label-free per-dataset summaries, matched coverage, equally
weighted family summaries, and MLP epoch metadata. Partial benchmark coverage
stays explicit and should be regenerated after the remaining TPL XGBoost cells
finish.

The cached-feature validity check is:

```bash
python Experiments/11_interpretation/feature_validity_audit.py
```

`artifact_integrity.py` checks the generated CSV schemas and numeric values and
writes `results/artifact_integrity.json`.

It records failed encodings, zero rows, dimensions, and sparsity for all five
label-free clustering representations without refitting clustering models.
The companion `results/feature_validity_failure_rates.pdf` is a compact
presentation figure for this quality-control check.

The completed E7 serialization/reversal perturbation control is in
`Experiments/07_robustness/results/perturb.csv`. It samples 2,000 reactions
from each of four datasets and records validity alongside identity and
similarity under randomized serialization and reaction reversal.

`classification_progress.py` writes `results/classification_progress.json`,
using the benchmark queue's expected-cell inventory. The current snapshot
separates the **106** remaining XGBoost benchmark cells into TPL UB (**49**)
and TPL B (**57**), so partial coverage is auditable.

`results/CURRENT_RESULTS.md` is a compact human-readable comparison of the
completed label-free classification records. It labels XGBoost rankings as
provisional until those TPL cells are complete.
`tpl_partial_analysis.py` produces `TPL_PARTIAL_ANALYSIS.md`, a live stress-case
summary from the completed TPL XGBoost log cells and renders
`results/tpl_partial_margins.png`; it is explicitly provisional.

The paper-facing interpretation is summarized in `PAPER_READY_SYNTHESIS.md`.
`applicability_analysis.py` writes `APPLICABILITY_ANALYSIS.md` and a JSON
summary of model-specific wins, losses, and clustering exceptions; it is a
read-only descriptive analysis of completed rows.
`plot_mechanism_schematic.py` renders the conceptual signed-difference figure
at `results/synrfp_mechanism_schematic.png` for the manuscript or slides.
Fixed-rule chemical examples are rendered in
`results/cluster_examples_mech2_panel.png` and
`results/cluster_examples_schneider_b_panel.png`; these panels are illustrative
replays and are not additional benchmark replicates.

`tpl_tail_eta.py` estimates the TPL tail from observed per-fingerprint fit
durations. The current conservative estimate is about **38.9 hours for TPL
UB** and **47.6 hours for TPL B**, with both workers running in parallel;
regenerate the JSON before quoting an ETA.

`clustering_stability_audit.py` summarizes the 20 matched 80%-subsample
K-means runs per endpoint and writes paired-difference tables plus an AMI
figure. It is a read-only analysis of completed clustering outputs.

`paired_inference.py` adds descriptive paired bootstrap 95% intervals and
exact two-sided sign tests for those clustering contrasts and the complete kNN
delta-versus-union contrast. Intervals are conditional on the observed paired
runs; they are not population-level confidence intervals across datasets.

`claim_audit.py` checks the headline AMI and kNN statements against the saved
tables and writes `results/CLAIM_AUDIT.md` plus a machine-readable JSON record.
The large-data claim intentionally records SynRFP's 4/5 result because DRFP
leads on TPL UB.

`representation_controls.py` records reversal and unchanged-spectator controls
for default SynRFP, unsigned-delta SynRFP, and DRFP encodings in
`results/representation_controls.json`.

All of these read-only audits can be regenerated with:

```bash
Experiments/11_interpretation/run_audit.sh
```

`FINAL_READINESS.md` records which analyses are complete, provisional, or
still queued at the time of the latest snapshot.

The main files are:

- `EXPLANATORY_RESULTS.md`: manuscript-ready summary of reaction-level replay,
  preprocessing sensitivity, hash-seed sensitivity, and limitations.
- `SYNRFP_EXPLANATION.md`: manuscript-ready mechanism, hierarchy, and caveat
  wording.
- `SYNTHESIS_RECOMMENDATION.md`: decision on which explanatory analyses are
  still worth adding and which should remain deferred.
- `results/delta_union_evidence.pdf`: paired classification and clustering
  delta-versus-union contrasts.
- `results/knn_factor_marginals.csv`: factorial main effects.
- `results/knn_mode_contrast_summary.csv`: endpoint-level paired contrasts.
- `results/clustering_hierarchy_ami.csv`: SynTemp and mechanistic hierarchy
  profiles.
- `results/clustering_ami_ari.csv`: AMI and ARI summaries.
- `results/benchmark_completed_summary_partial.csv`: interim benchmark summary
  with explicit completed-cell counts. Regenerate it after the remaining TPL
  XGBoost cells finish; it is not a final ranking while counts differ.

The interpretation package keeps the default SynRFP configuration frozen. Any
new controls selected after viewing these results should be labelled
exploratory and should not replace the primary benchmark.

`cross_task_audit.py` compares the frozen SynRFP-versus-baseline margins on
the same endpoints for classification and small-endpoint K-means. It is a
descriptive consistency check, not a pooled significance test.

`cluster_assignment_replay.py` saves fixed-seed reaction-level K-means
assignments for selected endpoints. After replay, `select_cluster_examples.py`
selects the largest clusters and one deterministic modal-label reaction per
cluster, while retaining cluster size and purity in
`results/cluster_examples.csv`. These examples are explanatory evidence and
must be shown with their validity and fragmentation context.

`summarize_replays.py` combines replay metadata into
`results/cluster_replay_summary.csv` and a compact AMI/ARI figure for the
mechanistic and Schneider examples.

`hash_seed_sensitivity.py` measures real-reaction bit changes across fixed
SynRFP hash seeds; the current Schneider B result is in
`results/hash_seed_sensitivity_schneider_b.csv`.

`reaction_center_protocol_audit.py` is a small read-only pilot that compares
the primary original-reaction/radius-2 convention with the historical E8
updated-reaction/radius-3 convention. It does not replace the full
reaction-centre analysis.

`cost_summary.py` converts the cached feature-shape audit into a compact
payload, validity, and sparsity comparison in
`results/feature_cost_summary.csv` and
`results/feature_cost_by_fingerprint.csv`.

`plot_reaction_center.py` renders the full corrected E8 result as
`results/reaction_center_localization.png`/`.pdf`, combining the
reaction-level distribution with class-level localization summaries. The
underlying class table is `results/reaction_center_class_summary.csv`.

`protocol_audit.py` records the fixed hash seed, grouped CV seed scheme,
Track B fallback seed, and factorial cell count, and checks saved result
metadata against the declared seeds.

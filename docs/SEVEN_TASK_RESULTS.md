# Seven-task factorial and fingerprint results

The completed construction factorial contains **20,448 logical result cells**
across 72 SynRFP configurations. The completed main fingerprint comparison
contains **8,238 cells** across the same seven application tasks.

| Task | Factorial cells | Main benchmark cells | Main comparison summary |
| --- | ---: | ---: | --- |
| Classification | 5,040 | 4,200 | [Results](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/classification_summary.csv) |
| Clustering | 4,320 | 1,440 | [Results](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/clustering_summary.csv) |
| Yield prediction | 4,608 | 384 | [Results](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/yield_summary.csv) |
| Barrier prediction | 3,240 | 1,944 | [Results](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/barrier_summary.csv) |
| Feasibility | 360 | 30 | [Results](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/feasibility_summary.csv) |
| Reagent prediction | 720 | 60 | [Results](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/reagent_summary.csv) |
| Condition prediction | 2,160 | 180 | [Results](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/condition_summary.csv) |
| **Total** | **20,448** | **8,238** | |

## Factorial tables and analyses

- [Classification: cells, configuration summaries and interaction effects](../Experiments/22_all_endpoint_factorial/publication_metadata/all_endpoint_factorial_full_v1/classification/).
- [Clustering: cells, interaction effects and initialization stability](../Experiments/22_all_endpoint_factorial/publication_metadata/all_endpoint_factorial_full_v1/clustering/).
- [Yield: cells, configuration summaries and interaction effects](../Experiments/22_all_endpoint_factorial/publication_metadata/all_endpoint_factorial_full_v1/yield/).
- [Barrier, feasibility, reagent and condition: combined four-task tables](../Experiments/30_cross_task_factorial/publication_metadata/cross_task_factorial_v1/). Filter the `task` column for an individual task. These four tasks contribute 6,480 cells.
- [Seven-task candidate analysis, endpoint summaries and family-balanced rankings](../Experiments/30_cross_task_factorial/publication_metadata/cross_task_factorial_v1/common_representative_analysis/report.md).
- [Matched combination-mode effects and bounded token analysis](../Experiments/31_ncs_evidence/publication_metadata/ncs_evidence_v1/report.md).
- [All 72 configurations and the exploratory three-candidate shortlist](../Experiments/32_shared_preset_review/publication_metadata/shared_preset_review_v1/report.md).
- [Frozen shared preset and decision provenance](../Experiments/33_shared_preset_decision/publication_metadata/shared_preset_decision_v1/report.md).

Classification, clustering and yield include interaction orders 1–4. The four
study30 tasks include the complete 72-configuration grid and paired effects
against the fixed WL/r2/parity/delta control. Matched mode effects and common
rankings are supplied separately. A full higher-order interaction decomposition
for those four tasks is outside these completed exports.

The grid crosses three tokenizers (WL, Morgan and Path), two radii (2 and 3),
three sketches (parity, b-bit MinHash and controlled CW), and four combination
modes (delta, union, delta_union and original roles), at 2,048 total bits and
fingerprint seed 1. The original roles representation allocates 1,024 bits to
signed change and 1,024 to shared token context. The retired joint-role variant
is excluded. [Construction and evaluation protocol](../Experiments/30_cross_task_factorial/PROTOCOL.md).

## Main fingerprint comparison

[Main report](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/report.md)
and [all per-cell and summary tables](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/).
Each task compares six fingerprints: SynRFP, DRFP, RDKit structural, RDKit
difference, RXNFP-MLM and cRXNFP. Seven denotes the number of tasks.

The shared SynRFP preset is WL / radius 2 / parity / signed delta / 2,048 bits /
fingerprint seed 1. Its completed benchmark uses the native task populations,
partitions and learner settings documented in the
[main protocol](../Experiments/34_final_comparison/PROTOCOL.md).

## Interpretation and artifact scope

Configuration selection is exploratory because earlier test results were
viewed. Fold, optimizer-seed and clustering-initialization variation have
different meanings; preserve the original protocols and aggregate them as
documented. Classification factorials use one grouped five-fold evaluation;
the main comparison also contains repeated 5×5 grouped CV and kNN. Clustering
factorials use five initializations with a row-derived K; main clustering uses
the declared known class count and ten initializations. Yield retains its 64
native partitions. Logical cells include verified reuse and dependent evaluations.

For barrier results, keep target units and endpoint effects explicit. The
historical study31 task-level raw RMSE-effect averages do not establish a
unit-normalized effect across sources whose units remain incompletely verified.
Use its endpoint effects or the study32 normalized rank/regret diagnostics for
cross-source interpretation. Other task metrics are also kept separate.

Historical reports describe their completion date: study33's pre-benchmark
freeze and study31's incomplete learned-reference inventory remain unchanged.
The current main fingerprint completion authority is study34. The ongoing
full-main SynCluster extension and unfinished learned-model references have
separate completion boundaries and contribute no cells to the totals above.

This release contains protocols, analysis sources, scalar result tables and
completion/hash metadata. Model weights, raw datasets, feature caches and
prediction/membership arrays remain local. Existing scientific reports and
receipts are preserved. [Machine-readable publication inventory](SEVEN_TASK_RESULTS.json).

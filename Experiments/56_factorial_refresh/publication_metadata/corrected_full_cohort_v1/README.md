# Corrected full-cohort seven-task factorial

The primary report closed on 26 October 2026. It contains 20,448 cells across
72 configurations, including 3,960 corrected full-cohort replacements and
16,488 unchanged cells. The declared graph-native Morgan ECFP count profile
is retained; historical pre-correction Morgan values are not relabelled.

| Task | Primary cells |
| --- | ---: |
| Classification, MLP | 5,040 |
| Clustering, class-count-blind | 4,320 |
| Yield | 4,608 |
| Barrier | 3,240 |
| Feasibility | 360 |
| Reagent | 720 |
| Condition | 2,160 |

`cells.csv` contains normalized analysis records; `canonical_cells.csv` retains
native metrics and result origins. The remaining tables provide endpoint and
family summaries, ranks and regrets, matched contrasts, and interaction
decompositions through order four. `configurations.json` describes all 72
constructions. Selection remains exploratory; new seeds on viewed data do not
establish external confirmation. Barrier units and task metrics are not pooled.
The existing yield factorial does not include a separate USPTO configuration sweep.

The separate `knn_aux/` report closed on 28 October 2026 at 05:10 UTC. It
summarizes the 5,040-cell fixed-k=5 Tanimoto classification factorial after all
360 TPL coverage replacements. Its six summary/interaction CSVs are exported;
its full result and prediction provenance remains local. These learner results
are not substituted into the primary MLP rankings.

Completion metadata binds the native local receipts and the exported scalar
hashes. Export verification checks the scalar hashes and complete row counts;
it does not claim a new replay of every saved model. Weights, raw datasets,
features and prediction arrays remain local. Historical releases are preserved.

The [completed main comparison](../../../65_default_optimal/publication_metadata/default_optimal_v1/README.md)
contains both shared-default and task-optimal SynRFP rows for all seven tasks.

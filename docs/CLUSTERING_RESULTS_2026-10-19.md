# Completed clustering results

Publication snapshot: 19 October 2026. These are completed analyses with
different populations and questions; their scores should not be pooled.

| Study | Completed scope | Results |
| --- | --- | --- |
| Main fingerprint comparison | 1,440 cells: 12 endpoints, six fingerprints, ten initializations, two populations; known class count | [Summary](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/clustering_summary.csv), [cells](../Experiments/34_final_comparison/publication_metadata/final_comparison_v1/clustering_cells.csv) |
| Full construction factorial | 4,320 cells: 72 configurations, 12 panels, five initializations; cluster count derived from row count | [Report](../Experiments/22_all_endpoint_factorial/publication_metadata/all_endpoint_factorial_full_v1/clustering/report.md), [configuration summary](../Experiments/22_all_endpoint_factorial/publication_metadata/all_endpoint_factorial_full_v1/clustering/configuration_summary.csv) |
| Native SynCluster reference | All 50,016 author-source rows audited; 43,381 common admitted rows; ten matched SynRFP fits | [Comparison](../Experiments/35_specialist_references/syncluster/publication_metadata/full_reference_v1/comparison.json), [cells](../Experiments/35_specialist_references/syncluster/publication_metadata/full_reference_v1/comparison_cells.csv) |
| Representative-selection utility | 280 selections and 560 fixed downstream fits: two endpoints, six fingerprints plus uniform random, four budgets, five selection seeds, two evaluator panels | [Report](../Experiments/37_ncs_clustering_utility/publication_metadata/ncs_clustering_utility_v2/report.md), [summary](../Experiments/37_ncs_clustering_utility/publication_metadata/ncs_clustering_utility_v2/heldout_summary.csv), [cells](../Experiments/37_ncs_clustering_utility/publication_metadata/ncs_clustering_utility_v2/heldout_cells.csv) |

The main table was already published in commit
`40b6bb581d46b3fde6ca71a84051486e313e1525`. This update adds the three other
completed result packages, including their existing completion receipts and
declared output hashes. The [publication manifest](CLUSTERING_PUBLICATION_2026-10-19.json)
binds the exact files in this update. Models, raw data, feature caches, cluster
assignment arrays and downstream prediction arrays remain local.

## Interpretation boundaries

- The main table uses known K. Its two populations and aggregate AMI/ARI remain
  separate. SynRFP is competitive; the table does not support a universal win.
- The factorial uses five clustering initializations, not five CV folds. It is
  exploratory construction analysis; its leading configuration does not replace
  the main table's frozen shared preset. The interaction and stability CSVs retain
  all configurations and panels.
- Native SynCluster has higher AMI on its author panel: raw SynCluster 0.476161
  versus matched SynRFP 0.422204 +/- 0.006402. Different representations,
  algorithms and resolutions make this an end-to-end reference comparison.
- Utility uses all 30,012 USPTO50K-B and 115,385 ECReact-L3 training candidates,
  with 10,002 and 35,011 grouped test rows. Selection receives no endpoint labels
  or known class count. The fixed DRFP and RXNFP-MLM downstream evaluators are
  separate panels. All six fingerprint comparators retain the same source rows.
- SynRFP utility improves mean MCC over uniform selection in 14/16
  dataset/evaluator/budget settings, but has the highest numerical mean among all
  selectors in only 1/16 settings. Five selection seeds share one outer test;
  their SD is descriptive. These are mixed utility results, not independent-fold
  significance, a mechanistic discovery claim or untouched confirmation.

The existing experiment05 archive remains historical. Two older untracked
exploratory exports (the earlier 1,080-cell factorial and the 48-checkpoint
BitBIRCH consolidation pilot) are outside this focused current-results update;
they are not being promoted to the current canonical comparison or deleted.
This publication adds no new scientific fits and does not affect running
specialist references or the SynRXN D-MPNN extension.
